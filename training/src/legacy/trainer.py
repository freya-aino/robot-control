import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from os import cpu_count, environ
from typing import Dict

import mlflow
import numpy as np
import torch as T
import torch.nn as nn
from mujoco import MjData
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.agents import PPOAgent
from src.common.datastructure import UnrollData
from src.common.environments import MujocoEnv
from src.common.experiments import Experiment



class MultiEnvTrainer:
    def __init__(
        self,
        environment: MujocoEnv,
        agent: PPOAgent,
        experiment: Experiment,
        copies: int,
        optimizer,
        device: T.device,
    ):
        self.optimizer = optimizer
        self.device = device
        self.proto_agent = deepcopy(agent.to(device))

        self.env_pool_unroll_inputs = [
            {
                "agent": deepcopy(agent.to("cpu")),
                "environment": deepcopy(environment),
                "experiment": deepcopy(experiment),
            }
            for _ in range(copies)
        ]

    @staticmethod
    def _unroll_step(
        agent: PPOAgent,
        environment: MujocoEnv,
        experiment: Experiment,
        observation: T.Tensor,
    ):
        # decide which action to take depending on the environment observation (robot state)
        logits, action = agent.get_action(observation=observation)

        # squeeze the batch dimension and the sequence dimension
        action = action.cpu().squeeze().to(device="cpu")
        logits = logits.cpu().squeeze().to(device="cpu")

        # take a step in the environment and get its return values like the local reward for taking that action.
        new_observation, mj_data = environment.step(action=action)
        rewards = experiment(mj_data=mj_data)
        scaled_reward_mean = T.Tensor([*rewards.values()]).mean()

        return {
            "observation": new_observation,
            "logits": logits,
            "action": action,
            "reward": scaled_reward_mean,
            "raw_rewards": rewards,
        }

    @staticmethod
    def _unroll(
        agent: PPOAgent,
        environment: MujocoEnv,
        experiment: Experiment,
        unroll_length: int,
        seed: int,
    ):
        """Return step data over multple unrolls."""

        agent.eval()

        unroll_data = UnrollData.initialize_empty(
            unroll_length=unroll_length,
            observation_size=environment.observation_space_size,
            action_size=environment.action_space_size,
        )

        # env warmup
        environment.reset(seed=seed)
        action = T.Tensor(environment.action_space.sample()).to(dtype=T.float32)
        last_observation, mj_data = environment.step(action=action)
        last_observation = last_observation.to(dtype=T.float32)

        mean_metrics = {
            **{r: 0.0 for r in experiment(mj_data=mj_data)},
            **{k: 0.0 for k in get_all_env_metrics(mj_data)},
        }
        for j in range(0, unroll_length):
            with T.no_grad():
                step_out = MultiEnvTrainer._unroll_step(
                    agent=agent,
                    environment=environment,
                    experiment=experiment,
                    observation=last_observation,
                )

            # log raw rewards
            for r in step_out["raw_rewards"]:
                mean_metrics[r] += step_out["raw_rewards"][r] / unroll_length

            # log observation and actions
            env_metrics = get_all_env_metrics(mj_data)
            for k in env_metrics:
                mean_metrics[k] += env_metrics[k] / unroll_length

            # track unroll data
            unroll_data.observation[j, :] = last_observation[:]
            unroll_data.logits[j, :] = step_out["logits"][:]
            unroll_data.actions[j, :] = step_out["action"][:]
            unroll_data.rewards[j, :] = step_out["reward"]

            # set last observation
            last_observation = step_out["observation"]

        return unroll_data, mean_metrics

    @staticmethod
    def _batch_unroll(
        agents: Dict[str, PPOAgent],
        environments: Dict[str, MujocoEnv],
        experiments: Dict[str, Experiment],
        unroll_length: int,
        seed: int,
    ):
        """Return step data over multple unrolls."""

        for k in agents:
            agents[k].eval()

        unroll_datas = {}
        last_observations = {}
        mj_datas = {}
        for k in environments:
            unroll_datas[k] = UnrollData.initialize_empty(
                unroll_length=unroll_length,
                observation_size=environments[k].observation_space_size,
                action_size=environments[k].action_space_size,
            )

            # env warmup
            environments[k].reset(seed=seed)
            action = T.Tensor(environments[k].action_space.sample()).to(dtype=T.float32)
            last_observation, mj_data = environments[k].step(action=action)

            last_observations[k] = last_observation.to(dtype=T.float32)
            mj_datas[k] = mj_data

        # mean_metrics = {
        #     **{r: 0.0 for r in experiment(mj_data=mj_data)},
        #     **{k: 0.0 for k in get_all_env_metrics(mj_data)},
        # }
        for j in range(0, unroll_length):
            with T.no_grad():
                futures = {}
                with ThreadPoolExecutor(max_workers=min(cpu_count(), 4)) as pool:
                    for k in environments:
                        f = pool.submit(
                            MultiEnvTrainer._unroll_step,
                            agent=agents[k],
                            environment=environments[k],
                            experiment=experiments[k],
                            observation=last_observations[k],
                        )
                        futures[f] = k

                    step_outs = {}
                    for fut in as_completed(futures):
                        k = futures[fut]
                        step_outs[k] = fut.result()

            # # log raw rewards
            # for r in step_out["raw_rewards"]:
            #     mean_metrics[r] += step_out["raw_rewards"][r] / unroll_length

            # # log observation and actions
            # env_metrics = get_all_env_metrics(mj_data)
            # for k in env_metrics:
            #     mean_metrics[k] += env_metrics[k] / unroll_length

            # track unroll data
            for k in environments:
                unroll_datas[k].observation[j, :] = last_observations[k][:]
                unroll_datas[k].logits[j, :] = step_outs[k]["logits"][:]
                unroll_datas[k].actions[j, :] = step_outs[k]["action"][:]
                unroll_datas[k].rewards[j, :] = step_outs[k]["reward"]

                # set last observation
                last_observations[k] = step_outs[k]["observation"]

        # return unroll_data, mean_metrics

    def train(
        self,
        epochs: int,
        train_batch_size: int,
        num_unrolls: int,
        unroll_length: int,
        minibatch_size: int,
        entropy_loss_scale: float,
        value_loss_scale: float,
        policy_loss_scale: float,
        seed: int,
        max_gradient_norm: float,
    ):
        assert unroll_length % minibatch_size == 0, (
            "unroll_length must be divisible by minibatch_size"
        )
        num_minibatches = unroll_length // minibatch_size

        assert (num_unrolls * num_minibatches) % train_batch_size == 0, (
            "(num_unrolls * num_minibatches) must be divisible by train_batch_size"
        )

        # pin torch threads
        T.set_num_threads(1)
        environ["OMP_NUM_THREADS"] = "1"

        # reset random seed for training
        np.random.seed(seed)
        T.random.manual_seed(seed)

        for e in tqdm(range(epochs), "training"):
            # --- UNROLL
            current_seed = random.randint(0, 10000000)

            #     dt = time.time()

            #     MultiEnvTrainer._batch_unroll(
            #         agents={
            #             f"{i}": self.env_pool_unroll_inputs[i]["agent"]
            #             for i in range(len(self.env_pool_unroll_inputs))
            #         },
            #         environments={
            #             f"{i}": self.env_pool_unroll_inputs[i]["environment"]
            #             for i in range(len(self.env_pool_unroll_inputs))
            #         },
            #         experiments={
            #             f"{i}": self.env_pool_unroll_inputs[i]["experiment"]
            #             for i in range(len(self.env_pool_unroll_inputs))
            #         },
            #         unroll_length=unroll_length,
            #         seed=current_seed,
            #     )

            #     print(
            #         f"avg time taken per unroll: {(time.time() - dt) / len(self.env_pool_unroll_inputs)}"
            #     )

            unrolls = []
            metrics = []
            unrolls_todo = num_unrolls
            with ThreadPoolExecutor(max_workers=min(cpu_count(), 4)) as pool:
                while unrolls_todo > 0:
                    print(f"found {unrolls_todo} unrolls to do")

                    dt = time.time()

                    futures = []
                    for ind in range(
                        min(len(self.env_pool_unroll_inputs), unrolls_todo)
                    ):
                        futures.append(
                            pool.submit(
                                MultiEnvTrainer._unroll,
                                **self.env_pool_unroll_inputs[ind],
                                unroll_length=unroll_length,
                                seed=current_seed,
                            )
                        )

                    print(f"waiting for {len(futures)} futures")

                    for future in as_completed(futures):
                        unroll, metrics_ = future.result()
                        unrolls.append(unroll)
                        metrics.append(metrics_)

                    print(
                        f"avg time for unroll {(time.time() - dt) / min(len(self.env_pool_unroll_inputs), unrolls_todo)}"
                    )

                    unrolls_todo = num_unrolls - len(unrolls)

        # for _ in range(num_unrolls):
        #     unroll, mean_metrics_ = self.unroll(unroll_length=unroll_length)
        #     unrolls.append(unroll)
        #     if len(mean_metrics) == 0:
        #         mean_metrics = {
        #             k: mean_metrics_[k] / num_unrolls for k in mean_metrics_
        #         }
        #     else:
        #         mean_metrics = {
        #             k: mean_metrics[k] + (mean_metrics_[k] / num_unrolls)
        #             for k in mean_metrics_
        #         }

        # for k in mean_metrics:
        #     mlflow.log_metric(k, mean_metrics[k], step=e)

        # # mlflow.log_metric(
        # #     "mean_time_per_unroll_step_ms",
        # #     (time.perf_counter() - time_to_unroll)
        # #     / (num_unrolls * unroll_length)
        # #     * 1000,
        # #     step=e,
        # # )

        # # --- CREATE DATASET

        # # convert the full sequences to sequence parts (minibatches)
        # multi_unroll_dataset = MultiUnrollDataset(
        #     unrolls=unrolls,
        #     minibatch_size=minibatch_size,
        #     num_minibatches=num_minibatches,
        #     minibatched=True,
        # )
        # multi_unroll_dataset.validate()

        # mlflow.log_metric(
        #     "mean_reward",
        #     multi_unroll_dataset.rewards.mean().detach().cpu().item(),
        #     step=e,
        # )

        # dataloader = DataLoader(
        #     multi_unroll_dataset,
        #     batch_size=train_batch_size,
        #     shuffle=True,
        #     pin_memory=True,
        # )

        # # --- TRAINING

        # self.agent.train()

        # loss_averages = {
        #     "policy_loss": 0,
        #     "value_loss": 0,
        #     "entropy_loss": 0,
        # }
        # for _, batch in enumerate(dataloader):
        #     observations = batch["observations"].to(self.device)
        #     logits = batch["logits"].to(self.device)
        #     actions = batch["actions"].to(self.device)
        #     rewards = batch["rewards"].to(self.device)

        #     losses = self.agent(
        #         observations=observations,
        #         logits=logits,
        #         actions=actions,
        #         rewards=rewards,
        #     )

        #     # logging
        #     loss_averages["policy_loss"] += (
        #         losses["policy_loss"].detach().cpu().item()
        #     )
        #     loss_averages["value_loss"] += (
        #         losses["value_loss"].detach().cpu().item()
        #     )
        #     loss_averages["entropy_loss"] += (
        #         losses["entropy_loss"].detach().cpu().item()
        #     )

        #     policy_loss = losses["policy_loss"] * policy_loss_scale
        #     value_loss = losses["value_loss"] * value_loss_scale
        #     entropy_loss = losses["entropy_loss"] * entropy_loss_scale

        #     assert ~policy_loss.isnan(), f"policy_loss is NaN"
        #     assert ~policy_loss.isinf(), f"policy_loss is Inf"
        #     assert ~value_loss.isnan(), f"value_loss is NaN"
        #     assert ~value_loss.isinf(), f"value_loss is Inf"
        #     assert ~entropy_loss.isnan(), f"entropy_loss is NaN"
        #     assert ~entropy_loss.isinf(), f"entropy_loss is Inf"

        #     # sum loss and backpropagate
        #     loss = policy_loss + value_loss + entropy_loss
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     nn.utils.clip_grad_norm_(
        #         self.agent.parameters(), max_norm=max_gradient_norm
        #     )
        #     self.optimizer.step()

        # for k in loss_averages:
        #     mlflow.log_metric(k, loss_averages[k], step=e)


class Trainer:
    def __init__(
        self,
        environment: MujocoEnv,
        experiment: Experiment,
        agent: PPOAgent,
        device: T.device,
        max_gradient_norm: float,
        seed: int,
        optimizer,
    ) -> None:
        # -- variables
        self.environment = environment
        self.experiment = experiment
        self.agent = agent.to(device)
        self.device = device
        self.seed = seed

        # -- set up and grad norm
        self.optimizer = optimizer
        self.max_gradient_norm = max_gradient_norm

    def unroll_step(self, observation: T.Tensor):
        # decide which action to take depending on the environment observation (robot state)
        logits, action = self.agent.get_action(observation=observation)

        # squeeze the batch dimension and the sequence dimension
        action = action.cpu().squeeze().to(device="cpu")
        logits = logits.cpu().squeeze().to(device="cpu")

        # take a step in the environment and get its return values like the local reward for taking that action.
        new_observation, mj_data = self.environment.step(action=action)
        rewards = self.experiment(mj_data=mj_data)
        scaled_reward_mean = T.Tensor([*rewards.values()]).mean()

        return {
            "observation": new_observation,
            "logits": logits,
            "action": action,
            "reward": scaled_reward_mean,
            "raw_rewards": rewards,
        }

    def unroll(self, unroll_length: int):
        """Return step data over multple unrolls."""

        unroll_data = UnrollData(
            unroll_length=unroll_length,
            observation_size=self.environment.observation_space_size,
            action_size=self.environment.action_space_size,
        )

        # env warmup
        _ = self.environment.reset(seed=None)
        action = T.Tensor(self.environment.action_space.sample()).to(dtype=T.float32)
        last_observation, mj_data = self.environment.step(action=action)
        last_observation = last_observation.to(dtype=T.float32)

        mean_metrics = {
            **{r: 0.0 for r in self.experiment(mj_data=mj_data)},
            **{k: 0.0 for k in get_all_env_metrics(mj_data)},
        }
        for j in range(0, unroll_length):
            step_out = self.unroll_step(
                observation=last_observation.to(device=self.device)
            )

            # log raw rewards
            for r in step_out["raw_rewards"]:
                mean_metrics[r] += step_out["raw_rewards"][r] / unroll_length

            # log observation and actions
            env_metrics = get_all_env_metrics(mj_data)
            for k in env_metrics:
                mean_metrics[k] += env_metrics[k] / unroll_length

            # track unroll data
            unroll_data.observation[j, :] = last_observation[:]
            unroll_data.logits[j, :] = step_out["logits"][:]
            unroll_data.actions[j, :] = step_out["action"][:]
            unroll_data.rewards[j, :] = step_out["reward"]

            # set last observation
            last_observation = step_out["observation"]

        return unroll_data, mean_metrics

    def train(
        self,
        epochs: int,
        train_batch_size: int,
        num_unrolls: int,
        unroll_length: int,
        minibatch_size: int,
        entropy_loss_scale: float,
        value_loss_scale: float,
        policy_loss_scale: float,
    ):
        assert unroll_length % minibatch_size == 0, (
            "unroll_length must be divisible by minibatch_size"
        )
        num_minibatches = unroll_length // minibatch_size

        assert (num_unrolls * num_minibatches) % train_batch_size == 0, (
            "(num_unrolls * num_minibatches) must be divisible by train_batch_size"
        )

        # reset random seed for training
        np.random.seed(self.seed)
        T.random.manual_seed(self.seed)

        for e in tqdm(range(epochs), "training"):
            # --- UNROLL

            # time_to_unroll = time.perf_counter()

            # Unroll a couple of times
            self.agent.eval()
            with T.no_grad():
                unrolls = []
                mean_metrics = {}
                for _ in range(num_unrolls):
                    unroll, mean_metrics_ = self.unroll(unroll_length=unroll_length)
                    unrolls.append(unroll)
                    if len(mean_metrics) == 0:
                        mean_metrics = {
                            k: mean_metrics_[k] / num_unrolls for k in mean_metrics_
                        }
                    else:
                        mean_metrics = {
                            k: mean_metrics[k] + (mean_metrics_[k] / num_unrolls)
                            for k in mean_metrics_
                        }

            for k in mean_metrics:
                mlflow.log_metric(k, mean_metrics[k], step=e)

            # mlflow.log_metric(
            #     "mean_time_per_unroll_step_ms",
            #     (time.perf_counter() - time_to_unroll)
            #     / (num_unrolls * unroll_length)
            #     * 1000,
            #     step=e,
            # )

            # --- CREATE DATASET

            # # convert the full sequences to sequence parts (minibatches)
            # multi_unroll_dataset = MultiUnrollDataset(
            #     unrolls=unrolls,
            #     minibatch_size=minibatch_size,
            #     num_minibatches=num_minibatches,
            #     minibatched=True,
            # )
            # multi_unroll_dataset.validate()

            # mlflow.log_metric(
            #     "mean_reward",
            #     multi_unroll_dataset.rewards.mean().detach().cpu().item(),
            #     step=e,
            # )

            # dataloader = DataLoader(
            #     multi_unroll_dataset,
            #     batch_size=train_batch_size,
            #     shuffle=True,
            #     pin_memory=True,
            # )

            # # --- TRAINING

            # self.agent.train()

            # loss_averages = {
            #     "policy_loss": 0,
            #     "value_loss": 0,
            #     "entropy_loss": 0,
            # }
            # for _, batch in enumerate(dataloader):
            #     observations = batch["observations"].to(self.device)
            #     logits = batch["logits"].to(self.device)
            #     actions = batch["actions"].to(self.device)
            #     rewards = batch["rewards"].to(self.device)

            #     losses = self.agent(
            #         observations=observations,
            #         logits=logits,
            #         actions=actions,
            #         rewards=rewards,
            #     )

            #     # logging
            #     loss_averages["policy_loss"] += (
            #         losses["policy_loss"].detach().cpu().item()
            #     )
            #     loss_averages["value_loss"] += (
            #         losses["value_loss"].detach().cpu().item()
            #     )
            #     loss_averages["entropy_loss"] += (
            #         losses["entropy_loss"].detach().cpu().item()
            #     )

            #     policy_loss = losses["policy_loss"] * policy_loss_scale
            #     value_loss = losses["value_loss"] * value_loss_scale
            #     entropy_loss = losses["entropy_loss"] * entropy_loss_scale

            #     assert ~policy_loss.isnan(), f"policy_loss is NaN"
            #     assert ~policy_loss.isinf(), f"policy_loss is Inf"
            #     assert ~value_loss.isnan(), f"value_loss is NaN"
            #     assert ~value_loss.isinf(), f"value_loss is Inf"
            #     assert ~entropy_loss.isnan(), f"entropy_loss is NaN"
            #     assert ~entropy_loss.isinf(), f"entropy_loss is Inf"

            #     # sum loss and backpropagate
            #     loss = policy_loss + value_loss + entropy_loss
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     nn.utils.clip_grad_norm_(
            #         self.agent.parameters(), max_norm=self.max_gradient_norm
            #     )
            #     self.optimizer.step()

            # for k in loss_averages:
            #     mlflow.log_metric(k, loss_averages[k], step=e)
