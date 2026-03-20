from os import environ, unsetenv

# set xla flags for some GPUs
xla_flags = environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
environ["XLA_FLAGS"] = xla_flags
environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

# set render variable
unsetenv("WAYLAND_DISPLAY")
environ["DISPLAY"] = ":0"
environ["MESA_BACKEND"] = "glx"
environ["GLFW_LIBDECOR"] = "0"
# environ["MUJOCO_GL"] = "egl"

# -----------------------------------------------


import logging
import time
import random
import warnings
import hydra
import jax
import mlflow
import numpy as np
import torch as T
import torch.nn.functional as F
import tqdm
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten
from torch import optim
from sys import platform

from src.common.agents import PPOAgent, PPOAgentTorcRL
from src.common.environments import MujocoMjxEnv
from src.common.datastructure import UnrollData
from src.common.experiments import MjxExperiment, Go2WalkingExperiment
from src.common.util import get_azureml_mlflow_tracking_uri


# -----------------------------------------------

def reset_global_seed(seed: int):
    # random.seed(seed)
    T.random.manual_seed(seed)
    np.random.seed(seed)


def unroll(
    agent: T.nn.Module,
    environment: MujocoMjxEnv,
    experiment: MjxExperiment,
    unroll_data: UnrollData,
):
    
    agent = agent.eval()
    
    with T.no_grad():

        # reset environment
        jax_seed = T.randint(low=0, high=2 ** 32, size=[1], dtype=T.uint32).item()
        observation = environment.reset(seed=jax_seed)

        # unroll step
        for i in tqdm.tqdm(range(unroll_data.observations.shape[1]), desc="Unrolling", position=1, leave=False):

            new_observation, action, logits = environment.unroll_step(
                agent=agent,
                observation=observation,
            )

            reward = experiment.calculate_reward(environment.mjx_data)

            unroll_data.update(
                unroll_step=i,
                observation=observation,
                action=action,
                logits=logits,
                reward=reward,
            )

            observation = new_observation

    return unroll_data

def train(
    agent: T.nn.Module,
    unroll_data: UnrollData,
    epochs: int,
    optimizer: T.optim.Optimizer,
    lr_scheduler: T.optim.lr_scheduler,
    grad_clip_norm: float,
    batch: int
):

    assert unroll_data.validate(), "unroll data is not valid, some NaN or inf values found"
    
    agent = agent.train()
    agent.update_normalization(unroll_data.observations)

    for e in tqdm.tqdm(range(epochs), desc="Training", position=1, leave=False):
        loss, raw_losses = agent.train_step(unroll_data=unroll_data)

        optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        lr_scheduler.step()

        mlflow.log_metric("learning_rate", lr_scheduler.get_last_lr()[0], step=(epochs*batch)+e)
        mlflow.log_metric("loss", loss, step=(epochs*batch)+e)
        mlflow.log_metrics(raw_losses, step=(epochs*batch)+e)


def validate(
    agent: T.nn.Module,
    environment: MujocoMjxEnv,
    experiment: MjxExperiment,
    unroll_length: int,
    batch: int,
):

    agent = agent.eval()

    with T.no_grad():

        # reset environment
        jax_seed = T.randint(low=0, high=2 ** 32, size=[1], dtype=T.uint32).item()
        observation = environment.reset(seed=jax_seed)

        rewards = []
        for i in tqdm.tqdm(range(unroll_length), desc="Validating", position=1, leave=False):

            new_observation, _, _ = environment.unroll_step(
                agent=agent,
                observation=observation,
                eval=True
            )

            reward = experiment.calculate_reward(environment.mjx_data)
            rewards.append(reward.detach().cpu())

            observation = new_observation

        mlflow.log_metric("validation_reward_mean", T.stack(rewards).mean().item(), step=batch)
        mlflow.log_metric("validation_reward_variance", T.stack(rewards).var().item(), step=batch)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- CONFIGURATIONS -----------")

    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    logging.basicConfig(level=logging.WARNING) # affects all modules
    # or, only affect mlflow’s own logger:
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- SYSTEM VARIABLES -----------")

    # set xla flags for some GPUs
    environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{cfg.xla_gpu_memory_fraction}"

    print(f"[INFO]: XLA_FLAGS={environ['XLA_FLAGS']}")
    print(f"[INFO]: XLA_PYTHON_CLIENT_PREALLOCATE={environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")
    print(f"[INFO]: XLA_PYTHON_CLIENT_MEM_FRACTION={environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- SETTING VARIABLES -----------")

    if platform == "win32":
        warnings.warn("Disabled GPU for win32 systems for both pytoch and jax because of jax not having cuda available on windows")
    warnings.warn("only 1 device is used at a time, so 1 cpu or 1 gpu, change if need more")

    device = cfg.device if platform != "win32" else "cpu"
    random_seed = cfg.random_seed
    total_batches = int(cfg.total_timesteps // (cfg.num_parallel_environments * cfg.batch_unroll_length))

    print(f"[INFO]: total number of batches: {total_batches}")

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- CONFIGURING JAX -----------")

    # TODO: set jax cuda memory allocation percentage
    jax.config.update("jax_platforms", device)
    jax.config.update("jax_default_device", jax.devices(device)[0])
    # jax.config.update("jax_distributed_debug", True)
    # jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_disable_most_optimizations", True) # reduce non determinism
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_enable_x64", False)
    # jax.config.update("jax_numpy_rank_promotion", "raise")
    jax.config.update("jax_default_matmul_precision", "float32")

    print(f"jax devices: {jax.devices()}")
    print(f"jax backend: {jax.default_backend()}")

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- INITIALIZE RANDOM GENERATORS -----------")

    # sanity check seed setting
    reset_global_seed(random_seed)
    t_seed_a = T.randint(low=0, high=2**32, size=[1], dtype=T.uint32).item()
    np_seed_a = np.random.randint(low=0, high=2**32, size=[1], dtype=np.uint32).item()
    reset_global_seed(random_seed)
    t_seed_b = T.randint(low=0, high=2**32, size=[1], dtype=T.uint32).item()
    np_seed_b = np.random.randint(low=0, high=2**32, size=[1], dtype=np.uint32).item()
    print(f"torch seed: {t_seed_a} == {t_seed_b}")
    print(f"numpy seed: {np_seed_a} == {np_seed_b}")

    # set cudnn for reproducibility
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- SETUP ENVIRONMENT, AGENT, EXPERIMENT AND UNROLL DATA -----------")

    unroll_data = UnrollData(
        num_unrolls=cfg.environment.num_parallel_environments,
        unroll_length=cfg.batch_unroll_length,
        observation_size=cfg.environment.observation_size,
        action_size=cfg.environment.action_size,
    ).to(device=device, dtype=T.float32)
    # agent = PPOAgent(**cfg.agent).to(device=device, dtype=T.float32)
    agent = PPOAgentTorcRL(**cfg.agent).to(device=device, dtype=T.float32)
    environment = MujocoMjxEnv(**cfg.environment)
    experiment = Go2WalkingExperiment(mjx_model = environment.mjx_model, **cfg.experiment)
    optimizer = optim.Adam(agent.parameters(), **cfg.optimizer)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **cfg.lr_scheduler)

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- ENVIRONMENT INFO -----------")

    print(f"[INFO]: simulation timestep: mjx = {environment.mjx_model.opt.timestep}; mj = {environment.mj_model.opt.timestep}")
    print(f"[INFO]: ctrl range: {environment.mjx_model.actuator_ctrlrange.copy()}")

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- MLFLOW SETUP -----------")

    if cfg.mlflow_tracking_uri == "azureml":
        mlflow.set_tracking_uri(get_azureml_mlflow_tracking_uri())
    else:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    # set experiment
    if not mlflow.get_experiment_by_name(cfg.experiment_name):
        mlflow.create_experiment(cfg.experiment_name)
    mlflow.set_experiment(cfg.experiment_name)
    print(f"[INFO]: mlflow experiment set to: '{cfg.experiment_name}'")

    # enable system metric logging (cpu util, etc.)
    mlflow.enable_system_metrics_logging()

    # ----------------------------------------------------------------------------------------------------------------
    print("----------- TRAINING LOOP -----------")

    # reset seed once again since model generation would always change the training seeds
    reset_global_seed(random_seed)

    # reset environment and jit warmup
    with T.no_grad():
        jax_seed = T.randint(low=0, high=2**32, size=[1], dtype=T.uint32).item()
        observation = environment.reset(seed=jax_seed)
        environment.unroll_step(agent=agent, observation=observation)

    # run training
    with mlflow.start_run():

        # log params
        mlflow.log_params(flatten(cfg, reducer="dot"))


        for batch in tqdm.tqdm(range(total_batches), desc="Batches", position=0):

            # unroll step
            unroll_data = unroll(
                agent=agent,
                environment=environment,
                experiment=experiment,
                unroll_data=unroll_data,
            )

            mlflow.log_metric("reward_mean", unroll_data.rewards.mean(1).mean().detach().item(), step=batch)
            mlflow.log_metric("reward_variance", unroll_data.rewards.var(1).mean().detach().item(), step=batch)
            mlflow.log_metric("action_variance", unroll_data.actions.var(1).mean().detach().item(), step=batch)
            mlflow.log_metric("logit_variance", unroll_data.logits.var(1).mean().detach().item(), step=batch)
            mlflow.log_metric("observation_variance", unroll_data.observations.var(1).mean().detach().item(), step=batch)
            mlflow.log_metric("action_mean", unroll_data.actions.mean(1).mean().detach().item(), step=batch)
            mlflow.log_metric("logit_mean", unroll_data.logits.mean(1).mean().detach().item(), step=batch)
            mlflow.log_metric("observation_mean", unroll_data.observations.mean(1).mean().detach().item(), step=batch)



            train(
                agent=agent,
                unroll_data=unroll_data,
                epochs=cfg.train_epochs_per_batch,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_clip_norm=cfg.max_gradient_norm,
                batch=batch,
            )

            if batch % cfg.validation_interval == 0:
                validate(
                    agent=agent,
                    environment=environment,
                    experiment=experiment,
                    unroll_length=cfg.batch_unroll_length,
                    batch=batch,
                )


if __name__ == "__main__":
    main()

