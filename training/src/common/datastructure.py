import torch as T
from tensordict import TensorDict
from torch import nn, Tensor, zeros, zeros_like, ones_like
from typing import Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset
import numpy as np

from src.common.util import logits_to_normal

class UnrollData(nn.Module):

    def __init__(
        self,
        num_unrolls: int,
        unroll_length: int,
        observation_size: int,
        action_size: int,
    ):
        super().__init__()

        self.observations = nn.Parameter(zeros(size=[num_unrolls, unroll_length, observation_size]), requires_grad=False)
        self.logits = nn.Parameter(zeros(size=[num_unrolls, unroll_length, action_size * 2]), requires_grad=False)
        self.actions = nn.Parameter(zeros(size=[num_unrolls, unroll_length, action_size]), requires_grad=False)
        self.rewards = nn.Parameter(zeros(size=[num_unrolls, unroll_length, 1]), requires_grad=False)

    def update(
        self,
        unroll_step: int,
        observation: Tensor,
        logits: Tensor,
        action: Tensor,
        reward: Tensor,
    ):
        
        if observation.dim() == 2:
            observation = observation.unsqueeze(1)
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        if reward.dim() == 2:
            reward = reward.unsqueeze(1)
            

        self.observations[:, unroll_step:unroll_step+1, :] = observation
        self.logits[:, unroll_step:unroll_step+1, :] = logits
        self.actions[:, unroll_step:unroll_step+1, :] = action
        self.rewards[:, unroll_step:unroll_step+1, :] = reward

    def as_tensor_dict(self):

        bs, ss, _ = self.observations.shape
        ss -= 1 # -1 because we shift the sequence by one

        # action_log_prob = logits_to_normal(self.logits).log_prob(self.actions)
        obs_t = self.observations[:, :-1, :].reshape(bs * ss, -1)
        act_t = self.actions[:, :-1, :].reshape(bs * ss, -1)
        obs_t_1 = self.observations[:, 1:, :].reshape(bs * ss, -1)
        rew_t_1 = self.rewards[:, 1:, :].reshape(bs * ss, -1)

        return TensorDict({
            "done": zeros_like(rew_t_1).to(dtype=T.bool),  # TODO
            "observation": obs_t,
            "action": act_t,
            "next": {
                "done": zeros_like(rew_t_1).to(dtype=T.bool), # TODO
                "terminated": zeros_like(rew_t_1).to(dtype=T.bool), # TODO
                "reward": rew_t_1,
                "observation": obs_t_1,
                # "step_count": T.linspace(0, ss-1, ss).unsqueeze(0).repeat([bs, 1]),
            },
            # "truncated" # TODO
        },
        batch_size=obs_t.shape[0]
        )

        # bs, ss, _ = self.observations.shape
        # ss -= 1 # -1 because we shift the sequence by one
        #
        # # action_log_prob = logits_to_normal(self.logits).log_prob(self.actions)
        # obs_t = self.observations[:, :-1, :]
        # obs_t_1 = self.observations[:, 1:, :]
        # act_t = self.actions[:, :-1, :]
        # rew_t_1 = self.rewards[:, :-1, :]
        #
        # return TensorDict({
        #     "done": zeros_like(rew_t_1).to(dtype=T.bool),  # TODO
        #     "observation": obs_t,
        #     "action": act_t,
        #     "next": {
        #         "done": zeros_like(rew_t_1).to(dtype=T.bool), # TODO
        #         "terminated": zeros_like(rew_t_1).to(dtype=T.bool), # TODO
        #         "reward": rew_t_1,
        #         "observation": obs_t_1,
        #         # "step_count": T.linspace(0, ss-1, ss).unsqueeze(0).repeat([bs, 1]),
        #     },
        #     # "truncated" # TODO
        # },
        # batch_size=obs_t.shape[:2]
        # )

    def validate(self):
        is_observation_good = ~(self.observations.isnan().any() | self.observations.isinf().any())
        is_logits_good = ~(self.logits.isnan().any() | self.logits.isinf().any())
        is_actions_good = ~(self.actions.isnan().any() | self.actions.isinf().any())
        is_rewards_good = ~(self.rewards.isnan().any() | self.rewards.isinf().any())
        if is_observation_good and is_logits_good and is_actions_good and is_rewards_good:
            return True
        else:
            print("UnrollData validation failed:")
            print(f"observations nan: {self.observations.isnan().sum().item()}, inf: {self.observations.isinf().sum().item()}")
            print(f"logits nan:       {self.logits.isnan().sum().item()},       inf: {self.logits.isinf().sum().item()}")
            print(f"actions nan:      {self.actions.isnan().sum().item()},      inf: {self.actions.isinf().sum().item()}")
            print(f"rewards nan:      {self.rewards.isnan().sum().item()},      inf: {self.rewards.isinf().sum().item()}")
            return False

    # TODO - do a type of validation that check for abnormal values during training (e.g. actions outside of bounds or logits very small or very big)


# class MultiUnrollDataset(Dataset):
#     def __init__(
#         self,
#         unrolls: Sequence[UnrollData],
#         num_minibatches: int,
#         minibatch_size: int,
#         minibatched: bool = True,
#     ):
#         if not minibatched:
#             raise NotImplementedError
#
#         self.observations = stack([u.observation for u in unrolls], dim=0)
#         self.logits = stack([u.logits for u in unrolls], dim=0)
#         self.actions = stack([u.actions for u in unrolls], dim=0)
#         self.rewards = stack([u.rewards for u in unrolls], dim=0)
#
#         self.preprocess(
#             num_unrolls=len(unrolls),
#             num_minibatches=num_minibatches,
#             minibatch_size=minibatch_size,
#         )
#
#     def __len__(self):
#         return self.observations.shape[0]
#
#     def __getitem__(self, idx: int):
#         return {
#             "observations": self.observations[idx],
#             "logits": self.logits[idx],
#             "actions": self.actions[idx],
#             "rewards": self.rewards[idx],
#         }
#
#     def preprocess(self, num_unrolls: int, num_minibatches: int, minibatch_size: int):
#         observations = self.observations.view(
#             [num_unrolls, num_minibatches, minibatch_size, -1]
#         )
#         logits = self.logits.view([num_unrolls, num_minibatches, minibatch_size, -1])
#         actions = self.actions.view([num_unrolls, num_minibatches, minibatch_size, -1])
#         rewards = self.rewards.view([num_unrolls, num_minibatches, minibatch_size, -1])
#
#         ll = num_unrolls * num_minibatches
#         self.observations = observations.reshape([ll, minibatch_size, -1])
#         self.logits = logits.reshape([ll, minibatch_size, -1])
#         self.actions = actions.reshape([ll, minibatch_size, -1])
#         self.rewards = rewards.reshape([ll, minibatch_size, -1])
#
#     def validate(self):
#         assert ~self.actions.isnan().any(), "Action contains NaN values"
#         assert ~self.actions.isinf().any(), "Action contains infinite values"
#         assert ~self.rewards.isnan().any(), "Reward contains NaN values"
#         assert ~self.rewards.isinf().any(), "Reward contains infinite values"
#         assert ~self.logits.isnan().any(), "Logits contains NaN values"
#         assert ~self.logits.isinf().any(), "Logits contains infinite values"
#         assert ~self.observations.isnan().any(), "Observations contain NaN values"
#         assert ~self.observations.isinf().any(), "Observations contain infinite values"
