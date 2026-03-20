import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from src.common.datastructure import UnrollData
from src.common.networks import BasicPolicyValueNetwork
from src.common.util import logits_to_normal, jacobian_entropy, jacobian_log_prob

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss

class PPOAgentTorcRL(nn.Module):

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        network_hidden_size: int,
        num_hidden_layers: int,
        discounting: float,
        lambda_: float,
        epsilon: float,
        moving_average_window_size: int,
        reward_scaling: float,
        train_sequence_length: int,
        policy_loss_scale: float,
        value_loss_scale: float,
        entropy_loss_scale: float,
    ):
        super(PPOAgentTorcRL, self).__init__()

        network = BasicPolicyValueNetwork(
            input_size=observation_size,
            value_output_size=1,
            policy_output_size=action_size * 2,
            num_hidden_layers=num_hidden_layers,
            hidden_size=network_hidden_size,
        )

        value_module = ValueOperator(
            module=network.value_network,
            in_keys=["observation"],
        )

        policy_module_ = TensorDictModule(
            nn.Sequential(
                network.policy_network,
                NormalParamExtractor()
            ),
            in_keys=["observation"],
            out_keys=["loc", "scale"]
        )

        policy_module = ProbabilisticActor(
            module=policy_module_,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": -1.0,
                "high": 1.0
            },
            return_log_prob=True,
        )

        self.gae = GAE(
            gamma=discounting,
            lmbda=lambda_,
            value_network=value_module,
            differentiable=False,
            average_gae=True
        )

        self.loss_module = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=epsilon,
            entropy_coeff=entropy_loss_scale,
            critic_coeff=value_loss_scale,
            loss_critic_type="smooth_l1",
        )

        self.running_mean = nn.Parameter(T.zeros(observation_size), requires_grad=False)
        self.running_variance = nn.Parameter(T.ones(observation_size), requires_grad=False)
        self.num_steps = nn.Parameter(T.tensor(0.0), requires_grad=False)

    def update_normalization(self, observation):
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = T.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean[:] = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = T.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance[:] = self.running_variance + var_diff

    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = T.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    def train_step(self, unroll_data: UnrollData):

        unroll_data.observations[:] = self.normalize(unroll_data.observations)

        data = unroll_data.as_tensor_dict()

        with T.no_grad():
            data = self.loss_module.actor_network(data)
            data = self.gae(data)

        out = self.loss_module(data)

        return (
            out["loss_objective"]
            + out["loss_critic"]
            + out["loss_entropy"]
        ,
            {
                "policy_loss": out["loss_objective"],
                "value_loss": out["loss_critic"],
                "entropy_loss": out["loss_entropy"],
            }
        )

    def get_action_and_logits(self, observation: T.Tensor, eval: bool = False):
        loc, scale, action, log_prob = self.loss_module.actor_network(observation)
        logits = T.cat([loc, scale], -1)

        if eval:
            return loc, logits
        else:
            return action, logits

    def postprocess(self, action):
        return action.clip(-1, 1)
    
    def forward(self, observation: T.Tensor):
        self.get_action_and_logits(observation, eval=True)



class PPOAgent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        network_hidden_size: int,
        num_hidden_layers: int,
        discounting: float,
        lambda_: float,
        epsilon: float,
        moving_average_window_size: int,
        reward_scaling: float,
        train_sequence_length: int,
        policy_loss_scale: float,
        value_loss_scale: float,
        entropy_loss_scale: float,
    ):
        super(PPOAgent, self).__init__()

        self.network = BasicPolicyValueNetwork(
            input_size=observation_size,
            policy_output_size=action_size * 2,
            value_output_size=1,
            num_hidden_layers=num_hidden_layers,
            hidden_size=network_hidden_size,
        )

        self.moving_average_window_size = moving_average_window_size
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.epsilon = epsilon

        self.policy_loss_scale = policy_loss_scale
        self.value_loss_scale = value_loss_scale
        self.entropy_loss_scale = entropy_loss_scale

        # decay matrix
        decay_factor = self.discounting * lambda_
        self.decay_factor_matrix = Parameter(
            T.triu(
                decay_factor
                ** T.clamp(
                    T.arange(train_sequence_length - 1).unsqueeze(0)
                    - T.arange(train_sequence_length - 1).unsqueeze(1),
                    min=0,
                )
            ).unsqueeze(0),
            requires_grad=False,
        )

        self.running_mean = nn.Parameter(T.zeros(observation_size), requires_grad=False)
        self.running_variance = nn.Parameter(T.ones(observation_size), requires_grad=False)
        self.num_steps = nn.Parameter(T.tensor(0.0), requires_grad=False)

    def update_normalization(self, observation):
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = T.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean[:] = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = T.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance[:] = self.running_variance + var_diff

    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = T.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    def create_distribution(self, logits: T.Tensor):
        return logits_to_normal(logits)
        # return Normal(
        #     loc=loc,
        #     scale=0.001 # TODO for simulating a real world discrete control signal
        # )

    def get_action_and_logits(self, observation: T.Tensor, eval: bool = False):
        observation = self.normalize(observation)
        logits = self.network(observation)
        dist = self.create_distribution(logits)
        action = dist.rsample()

        if eval:
            return dist.loc, logits
        else:
            return action, logits

    def train_step(self, unroll_data: UnrollData):
        observations = self.normalize(unroll_data.observations)
        logits = unroll_data.logits
        actions = unroll_data.actions
        rewards = unroll_data.rewards * self.reward_scaling

        # # moving average rewards
        # rewards = F.avg_pool1d(
        #     rewards.transpose(1, 2),
        #     stride=1,
        #     kernel_size=self.moving_average_window_size,
        #     padding=self.moving_average_window_size // 2
        # ).transpose(1, 2)[:, :-1, :]

        # # normalize rewards
        # std, mu = T.std_mean(rewards, dim=1, keepdim=True)
        # rewards = (rewards - mu) / std

        # compute policy logits and values
        policy_logits = self.network(observations)
        values = self.network.value_forward(observations)

        # format the individual sequences
        values_t_1 = values[:, 1:]
        bootstrap_value = values[:, -1:]
        values_t = values[:, :-1]
        policy_logits = policy_logits[:, :-1]
        rewards = rewards[:, :-1]
        actions = actions[:, :-1]
        logits = logits[:, :-1]

        behaviour_dist = self.create_distribution(logits)
        behaviour_action_log_probs = jacobian_log_prob(behaviour_dist, actions)
        policy_dist = self.create_distribution(policy_logits)
        policy_action_log_probs = jacobian_log_prob(policy_dist, actions)

        # compute GAE
        with T.no_grad():
            # calculate deltas for each timestep simultainously
            deltas = rewards + self.discounting * values_t_1 - values_t

            # calculate discounted deltas
            discounted_deltas = T.matmul(self.decay_factor_matrix, deltas)

            vs_t = discounted_deltas + values_t
            vs_t_1 = T.cat([vs_t[:, 1:], bootstrap_value], 1)
            advantages = rewards + self.discounting * vs_t_1 - values_t

        rho_s = T.exp(policy_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -T.mean(T.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        value_loss = F.smooth_l1_loss(vs_t, values_t).mean()

        # Entropy loss
        entropy_loss = -jacobian_entropy(policy_dist).mean()

        return (
            policy_loss * self.policy_loss_scale
            + value_loss * self.value_loss_scale
            + entropy_loss * self.entropy_loss_scale,
        {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
        })

    def postprocess(self, action):
        return F.tanh(action)
    
    def forward(self, observation: T.Tensor):
        self.get_action_and_logits(observation, eval=True)


# class PPOAgentGoogle(nn.Module):
#   """Standard PPO Agent with GAE and observation normalization."""
#     def __init__(
#         # self,
#         # policy_layers: Sequence[int],
#         # value_layers: Sequence[int],
#         # device: str):
#         # self,
#         # observation_size: int,
#         # action_size: int,
#         # network_hidden_size: int,
#         # num_hidden_layers: int,
#         # discounting: float,
#         # lambda_: float,
#         # epsilon: float,
#         # moving_average_window_size: int,
#         # reward_scaling: float,
#         # train_sequence_length: int,
#         # policy_loss_scale: float,
#         # value_loss_scale: float,
#         # entropy_loss_scale: float,
#         # softplus_sharpness_factor: float):
#         super(PPOAgentGoogle, self).__init__()

#         # self.num_steps = T.zeros(())
#         # self.running_mean = T.zeros(policy_layers[0])
#         # self.running_variance = T.zeros(policy_layers[0])

#         self.entropy_cost = entropy_loss_scale
#         self.discounting = discounting
#         self.reward_scaling = reward_scaling
#         self.lambda_ = 0.95
#         self.epsilon = 0.3
#         # self.device = device
    
#     def dist_create(self, logits):
#         """Normal followed by tanh.
#         torch.distribution doesn't work with torch.jit, so we roll our own."""
#         loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
#         scale = F.softplus(scale) + .001
#         return loc, scale

#     def dist_sample_no_postprocess(self, loc, scale):
#         return T.normal(loc, scale)

#     @classmethod
#     def dist_postprocess(cls, x):
#         return T.tanh(x)

#     def dist_entropy(self, loc, scale):
#         log_normalized = 0.5 * math.log(2 * math.pi) + T.log(scale)
#         entropy = 0.5 + log_normalized
#         entropy = entropy * T.ones_like(loc)
#         dist = T.normal(loc, scale)
#         log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
#         entropy = entropy + log_det_jacobian
#         return entropy.sum(dim=-1)

#     def dist_log_prob(self, loc, scale, dist):
#         log_unnormalized = -0.5 * ((dist - loc) / scale).square()
#         log_normalized = 0.5 * math.log(2 * math.pi) + T.log(scale)
#         log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
#         log_prob = log_unnormalized - log_normalized - log_det_jacobian
#         return log_prob.sum(dim=-1)

#     def update_normalization(self, observation):
#         self.num_steps += observation.shape[0] * observation.shape[1]
#         input_to_old_mean = observation - self.running_mean
#         mean_diff = T.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
#         self.running_mean = self.running_mean + mean_diff
#         input_to_new_mean = observation - self.running_mean
#         var_diff = T.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
#         self.running_variance = self.running_variance + var_diff

#     def normalize(self, observation):
#         variance = self.running_variance / (self.num_steps + 1.0)
#         variance = T.clip(variance, 1e-6, 1e6)
#         return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

#     def get_logits_action(self, observation):
#         observation = self.normalize(observation)
#         logits = self.policy(observation)
#         loc, scale = self.dist_create(logits)
#         action = self.dist_sample_no_postprocess(loc, scale)
#         return logits, action

#   def compute_gae(
#         self,
#         truncation, 
#         termination, 
#         reward, 
#         values,
#         bootstrap_value
#     ):
#     truncation_mask = 1 - truncation
#     # Append bootstrapped value to get [v1, ..., v_t+1]
#     values_t_plus_1 = T.cat(
#         [values[1:], T.unsqueeze(bootstrap_value, 0)], dim=0)
#     deltas = reward + self.discounting * (
#         1 - termination) * values_t_plus_1 - values
#     deltas *= truncation_mask

#     acc = T.zeros_like(bootstrap_value)
#     vs_minus_v_xs = T.zeros_like(truncation_mask)

#     for ti in range(truncation_mask.shape[0]):
#       ti = truncation_mask.shape[0] - ti - 1
#       acc = deltas[ti] + self.discounting * (
#           1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc
#       vs_minus_v_xs[ti] = acc

#     # Add V(x_s) to get v_s.
#     vs = vs_minus_v_xs + values
#     vs_t_plus_1 = T.cat([vs[1:], T.unsqueeze(bootstrap_value, 0)], 0)
#     advantages = (reward + self.discounting *
#                   (1 - termination) * vs_t_plus_1 - values) * truncation_mask
#     return vs, advantages

#   def loss(self, td: Dict[str, T.Tensor]):
#     observation = self.normalize(td['observation'])
#     policy_logits = self.policy(observation[:-1])
#     baseline = self.value(observation)
#     baseline = T.squeeze(baseline, dim=-1)

#     # Use last baseline value (from the value function) to bootstrap.
#     bootstrap_value = baseline[-1]
#     baseline = baseline[:-1]
#     reward = td['reward'] * self.reward_scaling
#     termination = td['done'] * (1 - td['truncation'])

#     loc, scale = self.dist_create(td['logits'])
#     behaviour_action_log_probs = self.dist_log_prob(loc, scale, td['action'])
#     loc, scale = self.dist_create(policy_logits)
#     target_action_log_probs = self.dist_log_prob(loc, scale, td['action'])

#     with T.no_grad():
#       vs, advantages = self.compute_gae(
#           truncation=td['truncation'],
#           termination=termination,
#           reward=reward,
#           values=baseline,
#           bootstrap_value=bootstrap_value)

#     rho_s = T.exp(target_action_log_probs - behaviour_action_log_probs)
#     surrogate_loss1 = rho_s * advantages
#     surrogate_loss2 = rho_s.clip(1 - self.epsilon,
#                                  1 + self.epsilon) * advantages
#     policy_loss = -T.mean(T.minimum(surrogate_loss1, surrogate_loss2))

#     # Value function loss
#     v_error = vs - baseline
#     v_loss = T.mean(v_error * v_error) * 0.5 * 0.5

#     # Entropy reward
#     entropy = T.mean(self.dist_entropy(loc, scale))
#     entropy_loss = self.entropy_cost * -entropy

#     return policy_loss + v_loss + entropy_loss