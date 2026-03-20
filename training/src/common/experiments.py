import numpy as np
from mujoco.mjx import Model as MjxModel
from mujoco.mjx import Data as MjxData
from torch import from_numpy, nan_to_num

# explanation for mjData fields: https://bhaswanth-a.github.io/posts/mujoco-basics/


# qpos	Current joint positions (state vector).
# qvel	Joint velocities – how fast each DoF moves.
# qacc	Joint accelerations – derivative of qvel.
# ctrl	Control signals sent to actuators (the policy output).
# act	Raw actuator activations before scaling.
# act_dot	Time‑derivative of actuator activations (useful for smoothness penalties).
# actuator_force	Forces/torques actually produced by actuators.
# qfrc_applied	External forces directly applied to joints (e.g., perturbations).
# xfrc_applied	External forces/torques applied to bodies (world‑frame).
# xpos	Global positions of bodies – useful for pose‑based rewards.
# xquat	Global orientations of bodies (quaternion).
# xmat	Rotation matrices of bodies – alternative orientation view.
# cvel	Spatial velocity (angular + linear) of each body in its local frame.
# geom_xpos	World positions of geometry primitives (collision/contact points).
# geom_xmat	Orientations of those geometry primitives.
# sensordata	Raw sensor readings (force, torque, touch, etc.) supplied to the agent.
# subtree_com	Center‑of‑mass of each kinematic subtree – helps compute balance.
# subtree_mass	Mass of each subtree – useful for dynamics‑aware policies.
# cfrc_ext	Contact forces/torques from the environment (e.g., ground reaction).
# cfrc_int	Internal constraint forces (joint limits, tendon constraints).
# ten_length	Current tendon lengths – can be part of state for compliant robots.
# ten_velocity	Tendon length rates – useful for modeling elastic elements.
# eq_active	Which equality constraints are currently active (helps debug infeasibilities).
# plugin_state	State of custom plugins (e.g., vision or proprioception modules).


# we have
class MjxExperiment:

    def __init__(self, mjx_model: MjxModel):
        self.mjx_model = mjx_model
        # self.initial_mjx_data = self.parse_mjx_data(initial_mjx_data)

        self.actuator_name_dict = {self.get_name_by_idx(i): int(i) for i in mjx_model.name_actuatoradr}
        self.body_name_dict = {self.get_name_by_idx(i): int(i) for i in mjx_model.name_bodyadr}
        # mjx_model.name_geomadr
        # mjx_model.name_jntadr
        # mjx_model.name_keyadr
        # mjx_model.name_siteadr
        # mjx_model.name_sensoradr
        # mjx_model.name_meshadr
        # TODO: we have not the below
        # environment.mjx_model.name_camadr     # cameras
        # environment.mjx_model.name_eqadr      # ?
        # environment.mj_model.name_excludeadr  # ?
        # environment.mjx_model.name_hfieldadr  # hight field ?
        # environment.mjx_model.name_tendonadr  # tendons
        # environment.mjx_model.name_tupleadr   # ?
        # environment.mjx_model.name_pairadr    # ?
        # environment.mjx_model.name_numericadr # ?

    def get_name_by_idx(self, idx: int):
        return self.mjx_model.names[idx:].split(b'\x00')[0].decode("utf-8")

    # def parse_mjx_data(self, mjx_data: MjxData):
    #     return {
    #         "actuator": {
    #             "act": dict(zip(self.actuator_name_dict, mjx_data.act)),
    #             "act_dot": dict(zip(self.actuator_name_dict, mjx_data.act_dot)),
    #             "force": dict(zip(self.actuator_name_dict, mjx_data.actuator_force)),
    #         },
    #         "body": {
    #             "xpos": dict(zip(self.body_name_dict, mjx_data.xpos)),
    #             "xquat": dict(zip(self.body_name_dict, mjx_data.xquat)),
    #         },
    #         "geom": {
    #             # dict(zip(self.body_name_dict, mjx_data.geom_xpos)),
    #             # dict(zip(self.body_name_dict, mjx_data.geom_xmat)),
    #             # TODO
    #         },
    #     }

    def calculate_reward(self, data: MjxData):
        raise NotImplementedError

    # def bodypart_height_reward(self, parsed_data: dict, bodypart_name: str):

    #     # num_dims = len(parsed_data["body"]["xpos"][bodypart_name].shape)
    #     # if num_dims > 1:
    #     torso_z = parsed_data["body"]["xpos"][bodypart_name][:, -1]
    #     # else:
    #         # torso_z = parsed_data["body"]["xpos"][bodypart_name][-1]

    #     print(f'torso_z: {parsed_data["body"]["xpos"][bodypart_name].shape}')

    #     return torso_z

    def bodypart_height_reward(self, data: MjxData, bodypart_id: int):
        return data.xpos[:, bodypart_id, 2]

    def energy_reward(self, data: MjxData):
        return -np.abs(data.actuator_force * data.ctrl).mean(-1)

    # TODO
    # def bodypart_distance_reward(self, data: MjData) -> float:
    #     from_positions = np.stack([dxata.body(n).xpos for n in self.body_names_from])
    #     to_positions = np.stack([data.body(n).xpos for n in self.body_names_to])
    #     dist = np.mean(np.abs(from_positions.mean(axis=0) - to_positions.mean(axis=0)))
    #     return super().__call__(dist)

    def torso_distance_from_origin_reward(self, data: MjxData, torso_id: int):
        xy_pos = data.xpos[:, torso_id, :2]
        dist = np.abs(xy_pos).mean(-1)
        return np.floor(dist)

    def body_part_variance(self, data: MjxData):
        return np.var(data.xpos, axis=-1).mean(-1)

class Go2WalkingExperiment(MjxExperiment):

    def __init__(self, mjx_model: MjxModel, torso_name: str, energy_reward_scale: float,
                 torso_height_reward_scale: float, torso_distance_from_origin_reward_scale: float):
        self.torso_name = torso_name
        self.energy_reward_scale = energy_reward_scale
        self.torso_height_reward_scale = torso_height_reward_scale
        self.torso_distance_from_origin_reward_scale = torso_distance_from_origin_reward_scale

        super().__init__(mjx_model=mjx_model)

    def calculate_reward(self, data: MjxData):

        torso_id = self.body_name_dict[self.torso_name]

        reward = (
            # self.body_part_variance(data) +
            self.bodypart_height_reward(data, torso_id) * self.torso_height_reward_scale
            + self.energy_reward(data) * self.energy_reward_scale
            + self.torso_distance_from_origin_reward(data, torso_id) * self.torso_distance_from_origin_reward_scale
        )
        t_reward = from_numpy(reward.__array__().copy()).unsqueeze(1)
        return t_reward


# def calc_angle(body_rotation_quat: np.NDArray[np.float32], angle_vector: np.NDArray[np.float32] = np.array([1.0, 0.0, 0.0])) -> float:
#
#     rotated_angle_vector = R.from_quat(body_rotation_quat).apply(angle_vector)
#
#     if np.isclose(rotated_angle_vector, angle_vector, atol=1e-4).all():
#         return 0.0
#
#     frac = np.dot(rotated_angle_vector, angle_vector) / (
#         np.linalg.norm(rotated_angle_vector) * np.linalg.norm(angle_vector)
#     )
#     return np.arccos(frac) / np.pi

# class BaseOrientationReward(Reward):
#     def __init__(
#         self,
#         body_index: str,
#         scale: float,
#         angle_vector: NDArray[np.float32] = np.array([1.0, 0.0, 0.0]),
#     ):
#         # self.body_name = body_name
#         self.angle_vector = angle_vector
#         super().__init__(scale=scale)
#
#     def __call__(self, data: MjData) -> float:
#         body_quat = data.body(self.body_name).xquat
#         assert body_quat.sum() > 0, (
#             "body rotation quaternion is not initialized at this point"
#         )
#         reward = 1 - calc_angle(
#             body_rotation_quat=body_quat, angle_vector=self.angle_vector
#         )
#         return super().__call__(reward=reward)
#
# class JointLimitReward(Reward):
#     def __init__(
#         self,
#         mjx_model: MjxModel,
#         scale: float = 1.0,
#         # joint_names: List = [
#         #     "FL_calf_joint",
#         #     "FL_hip_joint",
#         #     "FL_thigh_joint",
#         #     "FR_calf_joint",
#         #     "FR_hip_joint",
#         #     "FR_thigh_joint",
#         #     "RL_calf_joint",
#         #     "RL_hip_joint",
#         #     "RL_thigh_joint",
#         #     "RR_calf_joint",
#         #     "RR_hip_joint",
#         #     "RR_thigh_joint",
#         # ],
#     ):
#         # self.joint_indecies = joint_indecies
#         # self.joint_ranges_min, self.joint_ranges_max = np.stack(
#         #     [mj_model.joint(n).range for n in joint_names]
#         # ).T
#         super().__init__(scale=scale)
#
#     def __call__(self, data: MjData):
#         current_joint_pos = np.concatenate(
#             [data.joint(n).qpos for n in self.joint_names]
#         )
#         joint_min_max_scaled = (current_joint_pos - self.joint_ranges_min) / (
#             self.joint_ranges_max - self.joint_ranges_min
#         )
#         joint_scaled_to_center = ((joint_min_max_scaled - 0.5) * 2) ** 2
#         return super().__call__(-joint_scaled_to_center.mean())
#
