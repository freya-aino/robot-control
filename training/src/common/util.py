import os
import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchrl.modules import TanhNormal

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import json


def save_as_onnx(agent: nn.Module, path: str, input_size: int):
    # get_action_and_logits(self, observation: T.Tensor, eval: bool = False)
    example_observation = (T.rand(1, 1, input_size),)
    onnx_model = T.onnx.export(agent, example_observation, dynamo=True)
    onnx_model.save(path)

def get_azureml_mlflow_tracking_uri():

    assert os.path.exists("./config.json"), "config.json file not found in project root"

    with open("./config.json", "r") as f:
        config = json.load(f)

    # Enter information about your Azure Machine Learning workspace.
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    workspace_name = config["workspace_name"]

    ml_client = MLClient(credential=DefaultAzureCredential(),
                            subscription_id=subscription_id,
                            resource_group_name=resource_group,
                            workspace_name=workspace_name)

    tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri

    assert tracking_uri, "tracking uri is None"

    return tracking_uri



def logits_to_normal(logits: T.Tensor, beta: float = 1.0) -> Normal:
    loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
    scale = F.softplus(scale, beta=beta)
    return Normal(loc=loc, scale=scale)

def logits_to_tanh_normal(logits: T.Tensor, beta: float = 1.0) -> TanhNormal:
    loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
    scale = F.softplus(scale, beta=beta)
    return TanhNormal(loc=loc, scale=scale)

def jacobian_entropy(dist: Normal):
    log_normalized = 0.5 * math.log(2 * math.pi) + T.log(dist.scale)
    entropy = 0.5 + log_normalized
    entropy = entropy * T.ones_like(dist.loc)
    sample = dist.rsample()
    # entropy = dist.entropy()
    jacobian = 2 * (math.log(2) - sample - F.softplus(-2 * sample))

    return (entropy + jacobian).sum(-1)

def jacobian_log_prob(dist: Normal, sample: T.Tensor):
    log_unnormalized = -0.5 * ((sample - dist.loc) / dist.scale).square()
    log_normalized = 0.5 * math.log(2 * math.pi) + T.log(dist.scale)
    # log_p = dist.log_prob(sample)
    jacobian = 2 * (math.log(2) - sample - F.softplus(-2 * sample))
    # return (log_p - jacobian).sum(dim=-1).mean(dim=0)
    return (log_unnormalized - log_normalized - jacobian).sum(dim=-1).mean()
