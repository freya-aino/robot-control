from multiprocessing import Value
import torch as T
import torch.nn as nn
from typing import Tuple


class NetworkBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, final_act_f):
        super().__init__()

        # self.bn = nn.BatchNorm1d(input_size)
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)

        self.nn1 = nn.Linear(input_size, input_size)
        self.nn2 = nn.Linear(input_size, input_size)
        self.nn3 = nn.Linear(input_size, output_size)

        self.final_act = final_act_f()
        self.act = nn.SiLU()

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.act(self.ln1(self.nn1(x)))
        x = self.act(self.ln2(self.nn2(x)))
        x = self.final_act(self.nn3(x))
        return x
        # return self.final_act(self.nn3(self.act(self.ln(self.nn2(x)))))


class BasicPolicyValueNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_hidden_layers: int,
        policy_output_size: int,
        value_output_size: int,
        hidden_size: int,
    ):
        super().__init__()

        self.policy_network = nn.Sequential(
            NetworkBlock(input_size, hidden_size, nn.SiLU),
            *[
                NetworkBlock(hidden_size, hidden_size, nn.SiLU)
                for _ in range(num_hidden_layers)
            ],
            NetworkBlock(hidden_size, policy_output_size, nn.Identity),
        )

        self.value_network = nn.Sequential(
            NetworkBlock(input_size, hidden_size, nn.SiLU),
            *[
                NetworkBlock(hidden_size, hidden_size, nn.SiLU)
                for _ in range(num_hidden_layers)
            ],
            NetworkBlock(hidden_size, value_output_size, nn.Identity),
        )

    def policy_forward(self, x: T.Tensor) -> T.Tensor:
        return self.policy_network(x)

    def value_forward(self, x: T.Tensor) -> T.Tensor:
        return self.value_network(x)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.policy_forward(x)