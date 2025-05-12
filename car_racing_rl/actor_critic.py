import functools
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):

    class Output(NamedTuple):

        action_logits: torch.Tensor
        """`(batch_size, n_discrete_actions)`"""

        values: torch.Tensor
        """`(batch_size, 1)`"""

    class Step(NamedTuple):

        action_index: int

        action: np.ndarray
        """`(action_dim,)`"""

        action_log_prob: float

        value: float

    _ACTION_TABLE = np.array(
        [
            [-1.0, 0.0, 0.0],  # Turn left (hard)
            [-0.5, 0.0, 0.0],  # Turn left (soft)
            [1.0, 0.0, 0.0],  # Turn right (hard)
            [0.5, 0.0, 0.0],  # Turn right (soft)
            [0.0, 0.0, 0.8],  # Brake (hard)
            [0.0, 0.0, 0.4],  # Brake (soft)
            [0.0, 1.0, 0.0],  # Accelerate (hard)
            [0.0, 0.5, 0.0],  # Accelerate (soft)
            [0.0, 0.0, 0.0],  # No action
        ],
        dtype=np.float32,
    )

    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:

        super().__init__()

        self._action_space = action_space

        self.shared_cnn = nn.Sequential(
            # (batch_size, observation_channels, 96, 96)
            nn.Conv2d(observation_space.shape[0], 8, kernel_size=4, stride=2),
            # (batch_size, 8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            # (batch_size, 16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            # (batch_size, 32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            # (batch_size, 64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            # (batch_size, 128, 3, 3)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            # (batch_size, 256, 1, 1)
            nn.ReLU(),
            nn.Flatten(),
            # (batch_size, 256)
        )

        self.action_head = nn.Sequential(
            # (batch_size, 256)
            nn.Linear(256, 128),
            # (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, self._ACTION_TABLE.shape[0]),
            # (batch_size, n_discrete_actions)
        )

        self.value_head = nn.Sequential(
            # (batch_size, 256)
            nn.Linear(256, 128),
            # (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, 1),
            # (batch_size, 1)
        )

        relu_gain = nn.init.calculate_gain("relu")
        for module, gain in (
            (self.shared_cnn, relu_gain),
            (self.action_head[0], relu_gain),
            (self.action_head[-1], 1.0),
            (self.value_head[0], relu_gain),
            (self.value_head[-1], 1.0),
        ):
            module.apply(functools.partial(self._init_weights, gain=gain))

    @property
    def device(self) -> torch.device:
        return self.shared_cnn[0].weight.device

    def forward(self, observations: torch.Tensor) -> Output:

        # observations: (batch_size, observation_channels, 96, 96)

        # (batch_size, 256)
        latents = self.shared_cnn(observations)

        # (batch_size, n_discrete_actions)
        action_logits = self.action_head(latents)

        # (batch_size, 1)
        values = self.value_head(latents)

        return self.Output(action_logits, values)

    def step(self, observation: np.ndarray, temperature: float) -> Step:

        # (1, observation_channels, height, width)
        observation = torch.as_tensor(observation, device=self.device).unsqueeze(0)

        output: ActorCritic.Output = self(observation)

        # (n_discrete_actions,)
        action_logits = output.action_logits.cpu().squeeze(0)
        action_probs = nn.functional.gumbel_softmax(
            action_logits, tau=temperature, hard=True
        )

        action_index = action_probs.argmax()

        # (action_dim,)
        action = self._ACTION_TABLE[action_index.item()]

        action_log_prob = (
            torch.distributions.Categorical(logits=action_logits)
            .log_prob(action_index)
            .item()
        )

        value = output.values.cpu().item()

        return self.Step(action_index, action, action_log_prob, value)

    @staticmethod
    def _init_weights(module: nn.Module, gain: float) -> None:

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            return
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
