from typing import Literal, NamedTuple, TypedDict, override

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec

from .configurable import Configurable


class EnvConfig(TypedDict, total=False):
    reward_threshold: float
    max_episode_steps: int
    render_mode: Literal["rgb_array", "human", None]
    domain_randomize: bool
    action_interval: int
    frame_stack_size: int
    action_stack_size: int
    reward_history_size: int
    reward_history_min_threshold: float


class Env(Configurable[EnvConfig]):

    class Step(NamedTuple):
        observation: np.ndarray
        """`(frame_stack_size, height, width)`"""

        action_history: np.ndarray | None
        """`(action_stack_size, action_dim)`"""

        reward: float
        terminated: bool
        truncated: bool

    def __init__(self, **kwargs: EnvConfig) -> None:
        Configurable.__init__(self, **kwargs)

        env_spec = EnvSpec(
            id="CarRacing-v3",
            entry_point="car_racing_rl.car_racing:CarRacing",
            reward_threshold=self._config["reward_threshold"],
            max_episode_steps=self._config["max_episode_steps"],
        )
        self._env = gym.make(
            env_spec,
            render_mode=self._config["render_mode"],
            domain_randomize=self._config["domain_randomize"],
        )
        self._frame_stack: list[np.ndarray] = []
        self._action_stack: list[np.ndarray] = []
        self._reward_history = _RewardHistory(self._config["reward_history_size"])

    @property
    def observation_space(self) -> gym.Space:
        height, width, _ = self._env.observation_space.shape
        return gym.spaces.Box(
            0.0, 1.0, (self._config["frame_stack_size"], height, width)
        )

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        observation, _ = self._env.reset(seed=seed)
        observation = self._transform_observation(observation)
        self._frame_stack = [observation] * self._config["frame_stack_size"]

        action = np.zeros(self._env.action_space.shape, dtype=np.float32)
        self._action_stack = [action] * self._config["action_stack_size"]
        self._reward_history.reset()

        observation = np.stack(self._frame_stack)
        if len(self._action_stack) == 0:
            action_history = None
        else:
            action_history = np.stack(self._action_stack)
        return observation, action_history

    def step(self, action: np.ndarray):
        total_reward = 0.0

        for _ in range(self._config["action_interval"]):
            observation, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += reward
            self._reward_history.update(reward)
            if (
                self._reward_history.mean()
                < self._config["reward_history_min_threshold"]
            ):
                terminated = True
            if terminated or truncated:
                break

        observation = self._transform_observation(observation)
        self._frame_stack.pop(0)
        self._frame_stack.append(observation)
        observation = np.stack(self._frame_stack)

        if len(self._action_stack) == 0:
            action_history = None
        else:
            self._action_stack.pop(0)
            self._action_stack.append(action)
            action_history = np.stack(self._action_stack)

        return self.Step(
            observation, action_history, total_reward, terminated, truncated
        )

    def close(self) -> None:
        self._env.close()

    @staticmethod
    @override
    def default_config() -> EnvConfig:
        return {
            "render_mode": None,
            "reward_threshold": 900.0,
            "max_episode_steps": 1000,
            "domain_randomize": False,
            "action_interval": 8,
            "frame_stack_size": 4,
            "action_stack_size": 0,
            "reward_history_size": 100,
            "reward_history_min_threshold": -0.1,
        }

    @staticmethod
    def _transform_observation(observation: np.ndarray) -> np.ndarray:
        # observation: (height, width, 3), uint8

        # (height, width), uint8
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # (height, width), float32
        observation = observation.astype(np.float32)
        observation /= 255.0
        return observation


class _RewardHistory:

    def __init__(self, size: int) -> None:
        self._size = size
        self._history = np.zeros(size, dtype=np.float32)
        self._index = 0

    def reset(self) -> None:
        self._history.fill(0.0)
        self._index = 0

    def update(self, reward: float) -> None:
        self._history[self._index] = reward
        self._index = (self._index + 1) % self._size

    def mean(self) -> float:
        return self._history.mean()
