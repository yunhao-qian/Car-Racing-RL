from typing import NamedTuple, Literal

import cv2
import gymnasium as gym
import numpy as np


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

    def average(self) -> float:

        return self._history.mean()


class CarRacing:

    class Step(NamedTuple):

        observation: np.ndarray
        """`(frame_stack_size, height, width)`"""

        reward: float

        terminated: bool

        truncated: bool

    def __init__(
        self,
        render_mode: Literal["human", None],
        action_interval: int,
        frame_stack_size: int,
        reward_history_size: int,
        reward_history_min_threshold: float,
    ) -> None:

        self._env = gym.make(
            "CarRacing-v3", max_episode_steps=-1, render_mode=render_mode
        )
        self._action_interval = action_interval
        self._frame_stack_size = frame_stack_size
        self._frame_stack: list[np.ndarray] = []
        self._reward_history = _RewardHistory(reward_history_size)
        self._reward_history_min_threshold = reward_history_min_threshold

    @property
    def observation_space(self) -> gym.Space:
        height, width, channels = self._env.observation_space.shape
        assert channels == 3
        return gym.spaces.Box(0.0, 1.0, shape=(self._frame_stack_size, height, width))

    @property
    def action_space(self) -> gym.Space:

        return self._env.action_space

    def reset(self) -> np.ndarray:

        observation, _ = self._env.reset()
        observation = self._transform_observation(observation)
        self._frame_stack = [observation] * self._frame_stack_size
        self._reward_history.reset()
        return np.stack(self._frame_stack, axis=0)

    def step(self, action: np.ndarray) -> Step:

        total_reward = 0.0

        for _ in range(self._action_interval):
            observation, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += reward
            self._reward_history.update(reward)
            if self._reward_history.average() < self._reward_history_min_threshold:
                terminated = True
            if terminated or truncated:
                break

        observation = self._transform_observation(observation)
        self._frame_stack.pop(0)
        self._frame_stack.append(observation)
        observation = np.stack(self._frame_stack, axis=0)

        return self.Step(observation, total_reward, terminated, truncated)

    def close(self) -> None:

        self._env.close()

    @staticmethod
    def _transform_observation(observation: np.ndarray) -> np.ndarray:

        # observation: (height, width, 3), uint8

        # (height, width), uint8
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # (height, width), float32
        observation = observation.astype(np.float32)
        observation /= 255.0

        return observation
