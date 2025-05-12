import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, TypedDict, Unpack

import numpy as np
import scipy.signal
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .actor_critic import ActorCritic
from .car_racing import CarRacing


class _PPOBuffer:

    class Batch(NamedTuple):

        observations: torch.Tensor
        """`(batch_size, observation_shape...)`"""

        action_indices: torch.Tensor
        """`(batch_size,)`"""

        action_log_probs: torch.Tensor
        """`(batch_size,)`"""

        advantages: torch.Tensor
        """`(batch_size,)`"""

        returns: torch.Tensor
        """`(batch_size,)`"""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        size: int,
        gamma: float,
        gae_lambda: float,
    ) -> None:

        self._size = size
        self._gamma = gamma
        self._gae_lambda = gae_lambda

        self._observations = np.empty((size, *observation_shape), dtype=np.float32)
        self._action_indices = np.empty(size, dtype=np.int64)
        self._rewards = np.empty(size, dtype=np.float32)
        self._values = np.empty(size, dtype=np.float32)
        self._action_log_probs = np.empty(size, dtype=np.float32)
        self._advantages = np.empty(size, dtype=np.float32)
        self._returns = np.empty(size, dtype=np.float32)

        self._index = 0
        self._path_start_index = 0

    def store(
        self,
        observation: np.ndarray,
        action_index: int,
        reward: float,
        value: float,
        action_log_prob: float,
    ) -> None:

        self._observations[self._index] = observation
        self._action_indices[self._index] = action_index
        self._rewards[self._index] = reward
        self._values[self._index] = value
        self._action_log_probs[self._index] = action_log_prob

        self._index += 1

    def finish_path(self, last_value: float) -> None:

        index_slice = slice(self._path_start_index, self._index)
        rewards = np.append(self._rewards[index_slice], last_value)
        values = np.append(self._values[index_slice], last_value)

        # GAE-Lambda advantage estimation
        deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
        self._advantages[index_slice] = self._discounted_cumsum(
            deltas, self._gamma * self._gae_lambda
        )

        # Rewards-to-go (targets for the value function)
        self._returns[index_slice] = self._discounted_cumsum(rewards, self._gamma)[:-1]

        self._path_start_index = self._index

    def get(self, batch_size: int, device: torch.device) -> Iterator[Batch]:

        assert self._index == self._size, "Buffer is not full."
        self._index = 0
        self._path_start_index = 0

        # Advantage normalization
        self._advantages -= self._advantages.mean()
        self._advantages /= self._advantages.std()

        observations = torch.as_tensor(self._observations, device=device)
        action_indices = torch.as_tensor(self._action_indices, device=device)
        action_log_probs = torch.as_tensor(self._action_log_probs, device=device)
        advantages = torch.as_tensor(self._advantages, device=device)
        returns = torch.as_tensor(self._returns, device=device)

        indices = torch.randperm(self._size, device=device)

        for i_start in range(0, self._size, batch_size):
            i_stop = i_start + batch_size
            if i_stop > self._size:
                # Drop the last incomplete batch.
                break
            batch_indices = indices[i_start:i_stop]

            yield self.Batch(
                observations[batch_indices],
                action_indices[batch_indices],
                action_log_probs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
            )

    @staticmethod
    def _discounted_cumsum(x: np.ndarray, discount: float) -> np.ndarray:

        b = np.array([1.0], dtype=x.dtype)
        a = np.array([1.0, -discount], dtype=x.dtype)

        x = np.flip(x, axis=0)
        x = scipy.signal.lfilter(b, a, x, axis=0)
        x = np.flip(x, axis=0)

        return x


class PPOTrainingConfig(TypedDict, total=False):

    steps_per_epoch: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    temperature: float
    batch_size: int
    ratio_clip_range: float
    value_function_coeff: float
    entropy_coeff: float
    max_grad_norm: float
    checkpoint_interval: int


_DEFAULT_PPO_TRAINING_CONFIG: PPOTrainingConfig = {
    "steps_per_epoch": 2000,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "temperature": 1.0,
    "batch_size": 64,
    "ratio_clip_range": 0.2,
    "value_function_coeff": 0.5,
    "entropy_coeff": 0.01,
    "max_grad_norm": 0.5,
    "checkpoint_interval": 10,
}


def train_ppo(
    env: CarRacing,
    actor_critic: ActorCritic,
    epochs: int,
    log_dir: Path,
    checkpoint_dir: Path,
    **kwargs: Unpack[PPOTrainingConfig],
) -> None:

    config = _DEFAULT_PPO_TRAINING_CONFIG.copy()
    config.update(kwargs)

    writer = SummaryWriter(log_dir=str(log_dir))

    buffer = _PPOBuffer(
        env.observation_space.shape,
        config["steps_per_epoch"],
        config["gamma"],
        config["gae_lambda"],
    )

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config["learning_rate"])

    observation = env.reset()
    episode_return = 0.0
    episode_length = 0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        values: list[float] = []
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        actor_critic.eval()
        with torch.no_grad():
            for t in range(config["steps_per_epoch"]):
                actor_critic_step = actor_critic.step(
                    observation, config["temperature"]
                )

                env_step = env.step(actor_critic_step.action)
                episode_return += env_step.reward
                episode_length += 1

                buffer.store(
                    observation,
                    actor_critic_step.action_index,
                    env_step.reward,
                    actor_critic_step.value,
                    actor_critic_step.action_log_prob,
                )
                values.append(actor_critic_step.value)

                observation = env_step.observation

                epoch_ended = t + 1 == config["steps_per_epoch"]
                if env_step.terminated or env_step.truncated or epoch_ended:
                    if env_step.terminated or env_step.truncated:
                        last_value = 0.0
                    else:
                        last_value = actor_critic.step(
                            torch.as_tensor(observation, device=actor_critic.device),
                            config["temperature"],
                        ).value
                    buffer.finish_path(last_value)

                    if env_step.terminated:
                        episode_returns.append(episode_return)
                        episode_lengths.append(episode_length)

                    observation = env.reset()
                    episode_return = 0.0
                    episode_length = 0

        writer.add_scalar("episode/return", np.mean(episode_returns), epoch)
        writer.add_scalar("episode/length", np.mean(episode_lengths), epoch)
        writer.add_scalar("value", np.mean(values), epoch)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        total_losses: list[float] = []

        actor_critic.train()
        for batch in buffer.get(config["batch_size"], actor_critic.device):
            optimizer.zero_grad()

            output: ActorCritic.Output = actor_critic(batch.observations)

            action_distribution = torch.distributions.Categorical(
                logits=output.action_logits
            )

            # (batch_size,)
            action_log_probs = action_distribution.log_prob(batch.action_indices)
            ratios = (action_log_probs - batch.action_log_probs).exp()
            policy_losses_1 = batch.advantages * ratios
            policy_losses_2 = batch.advantages * ratios.clamp(
                1.0 - config["ratio_clip_range"], 1.0 + config["ratio_clip_range"]
            )
            policy_loss = -torch.min(policy_losses_1, policy_losses_2).mean()

            value_loss = (
                nn.functional.mse_loss(output.values.squeeze(1), batch.returns)
                * config["value_function_coeff"]
            )

            entropy_loss = (
                -action_distribution.entropy().mean() * config["entropy_coeff"]
            )

            total_loss = policy_loss + value_loss + entropy_loss
            total_loss.backward()

            nn.utils.clip_grad_norm_(actor_critic.parameters(), config["max_grad_norm"])
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(total_loss.item())

        writer.add_scalar("loss/policy", np.mean(policy_losses), epoch)
        writer.add_scalar("loss/value", np.mean(value_losses), epoch)
        writer.add_scalar("loss/entropy", np.mean(entropy_losses), epoch)
        writer.add_scalar("loss/total", np.mean(total_losses), epoch)

        if epoch % config["checkpoint_interval"] == 0 or epoch == epochs:
            epoch_dir = checkpoint_dir / f"epoch-{epoch}"
            epoch_dir.mkdir(exist_ok=True)

            with (epoch_dir / "epoch.txt").open("w", encoding="utf-8") as file:
                file.write(f"{epoch}")

            with (epoch_dir / "config.json").open("w", encoding="utf-8") as file:
                json.dump(config, file, indent=4)

            torch.save(actor_critic.state_dict(), epoch_dir / "actor_critic.pth")

            torch.save(optimizer.state_dict(), epoch_dir / "optimizer.pth")


def train_ppo_cli(
    action_interval: int = 8,
    frame_stack_size: int = 4,
    reward_history_size: int = 100,
    reward_history_min_threshold: float = -0.1,
    epochs: int = 1000,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    steps_per_epoch: int = _DEFAULT_PPO_TRAINING_CONFIG["steps_per_epoch"],
    gamma: float = _DEFAULT_PPO_TRAINING_CONFIG["gamma"],
    gae_lambda: float = _DEFAULT_PPO_TRAINING_CONFIG["gae_lambda"],
    learning_rate: float = _DEFAULT_PPO_TRAINING_CONFIG["learning_rate"],
    temperature: float = _DEFAULT_PPO_TRAINING_CONFIG["temperature"],
    batch_size: int = _DEFAULT_PPO_TRAINING_CONFIG["batch_size"],
    ratio_clip_range: float = _DEFAULT_PPO_TRAINING_CONFIG["ratio_clip_range"],
    value_function_coeff: float = _DEFAULT_PPO_TRAINING_CONFIG["value_function_coeff"],
    entropy_coeff: float = _DEFAULT_PPO_TRAINING_CONFIG["entropy_coeff"],
    max_grad_norm: float = _DEFAULT_PPO_TRAINING_CONFIG["max_grad_norm"],
    checkpoint_interval: int = _DEFAULT_PPO_TRAINING_CONFIG["checkpoint_interval"],
) -> None:

    env = CarRacing(
        None,
        action_interval,
        frame_stack_size,
        reward_history_size,
        reward_history_min_threshold,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCritic(env.observation_space, env.action_space)
    actor_critic.to(device)

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(log_dir, time_str)
    log_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir = Path(checkpoint_dir, time_str)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    train_ppo(
        env,
        actor_critic,
        epochs,
        log_dir,
        checkpoint_dir,
        steps_per_epoch=steps_per_epoch,
        gamma=gamma,
        gae_lambda=gae_lambda,
        learning_rate=learning_rate,
        temperature=temperature,
        batch_size=batch_size,
        ratio_clip_range=ratio_clip_range,
        value_function_coeff=value_function_coeff,
        entropy_coeff=entropy_coeff,
        max_grad_norm=max_grad_norm,
        checkpoint_interval=checkpoint_interval,
    )

    env.close()
