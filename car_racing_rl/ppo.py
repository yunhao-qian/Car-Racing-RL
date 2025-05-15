import json
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, Self, TypedDict, Unpack, override

import numpy as np
import scipy.signal
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .actor_critic import ActorCritic, ActorCriticConfig
from .configurable import Configurable
from .env import Env, EnvConfig


class PPOConfig(TypedDict, total=False):
    env_config: EnvConfig
    actor_critic_config: ActorCriticConfig
    log_dir: str | None
    checkpoint_dir: str | None
    checkpoint_interval: int
    learning_rate: float
    epoch_start: int
    epoch_stop: int
    steps_per_epoch: int
    gamma: float
    gae_lambda: float
    batch_size: int
    ratio_clip_epsilon: float
    policy_loss_weight: float
    value_loss_weight: float
    entropy_loss_weight: float
    max_grad_norm: float


class PPO(Configurable[PPOConfig]):

    def __init__(self, device: torch.device, **kwargs: Unpack[PPOConfig]) -> None:
        Configurable.__init__(self, **kwargs)

        self._env = Env(**self._config["env_config"])
        self._actor_critic = ActorCritic(**self._config["actor_critic_config"])
        self._actor_critic.to(device)
        self._optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self._actor_critic.parameters()),
            lr=self._config["learning_rate"],
        )

    def train(self) -> None:
        print("PPO configuration:")
        print(json.dumps(self._config, indent=4))

        buffer = _PPOBuffer(
            size=self._config["steps_per_epoch"],
            gamma=self._config["gamma"],
            gae_lambda=self._config["gae_lambda"],
            observation_shape=self._env.observation_space.shape,
            action_history_shape=(
                self._config["env_config"]["action_stack_size"],
                *self._env.action_space.shape,
            ),
            action_event_shape=self._actor_critic.action_head.action_event_shape(),
            action_event_dtype=self._actor_critic.action_head.action_event_dtype(),
        )

        trainable_params: list[nn.Parameter] = []
        for param_group in self._optimizer.param_groups:
            trainable_params.extend(param_group["params"])

        writer = SummaryWriter(
            self._config["log_dir"], purge_step=self._config["epoch_start"]
        )

        observation, action_history = self._env.reset()
        episode_return = 0.0
        episode_length = 0

        progress_bar = tqdm(total=self._config["steps_per_epoch"])

        for epoch in range(self._config["epoch_start"], self._config["epoch_stop"]):
            progress_bar.reset()
            progress_bar.set_description(
                f"Epoch {epoch + 1}/{self._config['epoch_stop']}"
            )

            stats: dict[str, list[float] | list[np.ndarray]] = {
                "actor_critic/action_event": [],
                "actor_critic/action_log_prob": [],
                "actor_critic/action": [],
                "actor_critic/value": [],
                "reward": [],
                "episode/return": [],
                "episode/length": [],
                "loss/policy": [],
                "loss/value": [],
                "loss/entropy": [],
                "loss/total": [],
                "ratio/ratio": [],
                "ratio/clip_fraction": [],
                "train/grad_norm": [],
                "train/approx_kl": [],
                "train/explained_variance": [],
            }

            self._actor_critic.eval()
            with torch.no_grad():
                for step in range(self._config["steps_per_epoch"]):
                    actor_critic_step = self._actor_critic.step(
                        observation, action_history
                    )

                    env_step = self._env.step(actor_critic_step.action)
                    episode_return += env_step.reward
                    episode_length += 1

                    buffer.store(
                        observation,
                        action_history,
                        actor_critic_step.action_event,
                        env_step.reward,
                        actor_critic_step.value,
                        actor_critic_step.action_log_prob,
                    )
                    stats["actor_critic/action_event"].append(
                        actor_critic_step.action_event
                    )
                    stats["actor_critic/action_log_prob"].append(
                        actor_critic_step.action_log_prob
                    )
                    stats["actor_critic/action"].append(actor_critic_step.action)
                    stats["actor_critic/value"].append(actor_critic_step.value)
                    stats["reward"].append(env_step.reward)

                    observation = env_step.observation
                    action_history = env_step.action_history

                    progress_bar.update()

                    done = env_step.terminated or env_step.truncated
                    epoch_ended = step + 1 == self._config["steps_per_epoch"]
                    if done or epoch_ended:
                        if done:
                            last_value = 0.0
                            stats["episode/return"].append(episode_return)
                            stats["episode/length"].append(episode_length)
                        else:
                            last_value = self._actor_critic.step(
                                observation, action_history
                            ).value

                        buffer.finish_path(last_value)

                        observation, action_history = self._env.reset()
                        episode_return = 0.0
                        episode_length = 0

            self._actor_critic.train()
            for batch in buffer.get(
                self._config["batch_size"], self._actor_critic.device
            ):
                self._optimizer.zero_grad()

                output: ActorCritic.Output = self._actor_critic(
                    batch.observations, batch.action_histories
                )

                # (batch_size, ...)
                action_log_probs = output.action_distribution.log_prob(
                    batch.action_events
                )
                # (batch_size,)
                if action_log_probs.dim() > 1:
                    sum_dim = tuple(range(1, action_log_probs.dim()))
                    action_log_probs = action_log_probs.sum(dim=sum_dim)

                # (batch_size,)
                ratio = (action_log_probs - batch.action_log_probs).exp()
                policy_loss_1 = batch.advantages * ratio
                policy_loss_2 = batch.advantages * ratio.clamp(
                    1.0 - self._config["ratio_clip_epsilon"],
                    1.0 + self._config["ratio_clip_epsilon"],
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                policy_loss = policy_loss * self._config["policy_loss_weight"]

                value_loss = nn.functional.mse_loss(output.values, batch.returns)
                value_loss = value_loss * self._config["value_loss_weight"]

                if self._config["entropy_loss_weight"] == 0.0:
                    # Some distributions do not implement entropy(), and calling it will
                    # raise NotImplementedError.
                    entropy_loss = torch.tensor(0.0, device=self._actor_critic.device)
                else:
                    entropy_loss = -output.action_distribution.entropy().mean()
                    entropy_loss = entropy_loss * self._config["entropy_loss_weight"]

                total_loss = policy_loss + value_loss + entropy_loss
                total_loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(
                    trainable_params, self._config["max_grad_norm"]
                ).item()
                self._optimizer.step()

                with torch.no_grad():
                    stats["loss/policy"].append(policy_loss.item())
                    stats["loss/value"].append(value_loss.item())
                    stats["loss/entropy"].append(entropy_loss.item())
                    stats["loss/total"].append(total_loss.item())
                    ratio = ratio.detach().cpu().numpy()
                    stats["ratio/ratio"].append(ratio)
                    stats["ratio/clip_fraction"].append(
                        (ratio < 1.0 - self._config["ratio_clip_epsilon"])
                        | (ratio > 1.0 + self._config["ratio_clip_epsilon"])
                    )
                    stats["train/grad_norm"].append(grad_norm)
                    stats["train/approx_kl"].append(
                        (batch.action_log_probs - action_log_probs).mean().item()
                    )
                    stats["train/explained_variance"].append(
                        1.0
                        - (
                            (output.values - batch.returns).var(correction=0)
                            / batch.returns.var(correction=0)
                        ).item()
                    )

            self._log_config_and_stats(writer, stats, epoch)

            if (
                epoch % self._config["checkpoint_interval"] == 0
                or epoch == self._config["epoch_stop"] - 1
            ):
                self._save_checkpoint(epoch)

    @staticmethod
    def from_checkpoint(
        checkpoint_dir: Path,
        skip_optimizer: bool,
        device: torch.device,
        **kwargs: Unpack[PPOConfig],
    ) -> Self:
        with (checkpoint_dir / "config.json").open(encoding="utf-8") as file:
            config: PPOConfig = json.load(file)
        for key, value in kwargs.items():
            match key:
                case "env_config":
                    config["env_config"].update(value)
                case "actor_critic_config":
                    config["actor_critic_config"].update(value)
                case _:
                    config[key] = value

        ppo = PPO(device, **config)
        ppo._actor_critic.load_state_dict(
            torch.load(checkpoint_dir / "actor_critic.pth", map_location=device)
        )
        if not skip_optimizer:
            ppo._optimizer.load_state_dict(
                torch.load(checkpoint_dir / "optimizer.pth", map_location=device)
            )

        # Override the learning rate if specified in kwargs.
        if "learning_rate" in kwargs:
            for param_group in ppo._optimizer.param_groups:
                if param_group["lr"] == config["learning_rate"]:
                    continue
                print(
                    "Overriding learning rate: "
                    f"{param_group['lr']} -> {config['learning_rate']}"
                )
                param_group["lr"] = config["learning_rate"]

        return ppo

    @staticmethod
    @override
    def default_config() -> PPOConfig:
        return {
            "env_config": Env.default_config(),
            "actor_critic_config": ActorCritic.default_config(),
            "log_dir": None,
            "checkpoint_dir": None,
            "checkpoint_interval": 10,
            "learning_rate": 3e-4,
            "epoch_start": 0,
            "epoch_stop": 1000,
            "steps_per_epoch": 2000,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "batch_size": 64,
            "ratio_clip_epsilon": 0.2,
            "policy_loss_weight": 1.0,
            "value_loss_weight": 0.5,
            "entropy_loss_weight": 0.01,
            "max_grad_norm": 0.5,
        }

    def _log_config_and_stats(
        self,
        writer: SummaryWriter,
        stats: dict[str, list[float] | list[np.ndarray]],
        epoch: int,
    ) -> None:
        for tag in (
            "learning_rate",
            "policy_loss_weight",
            "value_loss_weight",
            "entropy_loss_weight",
            "max_grad_norm",
        ):
            writer.add_scalar(f"config/{tag}", self._config[tag], epoch)

        for tag, stat_list in stats.items():
            if tag.startswith("ratio/"):
                stat_array = np.concatenate(stat_list)
            else:
                stat_array = np.stack(stat_list)

            match tag:
                case "actor_critic/action_event" | "actor_critic/action":
                    writer.add_histogram(tag, stat_array, epoch)
                case (
                    "actor_critic/action_log_prob"
                    | "actor_critic/value"
                    | "reward"
                    | "episode/return"
                    | "episode/length"
                    | "ratio/ratio"
                ):
                    writer.add_scalar(f"{tag}/mean", stat_array.mean(), epoch)
                    writer.add_scalar(f"{tag}/std", stat_array.std(), epoch)
                    writer.add_scalar(f"{tag}/min", stat_array.min(), epoch)
                    writer.add_scalar(f"{tag}/max", stat_array.max(), epoch)
                case _:
                    writer.add_scalar(tag, stat_array.mean(), epoch)

    def _save_checkpoint(self, epoch: int) -> None:
        epoch_dir = Path(self._config["checkpoint_dir"], f"epoch-{epoch:04d}")
        epoch_dir.mkdir(parents=True, exist_ok=True)

        with (epoch_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(self._config, file, indent=4)

        torch.save(self._actor_critic.state_dict(), epoch_dir / "actor_critic.pth")
        torch.save(self._optimizer.state_dict(), epoch_dir / "optimizer.pth")


class _PPOBuffer:

    class Batch(NamedTuple):
        observations: torch.Tensor
        """`(batch_size, observation_shape...)`"""

        action_histories: torch.Tensor | None
        """`(batch_size, action_history_shape...)`"""

        action_events: torch.Tensor
        """`(batch_size, action_event_shape...)`"""

        action_log_probs: torch.Tensor
        """`(batch_size,)`"""

        advantages: torch.Tensor
        """`(batch_size,)`"""

        returns: torch.Tensor
        """`(batch_size,)`"""

    def __init__(
        self,
        size: int,
        gamma: float,
        gae_lambda: float,
        observation_shape: tuple[int, ...],
        action_history_shape: tuple[int, ...],
        action_event_shape: tuple[int, ...],
        action_event_dtype: np.dtype,
    ) -> None:
        self._size = size
        self._gamma = gamma
        self._gae_lambda = gae_lambda

        self._observations = np.empty((size, *observation_shape), dtype=np.float32)
        if action_history_shape[0] == 0:
            self._action_histories = None
        else:
            self._action_histories = np.empty(
                (size, *action_history_shape), dtype=np.float32
            )
        self._action_events = np.empty(
            (size, *action_event_shape), dtype=action_event_dtype
        )
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
        action_history: np.ndarray | None,
        action_event: int | np.ndarray,
        reward: float,
        value: float,
        action_log_prob: float,
    ) -> None:
        self._observations[self._index] = observation
        if action_history is not None:
            self._action_histories[self._index] = action_history
        self._action_events[self._index] = action_event
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
        assert self._index == self._size, "Buffer is not full"
        self._index = 0
        self._path_start_index = 0

        # Advantage normalization
        self._advantages -= self._advantages.mean()
        self._advantages /= self._advantages.std()

        observations = torch.as_tensor(self._observations, device=device)
        if self._action_histories is None:
            action_histories = None
        else:
            action_histories = torch.as_tensor(self._action_histories, device=device)
        action_events = torch.as_tensor(self._action_events, device=device)
        action_log_probs = torch.as_tensor(self._action_log_probs, device=device)
        advantages = torch.as_tensor(self._advantages, device=device)
        returns = torch.as_tensor(self._returns, device=device)

        indices = torch.randperm(self._size, device=device)

        for start in range(0, self._size, batch_size):
            stop = start + batch_size
            if stop > self._size:
                # Drop the last incomplete batch.
                break
            batch_indices = indices[start:stop]

            yield self.Batch(
                observations[batch_indices],
                None if action_histories is None else action_histories[batch_indices],
                action_events[batch_indices],
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
