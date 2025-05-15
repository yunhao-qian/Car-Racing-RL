import json
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, Self, TypedDict, Unpack, override

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .actor_critic import ActorCritic, ActorCriticConfig
from .configurable import Configurable
from .env import Env, EnvConfig
from .ppo import PPOConfig


class BehaviorCloningConfig(TypedDict):
    env_config: EnvConfig
    teacher_checkpoint_dir: str | None
    actor_critic_config: ActorCriticConfig
    log_dir: str | None
    checkpoint_dir: str | None
    checkpoint_interval: int
    learning_rate: float
    epoch_start: int
    epoch_stop: int
    steps_per_epoch: int
    batch_size: int
    action_loss_weight: float
    value_loss_weight: float
    entropy_loss_weight: float
    max_grad_norm: float


class BehaviorCloning(Configurable[BehaviorCloningConfig]):

    def __init__(
        self, device: torch.device, **kwargs: Unpack[BehaviorCloningConfig]
    ) -> None:
        Configurable.__init__(self, **kwargs)

        self._env = Env(**self._config["env_config"])

        # Load the teacher model.
        with Path(self._config["teacher_checkpoint_dir"], "config.json").open(
            encoding="utf-8"
        ) as file:
            teacher_config: PPOConfig | BehaviorCloningConfig = json.load(file)
        self._teacher = ActorCritic(**teacher_config["actor_critic_config"])
        self._teacher.to(device)
        self._teacher.load_state_dict(
            torch.load(
                Path(self._config["teacher_checkpoint_dir"], "actor_critic.pth"),
                map_location=device,
            )
        )
        self._teacher.eval()
        self._teacher.requires_grad_(False)

        # Create the student model.
        self._student = ActorCritic(**self._config["actor_critic_config"])
        self._student.to(device)
        self._student.train()

        self._optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self._student.parameters()),
            lr=self._config["learning_rate"],
        )

    def train(self) -> None:
        print("Behavior cloning configuration:")
        print(json.dumps(self._config, indent=4))

        buffer = _BehaviorCloningBuffer(
            size=self._config["steps_per_epoch"],
            observation_shape=self._env.observation_space.shape,
            action_history_shape=(
                self._config["env_config"]["action_stack_size"],
                *self._env.action_space.shape,
            ),
            action_dim=self._env.action_space.shape[0],
        )

        trainable_params: list[nn.Parameter] = []
        for param_group in self._optimizer.param_groups:
            trainable_params.extend(param_group["params"])

        writer = SummaryWriter(
            self._config["log_dir"], purge_step=self._config["epoch_start"]
        )

        observation, action_history = self._env.reset()

        progress_bar = tqdm(total=self._config["steps_per_epoch"])

        for epoch in range(self._config["epoch_start"], self._config["epoch_stop"]):
            progress_bar.reset()
            progress_bar.set_description(
                f"Epoch {epoch + 1}/{self._config['epoch_stop']}"
            )

            with torch.no_grad():
                for step in range(self._config["steps_per_epoch"]):
                    teacher_step = self._teacher.step(observation, action_history)
                    env_step = self._env.step(teacher_step.action)

                    buffer.store(
                        observation,
                        action_history,
                        teacher_step.action,
                        teacher_step.value,
                    )

                    observation = env_step.observation
                    action_history = env_step.action_history

                    progress_bar.update()

                    done = env_step.terminated or env_step.truncated
                    epoch_ended = step + 1 == self._config["steps_per_epoch"]
                    if done or epoch_ended:
                        observation, action_history = self._env.reset()

            stats: dict[str, list[float]] = {
                "loss/action": [],
                "loss/value": [],
                "loss/entropy": [],
                "loss/total": [],
                "train/grad_norm": [],
            }

            for batch in buffer.get(self._config["batch_size"], self._student.device):
                self._optimizer.zero_grad()

                output: ActorCritic.Output = self._student(
                    batch.observations, batch.action_histories
                )

                action_loss = (
                    -output.action_distribution.log_prob(batch.actions)
                    .mean(dim=0)
                    .sum()
                )
                action_loss = action_loss * self._config["action_loss_weight"]

                value_loss = nn.functional.mse_loss(output.values, batch.values)
                value_loss = value_loss * self._config["value_loss_weight"]

                if self._config["entropy_loss_weight"] == 0.0:
                    # Some distributions do not implement entropy(), and calling it will
                    # raise NotImplementedError.
                    entropy_loss = torch.tensor(0.0, device=self._student.device)
                else:
                    entropy_loss = -output.action_distribution.entropy().mean()
                    entropy_loss = entropy_loss * self._config["entropy_loss_weight"]

                total_loss = action_loss + value_loss + entropy_loss
                total_loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(
                    trainable_params, self._config["max_grad_norm"]
                ).item()
                self._optimizer.step()

                stats["loss/action"].append(action_loss.item())
                stats["loss/value"].append(value_loss.item())
                stats["loss/entropy"].append(entropy_loss.item())
                stats["loss/total"].append(total_loss.item())
                stats["train/grad_norm"].append(grad_norm)

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
        **kwargs: Unpack[BehaviorCloningConfig],
    ) -> Self:
        with (checkpoint_dir / "config.json").open(encoding="utf-8") as file:
            config: BehaviorCloningConfig = json.load(file)
        for key, value in kwargs.items():
            match key:
                case "env_config":
                    config["env_config"].update(value)
                case "actor_critic_config":
                    config["actor_critic_config"].update(value)
                case _:
                    config[key] = value

        behavior_cloning = BehaviorCloning(device, **config)
        behavior_cloning._student.load_state_dict(
            torch.load(checkpoint_dir / "actor_critic.pth", map_location=device)
        )
        if not skip_optimizer:
            behavior_cloning._optimizer.load_state_dict(
                torch.load(checkpoint_dir / "optimizer.pth", map_location=device)
            )

        # Override the learning rate if specified in kwargs.
        if "learning_rate" in kwargs:
            for param_group in behavior_cloning._optimizer.param_groups:
                if param_group["lr"] == config["learning_rate"]:
                    continue
                print(
                    "Overriding learning rate: "
                    f"{param_group['lr']} -> {config['learning_rate']}"
                )
                param_group["lr"] = config["learning_rate"]

        return behavior_cloning

    @staticmethod
    @override
    def default_config() -> BehaviorCloningConfig:
        return {
            "env_config": Env.default_config(),
            "teacher_checkpoint_dir": None,
            "actor_critic_config": ActorCritic.default_config(),
            "log_dir": None,
            "checkpoint_dir": None,
            "checkpoint_interval": 10,
            "learning_rate": 3e-4,
            "epoch_start": 0,
            "epoch_stop": 1000,
            "steps_per_epoch": 2000,
            "batch_size": 64,
            "action_loss_weight": 1.0,
            "value_loss_weight": 0.5,
            "entropy_loss_weight": 0.01,
            "max_grad_norm": 0.5,
        }

    def _log_config_and_stats(
        self, writer: SummaryWriter, stats: dict[str, list[float]], epoch: int
    ) -> None:
        for tag in (
            "learning_rate",
            "action_loss_weight",
            "value_loss_weight",
            "max_grad_norm",
        ):
            writer.add_scalar(f"config/{tag}", self._config[tag], epoch)

        for tag, stat_list in stats.items():
            writer.add_scalar(tag, np.mean(stat_list), epoch)

    def _save_checkpoint(self, epoch: int) -> None:
        epoch_dir = Path(self._config["checkpoint_dir"], f"epoch-{epoch:04d}")
        epoch_dir.mkdir(parents=True, exist_ok=True)

        with (epoch_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(self._config, file, indent=4)

        torch.save(self._student.state_dict(), epoch_dir / "actor_critic.pth")
        torch.save(self._optimizer.state_dict(), epoch_dir / "optimizer.pth")


class _BehaviorCloningBuffer:

    class Batch(NamedTuple):
        observations: torch.Tensor
        """`(batch_size, observation_shape...)`"""

        action_histories: torch.Tensor | None
        """`(batch_size, action_history_shape...)`"""

        actions: torch.Tensor
        """`(batch_size, action_dim)`"""

        values: torch.Tensor
        """`(batch_size,)`"""

    def __init__(
        self,
        size: int,
        observation_shape: tuple[int, ...],
        action_history_shape: tuple[int, ...],
        action_dim: int,
    ) -> None:
        self._size = size

        self._observations = np.empty((size, *observation_shape), dtype=np.float32)
        if action_history_shape[0] == 0:
            self._action_histories = None
        else:
            self._action_histories = np.empty(
                (size, *action_history_shape), dtype=np.float32
            )
        self._actions = np.empty((size, action_dim), dtype=np.float32)
        self._values = np.empty(size, dtype=np.float32)

        self._index = 0

    def store(
        self,
        observation: np.ndarray,
        action_history: np.ndarray | None,
        action: np.ndarray,
        value: float,
    ) -> None:
        self._observations[self._index] = observation
        if self._action_histories is not None:
            self._action_histories[self._index] = action_history
        self._actions[self._index] = action
        self._values[self._index] = value

        self._index += 1

    def get(self, batch_size: int, device: torch.device) -> Iterator[Batch]:
        assert self._index == self._size, "Buffer is not full"
        self._index = 0

        observations = torch.as_tensor(self._observations, device=device)
        if self._action_histories is None:
            action_histories = None
        else:
            action_histories = torch.as_tensor(self._action_histories, device=device)
        actions = torch.as_tensor(self._actions, device=device)
        values = torch.as_tensor(self._values, device=device)

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
                actions[batch_indices],
                values[batch_indices],
            )
