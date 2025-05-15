from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict, Unpack, override

import numpy as np
import torch
from torch import nn
from torch.distributions import (
    AffineTransform,
    Categorical,
    ComposeTransform,
    Dirichlet,
    Distribution,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

from .configurable import Configurable

_ACTION_DIM = 3


_DISCRETE_ACTION_TABLE = np.array(
    [
        [0.0, 0.0, 0.0],  # 0: do nothing
        [-0.6, 0.0, 0.0],  # 1: steer left
        [0.6, 0.0, 0.0],  # 2: steer right
        [0.0, 0.2, 0.0],  # 3: gas
        [0.0, 0.0, 0.8],  # 4: brake
    ],
    dtype=np.float32,
)
"""`(n_discrete_actions, action_dim)`"""


_MULTI_CATEGORICAL_ACTION_TABLE = np.array(
    [
        [-0.6, -0.3, 0.0, 0.3, 0.6],  # 0: steering
        [0.0, 0.2, 0.4, 0.6, 0.8],  # 1: gas
        [0.0, 0.2, 0.4, 0.6, 0.8],  # 2: braking
    ],
    dtype=np.float32,
)
"""`(action_dim, n_discrete_actions)`"""


class ActorCriticConfig(TypedDict, total=False):
    frame_stack_size: int
    action_stack_size: int
    action_distribution_type: Literal[
        "categorical", "dirichlet", "multi_categorical", "normal", "normal_tanh"
    ]
    frozen_weights: list[str] | None


class ActorCritic(nn.Module, Configurable[ActorCriticConfig]):

    class Output(NamedTuple):
        action_distribution: Distribution
        values: torch.Tensor
        """`(batch_size,)`"""

    class Step(NamedTuple):
        action_event: np.ndarray | int
        """`(action_dim,)` or `(n_discrete_actions,)` or scalar"""

        action_log_prob: float

        action: np.ndarray
        """`(action_dim,)`"""

        value: float

    action_head: "ActionHeadBase"

    def __init__(self, **kwargs: Unpack[ActorCriticConfig]) -> None:
        nn.Module.__init__(self)
        Configurable.__init__(self, **kwargs)

        self.cnn = nn.Sequential(
            # (batch_size, frame_stack_size, 96, 96)
            nn.Conv2d(self._config["frame_stack_size"], 8, kernel_size=4, stride=2),
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

        latent_features = 256 + self._config["action_stack_size"] * _ACTION_DIM
        match self._config["action_distribution_type"]:
            case "categorical":
                self.action_head = ActionHeadCategorical(latent_features)
            case "dirichlet":
                self.action_head = ActionHeadDirichlet(latent_features)
            case "multi_categorical":
                self.action_head = ActionHeadMultiCategorical(latent_features)
            case "normal":
                self.action_head = ActionHeadNormal(latent_features)
            case "normal_tanh":
                self.action_head = ActionHeadNormalTanh(latent_features)
            case _:
                raise ValueError(
                    "Invalid action distribution type: "
                    f"{self._config['action_distribution_type']}"
                )

        self.value_head = nn.Sequential(
            # (batch_size, 256)
            nn.Linear(latent_features, 128),
            # (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            # (batch_size, 64)
            nn.ReLU(),
            nn.Linear(64, 1),
            # (batch_size, 1)
        )

        self.apply(self._init_weights)
        self.freeze_weights()

    @property
    def device(self) -> torch.device:
        return self.cnn[0].weight.device

    @override
    def forward(
        self, observations: torch.Tensor, action_history: torch.Tensor | None
    ) -> Output:
        # observations: (batch_size, frame_stack_size, height, width)
        # action_history: (batch_size, action_stack_size, action_dim)

        # (batch_size, 256)
        latents = self.cnn(observations)
        if action_history is not None:
            # (batch_size, action_stack_size * action_dim)
            action_history = action_history.flatten(start_dim=1)
            # (batch_size, latent_features)
            latents = torch.cat((latents, action_history), dim=1)

        action_distribution: Distribution = self.action_head(latents)
        # (batch_size, 1) -> (batch_size,)
        values: torch.Tensor = self.value_head(latents).squeeze(dim=1)
        return self.Output(action_distribution, values)

    def step(
        self,
        observation: np.ndarray,
        action_history: np.ndarray | None,
        deterministic: bool = False,
    ) -> Step:
        # observation: (frame_stack_size, height, width)
        # action_history: (action_stack_size, action_dim)

        # (1, frame_stack_size, height, width)
        observation = torch.as_tensor(observation, device=self.device).unsqueeze(0)

        # (1, 256)
        latents = self.cnn(observation)
        if action_history is not None:
            # (1, action_stack_size * action_dim)
            action_history = (
                torch.as_tensor(action_history, device=self.device)
                .flatten()
                .unsqueeze(0)
            )
            # (1, latent_features)
            latents = torch.cat((latents, action_history), dim=1)

        action_head_step = self.action_head.step(latents, deterministic=deterministic)
        # (1,)
        value: torch.Tensor = self.value_head(latents)
        return self.Step(
            action_head_step.action_event,
            action_head_step.action_log_prob,
            action_head_step.action,
            value.item(),
        )

    def freeze_weights(self) -> None:
        frozen_names: list[str] = []
        for name, param in self.named_parameters():
            frozen = self._is_frozen_weight(name)
            if frozen:
                frozen_names.append(name)
            param.requires_grad_(not frozen)
        if len(frozen_names) > 0:
            print(f"Frozen weights: {', '.join(frozen_names)}")

    def unfreeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def load_weights(
        self, checkpoint_dir: Path, skipped_weights: Sequence[str] = ()
    ) -> None:
        state_dict = torch.load(
            checkpoint_dir / "actor_critic.pth", map_location=self.device
        )
        skipped_names: list[str] = []
        for name in tuple(state_dict.keys()):
            if any(
                map(
                    lambda pattern: name == pattern or name.startswith(f"{pattern}."),
                    skipped_weights,
                )
            ):
                skipped_names.append(name)
                del state_dict[name]
        if len(skipped_names) > 0:
            print(f"Skipped weights: {', '.join(skipped_names)}")
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict)

    @staticmethod
    @override
    def default_config() -> ActorCriticConfig:
        return {
            "frame_stack_size": 4,
            "action_stack_size": 0,
            "action_distribution_type": "categorical",
            "frozen_weights": None,
        }

    def _is_frozen_weight(self, name: str) -> bool:
        if self._config["frozen_weights"] is None:
            return False
        return any(
            map(
                lambda pattern: name == pattern or name.startswith(f"{pattern}."),
                self._config["frozen_weights"],
            )
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            return
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActionHeadBase(nn.Module, ABC):

    class Step(NamedTuple):
        action_event: np.ndarray | int
        """`(action_dim,)` or `(n_discrete_actions,)` or scalar"""

        action_log_prob: float

        action: np.ndarray
        """`(action_dim,)`"""

    @abstractmethod
    def action_event_shape(self) -> tuple[int, ...]: ...

    @abstractmethod
    def action_event_dtype(self) -> np.dtype: ...

    @abstractmethod
    def step(self, latents: torch.Tensor, deterministic: bool = False) -> Step: ...


class ActionHeadCategorical(ActionHeadBase):

    def __init__(self, latent_features: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            # (batch_size, latent_features)
            nn.Linear(latent_features, 128),
            # (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            # (batch_size, 64)
            nn.ReLU(),
            nn.Linear(64, _DISCRETE_ACTION_TABLE.shape[0]),
            # (batch_size, n_discrete_actions)
        )

    @override
    def forward(self, latents: torch.Tensor) -> Categorical:
        # latents: (batch_size, latent_features)

        # (batch_size, n_discrete_actions)
        logits = self.mlp(latents)
        return Categorical(logits=logits)

    @override
    def action_event_shape(self) -> tuple[int, ...]:
        return ()

    @override
    def action_event_dtype(self) -> np.dtype:
        return np.int64

    @override
    def step(
        self, latents: torch.Tensor, deterministic: bool = False
    ) -> ActionHeadBase.Step:
        # latents: (1, latent_features)

        action_distribution: Categorical = self(latents)
        # (1,)
        if not deterministic:
            action_event = action_distribution.sample()
        else:
            action_event = action_distribution.probs.argmax(dim=-1)
        action_log_prob = action_distribution.log_prob(action_event)

        # Scalar
        action_event = action_event.item()
        action_log_prob = action_log_prob.item()
        # (action_dim,)
        action = _DISCRETE_ACTION_TABLE[action_event]

        return self.Step(action_event, action_log_prob, action)


class ActionHeadDirichlet(ActionHeadBase):

    def __init__(self, latent_features: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            # (batch_size, latent_features)
            nn.Linear(latent_features, 128),
            # (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            # (batch_size, 64)
            nn.ReLU(),
            nn.Linear(64, _DISCRETE_ACTION_TABLE.shape[0]),
            # (batch_size, n_discrete_actions)
        )

    @override
    def forward(self, latents: torch.Tensor) -> Dirichlet:
        # latents: (batch_size, latent_features)

        # (batch_size, n_discrete_actions)
        concentration = self.mlp(latents)
        concentration = nn.functional.softplus(concentration)
        concentration = concentration + 1e-6
        return Dirichlet(concentration)

    @override
    def action_event_shape(self) -> tuple[int, ...]:
        return (_DISCRETE_ACTION_TABLE.shape[0],)

    @override
    def action_event_dtype(self) -> np.dtype:
        return np.float32

    @override
    def step(
        self, latents: torch.Tensor, deterministic: bool = False
    ) -> ActionHeadBase.Step:
        # latents: (1, latent_features)

        action_distribution: Dirichlet = self(latents)
        # (1, n_discrete_actions)
        if not deterministic:
            action_event = action_distribution.sample()
        else:
            action_event = action_distribution.mean
        # (1,)
        action_log_prob = action_distribution.log_prob(action_event)

        # (n_discrete_actions,)
        action_event = action_event.squeeze(0).cpu().numpy()
        # Scalar
        action_log_prob = action_log_prob.item()
        # (action_dim,)
        action = action_event @ _DISCRETE_ACTION_TABLE

        return self.Step(action_event, action_log_prob, action)


class ActionHeadMultiCategorical(ActionHeadBase):

    def __init__(self, latent_features: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            # (batch_size, latent_features)
            nn.Linear(latent_features, 128),
            # (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            # (batch_size, 64)
            nn.ReLU(),
            nn.Linear(64, _MULTI_CATEGORICAL_ACTION_TABLE.size),
            # (batch_size, action_dim * n_discrete_actions)
        )

    @override
    def forward(self, latents: torch.Tensor) -> Categorical:
        # latents: (batch_size, latent_features)

        # (batch_size, action_dim * n_discrete_actions)
        logits = self.mlp(latents)
        # (batch_size, action_dim, n_discrete_actions)
        logits = logits.reshape(
            logits.size(dim=0), *_MULTI_CATEGORICAL_ACTION_TABLE.shape
        )
        return Categorical(logits=logits)

    @override
    def action_event_shape(self) -> tuple[int, ...]:
        return (_ACTION_DIM,)

    @override
    def action_event_dtype(self) -> np.dtype:
        return np.int64

    @override
    def step(
        self, latents: torch.Tensor, deterministic: bool = False
    ) -> ActionHeadBase.Step:
        # latents: (1, latent_features)

        action_distribution: Categorical = self(latents)
        # (1, action_dim)
        if not deterministic:
            action_event = action_distribution.sample()
        else:
            action_event = action_distribution.probs.argmax(dim=-1)
        # (1, action_dim)
        action_log_prob = action_distribution.log_prob(action_event)

        # (action_dim,)
        action_event = action_event.squeeze(0).cpu().numpy()
        # Scalar
        action_log_prob = action_log_prob.sum().item()
        # (action_dim,)
        action = _MULTI_CATEGORICAL_ACTION_TABLE[[0, 1, 2], action_event]

        return self.Step(action_event, action_log_prob, action)


class ActionHeadNormal(ActionHeadBase):

    def __init__(self, latent_features: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            # (batch_size, latent_features)
            nn.Linear(latent_features, 128),
            # (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            # (batch_size, 64)
            nn.ReLU(),
            nn.Linear(64, _ACTION_DIM),
            # (batch_size, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros((_ACTION_DIM,), dtype=torch.float32))

    @override
    def forward(self, latents: torch.Tensor) -> Normal:
        # latents: (batch_size, latent_features)

        # (batch_size, action_dim)
        mean = self.mlp(latents)
        std = self.log_std.exp()
        return Normal(mean, std)

    @override
    def action_event_shape(self) -> tuple[int, ...]:
        return (_ACTION_DIM,)

    @override
    def action_event_dtype(self) -> np.dtype:
        return np.float32

    @override
    def step(
        self, latents: torch.Tensor, deterministic: bool = False
    ) -> ActionHeadBase.Step:
        # latents: (1, latent_features)

        action_distribution: Normal = self(latents)
        # (1, action_dim)
        if not deterministic:
            action_event = action_distribution.sample()
        else:
            action_event = action_distribution.mean
        # (1, action_dim)
        action_log_prob = action_distribution.log_prob(action_event)

        # (action_dim,)
        action_event = action_event.squeeze(0).cpu().numpy()
        # Scalar
        action_log_prob = action_log_prob.sum().item()
        # (action_dim,)
        action = action_event
        action = action.clip(min=[-1.0, 0.0, 0.0], max=1.0)

        return self.Step(action_event, action_log_prob, action)


class ActionHeadNormalTanh(ActionHeadBase):

    def __init__(self, latent_features: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            # (batch_size, latent_features)
            nn.Linear(latent_features, 128),
            # (batch_size, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            # (batch_size, 64)
            nn.ReLU(),
            nn.Linear(64, _ACTION_DIM),
            # (batch_size, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros((_ACTION_DIM,), dtype=torch.float32))

    @override
    def forward(self, latents: torch.Tensor) -> TransformedDistribution:
        # latents: (batch_size, latent_features)

        # (batch_size, action_dim)
        mean = self.mlp(latents)
        std = self.log_std.exp()
        normal = Normal(mean, std)

        transform = ComposeTransform(
            [
                TanhTransform(cache_size=1),
                AffineTransform(
                    torch.tensor(
                        [-1.0, 0.0, 0.0], dtype=torch.float32, device=latents.device
                    ),
                    torch.tensor(
                        [2.0, 1.0, 1.0], dtype=torch.float32, device=latents.device
                    ),
                ),
            ]
        )
        return TransformedDistribution(normal, transform)

    @override
    def action_event_shape(self) -> tuple[int, ...]:
        return (_ACTION_DIM,)

    @override
    def action_event_dtype(self) -> np.dtype:
        return np.float32

    @override
    def step(
        self, latents: torch.Tensor, deterministic: bool = False
    ) -> ActionHeadBase.Step:
        # latents: (1, latent_features)

        action_distribution: TransformedDistribution = self(latents)
        # (1, action_dim)
        if not deterministic:
            action_event = action_distribution.sample()
        else:
            action_event = action_distribution.base_dist.mean
            action_event = (
                torch.tensor(
                    [-1.0, 0.0, 0.0], dtype=torch.float32, device=latents.device
                )
                + torch.tensor(
                    [2.0, 1.0, 1.0], dtype=torch.float32, device=latents.device
                )
                * action_event.tanh()
            )
        # (1, action_dim)
        action_log_prob = action_distribution.log_prob(action_event)

        # (action_dim,)
        action_event = action_event.squeeze(0).cpu().numpy()
        # Scalar
        action_log_prob = action_log_prob.sum().item()
        # (action_dim,)
        action = action_event

        return self.Step(action_event, action_log_prob, action)
