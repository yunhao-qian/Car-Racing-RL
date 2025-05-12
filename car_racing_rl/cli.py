from pathlib import Path

import fire
import pygame
import torch

from .actor_critic import ActorCritic
from .car_racing import CarRacing
from .ppo import train_ppo_cli


def main() -> None:

    fire.Fire({"train": {"ppo": train_ppo_cli}, "run": _run_cli})


def _run_cli(
    checkpoint_dir: str,
    action_interval: int = 8,
    frame_stack_size: int = 4,
    temperature: float = 0.1,
) -> None:

    checkpoint_dir = Path(checkpoint_dir)

    env = CarRacing(
        "human",
        action_interval,
        frame_stack_size,
        reward_history_size=100,
        reward_history_min_threshold=-0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCritic(env.observation_space, env.action_space)
    actor_critic.to(device)
    actor_critic.load_state_dict(
        torch.load(checkpoint_dir / "actor_critic.pth", map_location=device)
    )

    with torch.no_grad():
        observation = env.reset()
        while not any(map(lambda event: event.type == pygame.QUIT, pygame.event.get())):
            actor_critic_step = actor_critic.step(observation, temperature)
            env_step = env.step(actor_critic_step.action)
            if env_step.terminated or env_step.truncated:
                observation = env.reset()
            else:
                observation = env_step.observation

    env.close()
