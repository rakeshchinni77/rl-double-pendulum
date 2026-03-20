"""Evaluation pipeline for trained PPO agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from src.agents.ppo_agent import PPOAgent
from src.env.environment import DoublePendulumEnv
from src.utils.gif_generator import capture_pygame_frame, save_gif


@dataclass
class EvaluateConfig:
	model_path: str
	reward_type: str = "shaped"
	episodes: int = 3
	max_steps: int = 1000
	deterministic: bool = True
	render: bool = False
	save_gif_enabled: bool = True
	gif_path: str = "media/agent_final.gif"
	fps: int = 30


def evaluate_pipeline(config: EvaluateConfig) -> Dict[str, Any]:
	"""Run policy evaluation and optionally save a rollout GIF."""
	model_path = Path(config.model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	render_mode = "human" if (config.render or config.save_gif_enabled) else None
	env = DoublePendulumEnv(reward_type=config.reward_type, render_mode=render_mode)
	model = PPOAgent.load(str(model_path), env=None)

	episode_rewards: List[float] = []
	frames = []

	for _ in range(config.episodes):
		obs = env.reset()
		done = False
		total_reward = 0.0
		step_count = 0

		while not done and step_count < config.max_steps:
			action, _ = model.predict(obs, deterministic=config.deterministic)
			obs, reward, terminated, truncated, _ = env.step(action)
			total_reward += float(reward)
			step_count += 1

			if config.render or config.save_gif_enabled:
				env.render()

			if config.save_gif_enabled:
				frame = capture_pygame_frame(env.screen)
				if frame is not None:
					frames.append(frame)

			done = bool(terminated or truncated)

		episode_rewards.append(total_reward)

	gif_output = None
	if config.save_gif_enabled:
		gif_output = save_gif(frames, config.gif_path, fps=config.fps)

	env.close()

	mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
	std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0

	return {
		"model_path": str(model_path),
		"episodes": config.episodes,
		"max_steps": config.max_steps,
		"mean_reward": mean_reward,
		"std_reward": std_reward,
		"gif_path": gif_output,
	}
