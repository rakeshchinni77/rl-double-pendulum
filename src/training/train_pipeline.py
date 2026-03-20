"""Training pipeline for PPO on DoublePendulumEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import gymnasium as gym

from stable_baselines3.common.monitor import Monitor

from src.agents.ppo_agent import PPOAgent
from src.env.environment import DoublePendulumEnv


class SB3CompatibilityWrapper(gym.Wrapper):
	"""Adapts reset/step signatures for stable-baselines3 compatibility."""

	def reset(self, **kwargs):
		out = self.env.reset()
		if isinstance(out, tuple) and len(out) == 2:
			return out
		return out, {}

	def step(self, action):
		out = self.env.step(action)
		if len(out) == 5:
			return out
		if len(out) == 4:
			obs, reward, done, info = out
			return obs, reward, done, False, info
		raise ValueError("Unexpected step() return signature")


@dataclass
class TrainConfig:
	timesteps: int = 200_000
	reward_type: str = "shaped"
	save_path: str = "models/ppo_model"
	log_dir: str = "logs"
	seed: int = 42
	learning_rate: float = 3e-4
	n_steps: int = 1024
	batch_size: int = 64
	gamma: float = 0.99
	device: str = "auto"
	verbose: int = 1


def _build_training_env(config: TrainConfig) -> Tuple[gym.Env, str, str]:
	"""Create wrapped env and logging paths used during training."""
	logs_dir = Path(config.log_dir)
	logs_dir.mkdir(parents=True, exist_ok=True)

	metrics_csv_path = logs_dir / f"training_metrics_{config.reward_type}.csv"
	monitor_csv_path = logs_dir / f"monitor_{config.reward_type}.csv"

	base_env = DoublePendulumEnv(
		reward_type=config.reward_type,
		enable_csv_logging=True,
		log_path=str(metrics_csv_path),
	)
	env = SB3CompatibilityWrapper(base_env)
	env = Monitor(env, filename=str(monitor_csv_path))
	return env, str(metrics_csv_path), str(monitor_csv_path)


def train_pipeline(config: TrainConfig) -> Dict[str, Any]:
	"""Run complete PPO training job and persist artifacts."""
	env, metrics_csv_path, monitor_csv_path = _build_training_env(config)

	# Keep PPO valid while honoring small CLI timesteps in smoke runs.
	adjusted_n_steps = max(8, min(config.n_steps, config.timesteps))
	adjusted_batch_size = min(config.batch_size, adjusted_n_steps)

	agent = PPOAgent(
		env=env,
		learning_rate=config.learning_rate,
		n_steps=adjusted_n_steps,
		batch_size=adjusted_batch_size,
		gamma=config.gamma,
		seed=config.seed,
		device=config.device,
		tensorboard_log=str(Path(config.log_dir) / "tb"),
		verbose=config.verbose,
	)

	agent.train(total_timesteps=config.timesteps)
	agent.save(config.save_path)

	env.close()

	return {
		"model_path": str(Path(config.save_path).with_suffix(".zip")),
		"reward_type": config.reward_type,
		"timesteps": config.timesteps,
		"metrics_csv": metrics_csv_path,
		"monitor_csv": monitor_csv_path,
	}
