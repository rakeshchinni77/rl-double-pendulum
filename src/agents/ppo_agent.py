"""PPO agent wrapper for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from stable_baselines3 import PPO


class PPOAgent:
	"""Thin wrapper around Stable-Baselines3 PPO for project-level consistency."""

	def __init__(
		self,
		env: Any,
		learning_rate: float = 3e-4,
		n_steps: int = 2048,
		batch_size: int = 64,
		gamma: float = 0.99,
		device: str = "auto",
		seed: Optional[int] = 42,
		tensorboard_log: Optional[str] = None,
		verbose: int = 1,
	) -> None:
		self.env = env
		self.model = PPO(
			policy="MlpPolicy",
			env=env,
			learning_rate=learning_rate,
			n_steps=n_steps,
			batch_size=batch_size,
			gamma=gamma,
			seed=seed,
			device=device,
			tensorboard_log=tensorboard_log,
			verbose=verbose,
		)

	def train(self, total_timesteps: int, progress_bar: bool = False) -> PPO:
		"""Train PPO policy for the configured environment."""
		self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
		return self.model

	def save(self, save_path: str) -> None:
		"""Save trained model to disk."""
		path_obj = Path(save_path)
		path_obj.parent.mkdir(parents=True, exist_ok=True)
		self.model.save(str(path_obj))

	@staticmethod
	def load(model_path: str, env: Any = None, device: str = "auto") -> PPO:
		"""Load a trained PPO model."""
		return PPO.load(model_path, env=env, device=device)
