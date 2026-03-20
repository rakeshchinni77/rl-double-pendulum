"""CLI entrypoint for training PPO on DoublePendulumEnv."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.training.train_pipeline import TrainConfig, train_pipeline


def _normalize_save_path(save_path: str) -> str:
	"""Normalize common path typo: 'modelsxxx' -> 'models/xxx'."""
	normalized = save_path.strip()
	if normalized.startswith("models") and not normalized.startswith(("models/", "models\\")):
		suffix = normalized[len("models"):].lstrip("/\\")
		if suffix:
			return str(Path("models") / suffix)
	return normalized


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
	"""Load YAML config file if present; return empty config otherwise."""
	path = Path(config_path)
	if not path.exists():
		return {}
	with path.open(mode="r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	return data if isinstance(data, dict) else {}


def _build_parser_with_config(config: Dict[str, Any]) -> argparse.ArgumentParser:
	training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
	ppo_cfg = config.get("ppo", {}) if isinstance(config.get("ppo", {}), dict) else {}

	parser = argparse.ArgumentParser(description="Train PPO on DoublePendulumEnv")
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	parser.add_argument(
		"--timesteps",
		type=int,
		default=int(os.getenv("TIMESTEPS", training_cfg.get("timesteps", 500000))),
	)
	parser.add_argument(
		"--reward_type",
		type=str,
		default=os.getenv("REWARD_TYPE", training_cfg.get("reward_type", "shaped")),
		choices=["baseline", "shaped"],
	)
	parser.add_argument(
		"--save_path",
		type=str,
		default=os.getenv("MODEL_PATH", training_cfg.get("save_path", "models/ppo_model")),
		help="Path without extension preferred; SB3 saves .zip model",
	)
	parser.add_argument("--log_dir", type=str, default=os.getenv("LOG_DIR", training_cfg.get("log_dir", "logs")))
	parser.add_argument("--seed", type=int, default=int(training_cfg.get("seed", 42)))
	parser.add_argument("--learning_rate", type=float, default=float(ppo_cfg.get("learning_rate", 3e-4)))
	parser.add_argument("--n_steps", type=int, default=int(ppo_cfg.get("n_steps", 1024)))
	parser.add_argument("--batch_size", type=int, default=int(ppo_cfg.get("batch_size", 64)))
	parser.add_argument("--gamma", type=float, default=float(ppo_cfg.get("gamma", 0.99)))
	parser.add_argument("--device", type=str, default=str(ppo_cfg.get("device", "auto")))
	parser.add_argument("--verbose", type=int, default=int(ppo_cfg.get("verbose", 1)))
	return parser


def build_arg_parser() -> argparse.ArgumentParser:
	# Parse --config first so YAML can provide centralized defaults.
	bootstrap_parser = argparse.ArgumentParser(add_help=False)
	bootstrap_parser.add_argument("--config", type=str, default="configs/config.yaml")
	bootstrap_args, _ = bootstrap_parser.parse_known_args()
	config = _load_yaml_config(bootstrap_args.config)
	return _build_parser_with_config(config)


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	normalized_save_path = _normalize_save_path(args.save_path)
	if normalized_save_path != args.save_path:
		print(f"Normalized save_path from '{args.save_path}' to '{normalized_save_path}'")

	config = TrainConfig(
		timesteps=args.timesteps,
		reward_type=args.reward_type,
		save_path=normalized_save_path,
		log_dir=args.log_dir,
		seed=args.seed,
		learning_rate=args.learning_rate,
		n_steps=args.n_steps,
		batch_size=args.batch_size,
		gamma=args.gamma,
		device=args.device,
		verbose=args.verbose,
	)

	result = train_pipeline(config)
	print("Training complete")
	print(f"model_path={result['model_path']}")
	print(f"model_path_abs={Path(result['model_path']).resolve()}")
	print(f"metrics_csv={result['metrics_csv']}")
	print(f"monitor_csv={result['monitor_csv']}")


if __name__ == "__main__":
	main()
