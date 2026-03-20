"""CLI entrypoint for evaluating trained PPO agent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from src.evaluation.evaluate_pipeline import EvaluateConfig, evaluate_pipeline


def _normalize_model_path(model_path: str) -> str:
	"""Normalize common escaped-path typos for model artifacts."""
	normalized = model_path.strip()

	if normalized.startswith("models") and not normalized.startswith(("models/", "models\\")):
		suffix = normalized[len("models"):].lstrip("/\\")
		if suffix:
			normalized = str(Path("models") / suffix)

	# If caller omitted .zip, use stable-baselines3 convention.
	if not normalized.endswith(".zip") and Path(normalized + ".zip").exists():
		normalized = normalized + ".zip"

	return normalized


def _normalize_gif_path(gif_path: str) -> str:
	"""Normalize common escaped-path typos for media outputs."""
	normalized = gif_path.strip()
	if normalized.startswith("media") and not normalized.startswith(("media/", "media\\")):
		suffix = normalized[len("media"):].lstrip("/\\")
		if suffix:
			return str(Path("media") / suffix)
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
	eval_cfg = config.get("evaluation", {}) if isinstance(config.get("evaluation", {}), dict) else {}

	parser = argparse.ArgumentParser(description="Evaluate PPO on DoublePendulumEnv")
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	parser.add_argument("--model_path", type=str, required=True, help="Path to trained PPO .zip model")
	parser.add_argument(
		"--reward_type",
		type=str,
		default=str(eval_cfg.get("reward_type", "shaped")),
		choices=["baseline", "shaped"],
		help="Environment reward function to use during evaluation",
	)
	parser.add_argument("--episodes", type=int, default=int(eval_cfg.get("episodes", 3)))
	parser.add_argument("--max_steps", type=int, default=int(eval_cfg.get("max_steps", 1000)))
	parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions")
	parser.add_argument("--render", action="store_true", help="Render interactive pygame window")
	parser.add_argument("--no_gif", action="store_true", help="Disable GIF generation")
	parser.add_argument("--gif_path", type=str, default=str(eval_cfg.get("gif_path", "media/agent_final.gif")))
	parser.add_argument("--fps", type=int, default=int(eval_cfg.get("fps", 30)))
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

	normalized_model_path = _normalize_model_path(args.model_path)
	normalized_gif_path = _normalize_gif_path(args.gif_path)
	if normalized_model_path != args.model_path:
		print(f"Normalized model_path from '{args.model_path}' to '{normalized_model_path}'")
	if normalized_gif_path != args.gif_path:
		print(f"Normalized gif_path from '{args.gif_path}' to '{normalized_gif_path}'")

	config = EvaluateConfig(
		model_path=normalized_model_path,
		reward_type=args.reward_type,
		episodes=args.episodes,
		max_steps=args.max_steps,
		deterministic=not args.stochastic,
		render=args.render,
		save_gif_enabled=not args.no_gif,
		gif_path=normalized_gif_path,
		fps=args.fps,
	)
	result = evaluate_pipeline(config)

	print("Evaluation complete")
	print(f"model_path={result['model_path']}")
	print(f"mean_reward={result['mean_reward']:.4f}")
	print(f"std_reward={result['std_reward']:.4f}")
	if result["gif_path"]:
		print(f"gif_path={result['gif_path']}")


if __name__ == "__main__":
	main()
