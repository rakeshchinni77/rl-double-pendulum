"""Plotting utilities for RL experiment analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load_metrics_csv(csv_path: str) -> pd.DataFrame:
	"""Load metrics with normalized columns: timesteps, mean_reward."""
	path = Path(csv_path)
	if not path.exists():
		raise FileNotFoundError(f"Metrics file not found: {path}")

	df = pd.read_csv(path)
	columns = {c.lower(): c for c in df.columns}

	if "timesteps" in columns and "mean_reward" in columns:
		out = df[[columns["timesteps"], columns["mean_reward"]]].copy()
		out.columns = ["timesteps", "mean_reward"]
		return out

	# Fallback for monitor-like outputs where reward may be named ep_rew_mean.
	if "timesteps" in columns and "ep_rew_mean" in columns:
		out = df[[columns["timesteps"], columns["ep_rew_mean"]]].copy()
		out.columns = ["timesteps", "mean_reward"]
		return out

	raise ValueError(
		f"CSV at {path} must contain columns (timesteps, mean_reward) "
		f"or (timesteps, ep_rew_mean). Found: {list(df.columns)}"
	)


def create_reward_comparison_plot(
	baseline_csv: str,
	shaped_csv: str,
	output_path: str = "reward_comparison.png",
) -> str:
	"""Generate comparison plot for baseline vs shaped reward runs."""
	baseline = _load_metrics_csv(baseline_csv)
	shaped = _load_metrics_csv(shaped_csv)

	plt.figure(figsize=(10, 6))
	plt.plot(
		baseline["timesteps"],
		baseline["mean_reward"],
		label="Baseline Reward",
		linewidth=2,
	)
	plt.plot(
		shaped["timesteps"],
		shaped["mean_reward"],
		label="Shaped Reward",
		linewidth=2,
	)

	plt.title("Reward Comparison: Baseline vs Shaped", fontsize=14)
	plt.xlabel("Timesteps", fontsize=12)
	plt.ylabel("Mean Reward", fontsize=12)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()

	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out, dpi=150)
	plt.close()
	return str(out)


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Create reward comparison plot")
	parser.add_argument("--baseline_csv", type=str, default="logs/training_metrics_baseline.csv")
	parser.add_argument("--shaped_csv", type=str, default="logs/training_metrics_shaped.csv")
	parser.add_argument("--output_path", type=str, default="reward_comparison.png")
	return parser


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()
	out = create_reward_comparison_plot(
		baseline_csv=args.baseline_csv,
		shaped_csv=args.shaped_csv,
		output_path=args.output_path,
	)
	print(f"Created plot: {out}")


if __name__ == "__main__":
	main()

