"""Logging utilities for RL training metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


class CSVMetricLogger:
	"""Append-only CSV logger for training metrics."""

	REQUIRED_COLUMNS = ["timesteps", "mean_reward"]

	def __init__(self, file_path: str) -> None:
		self.file_path = Path(file_path)
		self.file_path.parent.mkdir(parents=True, exist_ok=True)
		self._initialized = False

	def initialize(self) -> None:
		"""Create/overwrite CSV and write header."""
		with self.file_path.open(mode="w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(self.REQUIRED_COLUMNS)
		self._initialized = True

	def log(self, timesteps: int, mean_reward: float) -> None:
		"""Append one metric record."""
		if not self._initialized:
			self.initialize()

		with self.file_path.open(mode="a", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow([int(timesteps), float(mean_reward)])

	def read_rows(self) -> List[Dict[str, str]]:
		"""Read all rows for post-processing/plotting."""
		if not self.file_path.exists():
			return []
		with self.file_path.open(mode="r", newline="", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			return list(reader)

