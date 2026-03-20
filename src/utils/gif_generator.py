"""Utilities for capturing pygame frames and generating GIFs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
import pygame


def capture_pygame_frame(surface: Optional[pygame.Surface]) -> Optional[np.ndarray]:
	"""Capture current pygame surface as RGB ndarray (H, W, C)."""
	if surface is None:
		return None

	# pygame returns (W, H, C); transpose to (H, W, C) for imageio.
	frame = pygame.surfarray.array3d(surface)
	frame = np.transpose(frame, (1, 0, 2))
	return frame


def save_gif(frames: List[np.ndarray], output_path: str, fps: int = 30) -> str:
	"""Persist a list of RGB frames to GIF."""
	if not frames:
		raise ValueError("No frames provided for GIF generation")

	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	imageio.mimsave(str(output), frames, fps=fps)
	return str(output)
