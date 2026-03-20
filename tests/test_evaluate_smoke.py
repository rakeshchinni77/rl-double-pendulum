"""Phase 4 smoke test: validates evaluate.py and GIF creation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    model_path = root / "models" / "phase3_smoke_model.zip"
    gif_path = root / "media" / "phase4_eval_smoke.gif"

    if not model_path.exists():
        return 1

    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"

    cmd = [
        sys.executable,
        str(root / "evaluate.py"),
        "--model_path",
        str(model_path),
        "--reward_type",
        "shaped",
        "--episodes",
        "1",
        "--max_steps",
        "64",
        "--gif_path",
        str(gif_path),
    ]
    proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        return 1

    assert gif_path.exists(), f"Expected GIF not found: {gif_path}"
    assert gif_path.stat().st_size > 0, "GIF exists but is empty"

    print("✓ evaluate.py CLI smoke test passed")
    print(f"✓ gif artifact: {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
