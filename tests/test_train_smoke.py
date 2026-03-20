"""Phase 3 smoke test: validates train.py CLI args and artifact generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    model_path = root / "models" / "phase3_smoke_model"
    metrics_path = root / "logs" / "training_metrics_baseline.csv"

    cmd = [
        sys.executable,
        str(root / "train.py"),
        "--timesteps",
        "32",
        "--reward_type",
        "baseline",
        "--save_path",
        str(model_path),
    ]

    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        return 1

    model_zip = Path(str(model_path) + ".zip")
    assert model_zip.exists(), f"Expected model artifact not found: {model_zip}"
    assert metrics_path.exists(), f"Expected metrics CSV not found: {metrics_path}"

    print("✓ train.py CLI smoke test passed")
    print(f"✓ model artifact: {model_zip}")
    print(f"✓ metrics csv: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
