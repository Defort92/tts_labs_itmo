"""Apply the trained lab-2 pause model to new Russian text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pause_model import PausePredictor


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_model.joblib",
    )
    args = parser.parse_args()
    predictor = PausePredictor(args.model)
    print(json.dumps(predictor.predict_text(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
