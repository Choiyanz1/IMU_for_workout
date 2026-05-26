"""Quick smoke test: compare baseline vs velocity vs velocity+jerk features."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure scripts/ is on path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Now import the evaluate script
import evaluate_per_action_plain_rf_loso as eval_script


def run_variant(name: str, feature_mode: str = "baseline"):
    print(f"\n{'='*60}")
    print(f"VARIANT: {name}")
    print(f"FEATURE_MODE={feature_mode}")
    print(f"{'='*60}")

    os.environ["FEATURE_MODE"] = feature_mode

    # Override sys.argv and run
    original_argv = sys.argv
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    sys.argv = [
        "evaluate_per_action_plain_rf_loso.py",
        "--config", "config.yaml",
        "--subjects", "haoyu,kevin,yoru",
        "--output", f"artifacts/baseline_comparison/feature_test_{safe_name}",
    ]
    try:
        eval_script.main()
    finally:
        sys.argv = original_argv


def main():
    run_variant("BASELINE (63-dim)", "baseline")
    run_variant("+VELOCITY (126-dim)", "velocity")
    run_variant("+VELOCITY+JERK (189-dim)", "velocity_jerk")


if __name__ == "__main__":
    main()
