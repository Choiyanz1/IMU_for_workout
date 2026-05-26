from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.micro_macro_recognition import _load_config, _load_streams


def main() -> None:
    cfg = _load_config(Path("configs/micro_macro_recognition_stage3_40ep.yaml"))
    for modes in (["sets"], ["whole"], ["sets", "whole"]):
        streams, _, _ = _load_streams(cfg, modes)
        phase_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        total_samples = 0
        for _, df in streams:
            total_samples += int(len(df))
            if "phase" in df.columns:
                for k, v in df["phase"].astype(str).value_counts().to_dict().items():
                    phase_counts[str(k)] = phase_counts.get(str(k), 0) + int(v)
            if "action_type" in df.columns:
                for k, v in df["action_type"].astype(str).value_counts().to_dict().items():
                    action_counts[str(k)] = action_counts.get(str(k), 0) + int(v)
        print(f"modes={modes} streams={len(streams)} samples={total_samples}")
        print(f"phase_counts={phase_counts}")
        print(f"action_counts={action_counts}")
        print("---")

    streams, _, _ = _load_streams(cfg, ["whole"])
    print("whole-session examples")
    for stream_id, df in streams[:10]:
        print(stream_id)
        print(f"  samples={len(df)}")
        print(f"  phase_counts={df['phase'].astype(str).value_counts().to_dict()}")
        print(f"  action_counts={df['action_type'].astype(str).value_counts().head().to_dict()}")


if __name__ == "__main__":
    main()
