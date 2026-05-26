"""
Smoke test for rf_parser_ablation.py on a tiny subset.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from train.micro_macro_recognition import _load_streams
from scripts.new_c_pipeline.rf_parser_ablation import (
    ParserAblationConfig, run_ablation, PhaseCompareConfig,
)

raw = yaml.safe_load(open("config.yaml"))
streams, subjects, actions = _load_streams(raw, ["sets"])

# Use only 2 subjects: one for train, one for test
train_subjects = subjects[:1]
test_subjects = subjects[1:2]

train_streams = [(sid, df) for sid, df in streams if any(sid.startswith(f"{s}/") for s in train_subjects)]
test_streams = [(sid, df) for sid, df in streams if any(sid.startswith(f"{s}/") for s in test_subjects)]

print(f"Train subjects: {train_subjects}, streams: {len(train_streams)}")
print(f"Test subjects: {test_subjects}, streams: {len(test_streams)}")

cfg = ParserAblationConfig()
base_cfg = PhaseCompareConfig()

run_ablation(train_streams, test_streams, cfg, base_cfg, Path("artifacts/rf_parser_ablation_smoke"))
