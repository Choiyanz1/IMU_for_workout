#!/usr/bin/env python3
"""Extract phase classification quality metrics from existing results."""
import json
from pathlib import Path
import numpy as np

base = Path('artifacts/baseline_comparison/per_action_plain_rf_7fold')
actions = ['db_bench_press', 'db_biceps_curl', 'db_rdl', 'db_shoulder_press',
           'db_squat', 'db_triceps_curl', 'db_weighted_crunch', 'one_arm_db_row']

print('=' * 80)
print('Phase Classification Quality Analysis (Per-Action Plain RF, 7-fold LOSO)')
print('=' * 80)

all_accs = []
all_macro_f1s = []
all_start_mae = []
all_end_mae = []
all_transition_mae = []

per_action = {}

for action in actions:
    summary_file = base / action / 'summary.json'
    if not summary_file.exists():
        print(f"[WARN] {summary_file} not found")
        continue
    with open(summary_file) as f:
        data = json.load(f)

    fold_accs = [r['micro_sample_accuracy'] for r in data['fold_results']
                 if 'micro_sample_accuracy' in r]
    fold_macro_f1s = [r['micro_sample_macro_f1'] for r in data['fold_results']
                      if 'micro_sample_macro_f1' in r]
    fold_start = [r['start_mae_ms'] for r in data['fold_results']
                  if 'start_mae_ms' in r]
    fold_end = [r['end_mae_ms'] for r in data['fold_results']
                if 'end_mae_ms' in r]
    fold_trans = [r['transition_mae_ms'] for r in data['fold_results']
                  if 'transition_mae_ms' in r]

    per_action[action] = {
        'acc_mean': np.mean(fold_accs), 'acc_std': np.std(fold_accs),
        'f1_mean': np.mean(fold_macro_f1s), 'f1_std': np.std(fold_macro_f1s),
        'start_mean': np.mean(fold_start), 'start_std': np.std(fold_start),
        'end_mean': np.mean(fold_end), 'end_std': np.std(fold_end),
        'trans_mean': np.mean(fold_trans), 'trans_std': np.std(fold_trans),
        'rep_f1': data['overall']['rep_f1'],
    }

    all_accs.extend(fold_accs)
    all_macro_f1s.extend(fold_macro_f1s)
    all_start_mae.extend(fold_start)
    all_end_mae.extend(fold_end)
    all_transition_mae.extend(fold_trans)

print(f"\n### Overall Sample-Level Phase Classification (n={len(all_accs)} folds)")
print(f"  Sample Accuracy:   {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
print(f"  Sample Macro F1:  {np.mean(all_macro_f1s):.4f} ± {np.std(all_macro_f1s):.4f}")
print(f"\n### Boundary Localization Error (MAE in ms)")
print(f"  Start MAE:         {np.mean(all_start_mae):.1f} ± {np.std(all_start_mae):.1f} ms")
print(f"  End MAE:           {np.mean(all_end_mae):.1f} ± {np.std(all_end_mae):.1f} ms")
print(f"  Transition MAE:   {np.mean(all_transition_mae):.1f} ± {np.std(all_transition_mae):.1f} ms")

print(f"\n### Per-Action Breakdown")
print(f"{'Action':<25} {'Rep F1':>8} {'Sample Acc':>12} {'Macro F1':>10} {'Trans MAE':>10}")
print('-' * 70)
for action in actions:
    if action in per_action:
        p = per_action[action]
        print(f"{action:<25} {p['rep_f1']:>8.3f} {p['acc_mean']:>12.3f} "
              f"{p['f1_mean']:>10.3f} {p['trans_mean']:>10.1f}")

# Save to file
output = {
    "model": "Per-Action Plain RF",
    "n_folds": len(all_accs),
    "overall": {
        "sample_accuracy_mean": float(np.mean(all_accs)),
        "sample_accuracy_std": float(np.std(all_accs)),
        "sample_macro_f1_mean": float(np.mean(all_macro_f1s)),
        "sample_macro_f1_std": float(np.std(all_macro_f1s)),
        "start_mae_ms_mean": float(np.mean(all_start_mae)),
        "start_mae_ms_std": float(np.std(all_start_mae)),
        "end_mae_ms_mean": float(np.mean(all_end_mae)),
        "end_mae_ms_std": float(np.std(all_end_mae)),
        "transition_mae_ms_mean": float(np.mean(all_transition_mae)),
        "transition_mae_ms_std": float(np.std(all_transition_mae)),
    },
    "per_action": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per_action.items()},
}

out_path = Path("artifacts/baseline_comparison/phase_classification_quality.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n[OK] Saved to {out_path}")
