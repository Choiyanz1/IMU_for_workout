import json
from pathlib import Path
import numpy as np
from collections import defaultdict

base = Path('artifacts/baseline_comparison/per_action_plain_rf_7fold')
all_folds = []
for action_dir in base.iterdir():
    if action_dir.is_dir():
        for fold_file in action_dir.glob('fold_*.json'):
            d = json.load(open(fold_file))
            all_folds.append(d)

f1_50s = [d['micro_f1_at_50'] for d in all_folds if 'micro_f1_at_50' in d]
print(f'Overall IoU-F1@50 across {len(f1_50s)} folds: {np.mean(f1_50s):.4f} ± {np.std(f1_50s):.4f}')

by_action = defaultdict(list)
for d in all_folds:
    if 'micro_f1_at_50' in d and 'action' in d:
        by_action[d['action']].append(d['micro_f1_at_50'])

print('\nPer-action IoU-F1@50:')
for action in sorted(by_action.keys()):
    vals = by_action[action]
    print(f'  {action:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

# Also compute Rep F1 per action for comparison
by_action_rep = defaultdict(list)
for d in all_folds:
    if 'rep_f1' in d and 'action' in d:
        by_action_rep[d['action']].append(d['rep_f1'])

print('\nPer-action Rep F1:')
for action in sorted(by_action_rep.keys()):
    vals = by_action_rep[action]
    print(f'  {action:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
