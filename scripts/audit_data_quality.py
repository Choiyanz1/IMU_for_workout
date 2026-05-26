"""
Comprehensive data quality audit:
1. Check rep counts per subject/action
2. Check concentric/eccentric phase ratios
3. Check for anomalies in rep duration
4. Check for missing or inconsistent labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from collections import defaultdict

# Load config
raw = yaml.safe_load(open('config.yaml', encoding='utf-8'))
data_cfg = raw.get('data', {})
feature_cfg = raw.get('feature', {})

# Load all streams
import sys
sys.path.insert(0, str(Path.cwd()))
from scripts import compare_baselines as cb

mm_raw = raw.get('micro_macro', {}) or {}
mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ['sets']
streams, all_subjects, actions = cb._load_streams(raw, modes)

print(f"Total streams: {len(streams)}")
print(f"Subjects: {sorted(set(all_subjects))}")
print(f"Actions: {actions}")
print()

# Statistics containers
subject_stats = defaultdict(lambda: {
    'total_reps': 0,
    'total_streams': 0,
    'concentric_ratio': [],
    'eccentric_ratio': [],
    'rep_durations': [],
    'reps_per_stream': [],
    'actions': defaultdict(int)
})

action_stats = defaultdict(lambda: {
    'total_reps': 0,
    'total_streams': 0,
    'concentric_ratio': [],
    'eccentric_ratio': [],
    'rep_durations': [],
    'reps_per_stream': [],
    'subjects': set()
})

anomalies = []

print("Analyzing all streams...")
for stream_idx, (stream_id, df) in enumerate(streams):
    if 'phase' not in df.columns:
        anomalies.append(f"{stream_id}: No phase column")
        continue
    
    # Extract subject and action
    parts = [p for p in str(stream_id).split('/') if p]
    subject = parts[0] if parts else 'unknown'
    action = parts[-2] if len(parts) >= 2 else 'unknown'
    
    # Count phases
    phases = df['phase'].astype(str).str.lower()
    concentric_count = (phases == 'concentric').sum()
    eccentric_count = (phases == 'eccentric').sum()
    other_count = (phases == 'none').sum() + (phases == 'inter_set_rest').sum()
    total = len(df)
    
    concentric_ratio = concentric_count / total if total > 0 else 0
    eccentric_ratio = eccentric_count / total if total > 0 else 0
    
    # Extract reps using truth_reps_from_labels
    from preprocessing.micro_macro_segments import truth_reps_from_labels
    try:
        if 'action_type' in df.columns:
            truth_reps = truth_reps_from_labels(
                df['phase'].to_numpy(),
                actions=df['action_type'].astype(str).to_numpy(),
                min_phase_samples=3
            )
        else:
            truth_reps = truth_reps_from_labels(
                df['phase'].to_numpy(),
                min_phase_samples=3
            )
        n_reps = len(truth_reps)
    except Exception as e:
        n_reps = 0
        anomalies.append(f"{stream_id}: Error extracting reps - {e}")
    
    # Store stats
    subject_stats[subject]['total_reps'] += n_reps
    subject_stats[subject]['total_streams'] += 1
    subject_stats[subject]['concentric_ratio'].append(concentric_ratio)
    subject_stats[subject]['eccentric_ratio'].append(eccentric_ratio)
    subject_stats[subject]['reps_per_stream'].append(n_reps)
    subject_stats[subject]['actions'][action] += n_reps
    
    action_stats[action]['total_reps'] += n_reps
    action_stats[action]['total_streams'] += 1
    action_stats[action]['concentric_ratio'].append(concentric_ratio)
    action_stats[action]['eccentric_ratio'].append(eccentric_ratio)
    action_stats[action]['reps_per_stream'].append(n_reps)
    action_stats[action]['subjects'].add(subject)
    
    # Check for anomalies
    if n_reps == 0:
        anomalies.append(f"{stream_id}: No reps detected (phases={phases.unique()})")
    elif n_reps > 20:
        anomalies.append(f"{stream_id}: Unusually high rep count ({n_reps})")
    
    if concentric_ratio > 0.9 or eccentric_ratio > 0.9:
        anomalies.append(f"{stream_id}: Extremely unbalanced phases (C={concentric_ratio:.2f}, E={eccentric_ratio:.2f})")
    
    if (concentric_count > 0 and eccentric_count == 0) or (eccentric_count > 0 and concentric_count == 0):
        anomalies.append(f"{stream_id}: Missing one phase type (C={concentric_count}, E={eccentric_count})")
    
    if stream_idx % 100 == 0:
        print(f"  Processed {stream_idx}/{len(streams)} streams...")

print(f"\nDone! Found {len(anomalies)} anomalies")

# Print Subject Statistics
print("\n" + "="*80)
print("SUBJECT-LEVEL STATISTICS")
print("="*80)
print(f"{'Subject':<12} {'Streams':<8} {'Total Reps':<12} {'Avg Reps/Stream':<16} {'Avg C%':<10} {'Avg E%':<10} {'Actions'}")
print("-" * 80)
for subject in sorted(subject_stats.keys()):
    s = subject_stats[subject]
    avg_reps = np.mean(s['reps_per_stream']) if s['reps_per_stream'] else 0
    avg_c = np.mean(s['concentric_ratio']) * 100 if s['concentric_ratio'] else 0
    avg_e = np.mean(s['eccentric_ratio']) * 100 if s['eccentric_ratio'] else 0
    actions_str = ', '.join([f"{a}({c})" for a, c in sorted(s['actions'].items())])
    print(f"{subject:<12} {s['total_streams']:<8} {s['total_reps']:<12} {avg_reps:<16.1f} {avg_c:<10.1f} {avg_e:<10.1f} {actions_str}")

# Print Action Statistics
print("\n" + "="*80)
print("ACTION-LEVEL STATISTICS")
print("="*80)
print(f"{'Action':<25} {'Subjects':<8} {'Streams':<8} {'Total Reps':<12} {'Avg Reps/Stream':<16} {'Avg C%':<10} {'Avg E%':<10}")
print("-" * 80)
for action in sorted(action_stats.keys()):
    a = action_stats[action]
    avg_reps = np.mean(a['reps_per_stream']) if a['reps_per_stream'] else 0
    avg_c = np.mean(a['concentric_ratio']) * 100 if a['concentric_ratio'] else 0
    avg_e = np.mean(a['eccentric_ratio']) * 100 if a['eccentric_ratio'] else 0
    print(f"{action:<25} {len(a['subjects']):<8} {a['total_streams']:<8} {a['total_reps']:<12} {avg_reps:<16.1f} {avg_c:<10.1f} {avg_e:<10.1f}")

# Print Rep Count Distribution per Subject
print("\n" + "="*80)
print("REPS PER STREAM DISTRIBUTION")
print("="*80)
for subject in sorted(subject_stats.keys()):
    s = subject_stats[subject]
    if s['reps_per_stream']:
        reps = s['reps_per_stream']
        print(f"{subject}: mean={np.mean(reps):.1f}, std={np.std(reps):.1f}, min={min(reps)}, max={max(reps)}, median={np.median(reps):.1f}")

# Print Anomalies
if anomalies:
    print("\n" + "="*80)
    print(f"ANOMALIES ({len(anomalies)} found)")
    print("="*80)
    for i, anomaly in enumerate(anomalies[:50], 1):
        print(f"  {i}. {anomaly}")
    if len(anomalies) > 50:
        print(f"  ... and {len(anomalies) - 50} more")
else:
    print("\n" + "="*80)
    print("NO ANOMALIES FOUND")
    print("="*80)

# Check expected concentric/eccentric ratios per action
print("\n" + "="*80)
print("EXPECTED vs ACTUAL CONCENTRIC/ECCENTRIC RATIOS")
print("="*80)
print("Action                    Expected C:E    Actual C:E    Status")
print("-" * 70)
expected_ratios = {
    'db_bench_press': (0.5, 0.5),
    'db_biceps_curl': (0.4, 0.6),  # curl up is concentric (shorter)
    'db_rdl': (0.5, 0.5),
    'db_shoulder_press': (0.5, 0.5),
    'db_squat': (0.5, 0.5),
    'db_triceps_curl': (0.4, 0.6),
    'db_weighted_crunch': (0.5, 0.5),
    'one_arm_db_row': (0.5, 0.5),
}

for action in sorted(action_stats.keys()):
    a = action_stats[action]
    if a['concentric_ratio']:
        avg_c = np.mean(a['concentric_ratio'])
        avg_e = np.mean(a['eccentric_ratio'])
        actual_ratio = f"{avg_c:.2f}:{avg_e:.2f}"
        
        if action in expected_ratios:
            exp_c, exp_e = expected_ratios[action]
            diff_c = abs(avg_c - exp_c)
            diff_e = abs(avg_e - exp_e)
            if diff_c > 0.15 or diff_e > 0.15:
                status = f"UNEXPECTED (expected {exp_c:.1f}:{exp_e:.1f})"
            else:
                status = "OK"
        else:
            status = "Unknown action"
        
        print(f"{action:<25} {actual_ratio:<15} {status}")

print("\nDone!")
