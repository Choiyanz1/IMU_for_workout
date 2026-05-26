"""
List ALL problematic data files with exact paths for manual inspection.
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
from preprocessing.micro_macro_segments import truth_reps_from_labels

mm_raw = raw.get('micro_macro', {}) or {}
mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ['sets']
streams, all_subjects, actions = cb._load_streams(raw, modes)

print("=" * 100)
print("DETAILED DATA QUALITY REPORT - ALL PROBLEMATIC STREAMS")
print("=" * 100)
print()

problems = defaultdict(list)

for stream_idx, (stream_id, df) in enumerate(streams):
    parts = [p for p in str(stream_id).split('/') if p]
    subject = parts[0] if parts else 'unknown'
    action = parts[-2] if len(parts) >= 2 else 'unknown'
    set_name = parts[-1] if parts else 'unknown'
    
    # Try to reconstruct original CSV path
    # stream_id format: subject/session/action/set
    if len(parts) >= 4:
        csv_path = f"datasets/raw_data/{'/'.join(parts)}/*.csv"
        folder_path = f"datasets/raw_data/{'/'.join(parts)}"
    else:
        csv_path = f"datasets/raw_data/{stream_id}/*.csv"
        folder_path = f"datasets/raw_data/{stream_id}"
    
    issue_list = []
    
    # Check 1: Rep count anomalies
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
    except:
        n_reps = 0
        issue_list.append("Cannot extract reps")
    
    if n_reps == 0:
        issue_list.append("ZERO reps")
    elif n_reps == 1:
        issue_list.append(f"Only 1 rep (suspicious)")
    elif n_reps < 8:
        issue_list.append(f"Very few reps ({n_reps})")
    elif n_reps > 15:
        issue_list.append(f"Unusually many reps ({n_reps})")
    
    # Check 2: Phase balance
    phases = df['phase'].astype(str).str.lower()
    concentric_count = (phases == 'concentric').sum()
    eccentric_count = (phases == 'eccentric').sum()
    total = len(df)
    
    if concentric_count == 0 or eccentric_count == 0:
        issue_list.append(f"Missing one phase (C={concentric_count}, E={eccentric_count})")
    
    if total > 0:
        c_ratio = concentric_count / total
        e_ratio = eccentric_count / total
        if c_ratio > 0.85 or e_ratio > 0.85:
            issue_list.append(f"Extremely unbalanced phases (C={c_ratio:.1%}, E={e_ratio:.1%})")
    
    # Check 3: Data quality issues
    imu_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    available_imu = [c for c in imu_cols if c in df.columns]
    
    if available_imu:
        # Check for all zeros
        zero_counts = (df[available_imu] == 0).sum()
        if (zero_counts > len(df) * 0.5).any():
            cols_with_zeros = zero_counts[zero_counts > len(df) * 0.5].index.tolist()
            issue_list.append(f"Many zeros in: {cols_with_zeros}")
        
        # Check for NaN
        nan_counts = df[available_imu].isna().sum()
        if nan_counts.sum() > 0:
            cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
            issue_list.append(f"NaN in: {cols_with_nan}")
        
        # Check for constant values
        for col in available_imu:
            if df[col].nunique() <= 1:
                issue_list.append(f"Constant value in {col}")
    
    # Check 4: Duration
    duration_samples = len(df)
    if duration_samples < 100:  # Less than 1 second at 100Hz
        issue_list.append(f"Very short ({duration_samples} samples, {duration_samples/100:.1f}s)")
    elif duration_samples > 15000:  # More than 2.5 minutes
        issue_list.append(f"Very long ({duration_samples} samples, {duration_samples/100:.1f}s)")
    
    # Check 5: Missing columns
    required_cols = ['phase', 'action_type', 'subject_id']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issue_list.append(f"Missing columns: {missing_cols}")
    
    # If any issues found, record
    if issue_list:
        problems[stream_id] = {
            'subject': subject,
            'action': action,
            'set_name': set_name,
            'folder_path': folder_path,
            'csv_path': csv_path,
            'n_reps': n_reps,
            'duration_samples': duration_samples,
            'duration_sec': duration_samples / 100.0,
            'c_count': concentric_count,
            'e_count': eccentric_count,
            'issues': issue_list,
            'df_shape': df.shape,
            'columns': df.columns.tolist(),
        }

# Print summary
print(f"TOTAL STREAMS ANALYZED: {len(streams)}")
print(f"STREAMS WITH ISSUES: {len(problems)}")
print()

# Group by issue type
issue_counts = defaultdict(int)
for stream_id, info in problems.items():
    for issue in info['issues']:
        issue_counts[issue] += 1

print("ISSUE TYPE SUMMARY:")
print("-" * 80)
for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
    print(f"  {issue:<60} {count:>5} streams")
print()

# Print detailed list by subject
print("=" * 100)
print("DETAILED LIST BY SUBJECT")
print("=" * 100)
print()

for subject in sorted(set(info['subject'] for info in problems.values())):
    subject_problems = {k: v for k, v in problems.items() if v['subject'] == subject}
    if not subject_problems:
        continue
    
    print(f"--- SUBJECT: {subject} ({len(subject_problems)} problematic streams) ---")
    print()
    
    for stream_id, info in sorted(subject_problems.items()):
        print(f"  STREAM: {stream_id}")
        print(f"    Folder: {info['folder_path']}")
        print(f"    Action: {info['action']}")
        print(f"    Data shape: {info['df_shape']}")
        print(f"    Duration: {info['duration_samples']} samples ({info['duration_sec']:.1f}s)")
        print(f"    Reps detected: {info['n_reps']}")
        print(f"    Concentric samples: {info['c_count']}")
        print(f"    Eccentric samples: {info['e_count']}")
        print(f"    C:E ratio: {info['c_count']/max(1,info['c_count']+info['e_count']):.2f}:{info['e_count']/max(1,info['c_count']+info['e_count']):.2f}")
        print(f"    Issues:")
        for issue in info['issues']:
            print(f"      - {issue}")
        print()
    
    print("-" * 80)
    print()

# Print CSV paths for easy copy-paste
print("=" * 100)
print("FOLDER PATHS FOR MANUAL INSPECTION (one per line)")
print("=" * 100)
print()

for stream_id, info in sorted(problems.items()):
    print(info['folder_path'])

print()
print("=" * 100)
print(f"Total: {len(problems)} folders need manual inspection")
print("=" * 100)
