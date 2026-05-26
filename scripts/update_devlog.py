import json
import time
from pathlib import Path
from datetime import datetime

# Update dev-log with the 7-subject cleaned baseline result
log_path = Path("docs/dev-log.md")

new_entry = f"""

## {datetime.now().strftime('%Y-%m-%d')} - Phase 1a: 7-Subject LOSO Baseline (Cleaned Data)

### Data Quality Decision
- Manually cleaned non-standard reps with abnormal concentric-dominant phase ratios
- Excluded two subjects (tsenyu, ziho) due to persistent data quality issues
- Final dataset: 7 subjects, 8 actions, 226 streams

### Results (7-fold LOSO, 8 actions, Plain Causal RF)

| Subject | Rep F1 | Precision | Recall |
|---------|--------|-----------|--------|
| haoyu | 0.868 | 0.791 | 0.962 |
| thomas | 0.798 | 0.762 | 0.838 |
| yoru | 0.712 | 0.681 | 0.746 |
| kevin | 0.703 | 0.670 | 0.739 |
| yushuan | 0.672 | 0.632 | 0.718 |
| hsianshun | 0.634 | 0.594 | 0.679 |
| yanz | 0.577 | 0.525 | 0.640 |
| **Overall** | **0.706 ± 0.091** | **0.709** | **0.704** |

### Key Findings
1. **Best performers**: haoyu (0.87) and thomas (0.80) - standard, consistent movement patterns
2. **Worst performer**: yanz (0.58) - despite cleaning, cross-subject generalization remains challenging
3. **Mean F1 = 0.706** is adopted as the official Phase 1a baseline for Causal RF (plain)
4. Refiner historically improved on yushuan (+0.05-0.07) but current implementation underperforms

### Next Steps
- Complete Phase 1a baseline comparison: Peak Detection, SDTW, Sliding-window RF, BiLSTM
- Phase 1b: Modality Ablation (after Phase 1a complete)
- Phase 2/3: Deferred until Phase 1 complete
"""

if log_path.exists():
    content = log_path.read_text(encoding="utf-8")
    # Insert after the first heading
    lines = content.split("\n")
    # Find first ## heading
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("## "):
            insert_idx = i
            break
    
    new_content = "\n".join(lines[:insert_idx]) + new_entry + "\n".join(lines[insert_idx:])
    log_path.write_text(new_content, encoding="utf-8")
    print("Updated dev-log.md")
else:
    print("Warning: dev-log.md not found")
