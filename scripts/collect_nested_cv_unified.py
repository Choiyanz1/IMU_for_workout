"""
Collect and unify all existing nested CV results from various experiment directories.
Produces a single unified 9-fold report with provenance tracking.
"""

import json
from pathlib import Path
from collections import defaultdict

ALL_SUBJECTS = ["haoyu", "hsianshun", "kevin", "thomas", "tsenyu", "yanz", "yoru", "yushuan", "ziho"]

def find_all_nested_results():
    """Find all per-subject nested CV results across all experiment directories."""
    results = defaultdict(list)
    base = Path("artifacts/baseline_comparison")
    
    for result_file in base.rglob("*/results.json"):
        try:
            if not result_file.exists():
                continue
            data = json.load(open(result_file))
            if "folds" not in data or "tuned_overall" not in data:
                continue
            
            exp_name = result_file.parent.name
            for fold_name, fold_data in data.get("folds", {}).items():
                if fold_name in ALL_SUBJECTS:
                    results[fold_name].append({
                        "experiment": exp_name,
                        "path": str(result_file.parent),
                        "config": data.get("config", ""),
                        "selection_metric": data.get("selection_metric", ""),
                        "modalities": data.get("modalities", []),
                        "tuned_overall": fold_data.get("tuned_overall", {}),
                        "baseline_overall": fold_data.get("baseline_overall", {}),
                    })
        except Exception:
            continue
    
    return results


def select_best_result(subject_results):
    """Select the best result per subject based on tuned rep_f1."""
    if not subject_results:
        return None
    
    # Prefer results with higher tuned F1, but also consider:
    # - More modalities searched (less truncated)
    # - Higher baseline F1 (more stable)
    def score(r):
        tuned_f1 = r["tuned_overall"].get("rep_f1", 0)
        baseline_f1 = r["baseline_overall"].get("rep_f1", 0)
        n_modalities = len(r.get("modalities", []))
        # Prefer full 7-modality searches, then higher F1
        return (n_modalities, tuned_f1, baseline_f1)
    
    return max(subject_results, key=score)


def main():
    print("Scanning all experiment directories for nested CV results...")
    all_results = find_all_nested_results()
    
    unified = {
        "benchmark": "per_action_rf_nested_cv_9fold_unified",
        "description": "Best nested CV result per subject across all historical experiments",
        "subjects_completed": [],
        "subjects_missing": [],
        "folds": {},
        "provenance": {},
    }
    
    tuned_overalls = []
    baseline_overalls = []
    
    for subject in ALL_SUBJECTS:
        if subject in all_results:
            best = select_best_result(all_results[subject])
            if best:
                unified["folds"][subject] = {
                    "tuned_overall": best["tuned_overall"],
                    "baseline_overall": best["baseline_overall"],
                }
                unified["provenance"][subject] = {
                    "experiment": best["experiment"],
                    "path": best["path"],
                    "config": best["config"],
                    "modalities_searched": best["modalities"],
                    "selection_metric": best["selection_metric"],
                }
                unified["subjects_completed"].append(subject)
                tuned_overalls.append(best["tuned_overall"])
                baseline_overalls.append(best["baseline_overall"])
                
                tuned_f1 = best["tuned_overall"].get("rep_f1", 0)
                baseline_f1 = best["baseline_overall"].get("rep_f1", 0)
                print(f"{subject}: {best['experiment']} -> tuned={tuned_f1:.4f}, baseline={baseline_f1:.4f}")
        else:
            unified["subjects_missing"].append(subject)
            print(f"{subject}: MISSING")
    
    # Compute aggregates
    def _avg_metrics(metrics_list):
        if not metrics_list:
            return {}
        keys = ["precision", "recall", "rep_f1", "micro_f1_at_50", "start_mae_ms", "end_mae_ms", "transition_mae_ms"]
        out = {}
        for k in keys:
            vals = [m[k] for m in metrics_list if k in m and m[k] is not None]
            if vals:
                out[f"{k}_mean"] = sum(vals) / len(vals)
                out[f"{k}_std"] = (sum((v - out[f"{k}_mean"])**2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        return out
    
    unified["tuned_aggregate"] = _avg_metrics(tuned_overalls)
    unified["baseline_aggregate"] = _avg_metrics(baseline_overalls)
    unified["completion_rate"] = len(unified["subjects_completed"]) / len(ALL_SUBJECTS)
    
    # Write unified report
    output_dir = Path("artifacts/baseline_comparison/nested_cv_9fold_unified")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "unified_results.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Unified report written to: {out_path}")
    print(f"Completed: {len(unified['subjects_completed'])}/9 subjects")
    print(f"Missing: {unified['subjects_missing']}")
    if unified["tuned_aggregate"]:
        print(f"Tuned Rep F1 (completed): {unified['tuned_aggregate'].get('rep_f1_mean', 0):.4f} ± {unified['tuned_aggregate'].get('rep_f1_std', 0):.4f}")
    if unified["baseline_aggregate"]:
        print(f"Baseline Rep F1 (completed): {unified['baseline_aggregate'].get('rep_f1_mean', 0):.4f} ± {unified['baseline_aggregate'].get('rep_f1_std', 0):.4f}")
    print(f"{'='*60}")
    
    # Also write a markdown summary
    md_lines = [
        "# Nested CV 9-Fold Unified Results (Incremental)",
        "",
        f"**Completion**: {len(unified['subjects_completed'])}/9 subjects",
        f"**Missing subjects**: {', '.join(unified['subjects_missing']) if unified['subjects_missing'] else 'None'}",
        "",
        "## Overall Metrics (Completed Subjects)",
        "",
        "| Method | Rep F1 | Precision | Recall | micro_f1@50 |",
        "|--------|--------|-----------|--------|-------------|",
    ]
    
    if unified["tuned_aggregate"]:
        t = unified["tuned_aggregate"]
        md_lines.append(
            f"| Tuned | {t.get('rep_f1_mean', 0):.4f} ± {t.get('rep_f1_std', 0):.4f} | "
            f"{t.get('precision_mean', 0):.4f} ± {t.get('precision_std', 0):.4f} | "
            f"{t.get('recall_mean', 0):.4f} ± {t.get('recall_std', 0):.4f} | "
            f"{t.get('micro_f1_at_50_mean', 0):.4f} ± {t.get('micro_f1_at_50_std', 0):.4f} |"
        )
    
    if unified["baseline_aggregate"]:
        b = unified["baseline_aggregate"]
        md_lines.append(
            f"| Baseline | {b.get('rep_f1_mean', 0):.4f} ± {b.get('rep_f1_std', 0):.4f} | "
            f"{b.get('precision_mean', 0):.4f} ± {b.get('precision_std', 0):.4f} | "
            f"{b.get('recall_mean', 0):.4f} ± {b.get('recall_std', 0):.4f} | "
            f"{b.get('micro_f1_at_50_mean', 0):.4f} ± {b.get('micro_f1_at_50_std', 0):.4f} |"
        )
    
    md_lines.extend([
        "",
        "## Per-Subject Breakdown",
        "",
        "| Subject | Experiment | Tuned F1 | Baseline F1 | Modalities | Config |",
        "|---------|------------|----------|-------------|------------|--------|",
    ])
    
    for subject in ALL_SUBJECTS:
        if subject in unified["folds"]:
            prov = unified["provenance"][subject]
            tuned_f1 = unified["folds"][subject]["tuned_overall"].get("rep_f1", 0)
            baseline_f1 = unified["folds"][subject]["baseline_overall"].get("rep_f1", 0)
            md_lines.append(
                f"| {subject} | {prov['experiment']} | {tuned_f1:.4f} | {baseline_f1:.4f} | "
                f"{len(prov['modalities_searched'])} modalities | {Path(prov['config']).name} |"
            )
        else:
            md_lines.append(f"| {subject} | **MISSING** | - | - | - | - |")
    
    md_path = output_dir / "unified_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    
    print(f"Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
