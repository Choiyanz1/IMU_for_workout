"""Summarize rep_action_compare.json results."""
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/micro_macro_recognition/v2_testkevin/tcn/rep_action_compare.json")
data = json.loads(path.read_text())

print(f"Best model: {data['best_model']}")
print(f"Trusted flat labels: {data['trusted_flat_labels']}")
print()

totals = defaultdict(lambda: {"matched": 0, "correct": 0})
for sr in data["stream_results"]:
    for method in ["rep_complete_classifier", "rep_complete_hierarchical", "online_macro_aggregation", "hybrid_routing", "confidence_hybrid"]:
        if method not in sr:
            continue
        m = sr[method]
        n = m["matched_reps"]
        totals[method]["matched"] += n
        totals[method]["correct"] += round(m["accuracy"] * n)

print("=" * 70)
print("AGGREGATED RESULTS (all streams)")
print("=" * 70)
for method, v in totals.items():
    acc = v["correct"] / v["matched"] if v["matched"] > 0 else 0
    label = method.replace("_", " ").title()
    print(f"  {label:40s} matched={v['matched']:3d}  correct={v['correct']:3d}  accuracy={acc:.4f}")

print()
print("=" * 70)
print("PER-STREAM DETAIL")
print("=" * 70)
for sr in data["stream_results"]:
    d = sr["stream_dir"]
    if "streaming_eval" in d:
        d = d.split("streaming_eval")[-1].lstrip("/\\")
    clf = sr["rep_complete_classifier"]
    agg = sr["online_macro_aggregation"]
    n = clf["matched_reps"]
    if n == 0:
        continue
    ch = sr.get("confidence_hybrid", {})
    ch_acc = ch.get("accuracy", float("nan"))
    print(f"  {d:55s} n={n:2d}  clf={clf['accuracy']:.2f}  macro={agg['accuracy']:.2f}  conf_hyb={ch_acc:.2f}")
