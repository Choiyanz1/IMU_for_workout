from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _classify_stream(summary: dict, bias_threshold: float) -> dict[str, object]:
    pred_counts = {str(k): int(v) for k, v in (summary.get("pred_micro_counts") or {}).items()}
    gt_counts = {str(k): int(v) for k, v in (summary.get("gt_micro_counts") or {}).items()}
    pred_con = int(pred_counts.get("concentric", 0))
    pred_ecc = int(pred_counts.get("eccentric", 0))
    gt_con = int(gt_counts.get("concentric", 0))
    gt_ecc = int(gt_counts.get("eccentric", 0))
    pred_active = pred_con + pred_ecc
    gt_active = gt_con + gt_ecc
    pred_con_ratio = (pred_con / pred_active) if pred_active else 0.0
    pred_ecc_ratio = (pred_ecc / pred_active) if pred_active else 0.0
    collapse = "none"
    if pred_active > 0 and pred_con == 0:
        collapse = "all_eccentric"
    elif pred_active > 0 and pred_ecc == 0:
        collapse = "all_concentric"
    elif pred_ecc_ratio >= float(bias_threshold):
        collapse = "eccentric_dominant"
    elif pred_con_ratio >= float(bias_threshold):
        collapse = "concentric_dominant"
    return {
        "collapse": collapse,
        "pred_concentric": pred_con,
        "pred_eccentric": pred_ecc,
        "gt_concentric": gt_con,
        "gt_eccentric": gt_ecc,
        "pred_concentric_ratio": pred_con_ratio,
        "pred_eccentric_ratio": pred_ecc_ratio,
        "n_pred": _safe_float(summary.get("n_pred", 0.0)),
        "n_true": _safe_float(summary.get("n_true", 0.0)),
        "f1": _safe_float(summary.get("f1", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize phase-collapse patterns from streaming replay outputs.")
    parser.add_argument("--root", type=Path, required=True, help="Directory to recursively scan for streaming_summary.json files.")
    parser.add_argument("--bias-threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summaries = sorted(args.root.rglob("streaming_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No streaming_summary.json files found under {args.root}")

    rows: list[dict[str, object]] = []
    per_action: dict[str, dict[str, int]] = {}
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        input_path = Path(summary.get("input_path", ""))
        action = input_path.parent.name if input_path.is_dir() else input_path.parent.parent.name
        details = _classify_stream(summary, args.bias_threshold)
        bucket = per_action.setdefault(action, {
            "streams": 0,
            "all_eccentric": 0,
            "all_concentric": 0,
            "eccentric_dominant": 0,
            "concentric_dominant": 0,
            "zero_rep": 0,
            "zero_tp": 0,
        })
        bucket["streams"] += 1
        bucket[str(details["collapse"])] = bucket.get(str(details["collapse"]), 0) + (0 if str(details["collapse"]) == "none" else 1)
        if float(details["n_pred"]) == 0.0:
            bucket["zero_rep"] += 1
        if float(summary.get("tp", 0.0)) == 0.0:
            bucket["zero_tp"] += 1
        rows.append({
            "stream_dir": path.parent.as_posix(),
            "action": action,
            **details,
        })

    worst = sorted(rows, key=lambda r: (r["collapse"] == "none", r["f1"], r["n_pred"] - r["n_true"]))[: int(args.top_k)]
    payload = {
        "root": args.root.as_posix(),
        "stream_count": len(rows),
        "bias_threshold": float(args.bias_threshold),
        "per_action": per_action,
        "worst_streams": worst,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
