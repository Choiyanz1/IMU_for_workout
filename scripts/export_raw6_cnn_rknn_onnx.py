"""Export a RKNN-friendly ONNX from the raw6 CNN PyTorch checkpoint.

The training model uses reflect padding in each causal convolution. RKNN support
for non-constant Pad modes can be fragile, so this exporter keeps the same
weights but uses constant zero left padding for conversion. The live pipeline
already z-scores each window, so zero padding means "train-set mean" in
normalized space.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import onnx
except Exception:  # pragma: no cover - ONNX shape metadata cleanup is optional
    onnx = None

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RknnFriendlyCausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=0, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0), mode="constant", value=0.0))


class RknnFriendlySharedEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = RknnFriendlyCausalConv1d(in_ch, hidden, 5, 1)
        self.gn1 = nn.GroupNorm(8, hidden)
        self.conv2 = RknnFriendlyCausalConv1d(hidden, hidden, 5, 2)
        self.gn2 = nn.GroupNorm(8, hidden)
        self.conv3 = RknnFriendlyCausalConv1d(hidden, hidden, 5, 4)
        self.gn3 = nn.GroupNorm(8, hidden)
        self.conv4 = RknnFriendlyCausalConv1d(hidden, hidden, 5, 8)
        self.gn4 = nn.GroupNorm(8, hidden)
        self.conv5 = RknnFriendlyCausalConv1d(hidden, hidden, 5, 16)
        self.gn5 = nn.GroupNorm(8, hidden)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Conv1d(in_ch, hidden, 1) if in_ch != hidden else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.res_proj is None else self.res_proj(x)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.gn4(self.conv4(x)))
        x = self.dropout(x)
        x = F.relu(self.gn5(self.conv5(x)))
        x = self.dropout(x)
        return x + identity


class RknnFriendlyCausalCNNPhaseOnly(nn.Module):
    def __init__(self, in_ch: int = 6, hidden: int = 64, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = RknnFriendlySharedEncoder(in_ch, hidden, dropout)
        self.phase_head = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.phase_head(self.encoder(x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RKNN-friendly ONNX from raw6 CNN artifact.")
    parser.add_argument("--artifact", default="artifacts/deploy/raw6_cnn_top5_p5_current")
    parser.add_argument("--output", default=None, help="Default: <artifact>/model_rknn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact)
    output = Path(args.output) if args.output else artifact_dir / "model_rknn.onnx"
    checkpoint = torch.load(artifact_dir / "model.pt", map_location="cpu")
    model = RknnFriendlyCausalCNNPhaseOnly(
        checkpoint.get("input_channels", 6),
        checkpoint.get("hidden", 64),
        checkpoint.get("num_classes", 2),
        checkpoint.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, checkpoint.get("input_channels", 6), 300, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["imu"],
        output_names=["phase_logits"],
        opset_version=args.opset,
    )
    if onnx is not None:
        exported = onnx.load(output)
        for value, shape in [(exported.graph.input[0], [1, 6, 300]), (exported.graph.output[0], [1, 2, 300])]:
            dims = value.type.tensor_type.shape.dim
            for dim, size in zip(dims, shape):
                dim.dim_param = ""
                dim.dim_value = int(size)
        onnx.save(exported, output)
    else:
        print("[WARN] Could not rewrite static ONNX shapes because onnx is unavailable")
    metadata_path = artifact_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("files", {})["rknn_friendly_onnx"] = output.name
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OK] exported RKNN-friendly ONNX: {output}")


if __name__ == "__main__":
    main()
