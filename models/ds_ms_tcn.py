from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSMSTCNConfig:
    input_channels: int = 6
    micro_classes: int = 3
    macro_classes: int = 5
    num_filters: int = 64
    num_layers: int = 9
    kernel_size: int = 3
    dropout: float = 0.2
    causal: bool = True
    num_macro_stages: int = 4


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int, dropout: float, causal: bool = True) -> None:
        super().__init__()
        self.causal = bool(causal)
        self.left_padding = dilation * (kernel_size - 1)
        padding = 0 if self.causal else self.left_padding // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_in = F.pad(x, (self.left_padding, 0)) if self.causal else x
        out = F.relu(self.conv(conv_in))
        out = self.proj(out)
        return x + self.drop(out)


class SingleStageTCN(nn.Module):
    """Single-stage TCN used by each DS-MS-TCN stage.

    Input/output tensors use `(batch, time, channels)` at the public boundary.
    Internally Conv1d uses `(batch, channels, time)`.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        num_filters: int = 64,
        num_layers: int = 9,
        kernel_size: int = 3,
        dropout: float = 0.2,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    channels=num_filters,
                    dilation=2 ** i,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    causal=causal,
                )
                for i in range(num_layers)
            ]
        )
        self.out_proj = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x).transpose(1, 2)


class DSMSTCN(nn.Module):
    """Dual-scale multi-stage TCN.

    Stage 1 predicts micro labels from IMU. Stages 2-N predict/refine macro
    labels from a micro-probability sequence. Passing `external_micro_probs`
    lets DTW replace Stage 1 while keeping the macro stages unchanged.

    Args:
        cfg: DSMSTCNConfig with num_macro_stages controlling how many macro
             refinement stages are used (2 = no refinement, 4 = full pipeline).
    """

    def __init__(self, cfg: DSMSTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_macro_stages = int(cfg.num_macro_stages)
        self.stage1_micro = SingleStageTCN(
            input_channels=cfg.input_channels,
            num_classes=cfg.micro_classes,
            num_filters=cfg.num_filters,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            causal=cfg.causal,
        )
        self.stage2_macro = SingleStageTCN(
            input_channels=cfg.micro_classes,
            num_classes=cfg.macro_classes,
            num_filters=cfg.num_filters,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            causal=cfg.causal,
        )
        self.refinement_stages = nn.ModuleList()
        for _ in range(self.num_macro_stages - 2):
            self.refinement_stages.append(
                SingleStageTCN(
                    input_channels=cfg.macro_classes,
                    num_classes=cfg.macro_classes,
                    num_filters=cfg.num_filters,
                    num_layers=cfg.num_layers,
                    kernel_size=cfg.kernel_size,
                    dropout=cfg.dropout,
                    causal=cfg.causal,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        external_micro_probs: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        micro_logits = self.stage1_micro(x)
        micro_probs = (
            external_micro_probs
            if external_micro_probs is not None
            else torch.softmax(micro_logits, dim=-1)
        )
        macro_logits = self.stage2_macro(micro_probs)
        outputs = {
            "micro_logits": micro_logits,
            "micro_probs": micro_probs,
            "macro2_logits": macro_logits,
        }
        prev_logits = macro_logits
        for i, stage in enumerate(self.refinement_stages):
            prev_logits = stage(torch.softmax(prev_logits, dim=-1))
            outputs[f"macro{3 + i}_logits"] = prev_logits
        return outputs

    def single_stage_receptive_field(self) -> int:
        return 1 + (int(self.cfg.kernel_size) - 1) * sum(2 ** i for i in range(int(self.cfg.num_layers)))

    def total_receptive_field(self, include_micro_stage: bool = True) -> int:
        stages = self.num_macro_stages if include_micro_stage else self.num_macro_stages - 1
        return 1 + stages * (self.single_stage_receptive_field() - 1)

    def final_macro_logits(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        final_key = f"macro{self.num_macro_stages}_logits"
        return outputs.get(final_key, outputs["macro2_logits"])


class OnlineDSMSTCNPredictor:
    """Rolling-buffer causal inference helper.

    This intentionally recomputes the small buffer on each update. It is simple
    and useful for validating online behavior before optimizing deployment.
    """

    def __init__(
        self,
        model: DSMSTCN,
        imu_columns: Sequence[str],
        device: torch.device,
        buffer_size: int | None = None,
    ) -> None:
        if not model.cfg.causal:
            raise ValueError("OnlineDSMSTCNPredictor requires a causal DSMSTCN model.")
        self.model = model
        self.imu_columns = list(imu_columns)
        self.device = device
        self.buffer_size = int(buffer_size or model.total_receptive_field(include_micro_stage=True))
        self.buffer: deque[torch.Tensor] = deque(maxlen=self.buffer_size)
        self.model.eval()

    def reset(self) -> None:
        self.buffer.clear()

    def update(self, sample: Sequence[float]) -> Dict[str, torch.Tensor]:
        x = torch.as_tensor(sample, dtype=torch.float32)
        self.buffer.append(x)
        seq = torch.stack(tuple(self.buffer), dim=0)[None, :, :].to(self.device)
        with torch.inference_mode():
            out = self.model(seq)
        final_macro = self.model.final_macro_logits(out)
        return {
            "micro_probs": out["micro_probs"][0, -1].detach().cpu(),
            "macro_probs": torch.softmax(final_macro, dim=-1)[0, -1].detach().cpu(),
        }


def temporal_ce_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=ignore_index,
    )


def tmse_loss(logits: torch.Tensor, threshold: float = 4.0, eps: float = 1e-8) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1).clamp(min=torch.log(torch.tensor(eps, device=logits.device)))
    diff = torch.abs(log_probs[:, 1:, :] - log_probs[:, :-1, :])
    diff = torch.clamp(diff, max=float(threshold))
    return torch.mean(diff * diff)


def ds_ms_tcn_loss(
    outputs: Dict[str, torch.Tensor],
    micro_target: torch.Tensor,
    macro_target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.15,
    tmse_threshold: float = 4.0,
    include_micro_loss: bool = True,
) -> Dict[str, torch.Tensor]:
    macro_keys = [k for k in outputs if k.startswith("macro") and k.endswith("_logits")]
    micro_ce = temporal_ce_loss(outputs["micro_logits"], micro_target) if include_micro_loss else torch.tensor(0.0, device=macro_target.device)
    macro_losses: List[torch.Tensor] = []
    smooth_losses: List[torch.Tensor] = []
    for key in macro_keys:
        macro_losses.append(temporal_ce_loss(outputs[key], macro_target))
        smooth_losses.append(tmse_loss(outputs[key], threshold=tmse_threshold))
    macro_ce = torch.stack(macro_losses).sum()
    smooth = torch.stack(smooth_losses).sum()
    total = float(alpha) * micro_ce + macro_ce + float(beta) * smooth
    return {
        "loss": total,
        "micro_ce": micro_ce.detach(),
        "macro_ce": macro_ce.detach(),
        "tmse": smooth.detach(),
    }
