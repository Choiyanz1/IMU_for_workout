from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
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
        out = F.relu(self.conv(x))
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

    Stage 1 predicts micro labels from IMU. Stages 2-4 predict/refine macro
    labels from a micro-probability sequence. Passing `external_micro_probs`
    lets DTW replace Stage 1 while keeping the macro stages unchanged.
    """

    def __init__(self, cfg: DSMSTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage1_micro = SingleStageTCN(
            input_channels=cfg.input_channels,
            num_classes=cfg.micro_classes,
            num_filters=cfg.num_filters,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )
        self.stage2_macro = SingleStageTCN(
            input_channels=cfg.micro_classes,
            num_classes=cfg.macro_classes,
            num_filters=cfg.num_filters,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )
        self.stage3_macro = SingleStageTCN(
            input_channels=cfg.macro_classes,
            num_classes=cfg.macro_classes,
            num_filters=cfg.num_filters,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )
        self.stage4_macro = SingleStageTCN(
            input_channels=cfg.macro_classes,
            num_classes=cfg.macro_classes,
            num_filters=cfg.num_filters,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
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
        macro2_logits = self.stage2_macro(micro_probs)
        macro3_logits = self.stage3_macro(torch.softmax(macro2_logits, dim=-1))
        macro4_logits = self.stage4_macro(torch.softmax(macro3_logits, dim=-1))
        return {
            "micro_logits": micro_logits,
            "micro_probs": micro_probs,
            "macro2_logits": macro2_logits,
            "macro3_logits": macro3_logits,
            "macro4_logits": macro4_logits,
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
    micro_ce = temporal_ce_loss(outputs["micro_logits"], micro_target) if include_micro_loss else torch.tensor(0.0, device=macro_target.device)
    macro_losses: List[torch.Tensor] = []
    smooth_losses: List[torch.Tensor] = []
    for key in ("macro2_logits", "macro3_logits", "macro4_logits"):
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
