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
    semantic_micro_classes: int = 0
    macro_classes: int = 5
    num_filters: int = 64
    num_layers: int = 9
    kernel_size: int = 3
    dropout: float = 0.2
    causal: bool = True
    num_macro_stages: int = 4
    use_dual_micro_head: bool = False
    use_semantic_for_macro: bool = False
    use_rep_count_head: bool = False
    rep_count_head_dim: int = 64


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


class SingleStageFeatures(nn.Module):
    """Shared temporal feature extractor for a single DS-MS-TCN stage."""

    def __init__(
        self,
        input_channels: int,
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


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
        self.features = SingleStageFeatures(
            input_channels=input_channels,
            num_filters=num_filters,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
        )
        self.out_proj = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.out_proj(features.transpose(1, 2)).transpose(1, 2)


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
        self.use_dual_micro_head = bool(cfg.use_dual_micro_head) and int(cfg.semantic_micro_classes) > 0
        self.use_semantic_for_macro = bool(cfg.use_semantic_for_macro) and self.use_dual_micro_head
        if self.use_dual_micro_head:
            self.stage1_micro_features = SingleStageFeatures(
                input_channels=cfg.input_channels,
                num_filters=cfg.num_filters,
                num_layers=cfg.num_layers,
                kernel_size=cfg.kernel_size,
                dropout=cfg.dropout,
                causal=cfg.causal,
            )
            self.stage1_micro = nn.Conv1d(cfg.num_filters, cfg.micro_classes, kernel_size=1)
            self.stage1_semantic_micro = nn.Conv1d(cfg.num_filters, cfg.semantic_micro_classes, kernel_size=1)
        else:
            self.stage1_micro = SingleStageTCN(
                input_channels=cfg.input_channels,
                num_classes=cfg.micro_classes,
                num_filters=cfg.num_filters,
                num_layers=cfg.num_layers,
                kernel_size=cfg.kernel_size,
                dropout=cfg.dropout,
                causal=cfg.causal,
            )
        macro_input_channels = cfg.micro_classes + (cfg.semantic_micro_classes if self.use_semantic_for_macro else 0)
        self.stage2_macro = SingleStageTCN(
            input_channels=macro_input_channels,
            num_classes=cfg.macro_classes,
            num_filters=cfg.num_filters,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            causal=cfg.causal,
        )
        self.use_rep_count_head = bool(cfg.use_rep_count_head)
        if self.use_rep_count_head:
            self.rep_count_head = nn.Sequential(
                nn.Linear(cfg.num_filters, cfg.rep_count_head_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.rep_count_head_dim, 1),
            )
        else:
            self.rep_count_head = None
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
        external_semantic_micro_probs: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor]
        semantic_micro_logits = None
        semantic_micro_probs = None
        if self.use_dual_micro_head:
            stage1_features = self.stage1_micro_features(x)
            micro_logits = self.stage1_micro(stage1_features.transpose(1, 2)).transpose(1, 2)
            semantic_micro_logits = self.stage1_semantic_micro(stage1_features.transpose(1, 2)).transpose(1, 2)
            semantic_micro_probs = (
                external_semantic_micro_probs
                if external_semantic_micro_probs is not None
                else torch.softmax(semantic_micro_logits, dim=-1)
            )
        else:
            stage1_features = self.stage1_micro.features(x)
            micro_logits = self.stage1_micro.out_proj(stage1_features.transpose(1, 2)).transpose(1, 2)
        micro_probs = external_micro_probs if external_micro_probs is not None else torch.softmax(micro_logits, dim=-1)
        macro_input = torch.cat([micro_probs, semantic_micro_probs], dim=-1) if self.use_semantic_for_macro else micro_probs
        macro_logits = self.stage2_macro(macro_input)
        if self.use_rep_count_head and stage1_features is not None:
            pooled = stage1_features.mean(dim=1)
            rep_count_pred = self.rep_count_head(pooled).squeeze(-1)
            outputs = {
                "micro_logits": micro_logits,
                "micro_probs": micro_probs,
                "macro2_logits": macro_logits,
                "rep_count_pred": rep_count_pred,
            }
        else:
            outputs = {
                "micro_logits": micro_logits,
                "micro_probs": micro_probs,
                "macro2_logits": macro_logits,
            }
        if semantic_micro_logits is not None and semantic_micro_probs is not None:
            outputs["semantic_micro_logits"] = semantic_micro_logits
            outputs["semantic_micro_probs"] = semantic_micro_probs
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
        micro_smoothing_window: int = 1,
    ) -> None:
        if not model.cfg.causal:
            raise ValueError("OnlineDSMSTCNPredictor requires a causal DSMSTCN model.")
        self.model = model
        self.imu_columns = list(imu_columns)
        self.device = device
        self.buffer_size = int(buffer_size or model.total_receptive_field(include_micro_stage=True))
        self.buffer: deque[torch.Tensor] = deque(maxlen=self.buffer_size)
        self.micro_smoothing_window = max(1, int(micro_smoothing_window))
        self.micro_prob_buffer: deque[torch.Tensor] = deque(maxlen=self.micro_smoothing_window)
        self.model.eval()

    def reset(self) -> None:
        self.buffer.clear()
        self.micro_prob_buffer.clear()

    def update(self, sample: Sequence[float]) -> Dict[str, torch.Tensor]:
        x = torch.as_tensor(sample, dtype=torch.float32)
        self.buffer.append(x)
        seq = torch.stack(tuple(self.buffer), dim=0)[None, :, :].to(self.device)
        with torch.inference_mode():
            out = self.model(seq)
        final_macro = self.model.final_macro_logits(out)
        micro_probs = out["micro_probs"][0, -1].detach().cpu()
        self.micro_prob_buffer.append(micro_probs)
        if self.micro_smoothing_window > 1:
            micro_probs = torch.stack(tuple(self.micro_prob_buffer), dim=0).mean(dim=0)
        result = {
            "micro_probs": micro_probs,
            "macro_probs": torch.softmax(final_macro, dim=-1)[0, -1].detach().cpu(),
        }
        if "semantic_micro_probs" in out:
            result["semantic_micro_probs"] = out["semantic_micro_probs"][0, -1].detach().cpu()
        return result


def temporal_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=ignore_index,
        weight=class_weights,
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
    include_macro_loss: bool = True,
    micro_class_weights: torch.Tensor | None = None,
    micro_beta: float = 0.0,
    micro_tmse_threshold: float = 4.0,
    semantic_target: torch.Tensor | None = None,
    semantic_alpha: float = 0.0,
    semantic_class_weights: torch.Tensor | None = None,
    rep_count_target: torch.Tensor | None = None,
    rep_count_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    macro_keys = [k for k in outputs if k.startswith("macro") and k.endswith("_logits")]
    micro_ce = (
        temporal_ce_loss(outputs["micro_logits"], micro_target, class_weights=micro_class_weights)
        if include_micro_loss
        else torch.tensor(0.0, device=macro_target.device)
    )
    micro_tmse = (
        tmse_loss(outputs["micro_logits"], threshold=micro_tmse_threshold)
        if include_micro_loss and float(micro_beta) > 0.0
        else torch.tensor(0.0, device=macro_target.device)
    )
    semantic_ce = (
        temporal_ce_loss(outputs["semantic_micro_logits"], semantic_target, class_weights=semantic_class_weights)
        if semantic_target is not None and "semantic_micro_logits" in outputs and float(semantic_alpha) > 0.0
        else torch.tensor(0.0, device=macro_target.device)
    )
    macro_losses: List[torch.Tensor] = []
    smooth_losses: List[torch.Tensor] = []
    if include_macro_loss:
        for key in macro_keys:
            macro_losses.append(temporal_ce_loss(outputs[key], macro_target))
            smooth_losses.append(tmse_loss(outputs[key], threshold=tmse_threshold))
    macro_ce = torch.stack(macro_losses).sum() if macro_losses else torch.tensor(0.0, device=macro_target.device)
    smooth = torch.stack(smooth_losses).sum() if smooth_losses else torch.tensor(0.0, device=macro_target.device)
    rep_count_loss = torch.tensor(0.0, device=macro_target.device)
    if rep_count_target is not None and "rep_count_pred" in outputs and float(rep_count_weight) > 0.0:
        rep_count_loss = F.mse_loss(outputs["rep_count_pred"], rep_count_target.float())
    total = float(alpha) * micro_ce + float(micro_beta) * micro_tmse + float(semantic_alpha) * semantic_ce + macro_ce + float(beta) * smooth + float(rep_count_weight) * rep_count_loss
    return {
        "loss": total,
        "micro_ce": micro_ce.detach(),
        "micro_tmse": micro_tmse.detach(),
        "semantic_micro_ce": semantic_ce.detach(),
        "macro_ce": macro_ce.detach(),
        "tmse": smooth.detach(),
        "rep_count_loss": rep_count_loss.detach(),
    }


def _remap_legacy_single_stage_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(".features." in key for key in state_dict):
        return state_dict
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if ".out_proj" not in key:
            if key.startswith("stage1_micro."):
                new_key = key.replace("stage1_micro.", "stage1_micro.features.", 1)
            elif key.startswith("stage2_macro."):
                new_key = key.replace("stage2_macro.", "stage2_macro.features.", 1)
            elif key.startswith("refinement_stages."):
                parts = key.split(".")
                if len(parts) >= 3 and parts[2] != "features":
                    new_key = f"{parts[0]}.{parts[1]}.features." + ".".join(parts[2:])
        remapped[new_key] = value
    return remapped


def load_dsmstcn_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    remapped = _remap_legacy_single_stage_keys(state_dict)
    model.load_state_dict(remapped)
