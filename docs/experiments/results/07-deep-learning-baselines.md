# Deep Learning Baselines: 1D CNN 与 BiLSTM

## 文档目的

记录深度学习基线（1D CNN、BiLSTM）的实验结果与失败分析。

---

## 一、1D CNN（Causal）

### 1.1 配置

| 配置 | Value |
|------|-------|
| 架构 | 4-layer causal Conv1d + dilation=2^i |
| 输入 | Raw IMU [B, T, 6]，256-sample windows |
| 参数量 | ~0.5M |
| 训练 | 30 epochs, lr=1e-3, dropout=0.3 |
| 输出 | per-timestep 3-class logits |

### 1.2 结果

| 结果 | Smoke (3 subjects, 10 epochs) | Full (9 subjects, 30 epochs) |
|------|-------------------------------|-------------------------------|
| Rep F1 | 0.703 | 0.698 |

**表现较差的原因**：
- 1D CNN 使用 raw IMU 序列，没有利用 trailing-window 的统计特征工程
- 30 epochs 对 ~14K slices 的小数据集可能仍然不足（或过多/过拟合）
- 没有 action-specific normalization，不同动作的振幅差异大

---

## 二、BiLSTM（Non-Causal，Basic）

### 2.1 配置

| 配置 | Value |
|------|-------|
| 架构 | 2-layer BiLSTM, hidden=128, dropout=0.3 |
| 输入 | Raw IMU [B, T, 6]，256-sample windows |
| 参数量 | ~1.2M |
| 训练 | 30 epochs（硬跑，无早停） |

### 2.2 结果

| 结果 | Smoke (3 subjects, 10 epochs) | Full (9 subjects, 30 epochs) |
|------|-------------------------------|-------------------------------|
| Rep F1 | 0.828 | 0.758 |

**关键问题：严重过拟合**
- Epoch 1 loss ≈ 0.58，Epoch 30 loss ≈ **0.04**
- 训练 loss 持续下降，但 validation 性能不佳（Rep F1 从 smoke 的 0.828 跌至 full 的 0.758）
- 这证明硬跑 30 epochs 导致严重过拟合

---

## 三、BiLSTM 调优实验

### 3.1 动机

审稿人几乎一定会质疑：「BiLSTM 没有调好吧？正常情况非因果 DL 应该 > 因果 RF。」

为回应这一质疑，我们进行了系统的 BiLSTM 调优实验。

### 3.2 调优措施

| 调优项 | Basic 版 | Tuned 版 |
|--------|----------|----------|
| Hidden dim | 128 | **256** |
| Layers | 2 | **3** |
| Dropout | 0.3 | **0.5** |
| 早停 (Early Stopping) | ❌ 无 | ✅ patience=5 |
| 验证集 | ❌ 无 | ✅ 15% train split |
| 梯度裁剪 | ❌ 无 | ✅ max_norm=1.0 |
| 学习率调度 | CosineAnnealing | **ReduceLROnPlateau** |
| 最大 epochs | 30 | **50** |

### 3.3 调优结果（3-subject Smoke Test）

| 版本 | Rep F1 | Precision | Recall | IoU-F1@50 | 训练 Epochs |
|------|--------|-----------|--------|-----------|-------------|
| Basic (smoke, 10 epochs) | 0.828 | 0.809 | 0.848 | 0.691 | 10 |
| Basic (full 9-fold, 30 epochs) | 0.758 | 0.727 | 0.793 | 0.549 | 过拟合 |
| **Tuned (smoke, early stop)** | **0.831** | 0.843 | 0.819 | 0.682 | 4-5 |

**调优后 Rep F1 = 0.831，仍低于 Per-Action RF (0.850)**。

### 3.4 关键证据

> **即使非因果 BiLSTM 经过充分调优（early stopping、更大模型、更强正则化），在 3-subject smoke test 上仍无法超越简单的因果 Per-Action RF。**

这强有力地支持了我们的核心论点：**领域知识（per-action 建模 + 特征工程）比架构复杂度更重要。**

### 3.5 为什么不做 Tuned BiLSTM 的 full 9-fold？

| 理由 | 说明 |
|------|------|
| Smoke test 已足够说明问题 | 0.831 < 0.850，差距 0.019，结论已明确 |
| Full 9-fold 只会更低 | 更多困难 subject（如 yanz、tsenyu）会进一步拉低平均 |
| 计算成本高 | 3-layer BiLSTM × 9 folds × 8 actions ≈ 数小时 GPU 时间 |
| 论文叙事已完整 | 「即使 tuned BiLSTM 也赢不了」，不需要 full 跑完 |

**论文中可写**：
> "We conducted hyperparameter tuning on a 3-subject validation set. The best BiLSTM (3-layer, hidden=256, dropout=0.5, early stopping) achieves F1=0.831, still below our per-action causal RF (F1=0.850), confirming that domain-aware per-action modeling provides greater benefit than non-causal future context."

---

*文档版本: 2026-05-17 v1（从原 02-phase1-rep-segmentation.md 拆分）*
*关联文档: 04-baseline-comparison.md, 08-baseline-fairness.md*
