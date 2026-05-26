# Baseline Fairness Discussion

## 文档目的

回应审稿人可能提出的公平性质疑，解释为什么不同 baseline 使用不同的输入和架构是合理的。

---

## 一、为什么 BiLSTM / CNN 使用 raw IMU，而 RF 使用 trailing-window 特征？

这是**文献标准做法**的公平比较：
- **树方法**：通常配合手工特征工程（如 trailing-window statistics）
- **深度学习方法**：通常使用 raw 序列输入，让网络自动学习特征表示

如果我们强行让 BiLSTM 使用 trailing-window 特征，那就变成了「特征工程 + LSTM」，这不是文献中的标准 BiLSTM baseline。同理，如果让 RF 使用 raw IMU，那 RF 也不 work（因为它不是为序列设计的）。

---

## 二、为什么 BiLSTM / CNN / XGBoost / CatBoost 不做 per-action？

| 方法 | 是否做 per-action | 原因 |
|------|------------------|------|
| Per-Action Plain RF | ✅ **必须做** | 论文主推方法，Action-First 架构的核心 |
| XGBoost / CatBoost | ❌ Global | 梯度提升的公平比较应控制变量：相同架构下 RF > boosting |
| BiLSTM / CNN | ❌ Global | DL 的公平比较：让网络自己学习区分 action + phase |
| Peak Detection / SDTW | ❌ Global | 信号处理方法的文献标准做法 |

**关键逻辑分层**：
1. **第一层**（相同 global 架构）：RF (0.778) > XGBoost (0.726) > CatBoost (0.720) → **RF 分类器本身更优**
2. **第二层**（相同 RF 分类器）：Per-Action RF (0.850) > Global RF (0.778) → **Per-Action 架构是决定性改进**
3. **第三层**（上限对比）：Tuned BiLSTM (0.831) < Per-Action RF (0.850) → **简单 ML + 领域知识 > 复杂 DL**

---

## 三、如果强制所有方法都 per-action，会怎样？

| 问题 | 说明 |
|------|------|
| 破坏 baseline 身份 | BiLSTM per-action = 8 个独立 LSTM，这不是「BiLSTM baseline」|
| 部署不现实 | 8 个 BiLSTM 模型总参数量 ~10M，无法装入 64MB RAM |
| 叙事混乱 | 论文变成在比较「per-action 架构」而非「完整 pipeline」|
| 计算爆炸 | 9 folds × 8 actions × 多次调参 ≈ 不可承受 |

---

*文档版本: 2026-05-17 v1（从原 02-phase1-rep-segmentation.md 拆分）*
*关联文档: 04-baseline-comparison.md, 07-deep-learning-baselines.md*
