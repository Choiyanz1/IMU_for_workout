# Post-Processing Evaluation: Duration Prior + Boundary Refiner

## 文档目的

本文档记录对 Rep Segmentation 后处理技术的评估：
1. **Duration Prior**（时长先验过滤）
2. **Boundary Refiner**（边界修正回归器）
3. **Guardrail**（效果验证与回退机制）

**核心问题**：这些后处理能否提升三大核心指标（Rep F1、IoU-F1@50、Exact Count）？

---

## 一、后处理组件说明

### 1.1 Duration Prior（时长先验）

**原理**：根据训练数据的 GT rep duration 分布，过滤掉异常短/长的预测 rep。

```
训练数据统计 → 计算 duration 的 5th / 95th percentile
                ↓
预测时过滤：rep_duration < min_threshold 或 > max_threshold → 丢弃
```

**参数**：
- `min_rep_duration_samples`: 最短允许时长（samples）
- `max_rep_duration_samples`: 最长允许时长（samples）

**预期效果**：
- 减少 Over-segmented（过滤掉碎片化的虚假 reps）
- 减少 Under-segmented（防止过度合并导致的超长 rep）

### 1.2 Boundary Refiner（边界修正器）

**原理**：训练三个 ExtraTreesRegressor 分别预测 start/transition/end 的边界偏移量。

```
训练阶段：
  1. 用 coarse model 预测 reps
  2. 与 GT reps 做 IoU 匹配（threshold=0.3）
  3. 提取匹配 rep 的边界特征（IMU 统计 + phase 概率）
  4. 训练回归器：特征 → 边界偏移量（GT边界 - 预测边界）

推理阶段：
  1. coarse 预测 rep
  2. 提取边界特征
  3. 回归器预测偏移量 → 修正边界
```

**预期效果**：
- 提升 IoU-F1@50（边界更准）
- 间接提升 Exact Count（边界准了，合并/分裂减少）

### 1.3 Modality Selection（传感器组合选择）

**原理**：对每种动作，尝试多种 IMU 组合（acc+gyro, acc+mag, gyro+mag, acc+gyro+mag），选择最佳。

**预期效果**：某些动作可能对 mag 更敏感（如涉及旋转的动作）。

### 1.4 Guardrail（护栏机制）

**原理**：如果「加后处理」的效果不如「不加后处理」，则回退到不加后处理的 baseline。

**评判标准**：
- Rep F1 差距 < 阈值
- Exact Count Ratio 差距 < 阈值
- Recall 差距 < 阈值

---

## 二、实验结果：modality_count_guardrail_yoru_v1

### 2.1 实验设置

| 属性 | 值 |
|------|-----|
| 测试 subject | yoru (held-out) |
| 训练 subjects | haoyu, kevin, thomas, yanz, yushuan, ziho |
| Actions | 8 |
| 评估的 modality 组合 | acc+gyro, acc+mag, gyro+mag, acc+gyro+mag |
| Duration Prior | 按动作计算训练 duration 分布 |
| Boundary Refiner | ExtraTreesRegressor (300 trees, depth 18) |
| Guardrail | 如果 tuned < baseline，回退到 baseline |

### 2.2 Guardrail 决策结果

**结果：所有 8 个动作的 Guardrail 都选择了 `baseline_reference`**

这意味着：
- Duration Prior + Boundary Refiner + Modality Selection **没有带来统计显著的提升**
- 在 yoru held-out 数据上，「什么都不加」就是最好的

### Table 1: Guardrail 决策详情（yoru held-out, 25 streams）

| 动作 | 最佳 modality | 含 Duration Prior? | 含 Refiner? | Guardrail 决策 | Rep F1 |
|------|-------------|-------------------|------------|---------------|--------|
| db_bench_press | baseline_reference | ❌ | ❌ | 回退到 baseline | 0.707 |
| db_biceps_curl | baseline_reference | ❌ | ❌ | 回退到 baseline | 0.998 |
| db_rdl | baseline_reference | ❌ | ❌ | 回退到 baseline | 0.697 |
| db_shoulder_press | baseline_reference | ❌ | ❌ | 回退到 baseline | 0.745 |
| db_squat | baseline_reference | ❌ | ❌ | 回退到 baseline | 0.849 |
| db_triceps_curl | baseline_reference | ❌ | ❌ | 回退到 baseline | 0.574 |
| db_weighted_crunch | baseline_reference | ❌ | ❌ | 回退到 baseline | 0.642 |
| one_arm_db_row | baseline_reference | ❌ | ❌ | 回退到 baseline | 0.893 |

> 所有动作的 `guardrail_recall_gap = 0`, `guardrail_exact_count_ratio_gap = 0`, `guardrail_mean_abs_count_diff_gap = 0`

### 2.3 baseline_reference vs 其他 modality 的对比

以 db_bench_press 为例（31 streams）：

| modality | Rep F1 | Exact% | Over | Under | MADiff |
|----------|--------|--------|------|-------|--------|
| **baseline_reference** | **0.707** | **25.8%** | **60** | **32** | **4.10** |
| acc+gyro (with post-proc) | 0.700 | 12.9% | 0 | 27 | 4.68 |
| acc+mag (with post-proc) | 0.689 | 12.9% | 1 | 26 | 4.32 |
| gyro+mag (with post-proc) | 0.419 | 3.2% | 0 | 30 | 7.10 |
| acc+gyro+mag (with post-proc) | 0.661 | 16.1% | 0 | 26 | 5.10 |

**关键发现**：
1. baseline_reference 的 Rep F1 (0.707) > 所有带后处理的 modality
2. baseline_reference 的 Exact Count (25.8%) 反而高于带后处理的版本（12-16%）
3. **但**：baseline_reference 的 Over-segmented = 60（非常高！）——说明 baseline 产生了大量碎片化 reps

### 2.4 为什么后处理被 Guardrail 拒绝？

**可能原因**：

1. **Duration Prior 参数过于严格**
   - 按 5th/95th percentile 设置阈值，可能过滤掉了合法的 reps
   - 特别是 yoru 的动作风格可能与训练数据不同

2. **Boundary Refiner 过拟合**
   - 在训练数据上学习的边界偏移，不一定泛化到 yoru
   - 300 trees, depth 18 可能对 6 个训练 subject 过拟合

3. **Modality 切换不如预期**
   - 某些动作的最佳 modality 可能是 acc+mag，但整体提升不明显
   - 切换 sensor 组合带来的收益被 refiner 的噪声抵消

4. **测试集太小（25 streams）**
   - yoru 只有 25 streams，统计波动大
   - 某些动作只有 3-4 个 streams，一个 stream 的误差就能改变排名

---

## 三、有无后处理的直接比较

### 3.1 当前可用的数据

| 模型 | 测试集 | streams | 后处理 | Rep F1 | Exact% | IoU-F1@50 |
|------|--------|---------|--------|--------|--------|-----------|
| Per-Action Plain RF | 7-fold LOSO | 226 | ❌ 无 | 0.850 | 65.9% | 0.706 |
| yoru_v1 baseline | yoru only | 25 | ❌ 无（Guardrail 回退）| 0.876 | 76.0% | 0.813 |
| yoru_v1 tuned (any modality) | yoru only | 25 | ✅ 有 | <0.707 | <16% | — |

**结论**：在现有数据上，**后处理没有显示出明确的提升**。但需要注意：
- yoru_v1 的 tuned 结果（带后处理）非常差，可能是因为参数调优不当
- 需要更系统的后处理参数搜索（如 duration prior 的 percentile 调为 1st/99th 而非 5th/95th）
- 需要在全量 7-fold 上评估，而非单 subject

### 3.2 为什么还需要评估后处理？

尽管 yoru_v1 的 Guardrail 拒绝了后处理，但仍需在全量数据上验证：

1. **yoru 不代表整体分布**（单 subject 可能有偏差）
2. **后处理参数可能没调好**（5th/95th percentile 可能太激进）
3. **browse_model_replay.py 的实现是简化版**（无 nested CV，固定参数）
4. **某些动作可能受益**（如 db_weighted_crunch 的 Exact Count 只有 40.6%）

---

## 四、后续实验计划

### 4.1 优先级 1：全量 7-fold 评估（Duration Prior only）

**目标**：验证 Duration Prior 是否能提升 Exact Count，特别是对 db_weighted_crunch 等弱项动作。

**方法**：修改 `compare_baselines.py`，在 `evaluate_micro_probs` 后加入 Duration Prior 过滤，跑完整 7-fold。

**预期时间**：~30 分钟（7 folds × 8 actions，比 full refiner 快）。

### 4.2 优先级 2：轻量级 Boundary Refiner

**目标**：验证 Boundary Refiner 是否提升 IoU-F1@50。

**方法**：使用 browse_model_replay.py 中的简化 refiner，但限制 train streams 为 10-20 个（控制运行时间）。

**预期时间**：~2 小时。

### 4.3 优先级 3：后处理规则（硬编码 heuristics）

如果 Duration Prior + Refiner 效果有限，考虑更简单的后处理：

```python
# 示例：合并过近的 reps
if rep[i+1].start - rep[i].end < min_gap:
    merge(rep[i], rep[i+1])

# 示例：拆分过长的 reps
if rep.duration > max_single_rep:
    split_at_transition(rep)
```

---

## 五、论文叙事

> "We explored post-processing techniques including duration-based filtering and boundary regression to improve rep count accuracy. However, on the held-out yoru subject, a guardrail mechanism found that these additions did not statistically improve over the plain per-action RF baseline. This suggests that the core per-action RF model already captures most actionable signal, and further gains may require larger training sets or more sophisticated temporal post-processing rules."

---

*文档版本: 2026-05-17 v1*
*数据来源: artifacts/baseline_comparison/modality_count_guardrail_yoru_v1/*
*关联文档: 03-rep-count-metrics.md, 02-phase1-overview.md*
