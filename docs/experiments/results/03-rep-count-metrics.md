# Rep Count Metrics: 第三核心指标

## 文档目的

本文档分析 Rep Segmentation 的**数量准确性**——预测出的 rep 数量是否等于真实的 rep 数量。这是用户最直接感知的指标（「这组我做了 10 下，App 显示 7 下」）。

**核心论点**：Rep Count Accuracy 与 Rep F1、IoU-F1@50 **同级重要**，但之前未被充分强调。

---

## 一、指标定义

### 1.1 Exact Count Ratio（精确计数比例）

$$\text{Exact Count Ratio} = \frac{\text{预测数量} = \text{真实数量的 streams}}{\text{总 streams}}$$

### 1.2 Mean Absolute Count Difference（平均绝对数量差）

$$\text{MADiff} = \frac{1}{N} \sum_{i=1}^{N} |\text{Pred}_i - \text{True}_i|$$

### 1.3 Over-segmented / Under-segmented

| 类型 | 定义 | 用户感知 |
|------|------|----------|
| **Over-segmented** | 预测数 > 真实数 | 「App 多算了我做的次数」 |
| **Under-segmented** | 预测数 < 真实数 | 「App 少算了我做的次数」 |
| **Exact Count** | 预测数 = 真实数 | 「数量完全正确」 |

---

## 二、Per-Action Plain RF 的 Rep Count 结果

### Table 1: 按动作类型分解（7-fold LOSO, 226 streams）

| 动作 | Streams | Rep F1 | Exact% | Over | Under | Zero TP | MADiff | 评估 |
|------|---------|--------|--------|------|-------|---------|--------|------|
| db_biceps_curl | 25 | 0.998 | **96.0%** | 1 | 0 | 0 | 0.24 | 🟢 优秀 |
| one_arm_db_row | 34 | 0.909 | **85.3%** | 1 | 4 | 3 | 9.26 | 🟢 较好 |
| db_rdl | 32 | 0.905 | **65.6%** | 5 | 6 | 0 | 4.47 | 🟡 中等 |
| db_squat | 24 | 0.867 | **62.5%** | 7 | 2 | 0 | 1.12 | 🟡 中等 |
| db_bench_press | 32 | 0.846 | **62.5%** | 5 | 7 | 0 | 7.12 | 🟡 中等 |
| db_triceps_curl | 24 | 0.800 | **62.5%** | 7 | 2 | 0 | 1.62 | 🟡 中等 |
| db_shoulder_press | 23 | 0.888 | **52.2%** | 6 | 5 | 0 | 2.09 | 🟡 中等 |
| **db_weighted_crunch** | **32** | **0.600** | **40.6%** | **5** | **14** | **4** | **18.81** | 🔴 **最差** |
| **OVERALL** | **226** | **0.850** | **65.9%** | **37** | **40** | **7** | **6.23** | — |

### 2.1 关键发现

**1. 动作类型差异巨大**
- db_biceps_curl (96.0%) 几乎是完美的——动作模式清晰、周期性稳定
- db_weighted_crunch (40.6%) 是灾难性的——平均每组差 18.8 个 reps
- 原因：crunch 的 rest/active 切换模糊，short reps 容易被合并或遗漏

**2. Rep F1 高 ≠ Rep Count 准**
- db_weighted_crunch 的 Rep F1 = 0.600（已经较低），但 Exact Count 更低（40.6%）
- 说明模型能「找到」一些 reps（Recall 不差），但**数量完全不对**
- db_shoulder_press 的 Rep F1 = 0.888（很高），但 Exact Count = 52.2%——说明找到的 reps 边界不准，导致合并/分裂

**3. Under-segmented 是主要问题**
- 整体：Over=37 folds, Under=40 folds（大致平衡）
- 但 db_weighted_crunch：Under=14 >> Over=5（严重漏检）
- db_bench_press：Over=5, Under=7（略偏漏检）

---

## 三、与 IoU-F1@50 的关系

Rep Count 误差的主要来源：

| 误差类型 | 原因 | IoU-F1@50 表现 | Rep Count 表现 |
|----------|------|----------------|----------------|
| **边界太松** | 相邻 reps 被合并成一个 | 中等（IoU 覆盖）| Under-segmented |
| **边界太紧** | 一个 rep 被拆成多个 | 中等（IoU 覆盖）| Over-segmented |
| **漏检** | 完全没检测到某个 rep | 低（FN 高）| Under-segmented |
| **虚警** | 非 rep 区域被检测 | 低（FP 高）| Over-segmented |

> **洞察**：IoU-F1@50 衡量的是「找到 rep 后边界准不准」，而 Rep Count 衡量的是「总共找到多少个」。两者互补——高 IoU + 低 Count = 边界准但合并/分裂问题；高 Count + 低 IoU = 数量对但边界漂移。

---

## 四、部署目标与用户接受度

### 4.1 用户可接受的误差水平（行业参考）

| Exact Count Ratio | 用户体验 | 是否可接受 |
|-------------------|----------|------------|
| >90% | 「几乎每次都准」| ✅ 可接受 |
| 80-90% | 「偶尔不准」| ⚠️ 边缘 |
| 65-80% | 「经常需要手动修正」| ❌ 不可接受 |
| <65% | 「完全不相信 App」| ❌ 不可接受 |

**当前 65.9% 处于「不可接受」边缘。**

### 4.2 改进方向（按优先级）

| 优先级 | 改进点 | 预期效果 | 验证状态 |
|--------|--------|----------|----------|
| 1 | Duration Prior（过滤异常长短 reps）| 减少 Over/Under | ⏳ 待全量验证 |
| 2 | Boundary Refiner（修正 rep 边界）| 减少合并/分裂 | ⏳ 待全量验证 |
| 3 | 后处理规则（merge/split heuristics）| 修复明显错误 | ⏳ 未实现 |
| 4 | Per-action 阈值调优 | 按动作优化 min_phase_samples | ⏳ 未实现 |

---

## 五、modality_count_guardrail_yoru_v1 的 Rep Count 结果

### 5.1 结果说明

`modality_count_guardrail_yoru_v1` 尝试了以下后处理组合：
- Duration Prior（按动作过滤异常 duration）
- Boundary Refiner（ExtraTreesRegressor 修正边界）
- Modality Selection（选择最佳 sensor 组合）
- Guardrail（如果后处理效果不如 baseline，回退到无后处理）

**Guardrail 的最终决策**：**所有动作的 best config 都是 `baseline_reference`**——即**无 Duration Prior、无 Boundary Refiner、无 Modality 切换**。

这意味着在 yoru held-out 的 25 streams 上：
- Duration Prior + Boundary Refiner + Modality Selection **没有带来统计显著的提升**
- Guardrail 回退到了最原始的 Per-Action Plain RF

### 5.2 yoru_v1 的 Rep Count（baseline_reference）

| 指标 | 数值 | 测试集 |
|------|------|--------|
| Rep F1 | 0.876 | yoru held-out, 25 streams |
| Exact Count Ratio | **76.0%** (19/25) | yoru held-out |
| Over-segmented | 6 streams | — |
| Under-segmented | 0 streams | — |

> **注意**：76.0% > 65.9%，但测试集不同（yoru only vs 7 subjects, 25 vs 226 streams）。不能直接比较。

---

## 六、与 Rep F1 和 IoU-F1@50 的关系总结

| 维度 | Rep F1 (0.850) | IoU-F1@50 (0.706) | Exact Count (65.9%) |
|------|----------------|-------------------|---------------------|
| **衡量什么** | 能不能找到 rep | 找到的 rep 边界准不准 | 总共找到的数量对不对 |
| **用户感知** | 间接（看边界线） | 间接（看 phase 色块） | **直接（看 rep 计数）** |
| **当前短板** | 无（已足够） | 中等（可提升） | **明显（必须提升）** |
| **改进优先级** | 低 | 中 | **高** |

---

## 七、论文呈现建议

在论文 Results 章节，建议将 Rep Count 作为独立子节（与 Rep F1 和 IoU-F1@50 并列）：

```
4.3 Rep Count Accuracy
    - Table: Per-action exact count ratio
    - Figure: Bar chart of exact% by action
    - Analysis: Why crunch/shoulder press are hardest
    - Discussion: Target >90% for deployment
```

---

*文档版本: 2026-05-17 v1*
*数据来源: artifacts/baseline_comparison/per_action_plain_rf_7fold/*
*关联文档: 02-phase1-overview.md, 06-post-processing-evaluation.md*
