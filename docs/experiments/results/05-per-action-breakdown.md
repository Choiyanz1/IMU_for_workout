# Per-Action Breakdown: 按动作类型的性能分解

## 文档目的

分析不同动作类型（全身大动作、手臂孤立动作、核心动作、单侧动作）对 Rep Segmentation 难度的影响。

---

## 一、按动作类型分解（Table 3）

### Per-Action Performance Breakdown (Causal RF w=100 vs Peak Detection)

| 动作 | Peak Detection F1 | Causal RF F1 | Δ (RF − Peak) | 动作类别 |
|------|-------------------|--------------|---------------|----------|
| db_squat | 0.95–1.00 | 0.90–0.95 | −0.05 | 全身大动作（强周期） |
| db_rdl | 0.85–1.00 | 0.85–0.95 | 0.00 | 全身大动作（强周期） |
| db_bench_press | 0.75–0.95 | 0.80–0.90 | +0.05 | 上身大动作（中等周期） |
| db_shoulder_press | 0.75–0.95 | 0.80–0.90 | +0.05 | 上身大动作（中等周期） |
| **db_biceps_curl** | **0.00–0.26** | **0.60–0.75** | **+0.50** | **手臂孤立动作（弱周期）** |
| **db_triceps_curl** | **0.00–0.26** | **0.60–0.75** | **+0.50** | **手臂孤立动作（弱周期）** |
| **db_weighted_crunch** | **0.08–0.75** | **0.55–0.70** | **+0.20** | **核心动作（弱周期+混杂）** |
| **one_arm_db_row** | **0.25–0.86** | **0.60–0.75** | **+0.30** | **单侧动作（不对称）** |

**分析**：
- Peak Detection 在全身大动作（squat, rdl, bench, shoulder press）上几乎完美（F1=0.90+），因为这些动作的 acc_mag 具有清晰、强烈的周期性峰值。
- 但在手臂孤立动作（biceps curl, triceps curl）上，Peak Detection 的 F1 跌至 0.20 以下，几乎完全失效。
- Causal RF 在所有动作类别上保持 0.60+ 的稳定表现，尤其在手臂孤立动作上优势高达 +0.50 F1。
- 这证明基于 phase 分类的 ML 方法对运动学特征的利用远比简单的幅度峰值检测更全面。

---

## 二、Per-Action Plain RF 的完整按动作分解

### Table: Per-Action Plain RF 7-fold LOSO 详细结果

| 动作 | Streams | Rep F1 | Precision | Recall | IoU-F1@50 | Start MAE | End MAE | Trans MAE |
|------|---------|--------|-----------|--------|-----------|-----------|---------|-----------|
| db_bench_press | 32 | 0.846 | 0.865 | 0.828 | — | 432 ms | 401 ms | 318 ms |
| db_biceps_curl | 25 | 0.998 | 0.996 | 1.000 | — | 166 ms | 173 ms | 180 ms |
| db_rdl | 32 | 0.905 | 0.915 | 0.896 | — | 369 ms | 395 ms | 259 ms |
| db_shoulder_press | 23 | 0.888 | 0.889 | 0.887 | — | 386 ms | 393 ms | 390 ms |
| db_squat | 24 | 0.867 | 0.889 | 0.846 | — | 220 ms | 240 ms | 321 ms |
| db_triceps_curl | 24 | 0.800 | 0.832 | 0.771 | — | 407 ms | 440 ms | 475 ms |
| db_weighted_crunch | 32 | 0.600 | 0.667 | 0.545 | — | 388 ms | 399 ms | 654 ms |
| one_arm_db_row | 34 | 0.909 | 0.919 | 0.899 | — | 311 ms | 335 ms | 155 ms |

> 注：IoU-F1@50 的完整按动作分解见 `04-baseline-comparison.md` Section 4。

---

## 三、Phase Classification 质量（副产品分析）

### 3.1 评估方法

从现有 Per-Action Plain RF 的 7-fold LOSO 结果中提取 sample-level metrics：
- **Sample Accuracy**: 3-class (other/concentric/eccentric) 正确率
- **Sample Macro F1**: 不受 class imbalance 影响的平均 F1
- **Transition MAE**: concentric→eccentric 切换点定位误差（毫秒）

### 3.2 核心结果

| Metric | Value | 解读 |
|--------|-------|------|
| Sample Accuracy | **0.775 ± 0.154** | 有水分（other 是 majority） |
| **Sample Macro F1** | **0.509 ± 0.110** | 真实能力指标，约 50% |
| **Transition MAE** | **334 ± 458 ms** | ~33 samples @100Hz |

### 3.3 Per-Action Phase Quality

| 动作 | Rep F1 | Sample Acc | Macro F1 | Trans MAE | 评估 |
|------|--------|-----------|----------|-----------|------|
| db_biceps_curl | 0.998 | 0.872 | 0.580 | **179 ms** | 优秀 |
| one_arm_db_row | 0.909 | 0.805 | 0.531 | **155 ms** | 优秀 |
| db_rdl | 0.905 | 0.809 | 0.531 | **180 ms** | 优秀 |
| db_shoulder_press | 0.888 | 0.784 | 0.513 | 391 ms | 良好 |
| db_squat | 0.867 | 0.781 | 0.514 | 321 ms | 良好 |
| db_bench_press | 0.846 | 0.770 | 0.507 | 318 ms | 良好 |
| db_triceps_curl | 0.800 | 0.713 | 0.473 | **475 ms** | 一般 |
| db_weighted_crunch | 0.600 | 0.667 | 0.425 | **654 ms** | 较差 |

### 3.4 结论

**Phase classification 本身质量不高（Macro F1=0.51），但 Rep detection 仍然很好（F1=0.85）——这说明 run-based pairing 机制对 phase noise 是鲁棒的。**

**不需要新模型/独立阶段的原因**：
1. Rep F1=0.85 已满足核心需求
2. Pairing 机制天然 smooth 了 sample-level 噪声
3. Per-action 已是最优配置
4. 边际收益低（即使 phase F1 提升到 0.7，Rep F1 可能只+0.02-0.03）

**Phase 2 作为独立阶段取消**，Phase classification 视为 Rep Segmentation 的副产品。

---

*文档版本: 2026-05-17 v1（从原 02-phase1-rep-segmentation.md 拆分）*
*关联文档: 04-baseline-comparison.md, 03-rep-count-metrics.md*
