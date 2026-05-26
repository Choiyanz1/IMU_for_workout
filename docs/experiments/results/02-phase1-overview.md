# Phase 1 Overview: Rep Segmentation 三大核心指标

## 文档目的

本文档提供 Phase 1（Rep Segmentation）的**执行摘要**。所有详细分解请参阅对应子文档。

**实验日期**: 2026-05-16 ~ 2026-05-17
**数据**: 7 subjects, 8 actions, 226 streams（清洗后）
**协议**: 严格 7-fold LOSO（Leave-One-Subject-Out）

---

## 一、三大核心指标（Three Critical Metrics）

Rep Segmentation 的目标不仅是检测出 rep，还要**准确计数**。因此评估体系包含三个同等级的指标：

### Table 1: 三大核心指标总览

| 指标 | 英文 | 定义 | Per-Action Plain RF | 重要性 |
|------|------|------|---------------------|--------|
| **Rep F1** | Repetition Detection F1 | 基于 IoU>0.5 的 rep 匹配 F1 | **0.850** | ⭐ 检测能力 |
| **IoU-F1@50** | Micro Phase F1@50 | sample-level phase 序列重叠度 | **0.706** | ⭐ 边界精度 |
| **Exact Count Ratio** | Rep Count Accuracy | 预测 rep 数量 = GT 数量的比例 | **65.9%** | ⭐ 计数准确性 |

> **关键洞察**：Rep F1 (0.85) 很好，但 Exact Count (65.9%) 意味着每 3 组训练就有 1 组 rep 数量不对。这是部署前必须改进的指标。

---

## 二、部署候选

### 当前最佳模型

| 属性 | 值 |
|------|-----|
| 模型 | Per-Action Plain RF (Causal) |
| 输入 | 6-axis trailing-window (100 samples, 1.0s) |
| 树/深度 | 100 trees / depth 15 |
| 总大小 | ~1.6 MB (8 actions) |
| 推理延迟 | 1.0s (causal window) |
| 训练协议 | 7-fold LOSO |

### 部署就绪状态

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Offline quality | ✅ | Rep F1=0.850, IoU-F1@50=0.706 |
| Streaming inference | ⏳ | 待验证（browse_model_replay.py 已开发） |
| Rep count accuracy | ⚠️ | 65.9% 需要改进（目标 >90%） |
| Model size | ✅ | 1.6MB 适合 64MB RAM |

---

## 三、快速结论

1. **Per-Action Plain RF 是最佳检测模型**（Rep F1=0.850, IoU-F1@50=0.706）
2. **Rep Count 是部署前必须改进的短板**（65.9% exact count）
3. **后处理（Duration Prior + Boundary Refiner）尚未在全量数据上验证**
4. **下一步**：评估后处理对 Rep Count 的提升效果 → 决定是否加入部署 pipeline

---

## 四、子文档速查

| 问题 | 参考文档 |
|------|----------|
| Rep Count 详细分析（按动作分解） | `03-rep-count-metrics.md` |
| 基线对比表（所有方法） | `04-baseline-comparison.md` |
| 按动作类型的性能差异 | `05-per-action-breakdown.md` |
| Duration Prior / Boundary Refiner 效果 | `06-post-processing-evaluation.md` |
| 为什么 BiLSTM < RF？ | `07-deep-learning-baselines.md` |
| 公平性讨论 | `08-baseline-fairness.md` |
| 特征重要性 | `04-feature-importance.md` |

---

*文档版本: 2026-05-17 v1*
*关联文档: 03-rep-count-metrics.md, 04-baseline-comparison.md, 06-post-processing-evaluation.md*
