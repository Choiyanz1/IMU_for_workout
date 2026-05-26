# 实验结果文档目录（Phase 1 Rep Segmentation 重组版）

## 文档定位

本文档说明 `docs/experiments/results/` 目录的用途和内容索引。

**本目录存放论文/最终报告要呈现的所有内容**：
- 实验数据与比较表格
- 背景解释与分析
- 使用各种方法的理由与证据
- 数据集来源与描述

**不包含**：
- 未来计划或待办事项
- 实验进展方向
- 技术实施细节（这些在 roadmap 和 dev-log 中）

---

## 文件索引（Phase 1 Rep Segmentation）

| 序号 | 文件名 | 内容 | 对应论文章节 |
|------|--------|------|-------------|
| 1 | `01-dataset-description.md` | 数据集来源、收集方式、清洗过程、最终分布 | Dataset |
| 2 | `02-phase1-overview.md` | **三大核心指标总览**、部署候选结论 | Executive Summary |
| 3 | `03-rep-count-metrics.md` | **Rep Count 数量准确性**（第三核心指标）、按动作分解 | Results: Rep Count |
| 4 | `04-baseline-comparison.md` | 基线对比总表、Causal RF 配置优化、梯度提升完整结果 | Results: Table 1-2 |
| 5 | `05-per-action-breakdown.md` | 按动作类型的性能分解、Peak Detection 对比 | Results: Table 3 |
| 6 | `06-post-processing-evaluation.md` | Duration Prior、Boundary Refiner、Guardrail 效果评估 | Results: Post-processing |
| 7 | `07-deep-learning-baselines.md` | 1D CNN、BiLSTM（Basic + Tuned）结果 | Results: DL Baselines |
| 8 | `08-baseline-fairness.md` | 基线比较公平性说明、输入差异解释 | Discussion: Fairness |
| 9 | `09-historical-archive.md` | 历史结果存档、版本演进记录 | Appendix |
| 10 | `06-post-processing-evaluation.md` | Duration Prior、Boundary Refiner、Guardrail 效果评估 | Results: Post-processing |
| 11 | `07-deep-learning-baselines.md` | 1D CNN、BiLSTM（Basic + Tuned）结果 | Results: DL Baselines |
| 12 | `08-baseline-fairness.md` | 基线比较公平性说明、输入差异解释 | Discussion: Fairness |
| 13 | `09-historical-archive.md` | 历史结果存档、版本演进记录 | Appendix |
| 14 | **`10-presentation-deck.md`** | **简报故事线：12 Slides 完整叙述 + 图表索引** | **Presentation** |
| 15 | `03-model-selection-rationale.md` | 为什么选 RF？候选模型对比、部署分析 | Method: Model Selection |
| 16 | `04-feature-importance.md` | RF 特征重要性分析、特征工程洞察 | Results: Feature Analysis |

> **注**：原 `02-phase1-rep-segmentation.md`（v4, 391 行）已拆分为上述 02-09。保留旧文件供回溯，但论文引用请以拆分后的版本为准。

---

## 三大核心指标速查

| 指标 | 数值 | 来源 |
|------|------|------|
| **Rep F1** | **0.850** | `04-baseline-comparison.md` Table 1 |
| **IoU-F1@50** | **0.706** | `04-baseline-comparison.md` Table 1 |
| **Exact Count Ratio** | **65.9%** (149/226 streams) | `03-rep-count-metrics.md` |

---

## 与 Roadmap 的关系

| 类型 | 路径 | 内容 |
|------|------|------|
| **Result（结果）** | `docs/experiments/results/*.md` | 论文要呈现的数据、分析、证据 |
| **Roadmap（计划）** | `docs/experiments/roadmap/*.md` | 实验设计、未来方向、待办事项 |
| **Log（日志）** | `docs/experiments/2026-*.md` | 每次实验的详细记录（按日期） |
| **Dev Log** | `docs/dev-log.md` | 开发过程记录、关键决策时间点 |

---

## 更新规则

- **Result 文档**：每次实验完成后更新数据表格，保留历史版本
- **Roadmap 文档**：规划未来实验时更新，记录已完成里程碑
- **Log 文档**：每次实验 session 结束后立即记录

---

*文档版本: 2026-05-17 v5（Phase 1 文档重组）*
