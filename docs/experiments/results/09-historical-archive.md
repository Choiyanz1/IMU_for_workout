# Historical Archive: Rep Segmentation 结果演进

## 文档目的

记录 Rep Segmentation 实验的历史版本演进，供论文 Appendix 或回溯参考。

---

## 一、版本演进表

| 日期 | 配置 | 数据 | Rep F1 | 备注 |
|------|------|------|--------|------|
| 2026-05-08 | w=50, n=50 | 9 subjects 未清洗 | 0.485 | sample_rate_hz=50 (bug) |
| 2026-05-10 | w=50, n=50 | 9 subjects 未清洗 | 0.702 | sample_rate_hz=100 (修复后) |
| 2026-05-12 | w=50, n=50 | 7 subjects 清洗 | 0.706 | 清洗后数据，7-subject |
| 2026-05-16 | w=100, n=100 | 7 subjects 清洗 | 0.778 | 增大 window_size 到 1.0s |
| **2026-05-17** | **w=100, n=100, per-action** | **7 subjects 清洗** | **0.850** | **per-action 训练 (Action-First 架构)** |

---

## 二、关键决策时间点

| 日期 | 决策 | 影响 |
|------|------|------|
| 2026-05-10 | 修复 sample_rate_hz bug (50→100) | F1 从 0.485 → 0.702 |
| 2026-05-12 | 数据清洗（去除异常 subject） | 排除 _tsenyu_temp, _ziho_temp 的噪声数据 |
| 2026-05-16 | window_size 从 50→100 (0.5s→1.0s) | F1 从 0.706 → 0.778 (+0.072) |
| 2026-05-17 | 采用 Per-Action 训练 | F1 从 0.778 → 0.850 (+0.072) |

---

## 三、废弃模型存档

| 模型 | 废弃日期 | 原因 | 替代 |
|------|---------|------|------|
| DS-MS-TCN (2026-05-08) | 2026-05-08 | Rep F1 太低 (~0.485)， poor recall | Per-Action Plain RF |
| 1D CNN (full 9-fold) | 2026-05-16 | Rep F1=0.698，远低于 RF | 不部署 |
| BiLSTM Basic (full 9-fold) | 2026-05-16 | 过拟合严重，F1=0.758 | 不部署 |
| yoru_v1 guardrail (all modalities) | 2026-05-17 | Guardrail 回退到 baseline，无提升 | 不采用 |

---

*文档版本: 2026-05-17 v1（从原 02-phase1-rep-segmentation.md 拆分）*
