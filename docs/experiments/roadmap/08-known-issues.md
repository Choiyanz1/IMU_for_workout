# 已知问题清单（Roadmap）

## 文档目的

**本文档记录项目中所有已知的设计缺陷、实现问题、实验异常和待修复项**。这是执行实验前的风险检查清单和后续任务追踪。

> **⚠️ 本文档属于 Roadmap（计划）**，记录待修复项和后续任务。
> 
> **实验结果与证据**请查阅 `docs/experiments/results/*.md`

**状态更新**：Phase 1a 大部分 baseline 已完成，多个历史问题已解决（Peak Detection 实现、DTW acc_mag 修复、Causal RF 配置优化）。当前主要剩余任务：SDTW/BiLSTM baseline、Per-action RF+Refiner 重跑。

**核心约束**：
1. 所有方法必须验证板子部署可行性
2. Baseline Comparison 可包含 non-causal 方法作为理论上限对照
3. 使用任何旧脚本前必须确认设计合理性
4. 任何大规模实验前必须先做 smoke test

---

## 一、模型相关问题

### 1.1 DS-MS-TCN 已放弃（不再是问题，但需记录）

| 项目 | 详情 |
|------|------|
| 问题 | 多任务目标错位：优化 frame-level phase classification，而非 rep-level boundary |
| 影响 | 在 held-out Kevin 上 Rep F1 仅 0.53，远不如 RF |
| 状态 | ✅ 已放弃，不再投入 |
| 记录位置 | `docs/experiments/2026-05-13-next-agent-handoff.md` |
| 使用建议 | **不使用**此脚本（`train/micro_macro_recognition.py`） |

### 1.2 Phase-only causal TCN 效果不佳

| 项目 | 详情 |
|------|------|
| 问题 | 虽然是因果模型，但 Rep F1 (0.69) < RF (0.76)，micro_f1@50 (0.45) < RF (0.64) |
| 影响 | 不能作为论文主推方法 |
| 状态 | ✅ 已明确为对照组（abandoned attempt） |
| 用途 | 放在 baseline comparison 表格最后一行，展示"我们尝试过但不够好" |
| 部署可行性 | ⚠️ 因果但模型较大，不适合 64MB RAM |

---

## 二、Baseline 相关问题

### 2.1 SDTW 评估路径可能不正确

| 项目 | 详情 |
|------|------|
| 问题 | 当前 `evaluation/rep_segmentation.py` 的评估方式可能有误：DTW 模板从单个 rep CSV 建立，但检测时可能也在单个 rep CSV 上做匹配，而不是 set-level stream |
| 影响 | 如果属实，DTW baseline 的结果不公平（在已知边界上匹配） |
| 状态 | ⚠️ 待验证 |
| 验证方法 | 检查 `_load_set_streams` 是否被主评估循环正确使用 |
| 修复方案 | 确保 DTW 检测在拼接后的 set stream 上进行 |

### 2.2 DTW 只用单一特征 ✅ 已修复

| 项目 | 详情 |
|------|------|
| 问题 | `_build_template_from_exemplar` 中 `dtw_feature = ranked[0]` 只使用变化量最大的单一特征 |
| 影响 | DTW 未充分利用多轴信息 |
| 状态 | ✅ **已修复** |
| 修复方案 | 统一使用 `acc_mag` 作为 DTW 的输入特征（这是 1D 方法的 natural input） |
| 备注 | `preprocessing/sdtw_rep_segmentation.py` 已更新 |

### 2.3 Peak Detection ✅ 已实现

| 项目 | 详情 |
|------|------|
| 问题 | 没有现成的 Magnitude Peak Detection baseline 脚本 |
| 影响 | 缺少最简单、最直觉的 baseline |
| 状态 | ✅ **已完成** |
| 脚本 | `scripts/evaluate_peak_baseline.py` |
| 结果 | 7-fold LOSO Rep F1 = **0.757 ± 0.073** |
| 重要 bug | ⚠️ `estimate_duration_prior` 曾错误使用连续 active segments，导致 F1=0.000（已修复） |
| 部署可行性 | ✅ 完全可部署（零模型参数） |

### 2.4 Sliding-window RF 和 BiLSTM

| 项目 | 详情 |
|------|------|
| 问题 | 两者都是非因果的，不可部署 |
| 影响 | - |
| 状态 | ✅ **保留作为理论上限对照**（非因果方法的性能上限） |
| 论文角色 | 证明"我们的 causal 方法接近非因果理论上限，且可以实时部署" |
| 使用方式 | 放入 baseline comparison 表格，但明确标注 ❌ 非因果 |

---

## 三、数据相关问题

### 3.1 rep0 文件

| 项目 | 详情 |
|------|------|
| 问题 | 数据集中仍有 307 个 `rep0_*.csv` 文件 |
| 影响 | rep0 文件可能包含 `none` / `inter_set_rest` 相位，污染 phase 模型训练 |
| 状态 | ✅ 用户表示关键文件已清理，rep0 不是问题 |
| 处理方式 | 无需额外处理 |

### 3.2 Rest 文件远大于 Rep 文件

| 项目 | 详情 |
|------|------|
| 问题 | Rest CSV 可达 5000-10000 samples，而 rep CSV 仅 200-400 samples |
| 影响 | 如果 z-score 基于整个 dataset（包含 rest），统计量会被 rest 主导 |
| 状态 | ✅ 已知，已有缓解措施 |
| 缓解措施 | z-score 只基于 active segments（concentric/eccentric）；评估时只计算包含 reps 的 streams |

### 3.3 磁力计信号质量差

| 项目 | 详情 |
|------|------|
| 问题 | MAG 的 CV 极低（~0.13），轴间高度冗余（mx-my 相关 -0.916） |
| 影响 | MAG-only 模型性能预期很差；AGM vs AG 的提升预期很小 |
| 状态 | ✅ 已知，这正是 Modality Ablation 要验证的假设 |
| 处理方式 | 在论文中明确报告 MAG 的低贡献 |

---

## 四、实验相关问题

### 4.1 历史实验结果波动大

| 项目 | 详情 |
|------|------|
| 问题 | DS-MS-TCN 同一架构不同运行：Rep F1 从 0.78 → 0.68 → 0.53 |
| 影响 | 训练不稳定，结果不可复现 |
| 状态 | ✅ 已放弃 TCN，不再相关 |
| 教训 | 对于 RF 实验，必须固定 random seed，确保结果可复现 |

### 4.2 缺少统计显著性检验

| 项目 | 详情 |
|------|------|
| 问题 | 历史实验报告只有单点估计，没有 confidence interval 或 significance test |
| 影响 | 无法判断改进是否显著 |
| 状态 | ⚠️ 将在新实验中修复 |
| 修复方案 | 9-fold LOSO 提供 mean±std；Wilcoxon signed-rank test 检验显著性 |

### 4.3 Modality-only 的 Per-action 最佳模态跨 subject 不稳定

| 项目 | 详情 |
|------|------|
| 问题 | `db_bench_press` 在 yushuan 最佳是 `acc+gyro`，在 kevin 最佳是 `acc` |
| 影响 | 所谓"per-action 最佳模态"可能只是过拟合到特定 subject |
| 状态 | ✅ 已知，支持"使用完整 6-axis 作为安全默认"的结论 |
| 处理方式 | 在论文中报告这一不稳定性，作为支持统一 6-axis 的证据 |

### 4.4 单 rep Action Classification 效果可能不好

| 项目 | 详情 |
|------|------|
| 问题 | 用户担心"用一个 rep 的范围去做动作辨识的效果其实很不好" |
| 影响 | Action Classification 可能是整个 pipeline 的第二个瓶颈 |
| 状态 | ⚠️ 待验证 |
| 验证方法 | Smoke test 先跑 3 subjects，看 per-rep Macro F1 |
| 关键认知 | **Per-rep F1 > 0.80 就够了**！因为可以利用 set-level 结构约束做后处理（majority vote / consecutive agreement），把 set-level accuracy 提升到 95%+ |
| 缓解方案 | 1) 先确认 per-rep F1 > 0.80；2) 实施 set-level majority voting；3) 在论文中同时报告 per-rep 和 set-level accuracy |

---

## 五、部署相关问题

### 5.1 LuckFox Pico Zero 部署验证

| 项目 | 详情 |
|------|------|
| 问题 | RF + Refiner 的 pipeline 只在 Python/sklearn 上验证，没有在目标硬件上运行 |
| 影响 | 无法确认实时性能、内存占用、功耗 |
| 状态 | ⚠️ 当前不在论文 scope 内，但需要在 future work 中提及 |
| 可行性分析 | RF (~100 trees, depth 10) ≈ 50-200 KB 模型；手写决策树遍历在 C 中可行；64MB RAM 足够 |
| 处理方式 | 论文中标注 "deployment validation on target hardware is future work" |

### 5.2 ONNX/RKNN 导出路径缺失

| 项目 | 详情 |
|------|------|
| 问题 | 现有 `deploy/export_onnx.py` 只支持 InertialStudent（旧模型），不支持 RF + Refiner |
| 影响 | 无法直接导出到 LuckFox |
| 状态 | ⚠️ 当前不在论文 scope 内 |
| 处理方式 | 论文中标注 "sklearn-based RF can be converted to ONNX via skl2onnx; refiner is lightweight regression" |

### 5.3 SDTW 的板子计算量

| 项目 | 详情 |
|------|------|
| 问题 | DTW 是 O(n²) 计算，在 30s × 100Hz = 3000 samples 的 stream 上可能太慢 |
| 影响 | SDTW 可能不适合实时部署 |
| 状态 | ⚠️ 需验证 |
| 处理方式 | 在 baseline comparison 中保留 SDTW 但标注 "实时性待验证"；如果不可行，只作为离线 baseline |

---

## 六、脚本验证清单（执行前必须完成）

在使用任何旧脚本执行 Phase 1 前，必须验证以下设计合理性：

### 6.1 `scripts/evaluate_causal_rf.py`

- [ ] trailing window 是否只用过去样本（严格 causal）？
- [ ] z-score 是否只从 train subjects 计算？
- [ ] 是否存在任何形式的 test subject 信息泄露？
- [ ] 随机种子是否固定（确保可复现）？

### 6.2 `scripts/benchmark_per_action_rf_refiner.py`

- [ ] refiner 的 edge_window 是否不超出 coarse boundary 范围？
- [ ] regression 特征是否不泄露 test subject 统计信息？
- [ ] 是否移除了 hyperparameter tuning（固定参数）？
- [ ] 是否存在 data leakage？

### 6.3 `preprocessing/sdtw_rep_segmentation.py`

- [ ] 评估是否在 set-level merged streams 上？
- [ ] `dtw_feature` 是否使用 `acc_mag`？
- [ ] duration prior 是否只从 train subjects 估计？
- [ ] threshold 是否只从 train subjects 校准？

### 6.4 `train/action_classification.py`

- [ ] 特征提取是否不泄露 test subject 统计信息？
- [ ] z-score 是否只从 train reps 计算？
- [ ] AutoGluon 的内层 tuning 是否不跨越 subject 边界？

### 6.5 不使用以下脚本（设计已确认不合理）

- [x] `train/micro_macro_recognition.py`（DS-MS-TCN 已放弃）

---

## 七、风险矩阵

| 问题 | 严重程度 | 发生概率 | 风险等级 | 缓解措施 | 状态 |
|------|---------|---------|---------|---------|------|
| DTW 评估路径错误 | 高 | 中 | 🔴 高 | 实验前验证 `_load_set_streams` | ⏳ 待验证 |
| 缺少 Peak Detection | 中 | 高 | 🟡 中 | 立刻实现 | ✅ **已完成** |
| 9-fold LOSO 时间过长 | 中 | 低 | 🟡 中 | 使用 outer-fold parallelism | ✅ 已优化到 7-subject |
| **RF 輸給 Peak Detection** | 高 | 中 | 🔴 高 | 执行 Contingency Plan A | ✅ **已解决**（RF 0.778 > Peak 0.757） |
| **所有方法 F1 < 0.7** | 高 | 低 | 🔴 高 | 停止實驗，修復數據 | ✅ **已解决**（RF 0.778） |
| **BiLSTM 只比 RF 好 < 5%** | - | - | 🟢 低 | **这是好消息！** 強調"接近理論上限" | ⏳ 待 BiLSTM 实验验证 |
| MAG-only 模型过弱 | 低 | 高 | 🟢 低 | 这正是要报告的"发现" | ✅ 已知 |
| 部署验证缺失 | 中 | 低 | 🟡 中 | 在论文 future work 中提及 | ✅ 已记录 |
| SDTW 板子计算量过大 | 中 | 中 | 🟡 中 | 只作为离线 baseline 对比 | ⏳ 待验证 |
| **单 rep Action Classification 效果差** | 高 | 中 | 🔴 高 | 使用 Set-level Majority Voting | ⏳ 待 Phase 3 验证 |

---

## 八、实验前检查清单（Phase 1a 已完成项标记 ✅）

在开始 Phase 1 实验前，必须完成以下检查：

- [x] 确认 `scripts/evaluate_causal_rf.py` 严格 causal 且无 leakage ✅
- [ ] 确认 `scripts/benchmark_per_action_rf_refiner.py` 无 leakage 且已移除 tuning
- [ ] 确认 DTW 评估路径在 set-level streams 上
- [x] 修复 DTW 的 single-feature 问题（改用 acc_mag）✅
- [x] 实现 Magnitude Peak Detection baseline ✅
- [x] 固定所有 random seeds（RF, numpy, python hash）✅
- [x] 验证 7-fold LOSO 的正确性（无 subject overlap）✅
- [x] **执行 7-subject 完整实验（haoyu, hsianshun, kevin, thomas, yoru, yushuan, yanz）** ✅
- [x] **验证 Causal RF 配置优化（window_size 50→100）** ✅
- [x] **确认 Causal RF (0.778) > Peak Detection (0.757)** ✅
- [ ] **重跑 Per-action RF+Refiner**（当前结果 0.619 异常低，可能是 bug）
- [ ] **跑 SDTW 7-subject LOSO**
- [ ] **跑 BiLSTM 7-subject LOSO**

---

*文档版本: 2026-05-16 v3*
*状态: 动态更新，每次发现新问题需追加*
*下一步: 完成剩余 baseline（SDTW, BiLSTM）；修复并验证 Per-action RF+Refiner*
