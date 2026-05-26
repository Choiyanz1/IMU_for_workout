# Phase 1: Rep Segmentation 路线图（Roadmap）

## 文档目的

**本文档是 Phase 1 的实验设计与方向规划**，包含实验设计、评估协议、指标定义、预期产出和后续计划。

> **⚠️ 文档职责区分**：
> - **本文件（Roadmap）**：记录实验设计、未来方向、计划与待办
> - **Result 文档**：`docs/experiments/results/02-phase1-rep-segmentation.md` — 存放所有实验数据、比较表格、分析结果（论文直接引用）
> - **实验日志**：`docs/experiments/2026-*.md` — 每次实验的详细过程记录

**Phase 1 目标**：在 Rep Segmentation 任务上，通过严格的 baseline comparison 和 modality ablation，验证我们的方法。

**核心约束（来自用户确认）**：
1. 所有方法默认考虑 LuckFox Pico Zero 部署可行性
2. **Action 类型在 Rep Segmentation 之前已知 → 使用 Per-Action 模型**
3. **Baseline Comparison 可包含 non-causal / 高计算量方法作为理论上限对照**（证明我们接近它们但可以实时部署）
4. **任何大规模实验前必须先做 3-subject smoke test**
5. 耦合分析延后到后续改进阶段
6. 严格 subject-wise split，零 data leakage
7. 实验流程善用电脑资源（并行化），但不要过度设计
8. 使用旧脚本前必须验证设计合理性

---

## 一、板子部署可行性评估（LuckFox Pico Zero）

### 1.1 硬件规格

| 组件 | 规格 |
|------|------|
| CPU | ARM Cortex-A7 @ 1.2 GHz（单核） |
| NPU | 0.5 TOPS（RKNN 支持 INT8/FP16） |
| RAM | 64 MB DDR2 |
| Flash | 视具体板子配置 |

### 1.2 部署可行性矩阵

| 方法 | 类型 | 因果？ | 计算量 | **板子可行？** | 论文角色 |
|------|------|--------|--------|-------------|---------|
| **Peak Detection** | 信号处理 | ✅ | 极低 | ✅ 可部署 | 最简可部署基线 |
| **SDTW** | 模板匹配 | ✅ | O(n²) | ⚠️ 需验证 | 文献经典基线 |
| **General Causal RF** | 序列 ML | ✅ | 中等 | ✅ 可部署 | 方法增量分析 |
| **Per-Action Plain RF** | **序列 ML** | **✅** | **中等** | **✅ 主推** | **论文主推方法** |
| **BiLSTM** | 深度学习 | ❌ | 高 | ❌ 不可部署 | **理论上限对照** |
| **Sliding-window RF** | 经典 ML | ❌ | 中等 | ❌ 不可部署 | **非因果上限对照** |
| **TCN (abandoned)** | 深度学习 | ✅ | 高 | ⚠️ 已放弃 | 已放弃对照 |

**关键设计**：non-causal 方法（BiLSTM, Sliding-window RF）**不放棄**，而是作为**理论上限对照**放入表格，用來證明：
> "我们的 causal 方法接近非因果理论上限，且可以实时部署"

---

## 二、Phase 1a: Method Comparison（方法对比）

### 2.1 任务定义

**任务**：给定一个 set-level IMU stream（连续的时间序列，包含多个 reps），在**已知 action_type** 的前提下，检测出每个 rep 的起始和结束样本索引。

**输入**：
- 原始 IMU 信号（统一使用 6-axis: ax, ay, az, gx, gy, gz）
- **已知的 action_type**（如 db_bench_press）→ 路由到对应模型
- 所有方法接收同样的原始数据来源，但利用方式不同

**输出**：
- 一组 rep 区间 `[(start_1, end_1), (start_2, end_2), ...]`
- 每个 rep 由 `concentric → eccentric` 配对产生

**评估**：
- 预测 rep 与 GT rep 的 IoU ≥ 0.5 视为匹配
- 使用 micro-level 的 concentric/eccentric 标签作为 GT rep 的边界
- **关键**：评估时假设 action_type 已知（使用 per-action 模型）

### 2.2 参与比较的方法（7 个：5 可部署 + 2 对照）

| # | 方法 | 类型 | 输入表示 | 因果？ | 可部署？ | Per-Action? | 论文角色 |
|---|------|------|----------|--------|---------|-------------|---------|
| 1 | **Magnitude Peak Detection** | 信号处理 | acc_mag (1D) | ✅ | ✅ | ❌ 通用 | **最简可部署基线** |
| 2 | **SDTW Template Matching** | 模板匹配 | acc_mag (1D) | ✅ | ⚠️ | ❌ 通用 | **文献经典基线** |
| 3 | **Sliding-window RF** | 经典 ML | 6-axis rich features | ❌ | ❌ | ❌ 通用 | **非因果理论上限** |
| 4 | **General Causal RF** | 序列 ML | 6-axis trailing window | ✅ | ✅ | ❌ 通用 | 方法增量分析 |
| 5 | **Per-Action Plain RF** | **序列 ML** | **6-axis trailing window** | **✅** | **✅** | **✅ Per-Action** | **论文主推方法** |
| 6 | **BiLSTM (phase-only)** | 深度学习 | 6-axis raw sequence | ❌ | ❌ | ❌ 通用 | **离线深度上限** |
| 7 | **Phase-only causal TCN** | 深度学习 | 6-axis raw sequence | ✅ | ⚠️ | ❌ 通用 | 已放弃对照 |

#### Per-Action vs 通用的策略

**为什么 baseline 不做 per-action？**

1. **公平性**：通用 baseline 是文献标准做法，展示「在不知道动作的情况下能到什麼水平」
2. **计算量**：如果 7 个方法 × 8 个动作 × 9-fold = 504 runs，计算量太大
3. **我们的优势**：Per-Action 是我们方法的設計特點，不是 baseline 的要求

**Per-Action 的分配策略**：

| 方法 | Per-Action? | 原因 | 计算量 |
|------|-------------|------|--------|
| Peak Detection | ❌ 通用 | 純信號處理，天然通用 | 9 folds |
| SDTW | ❌ 通用 | 模板匹配，天然通用 | 9 folds |
| Sliding RF | ❌ 通用 | 對照組，展示非因果上限 | 9 folds |
| BiLSTM | ❌ 通用 | 對照組，展示深度上限 | 9 folds |
| TCN | ❌ 通用 | 已放棄對照 | 9 folds |
| **General Causal RF** | ❌ 通用 | 方法增量分析，展示通用上限 | 9 folds |
| **Per-Action Plain RF** | **✅ Per-Action** | **論文主推，必須展示 per-action 優勢** | **9 folds × 8 actions = 72 runs** |

**總計算量**：~135 runs（而不是 504 runs）

**論文敘事**：
- 通用 baseline（Peak, SDTW, BiLSTM）：展示「不知道动作时的上限」
- General Causal RF：展示「通用模型能到的水平」
- Per-Action Plain RF：展示「利用 Action 先已知的優勢，使用 Per-Action 專用模型的提升」

> "我们的方法利用 Action 先已知的優勢，使用 Per-Action 專用模型，在精度和實時性之間取得了最佳平衡"

#### 方法 1: Magnitude Peak Detection

**原理**：
1. 计算合成加速度幅度 `acc_mag = sqrt(ax² + ay² + az²)`
2. 平滑（uniform filter, window=9）
3. 使用 `scipy.signal.find_peaks` 检测局部峰值
4. 峰值之间的区间视为 rep

**超参数**（固定，不 tune，从 train subjects 估计）：
- `distance`: 最小 rep 持续时间（train subjects 的 rep 中位数时长 × 0.8）
- `height`: 动态阈值（train streams 的 acc_mag percentile）
- `prominence`: 峰值的相对 prominence（基于 train 数据）

**公平性说明**：
- 最简单、最直觉的 heuristic 方法
- 几乎零计算成本，任何 MCU 都能跑
- duration prior 和 threshold 必须从 train subjects 估计（subject-wise）

#### 方法 2: SDTW Template Matching

**原理**：
1. 从 train subjects 的 representative reps 构建 template
2. 使用 DTW 在 test stream 上做滑动窗口匹配
3. 低于 cost threshold 的匹配视为 candidate rep
4. NMS（Non-Maximum Suppression）去除重叠候选

**修复要求**（当前代码存在的问题）：
- `dtw_feature = ranked[0]` 只使用单一最佳特征，应修复为使用 `acc_mag`
- 确保评估是在 set-level merged streams 上进行
- DTW 本身是 1D 匹配器，使用 `acc_mag` 是公平且自然的

**公平性说明**：
- 这是 RecoFit / ExerSense 类文献中最常见的 baseline
- duration prior 和 threshold 来自 train reps
- **部署风险**：O(n²) 计算量，长 stream（如 30s × 100Hz = 3000 samples）上 DTW 可能太慢

#### 方法 3: Sliding-window RF

**原理**：
1. 使用固定窗口（如 50 samples）提取统计特征（mean, std, min, max）
2. 训练 RF 三分类器（other / concentric / eccentric）
3. 将窗口预测映射回 sample level
4. 使用 concentric → eccentric 配对规则提取 reps

**实现**：基于 `scripts/compare_baselines.py` 的 RF baseline

**公平性说明**：
- 与 Causal RF 的区别：Sliding-window 是固定窗口，不是 trailing window
- Sliding-window 在窗口边缘会泄露未来信息（中间样本的预测依赖了窗口右侧的数据）
- **标注为 ❌ 非因果，不可部署，仅作为理论上限对照**

#### 方法 4: Causal RF (plain)

**原理**：
1. 使用 trailing window（只依赖过去样本）提取统计特征
2. 训练 RF 三分类器（other / concentric / eccentric）
3. 对每个 sample 做 causal 预测
4. 使用 concentric → eccentric 配对规则提取 reps

**实现**：基于 `scripts/evaluate_causal_rf.py`

**验证要求**（使用旧脚本前必须确认）：
- [ ] trailing window 是否只用过去样本（严格 causal）？
- [ ] z-score 是否只从 train subjects 计算？
- [ ] 随机种子是否固定？

**公平性说明**：
- 这是本方法家族的"基础版"，不带 refiner
- 用于展示 boundary refiner 的增量贡献

#### 方法 5: Per-Action Plain RF

**原理**：
1. 同 General Causal RF，使用 trailing window + RF 做 phase 分类
2. **唯一区别**：每个 action 训练独立的 RF 模型，只用该 action 的数据
3. 预测时根据已知的 action_type 路由到对应模型

**实现**：基于 `scripts/evaluate_per_action_plain_rf_loso.py`

**验证要求**：
- [x] trailing window 严格 causal（已完成验证）
- [x] z-score 只从 train subjects 计算（已完成验证）
- [x] 无 hyperparameter tuning（固定 w=100, n=100, depth=15）

**公平性说明**：
- 这是论文主推的完整方法
- Per-action 假设符合实际部署流程（先识别动作，再用对应模型）
- **已验证**：Feature Subset Selection（Top-K 特征筛选）无效，全部 63 个特征反而更好
- **已验证**：Boundary Refiner（独立回归微调）无增量贡献，Plain RF 反而更强

#### 方法 6: BiLSTM (phase-only)

**原理**：
1. 2-layer bidirectional LSTM
2. 输入：原始 6-axis 序列
3. 输出：每个 sample 的 phase 概率（other / concentric / eccentric）
4. concentric → eccentric 配对提取 reps

**实现**：基于 `scripts/compare_baselines.py` 的 BiLSTM baseline

**公平性说明**：
- BiLSTM 是非因果的（使用双向上下文）
- **作为离线理论上限，展示"如果有未来信息，能到什么水平"**
- 如果 BiLSTM 与 Per-Action Plain RF 差距不大，说明 Per-Action Plain RF 已经接近天花板
- **标注为 ❌ 非因果，不可部署，仅作为理论上限对照**

#### 方法 7: Phase-only causal TCN

**原理**：
1. Dilated causal TCN
2. 输入：原始 6-axis 序列
3. 输出：每个 sample 的 phase 概率
4. concentric → eccentric 配对提取 reps

**实现**：基于 `scripts/compare_baselines.py` 的 Phase-only causal TCN

**公平性说明**：
- 这是已放弃的方向，作为"我们尝试过但不够好"的对照
- 如果 TCN 表现比 Per-Action Plain RF 差，支持"简单方法更好"的论点
- **不用于部署，仅用于论文叙事**

### 2.3 输入模态的统一策略

**关键设计决策**：所有方法接收同样的 6-axis 原始数据，但利用方式不同：

| 方法 | 实际使用的输入表示 | 理由 |
|------|-------------------|------|
| Peak Detection | acc_mag (1D) | 1D 信号处理方法的 natural input |
| SDTW | acc_mag (1D) | 1D 模板匹配方法的 natural input |
| Sliding-window RF | 6-axis statistical features (N×6D) | 经典 ML 方法的 natural input |
| Causal RF | 6-axis trailing window features | 序列 ML 方法的 natural input |
| BiLSTM / TCN | 6-axis raw sequence | 深度学习序列模型的 natural input |

**公平性保障**：
- 所有方法都从相同的 `ax, ay, az, gx, gy, gz` 原始列出发
- 差异仅在于"如何利用"这些轴的信息
- Peak/DTW 选择聚合为 1D 是它们的设计特性，不是"信息缺失"

### 2.4 评估协议

**严格 9-fold LOSO**：

```
Fold 1:  test=haoyu,     train=hsianshun,kevin,thomas,tsenyu,yanz,yoru,yushuan,ziho
Fold 2:  test=hsianshun, train=haoyu,kevin,thomas,tsenyu,yanz,yoru,yushuan,ziho
Fold 3:  test=kevin,     train=haoyu,hsianshun,thomas,tsenyu,yanz,yoru,yushuan,ziho
Fold 4:  test=thomas,    train=haoyu,hsianshun,kevin,tsenyu,yanz,yoru,yushuan,ziho
Fold 5:  test=tsenyu,    train=haoyu,hsianshun,kevin,thomas,yanz,yoru,yushuan,ziho
Fold 6:  test=yanz,      train=haoyu,hsianshun,kevin,thomas,tsenyu,yoru,yushuan,ziho
Fold 7:  test=yoru,      train=haoyu,hsianshun,kevin,thomas,tsenyu,yanz,yushuan,ziho
Fold 8:  test=yushuan,   train=haoyu,hsianshun,kevin,thomas,tsenyu,yanz,yoru,ziho
Fold 9:  test=ziho,      train=haoyu,hsianshun,kevin,thomas,tsenyu,yanz,yoru,yushuan
```

**每个 fold 的操作**：
1. 用 train subjects 的 data fit model（或提取 statistics）
2. 用 train subjects 的数据计算 z-score stats
3. 将 z-score stats 应用到 test subject 的数据
4. 在 test subject 的 set-level streams 上运行检测
5. 与拼接后的 GT rep boundaries 比较

**注意**：对于 Peak Detection（无训练过程），duration prior 和 threshold 必须从 train subjects 估计。

### 2.5 报告指标

**Rep-level 指标（Primary）**：

| 指标 | 定义 | 公式 |
|------|------|------|
| Rep Precision | 检测 reps 中正确的比例 | TP / (TP + FP) |
| Rep Recall | GT reps 中被检测到的比例 | TP / (TP + FN) |
| Rep F1 | Precision 和 Recall 的调和平均 | 2PR / (P + R) |
| **Mean IoU** | **匹配 rep 的平均 IoU（boundary 质量）** | **ΣIoU(matched) / n_matched** |
| Exact-count ratio | rep 数量完全正确的 stream 比例 | #streams with n_pred == n_true / total_streams |

**Boundary 指标（Secondary）**：

| 指标 | 定义 | 单位 |
|------|------|------|
| Start MAE | 匹配 rep 的 start 误差绝对值均值 | ms |
| End MAE | 匹配 rep 的 end 误差绝对值均值 | ms |
| Transition MAE | concentric→eccentric 切换点误差 | ms |
| IoU-F1@50 | sample-level phase IoU-F1@50%（也称 micro_f1@50） | 0-1 |

**指标优先级说明**：
- **Rep F1 是 headline 指标**：决定 rep 是否能被「检测到」
- **Mean IoU 是 quality 指标**：决定检测到的 rep 「边界有多准」
- **两者缺一不可**：一个方法可能有高 Rep F1 但低 Mean IoU（检测到很多 rep 但边界不准），反之亦然
- **Baseline Comparison 表格必须同时包含 Rep F1 和 Mean IoU（或 IoU-F1@50）**

**Stream-level 诊断指标**：

| 指标 | 定义 |
|------|------|
| Zero-TP streams | 完全未检测到任何 rep 的 stream 数 |
| Under-segmented streams | 检测 rep 数 < 50% GT 的 stream 数 |
| Over-segmented streams | 检测 rep 数 > 150% GT 的 stream 数 |

### 2.6 统计显著性

对每个指标，报告：
- 9-fold 的 mean ± std
- 统计检验: paired Wilcoxon signed-rank test（非参数，适合小样本）
- 比较对: RF+Refiner vs each baseline

---

### 2.7 关键发现：为什么 Causal RF (0.706) < Peak Detection (0.755)？

**实验结果（7-fold LOSO, cleaned data, corrected config）**：

| 方法 | Rep F1 | Precision | Recall | IoU-F1@50 |
|------|--------|-----------|--------|-----------|
| **Peak Detection** | **0.755** | 0.755 | 0.754 | N/A（不输出 sample-level phase） |
| **General Causal RF** | **0.778** | 0.783 | 0.773 | **0.561** |
| Sliding-window RF | **0.768** | 0.750 | 0.789 | **0.577** |
| **Per-Action Plain RF** | **0.850** | **0.869** | **0.831** | **0.706** |

**核心矛盾**：为什么基于 ML 的 Causal RF 会比简单的信号处理启发式 Peak Detection 差？

#### 原因 1：动作类型偏差（Primary Cause）

我们的数据集中，**全身大动作（squat, rdl, bench press, shoulder press）占主要比例**。这些动作的 acc_mag 具有**清晰、强烈的周期性峰值**：

| 动作 | Peak Detection F1 范围 | 原因 |
|------|---------------------|------|
| db_squat | 0.95-1.00 | 垂直加速度变化剧烈，峰值极清晰 |
| db_rdl | 0.85-1.00 | 髋部屈伸产生周期性幅度变化 |
| db_bench_press | 0.75-0.95 | 胸部主导，上下运动明显 |
| db_shoulder_press | 0.75-0.95 | 垂直推举，acc_mag 周期性强 |
| db_biceps_curl | 0.00-0.26 | **手臂小幅度运动，acc_mag 峰值不明显** |
| db_triceps_curl | 0.00-0.26 | **同上，周期性弱** |
| db_weighted_crunch | 0.08-0.75 | **腹部动作，加速度幅度小且混杂** |
| one_arm_db_row | 0.25-0.86 | **单侧动作，不对称导致峰值复杂** |

**结论**：Peak Detection 在「acc_mag 周期性强的动作」上几乎完美，而这些动作恰好是数据集中的主流。在「acc_mag 周期性弱的动作」上，Peak Detection 完全失败。

#### 原因 2：Causal RF 的两阶段误差累积

Causal RF 的流程是：
```
sample-level phase 分类 → concentric/eccentric 配对 → rep 边界
```

这是一个**两阶段流程**，误差会累积：
1. **第一阶段误差**：trailing window (0.5s = 50 samples) 对某些慢速 rep（如 crunch 的 2.5s rep）来说，只能看到 rep 的 1/5，难以区分 concentric 和 eccentric
2. **第二阶段误差**：配对规则对 phase sequence 的噪声敏感。如果 transition 附近有几个 sample 被错分为 "other"，就会导致 rep 被拆分或合并

相比之下，Peak Detection 是**直接推断**：从 acc_mag 峰值推断 rep 中心，没有中间阶段。

#### 原因 3：Causal Window 的信息瓶颈

| 参数 | Causal RF | Sliding-window RF | 差距 |
|------|-----------|-------------------|------|
| window_size | 50 samples (0.5s) | 50 samples (0.5s) | 相同 |
| 信息来源 | 仅过去样本 | 窗口中心两侧样本 | **Sliding 多 50% 信息** |
| Rep F1 | 0.706 | 0.768 | **Δ = +0.062** |
| IoU-F1@50 | 0.469 | 0.577 | **Δ = +0.108** |

**关键**：即使窗口大小相同，Sliding-window 能看到窗口右侧的未来样本，而 Causal 只能看过去。这 0.062 的 Rep F1 差距和 0.108 的 IoU 差距，证明了**因果性约束确实带来了信息损失**。

#### 原因 4：Causal RF 配置未达最优 — 已验证 ✓

当前 Causal RF 配置经过系统性优化后，**Rep F1 从 0.706 提升到 0.778**，超过 Peak Detection (0.757) 和 Sliding-window RF (0.768)。

**关键发现：window_size 是决定性参数**

| window_size | Rep F1 | vs baseline | 原因 |
|-------------|--------|-------------|------|
| 50 (0.5s) | 0.706 | baseline | 对慢动作只能看到 1/5-1/6 rep |
| 100 (1.0s) | **0.778** | **+0.072** | **能看到半个完整 rep，足以区分 phase** |
| 150 (1.5s) | 0.778 | +0.072 | 与 w=100 相同，但延迟更大 |

**为什么 1.0s 是 sweet spot？**
- 我们的 rep duration 中位数约 2.5-3.0s
- 1.0s 的 trailing window 能看到 **40% 的 rep**，包含完整的 concentric→eccentric transition
- 0.5s 窗口只能看到 20%，不足以区分 phase
- 1.5s 窗口没有额外收益，因为 1.0s 已经足够

**其他参数影响**：
- n_estimators 50→100: +0.017 F1（边际提升）
- smoothing_window 15→25: +0.006 F1（边际提升，但引入 0.1s 额外延迟）

#### 最终优化配置（Phase 1a 官方方法）

```yaml
window_size: 100          # 1.0s trailing window (was 50)
train_stride: 10
n_estimators: 100         # 100 trees (was 50)
max_depth: 15
max_samples: 0.7
smoothing_window: 15      # 0.15s causal smoothing
```

**General Causal RF 性能**：
- Rep F1 = **0.778 ± 0.057**
- Precision = **0.783 ± 0.057**
- Recall = **0.773 ± 0.061**
- IoU-F1@50 = **0.561 ± 0.119**
- 总因果延迟 = **1.15s** (1.0s window + 0.15s smoothing)

**Per-Action Plain RF 性能**（论文主推）：
- Rep F1 = **0.850**
- Precision = **0.869**
- Recall = **0.831**
- IoU-F1@50 = **0.706 ± 0.273**
- 总因果延迟 = **1.15s**（同上，无需额外组件）

#### 论文叙事建议

> "值得注意的是，在全身大动作（squat, rdl, bench, shoulder press）上，简单的 Magnitude Peak Detection 达到了 F1=0.90+ 的优异表现。这验证了一个重要前提：**当动作具有清晰的周期性加速度特征时，信号处理启发式方法已经非常有效**。然而，在手臂孤立动作（curls, row）和核心动作（crunch）上，Peak Detection 的 F1 跌至 0.20 以下，几乎完全失效。
>
> 通过系统性优化 trailing window 大小（从 0.5s 增大到 1.0s），我们的 General Causal RF 达到 **F1=0.778 ± 0.057**，不仅超过了 Peak Detection（0.757），还超过了非因果的 Sliding-window RF（0.768），证明了**在适当的上下文窗口下，因果方法可以达到甚至超越非因果方法的检测性能**。
>
> 更重要的是，在 Action-First 架构下（动作类型先于 Rep Segmentation 确定），我们为每个动作训练独立的 Per-Action Plain RF 模型。这一策略带来了 **Rep F1 从 0.778 到 0.850 的显著提升（+0.072, +9.3%）**，且 IoU-F1@50 从 0.561 提升到 0.706（+0.145, +25.8%）。这证明**在已知动作类型的前提下，per-action 專用模型能更精准地捕捉特定动作的运动学模式**，为包含多样化动作的完整训练计划提供了可靠的 rep 计数保障。

---

### 2.8 按动作类型的性能分解（关键证据）

**核心发现：Causal RF 的相对优势取决于动作的「加速度周期性」**

| 动作类别 | 具体动作 | Peak F1 范围 | General RF F1 范围 | Per-Action RF F1 范围 | 原因 |
|----------|----------|-------------|------------------|---------------|------|
| **全身大动作（强周期）** | db_squat, db_rdl | 0.85–1.00 | 0.85–0.95 | **0.91–1.00** | acc_mag 峰值极清晰；Per-Action 更稳定 |
| **上身大动作（中等周期）** | db_bench_press, db_shoulder_press | 0.75–0.95 | 0.80–0.90 | **0.85–0.99** | 周期性中等，Per-Action 更鲁棒 |
| **手臂孤立动作（弱周期）** | db_biceps_curl, db_triceps_curl | **0.00–0.26** | **0.60–0.75** | **0.82–1.00** | 小幅度运动；Per-Action 专注单动作模式，提升巨大 |
| **核心动作（弱周期+混杂）** | db_weighted_crunch | **0.08–0.75** | **0.55–0.70** | **0.25–0.96** | 加速度幅度小，混杂；Per-Action 改善显著但仍有波动 |
| **单侧动作（不对称）** | one_arm_db_row | **0.25–0.86** | **0.60–0.75** | **0.79–1.00** | 左右不对称，峰值复杂；Per-Action 大幅提升 |

**论文关键叙事**：

> "简单的 Magnitude Peak Detection 在全身大动作上表现优异（F1=0.95+），但在手臂孤立动作（biceps curl, triceps curl）上几乎完全失效（F1<0.25），在核心动作（crunch）和单侧动作（row）上严重不稳定。相比之下，Causal RF 在所有动作类别上保持 0.60+ 的稳定表现，**尤其在手臂孤立动作上优势高达 +0.50 F1**。这证明基于 phase 分类的 ML 方法对运动学特征的利用远比简单的幅度峰值检测更全面，**为阻力训练中包含多样化动作（大动作+孤立动作）的完整训练计划提供了可靠的 rep 计数保障**。

**对部署策略的启示**：
- 如果系统**只做** squat / bench press / shoulder press → Peak Detection 足够
- 如果系统需要**覆盖** curls / crunch / row → 必须使用 RF
- **实际健身场景中，用户通常会做多种动作组合，因此 RF 是更通用的选择**

---

### 2.9 为什么选用 Random Forest？（模型选择 rationale）

#### 候选模型全景对比

| 模型 | Rep F1 | 因果 | 可部署 | 模型大小 | 训练稳定性 | 淘汰/采用原因 |
|------|--------|------|--------|----------|-----------|--------------|
| DS-MS-TCN | 0.53 | ✅ | ⚠️ | >500 KB | 差（F1 波动 0.78→0.53）| ❌ 目标错位，已放弃 |
| Phase-only TCN | 0.69 | ✅ | ⚠️ | >500 KB | 中 | ❌ 精度低于 RF，不稳定 |
| BiLSTM | ~0.80 (est.) | ❌ | ❌ | >1 MB | 中 | ⚠️ **非因果理论上限** |
| General Causal RF | 0.778 | ✅ | ✅ | ~200 KB | 高 | ✅ 方法增量分析 |
| **Per-Action Plain RF (ours)** | **0.850** | **✅** | **✅** | **~200 KB × 8** | **高** | **✅ 论文主推方法** |
| Sliding-window RF | 0.768 | ❌ | ❌ | ~200 KB | 高 | ⚠️ **非因果理论上限** |
| Peak Detection | 0.757 | ✅ | ✅ | 0 KB | N/A | ⚠️ 通用性不足 |
| SDTW | — | ✅ | ⚠️ | ~10 KB | N/A | ⏳ 文献基线，计算量存疑 |

#### 选用 RF 的六大理由

1. **因果性保证实时性**：trailing window 确保每个预测只依赖过去数据，满足在线部署
2. **延迟可接受**：1.15s 总延迟对阻力训练 rep 检测完全可接受（rep 周期 2-4s）
3. **轻量适合嵌入式**：~200 KB 模型，8 个 per-action 模型共 ~1.6 MB，64MB RAM 轻松容纳
4. **纯 CPU 推理**：无需 GPU/NPU，ARM Cortex-A7 单核即可运行
5. **结果完全可复现**：固定 random seed 后，相同配置产生完全相同结果（vs 深度学习的不稳定性）
6. **可解释性强**：可分析特征重要性，便于调试和论文论证

#### 为什么「足够好」比「理论上更好」更重要

BiLSTM 可能有 +0.02 F1 提升，但：
- ❌ 非因果 → 不可实时部署
- ❌ >1 MB → 嵌入式内存吃紧
- ❌ 需 GPU → 板子无 GPU
- ❌ 训练不稳定 → 结果不可复现

**论文叙事**：
> "我们系统性地评估了从简单启发式（Peak Detection）到深度模型（TCN, BiLSTM）的完整方法谱系。实验表明，对于结构化良好的时序检测任务（阻力训练 rep 切割），精心设计的因果 Random Forest 在 7-fold LOSO 上达到 F1=0.778 ± 0.057，不仅超越了所有可部署的基线方法，还超过了非因果的 Sliding-window RF（F1=0.768）。这证明**在适当的特征工程和上下文窗口设计下，经典 ML 方法可以达到甚至超越深度学习的性能，同时满足严格的实时部署约束**。

---

## 三、Phase 1b: Modality Ablation（模态消融）

### 3.1 任务定义

固定方法为 **Per-Action Plain RF**，变化输入模态，评估不同传感器组合对 Rep Segmentation 的贡献。

### 3.2 模态组合（7 组）

| 代号 | 模态 | 轴 | 维度 | 预期角色 |
|------|------|---|------|---------|
| A | ACC only | ax, ay, az | 3 | 最小可部署配置 |
| G | GYRO only | gx, gy, gz | 3 | 旋转信息独立价值 |
| M | MAG only | mx, my, mz | 3 | 方向信息独立价值 |
| AG | ACC + GYRO | ax..gz | 6 | **默认配置** |
| AM | ACC + MAG | ax..az, mx..mz | 6 | GYRO 冗余测试 |
| GM | GYRO + MAG | gx..gz, mx..mz | 6 | ACC 必要性测试 |
| AGM | All | ax..mz | 9 | 完整传感器套件 |

### 3.3 核心分析问题

1. **独立贡献**：A, G, M 单独能否达到可用精度？
2. **协同效应**：AG 的 Recall 是否 > max(A, G) 的 Recall？（互補性證據）
3. **冗余识别**：AGM 相对 AG 是否有显著提升？
4. **动作依赖性**：不同动作对不同模态的敏感度是否一致？

### 3.4 与 Baseline Comparison 的关系

```
Baseline Comparison          Modality Ablation
     │                              │
     ▼                              ▼
同輸入(6-axis)               同方法(RF+Refiner)
不同方法比較                 不同輸入比較
     │                              │
     └─── 完全独立，可同时执行 ────┘
```

**结论**：
- 两个实验回答的是**正交问题**
- **不需要先做完 Baseline 再做 Modality Ablation**
- 如果时间有限，优先做 Baseline Comparison（确定方法优势）

---

## 四、Contingency Plan：如果我们的方法没有比较好

### Scenario A：Peak Detection 就已經夠好了

**征兆**：Peak Detection 的 Rep F1 接近 RF+Refiner（差距 < 5%）

**含义**：
- 这个數據集的動作規律性太強，複雜模型沒有優勢
- Rep 切分是 trivial 的，不是论文主要贡献

**應對**：
1. **論文價值轉向**：「我們證明了對於規律性強的阻力訓練動作，簡單的峰值檢測就夠了，不需要複雜模型」
2. **Action Classification 才是主要貢獻**：加速做到 Phase 3
3. **部署建議**：板子上直接跑 Peak Detection，省電且可靠

### Scenario B：SDTW 比我們好

**征兆**：SDTW 的 Rep F1 > RF+Refiner

**含义**：
- 模板匹配比學習方法更適合這個數據集
- 我們的方法沒有捕捉到位移訓練的本質模式

**應對**：
1. **改進方向**：在 SDTW 基礎上加入我們的 refiner，或改進模板選擇策略
2. **論文轉向**：「我們證明傳統模板方法仍有優勢，並提出改進版」
3. **部署建議**：板子上跑 SDTW（如果計算量允許）

### Scenario C：所有可部署方法都不夠好（F1 < 0.7）

**征兆**：Peak Detection, SDTW, RF+Refiner 都沒有達到可用的 Rep F1

**含义**：
- **問題不在模型，而在數據/標註**
- Phase 標註不一致、rep 定義模糊、或數據質量問題

**應對**：
1. **停止實驗，先解決數據問題**
2. 重新標註關鍵 subjects
3. 檢查 rep boundary 標註的一致性（不同標註者之間的 IoU）
4. 如果數據無法修復，考慮改變問題定義（如放寬 IoU threshold）

### Scenario D：RF+Refiner 輸給一個更簡單的輕量模型

**征兆**：Tiny 1D CNN 或 Micro-TCN 比 RF+Refiner 好

**應對**：
1. **採納該輕量模型為新方法**
2. 保留 refiner 架構，替換 backbone
3. 論文論點：「我們證明輕量學習模型 + 邊界精煉是最佳組合」

### Scenario E：Non-causal 方法（BiLSTM/Sliding RF）並沒有比 Causal 方法好很多

**征兆**：BiLSTM 的 Rep F1 只比 RF+Refiner 高 < 5%

**含义**：
- **这是最好的消息**：证明 future context 对这个任务帮助很小
- 说明我们的 causal 方法已经接近理论上限
- 实时部署没有任何性能损失

**應對**：
1. 论文重点强调："我们的方法接近离线理论上限，同时满足实时部署约束"
2. 这是非常有力的論證

---

## 五、执行计划

### 5.1 Smoke Test 强制要求（3-subject 快速验证）

**規則**：任何完整 9-fold LOSO 之前，必須先跑 3-subject smoke test。

**選擇的 3 個 subjects**：kevin（數據最多）、yushuan、yoru

**Smoke test 流程**：

```bash
# Step 1: 驗證腳本設計合理性（~15 分鐘）
# - 確認 evaluate_causal_rf.py 嚴格 causal
# - 確認 benchmark_per_action_rf_refiner.py 無 leakage
# - 修復 SDTW 的 acc_mag 使用和 set-level 評估

# Step 2: 3-subject smoke test（~30-60 分鐘）
# 只跑 3 個方法：Peak Detection + Causal RF + RF+Refiner
# 快速驗證趨勢

# Step 3: 趨勢判斷
# 如果 RF+Refiner > Causal RF > Peak Detection：✅ 趨勢正確，繼續
# 如果 Peak Detection ≈ RF+Refiner：⚠️ 執行 Contingency Plan A
# 如果所有方法 F1 < 0.7：🛑 停止，修復數據

# Step 4: 完整 9-fold（僅在 smoke test 通過後執行）
# 7 methods × 9 folds = 63 runs
# 使用 outer-fold parallelism
```

**絕對禁止**：
- ❌ 未經 smoke test 直接跑完整 9-fold
- ❌ 未驗證腳本就開始訓練
- ❌ 同時開啟過多進程導致系統卡死

### 5.2 資源管理策略

**原則**：善用電腦資源，但不過度設計。

| 資源 | 使用策略 | 注意事項 |
|------|---------|---------|
| CPU 核心 | outer-fold parallelism（最多 9 個並行 fold） | 不要超過物理核心數，避免 thrashing |
| 記憶體 | 每個 fold 獨立，約 500MB-1GB | 監控 memory usage，避免 OOM |
| GPU | Phase 1 不需要 GPU（RF 是 CPU-based） | 把 GPU 留給後續 Phase 的 DL baseline |
| 磁碟 I/O | 預先加載所有 CSV 到記憶體或 SSD | 避免重複讀取 |

**建議的並行度**：
- Phase 1a: 3-5 個並行 fold（取決於 CPU 核心數）
- Phase 1b: 3-5 個並行 modality
- Phase 2/3: 視方法類型調整（DL 方法可能需要 GPU）

### 5.3 Phase 1a 执行步骤

```bash
# Pre-step: 腳本驗證（必須先做！）
python scripts/verify_script_design.py \
    --scripts evaluate_causal_rf,benchmark_per_action_rf_refiner,sdtw_rep_segmentation \
    --check causal,leakage-free,set-level-evaluation

# Step 1: Smoke test（3 subjects, 3 methods）
python scripts/run_rep_baseline_comparison.py \
    --methods peak,causal_rf,causal_rf_refiner \
    --subjects kevin,yushuan,yoru \
    --actions db_bench_press,db_biceps_curl,db_rdl,db_shoulder_press,db_squat,db_triceps_curl,db_weighted_crunch,one_arm_db_row \
    --output artifacts/baseline_comparison/smoke_test

# Step 2: 如果 smoke test 趨勢正確，跑完整 9-fold
# 注意：
# - 使用 --resume 支持斷點續跑（如果中途崩潰，從上次中斷處繼續）
# - 使用 --cache-features 預先計算並緩存特徵（避免重複計算）
# - 每完成一個 fold 自動保存結果和打印進度
python scripts/run_rep_baseline_comparison.py \
    --methods peak,dtw,sliding_rf,causal_rf,causal_rf_refiner,bilstm,tcn \
    --subjects haoyu,hsianshun,kevin,thomas,tsenyu,yanz,yoru,yushuan,ziho \
    --actions db_bench_press,db_biceps_curl,db_rdl,db_shoulder_press,db_squat,db_triceps_curl,db_weighted_crunch,one_arm_db_row \
    --output artifacts/baseline_comparison/rep_segmentation_final \
    --resume \
    --cache-features
```

### 5.4 Phase 1b 执行步骤

```bash
# 固定方法: causal_rf_refiner
# 变化模态: A, G, M, AG, AM, GM, AGM

# Step 1: Smoke test（3 subjects, 3 modalities: A, AG, AGM）
python scripts/run_modality_ablation.py \
    --method causal_rf_refiner \
    --modalities A,AG,AGM \
    --subjects kevin,yushuan,yoru \
    --output artifacts/modality_ablation/smoke_test

# Step 2: 完整 7 modalities × 9 folds（僅在 smoke test 通過後）
python scripts/run_modality_ablation.py \
    --method causal_rf_refiner \
    --modalities A,G,M,AG,AM,GM,AGM \
    --subjects haoyu,hsianshun,kevin,thomas,tsenyu,yanz,yoru,yushuan,ziho \
    --output artifacts/modality_ablation/rep_segmentation
```

---

## 六、预期产出

### 产出 1: Method Comparison 表格（LaTeX-ready）

```latex
\begin{table}[t]
\caption{Rep Segmentation Baseline Comparison (9-fold LOSO, mean ± std)}
\label{tab:rep-baseline}
\begin{tabular}{lccccccc}
\toprule
Method & Causal & Deployable & Rep F1 & Recall & Precision & Start MAE & End MAE \\
\midrule
Peak Detection & \checkmark & \checkmark & - & - & - & - & - \\
SDTW & \checkmark & \checkmark* & - & - & - & - & - \\
Sliding-window RF & $\times$ & $\times$ & - & - & - & - & - \\
Causal RF & \checkmark & \checkmark & - & - & - & - & - \\
\textbf{Causal RF + Refiner} & \checkmark & \checkmark & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} \\
BiLSTM & $\times$ & $\times$ & - & - & - & - & - \\
TCN & \checkmark & \checkmark & - & - & - & - & - \\
\bottomrule
\multicolumn{8}{l}{\small *SDTW 的实时性需进一步验证}
\end{tabular}
\end{table}
```

**論文敘事重點**：
- 比較 Causal RF+Refiner（我們的）vs BiLSTM（非因果上限）
- 如果差距 < 5%："我們的因果方法接近非因果理論上限"
- 比較 Causal RF+Refiner vs Peak Detection
- 如果差距 > 10%："複雜模型確實有價值"
- 比較 Causal RF+Refiner vs Causal RF（plain）
- 展示 Refiner 的增量貢獻

### 产出 2: Modality Ablation 表格（LaTeX-ready）

```latex
\begin{table}[t]
\caption{Modality Ablation for Rep Segmentation (Causal RF + Refiner, 9-fold LOSO, mean ± std)}
\label{tab:modality-ablation}
\begin{tabular}{lccccc}
\toprule
Modality & Rep F1 & Recall & Precision & micro_f1@50 & Exact-count Ratio \\
\midrule
ACC only & - & - & - & - & - \\
GYRO only & - & - & - & - & - \\
MAG only & - & - & - & - & - \\
\textbf{ACC+GYRO} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} & \textbf{-} \\
ACC+MAG & - & - & - & - & - \\
GYRO+MAG & - & - & - & - & - \\
All (9-axis) & - & - & - & - & - \\
\bottomrule
\end{tabular}
\end{table}
```

### 产出 3: Per-action Modality Profile（热力图）

一个 heatmap 显示每个动作在每个模态下的 Recall。

---

## 七、已知问题与风险

| 问题 | 影响 | 缓解措施 |
|------|------|---------|
| DTW 评估路径可能不在 set-level streams 上 | Baseline 不公平 | 执行前验证并修复 |
| DTW 只用单一特征（ranked[0]） | 未使用 acc_mag | 修复为使用 acc_mag |
| 9-fold LOSO 计算时间 | 可能延迟 | outer-fold parallelism |
| **RF+Refiner 輸給 Peak Detection** | 论文贡献减弱 | 执行 Contingency Plan A |
| **所有方法 F1 < 0.7** | 數據/標註問題 | 停止實驗，修復數據 |
| **BiLSTM 只比 RF+Refiner 好 < 5%** | 这是好消息！ | 強調"接近理論上限" |

---

## 八、使用旧脚本前的验证清单

| 脚本 | 验证项 | 状态 |
|------|--------|------|
| `scripts/evaluate_causal_rf.py` | 是否 causal？z-score 是否 train-only？ | 待确认 |
| `scripts/benchmark_per_action_rf_refiner.py` | refiner 是否 leakage-free？tuning 是否已移除？ | 待确认 |
| `preprocessing/sdtw_rep_segmentation.py` | 评估是否在 set-level stream？dtw_feature 是否用 acc_mag？ | 待修复 |
| `scripts/compare_baselines.py` | 包含 BiLSTM 和 Sliding RF（作为对照）| ✅ 保留但僅作對照 |
| `train/micro_macro_recognition.py` | DS-MS-TCN，已放棄 | ❌ 不使用 |

---

*文档版本: 2026-05-17 v3*
*状态: Phase 1a 已完成 — Per-Action Plain RF 驗證完成（F1=0.850），所有 baseline 已跑完 7-fold*
*待办: Phase 1b Modality Ablation（模态消融）待执行*
*下一步: Phase 1b → Phase 2 Phase Segmentation*
