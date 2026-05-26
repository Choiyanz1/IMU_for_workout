# 完整实验路线图：从模型验证到论文交付（Roadmap）

## 文档目的

**本文档是整个项目的顶层计划与方向**，记录从当前状态到最终论文 deliverables 的完整路径。包括各 Phase 的顺序、依赖关系、预期产出和时间估算。

> **⚠️ 文档职责区分**：
> - **本文件（Roadmap）**：记录计划、方向、依赖关系、时间估算
> - **Result 文档**：`docs/experiments/results/*.md` — 存放所有实验数据、比较表格、分析证据（论文直接引用）
> - **实验日志**：`docs/experiments/2026-*.md` — 每次实验的详细过程记录
> - **结果总索引**：`docs/experiments/results-tracking.md` — Result 文档中央索引

**核心约束（来自用户确认）**：
1. 所有模型默认部署到 LuckFox Pico Zero（64MB RAM, ARM Cortex-A7），不可行才考虑手机端
2. **Baseline Comparison 可包含 non-causal / 高计算量方法作为理论上限对照**
3. **任何大规模实验前必须先做 3-subject smoke test**
4. 耦合分析延后到后续改进阶段
5. 严格 subject-wise split，零 data leakage
6. 实验流程善用电脑资源（并行化），但不要过度设计
7. 使用旧脚本前必须验证设计合理性

---

## 一、当前项目状态（截至 2026-05-17）

### 已完成的工作

| 工作项 | 状态 | 关键产出 |
|--------|------|---------|
| 数据收集与清洗 | ✅ 基本完成 | 9 subjects, 8 actions, 3,903 CSV files |
| 文献调研 | ✅ 完成 | 10 篇相关论文，覆盖 RecoFit/MM-Fit/ExerSense 等 |
| Rep Segmentation 探索 | ✅ 完成 | **Per-Action Plain RF** (0.850 F1) 已验证为当前最强 |
| Phase 1a: Baseline Comparison | ✅ 完成 | Table 1-3 已产出，Rep F1 = 0.850 |
| Phase 1b: Modality Ablation | ⏸️ **已跳过** | Feature Importance 分析已证明 6-axis 最优 |
| Phase 2: Phase Segmentation | ⏸️ **已取消** | Phase classification 已内嵌在 Phase 1 的 RF 中，不需独立阶段 |
| TCN/DS-MS-TCN 尝试 | ✅ 已放弃 | 确认不适合本项目目标（目标错位、训练不稳定、Rep F1 低） |
| Phase Segmentation 框架 | ✅ 已有 | `train/phase_segmentation.py` |
| Action Classification 框架 | ✅ 已有 | `train/action_classification.py`, `train/hybrid_action_classifier.py` |

### 关键结论（已达成）

1. **DS-MS-TCN 不适合**：目标错位（优化 frame-level phase 而非 rep-level boundary）
2. **Per-Action Plain RF 是当前最佳可部署方法**：Rep F1 = **0.850**（7-fold LOSO, 7 subjects, 8 actions）
3. **Per-Action > General Causal RF**：0.850 vs 0.778（+0.072），per-action 训练是决定性改进
4. **General Causal RF > Peak Detection**：0.778 vs 0.757（+0.021），且 RF 在手臂/核心动作上优势高达 +0.50 F1
5. **General Causal RF > Sliding-window RF**：0.778 vs 0.768（+0.010），证明因果方法在适当配置下可超越非因果方法
6. **Feature Subset Selection 无效**：基于 per-action importance 的 Top-K 特征筛选反而降低性能（0.879 < 0.895），per-action 训练本身已提供足够正则化
5. **Modality 策略**：6-axis (ACC+GYRO) 是安全默认，MAG 价值有限（CV ~0.13，低动态范围）
6. **部署约束**：所有方法必须考虑 LuckFox Pico Zero 可行性，但 non-causal 方法可作为理论上限对照
7. **Action 先已知**：实际部署时 Action 在 Rep Segmentation 之前确定 → 使用 Per-Action 模型
8. **Action Verification 是可选功能**：不是核心功能，而是 UX 增强
9. **Peak Detection 实现有 bug**：duration_prior 计算错误导致 F1=0.000（已修复，修复后 F1=0.757）
10. **IoU-F1@50 必须纳入比较**：Rep F1  headline 指标，IoU-F1@50 是 boundary quality 指标，两者缺一不可
11. **Phase Classification 质量已评估**：Sample Macro F1=0.51（不高但够用），Transition MAE=334ms，Rep pairing 机制对 phase noise 鲁棒
12. **Phase 1b/2 已跳过**：Modality Ablation 被 Feature Importance 取代；Phase Segmentation 已内嵌在 Rep Segmentation 中

---

## 二、实验路线图总览（Action 先已知架构）

```
Stage 0: Action Selection（动作选择）
    │
    ├── 方式A: 用户手动选择（手机App）
    ├── 方式B: 系统前导识别（前2-3 reps 快速分类）
    └── 输出: 已知的 action_type
    │
    ▼（Action 已知 → 载入 Per-Action 模型）
Phase 1: Rep Segmentation 验证 ✅ 完成
    │
    ├── 1a: Baseline Comparison（Per-Action 方法对比） ✅ 完成
    │       产出：Table 1-3，Rep F1=0.850，IoU-F1@50=0.706
    │
    └── 1b: Modality Ablation ⏸️ 已跳过
            原因：Feature Importance 分析已证明 6-axis (ACC+GYRO) 最优，MAG 价值有限
    │
    ▼（Phase 1 完成）
Phase 2: Phase Segmentation 验证 ⏸️ 已取消
    │
    ├── 原因：Phase classification 已内嵌在 Phase 1 的 RF 中（3-class: other/concentric/eccentric）
    ├── Rep pairing 机制自动完成 concentric→eccentric 配对
    └── Phase classification 质量已评估：Macro F1=0.51（够用），不需独立优化
    │
    ▼（Phase 2 跳过）
Phase 3: Action Verification 验证 🔴 当前优先级
    │
    ├── 3a: Baseline Comparison
    │       目标：确定验证「是否做错动作」的最佳方法
    │       产出：动作验证对比表格
    │       注意：
    │         - 这是 UX 增强，不是核心功能
    │         - 单 rep 准确率 > 80% 即可（有 set-level 后处理）
    │
    └── 3b: [延后] 耦合分析
            说明：用户要求延后
    │
    ▼（Phase 1-3 完成后）
Phase 4: 端到端整合与可视化（最后做）
    │
    ├── 4a: 端到端系统验证
    │       目标：验证 Action-已知 → Rep → Phase → Verify 的完整流程
    │
    └── 4b: 可视化生成
            目标：生成论文所需的波形图
            产出：IMU切割rep、辨识动作与向心离心切割波形图
```

---

## 三、各 Phase 详细说明

### Phase 1: Rep Segmentation 验证 🔴 最高优先级

**目标**：
1. 在**可部署方法**中找到最佳的 Rep Segmentation 方法
2. 确定最优传感器模态组合
3. **如果我们的方法不是最好的，执行 Contingency Plan**

**时间估算**：
- 1a Baseline Comparison: ~135 runs（見下方計算）≈ 4-6 hours (with parallelization)
- 1b Modality Ablation: 7 modalities × 9 folds = 63 runs ≈ 4-6 hours
- 合计: 约 1 天（可并行）

**可部署 Baseline 列表（包含 non-causal 理论上限对照）**：

| Method | Input Repr | Causal | Deployable | Per-Action? | 论文角色 |
|--------|-----------|--------|-----------|-------------|---------|
| Magnitude Peak Detection | acc_mag (1D) | ✅ | ✅ | ❌ 通用 | **最简可部署基线** |
| SDTW Template Matching | acc_mag (1D) | ✅ | ⚠️ | ❌ 通用 | **文献经典基线** |
| Sliding-window RF | 6-axis rich features | ❌ | ❌ | ❌ 通用 | **非因果理论上限** |
| General Causal RF | 6-axis trailing window | ✅ | ✅ | ❌ 通用 | 方法增量分析 |
| **Per-Action Plain RF** | **6-axis trailing window** | **✅** | **✅** | **✅ Per-Action** | **论文主推方法** |
| BiLSTM (phase-only) | 6-axis sequence | ❌ | ❌ | ❌ 通用 | **离线深度上限** |
| Phase-only causal TCN | 6-axis sequence | ✅ | ⚠️ | ❌ 通用 | 已放弃对照 |

**Per-Action 策略**：
- Baseline（Peak, SDTW, BiLSTM, Sliding RF, TCN）：保持通用（公平對照）
- General Causal RF：展示「不知道动作时的上限」
- **Per-Action Plain RF：必須做 Per-Action（論文主推）**
- 總計算量：~135 runs（而不是 7×8×9=504 runs）

\* SDTW 的实时性需验证

**论文叙事重点**：
- 比較 Per-Action Plain RF vs BiLSTM（非因果上限）
  - 如果差距 < 5%："我們的因果方法接近非因果理論上限"
- 比較 Per-Action Plain RF vs Peak Detection
  - 如果差距 > 10%："複雜模型確實有價值"
- 比較 Per-Action Plain RF vs General Causal RF
  - 展示「Per-Action 策略」的增量貢獻（+0.072 F1, +0.145 IoU）

**关键分析**：
- Union/Intersection: **延后到后续改善阶段**
- Per-action profile: 哪些动作必须用 AG，哪些可以用 A-only？
- 统计显著性: paired Wilcoxon signed-rank test (Per-Action RF vs each baseline)
- **Phase 1b 已跳过**: Feature Importance 分析已证明 6-axis 最优

---

### Phase 2: Phase Segmentation 验证 ⏸️ 已取消（内嵌于 Phase 1）

**状态**: ⏸️ 已取消，不需要独立阶段

**原因**：
1. Per-Action Plain RF 已经直接输出 3-class phase probabilities (other/concentric/eccentric)
2. Rep pairing 机制自动完成 concentric→eccentric 配对，不需要独立的 phase segmentation
3. Phase classification 质量已评估：Macro F1=0.51（不高但够用），Transition MAE=334ms
4. Rep detection 对 phase noise 鲁棒：即使 phase classification 有噪声，Rep F1 仍达 0.85

**结论**：Phase classification 作为 Rep Segmentation 的"副产品"已足够，不需要额外优化

**延后事项**（未来改进阶段）：
- 如果需要更精确的 phase metrics（如 velocity-based analysis），可在已知 rep boundary 后做局部优化
- 当前不需要阻塞主线

---

### Phase 3: Action Verification 验证 🔴 当前优先级

---

### Phase 3: Action Verification 验证 🔴 当前优先级

**状态**: 🔴 当前正在执行（Phase 1/2 已完成或跳过）

**目标**：
1. 确定验证「是否做错动作」的最佳方法
2. 评估 Set-level Majority Voting 效果
3. **[延后]** 评估 Rep Segmentation 错误对 Verification 的影响

**前提**：Phase 1 已完成，Rep boundaries 已确定

**时间估算**：
- 3a Baseline Comparison: 4-5 methods × 9 folds ≈ 2-3 hours
- 3b Set-level Majority Voting: ~30 分钟
- 3c Coupling Analysis: **延后执行**
- 合计: 约半天

**产出表格**：

#### Table 4: Method Comparison (Action Verification)

| Method | Features | Per-rep Acc | Set-level Acc | Macro F1 |
|--------|----------|-------------|---------------|----------|
| Statistical + SVM | Hand-crafted (150+ dims) | - | - | - |
| Statistical + RF | Hand-crafted (150+ dims) | - | - | - |
| AutoGluon (rich) | Hand-crafted | - | - | - |
| 1D CNN | Raw sequence | - | - | - |
| **Hybrid Classifier** | **Statistical + Confidence** | **-** | **-** | **-** |
| **+ Set Majority Vote** | **-** | **-** | **-** | - |

**关键认知**：
- Action Verification 的输入是"已经切割好的单个 rep"
- **这不是核心功能**，而是 UX 增强（提醒用户是否做错动作）
- Per-rep F1 > 0.80 就足夠了，set-level 後處理可提升到 95%+
- **[延后]** 前導識別（前2-3 reps 識別動作）延後到耦合分析階段

---

### Phase 4: 端到端整合与可视化 🟢 最后做

**目标**：
1. 验证三阶段串行 pipeline 的整体效果
2. 生成论文所需的波形图

**前提**：Phase 1-3 全部完成

**时间估算**：
- 4a 端到端验证: 选择代表性 subjects/actions，运行完整 pipeline ≈ 2-3 hours
- 4b 可视化生成: 人工选择代表性样本，生成 SVG/HTML ≈ 2-4 hours
- 合计: 约 1 天

**产出**：

#### Deliverable 1: 波形图（IMU切割rep、辨识动作与向心离心切割波形图）

每个波形图应包含：
- 原始 IMU 信号（ACC + GYRO，用不同颜色/线型）
- Rep 边界线（垂直虚线，start/end）
- Phase 色块（concentric = 橙色，eccentric = 蓝色）
- Action 标签（每个 rep 上方标注）
- GT vs Pred 对比（可选：左右子图）

**代表性样本选择标准**：
- 至少 3 个不同动作
- 包含"好"的例子（高 IoU）和"差"的例子（低 IoU）
- 包含不同 held-out subjects

---

## 四、Smoke Test 强制要求

### 4.1 规则

**任何完整 9-fold LOSO 之前，必须先跑 3-subject smoke test。**

### 4.2 Smoke Test 流程

```
Step 1: 腳本驗證（~15 分鐘）
    ├── 確認舊腳本設計合理性
    ├── 檢查 causal / leakage-free / set-level-evaluation
    └── 修復已知問題（SDTW 的 acc_mag, set-level 路徑）

Step 2: 3-subject 快速驗證（~30-60 分鐘）
    ├── subjects: kevin（數據最多）、yushuan、yoru
    ├── methods: Peak Detection + Causal RF + RF+Refiner（Rep）
    └── 快速驗證趨勢

Step 3: 趨勢判斷
    ├── RF+Refiner > Causal RF > Peak Detection：✅ 趨勢正確，繼續完整實驗
    ├── Peak Detection ≈ RF+Refiner：⚠️ 執行 Contingency Plan A
    ├── 所有方法 F1 < 0.7：🛑 停止，修復數據
    └── BiLSTM 只比 RF+Refiner 好 < 5%：🎉 這是好消息！強調"接近理論上限"

Step 4: 完整 9-fold（僅在 smoke test 通過後執行）
    └── 使用 outer-fold parallelism
```

### 4.3 絕對禁止

- ❌ 未經 smoke test 直接跑完整 9-fold
- ❌ 未驗證腳本就開始訓練
- ❌ 同時開啟過多進程導致系統卡死

---

## 五、資源管理策略

### 5.1 原則

善用電腦資源，但不過度設計。

### 5.2 並行度建議

| 實驗 | 建議並行度 | 原因 |
|------|-----------|------|
| Phase 1a (RF-based) | 3-5 個並行 fold | RF 是 CPU-based，受 I/O 限制 |
| Phase 1b (RF-based) | 3-5 個並行 modality | 同上 |
| Phase 2 (RF/AutoGluon) | 3-5 個並行 fold | 同上 |
| Phase 3 (RF/AutoGluon) | 3-5 個並行 fold | 同上 |
| Phase 3 (1D CNN) | 1-2 個並行 fold | 需要 GPU，不要搶資源 |

### 5.3 監控指標

- CPU 使用率：不要持續 100%（留一些給系統）
- 記憶體使用率：不要超過 80%
- 磁碟 I/O：預先加載 CSV 到記憶體或 SSD

---

## 六、各 Phase 的依赖关系

```
Phase 1 (Rep) ───────────────────────────────────┐
    │                                               │
    ├── 直接产出: Table 2 (Rep 表格)               │
    │                                               │
    ├── 提供输入给 Phase 2: Rep boundaries         │
    │                                               │
    ├── 提供输入给 Phase 3: Rep boundaries         │
    │                                               │
    └── 提供数据给 Phase 4: 选择代表性样本        │
                                                    │
Phase 2 (Phase) ◄── 依赖 Phase 1 的 Rep boundaries │
    │                                               │
    └── 直接产出: Table 4 (向心离心表格)           │
                                                    │
Phase 3 (Action) ◄── 依赖 Phase 1 的 Rep boundaries│
    │                                               │
    └── 直接产出: Table 3 (动作识别表格)           │
                                                    │
Phase 4 (End-to-end + Viz) ◄── 依赖 Phase 1-3     │
    │                                               │
    └── 产出: 波形图 (Deliverable 1)              │
                                                    │
All Phases ───────────────────────────────────────┘
    │
    ▼
论文提交
```

---

## 七、时间估算汇总

| Phase | 内容 | 预估时间 | 可并行？ |
|-------|------|---------|---------|
| Smoke Test (Phase 1) | 3 subjects, 验证趋势 | 1-2 hrs | ✅ |
| Phase 1a | Rep Baseline Comparison | 3-5 hrs | ✅ outer-fold |
| Phase 1b | Modality Ablation | 3-5 hrs | ✅ outer-fold |
| Smoke Test (Phase 2/3) | 3 subjects, 验证趋势 | 1-2 hrs | ✅ |
| Phase 2 | Phase Segmentation | 2-3 hrs | ✅ outer-fold |
| Phase 3 | Action Verification | 2-3 hrs | ✅ outer-fold |
| Phase 4 | End-to-end + Visualization | 4-8 hrs | ❌ 需人工 |
| **合计** | | **15-25 hrs** | |

**時間優化說明**：
- 使用數據預加載（一次讀入，不再碰磁碟）可節省 ~30% 時間
- 使用特徵緩存（避免重複計算 z-score）可節省 ~20% 時間
- 使用斷點續跑（崩潰後不需要重跑已完成 fold）可避免時間浪費

**注意**：
- Phase 1a 和 1b 可以**完全并行**
- Phase 2 和 Phase 3 也可以与 Phase 1 部分并行（如果 Phase 1 先跑几个 fold 就拿到 rep boundaries）
- **Smoke test 總時間約 2-4 小時，但能避免浪費更多時間在錯誤的方向上**

---

## 八、风险评估与 Contingency Plan

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 9-fold LOSO 计算时间过长 | 延迟所有后续工作 | 使用 outer-fold parallelism（已支持） |
| Peak Detection / SDTW 实现有问题 | Baseline 不公平 | **Smoke test 先验证** |
| **RF+Refiner 輸給 Peak Detection** | 论文贡献减弱 | 执行 Contingency Plan A：转向 Action Classification 为主要贡献 |
| **所有方法 F1 < 0.7** | 數據/標註問題 | **Smoke test 发现后停止，修复數據** |
| Phase 2/3 的 baseline 太强 | 我们的方法无优势 | 这是一个好结果，说明问题不在 Action/Phase，而在 Rep |
| 板子部署不可行 | 方法不能落地 | 只比较可部署方法；必要时考虑手机端 offload |
| **BiLSTM 只比 RF+Refiner 好 < 5%** | - | **这是好消息！** 证明 causal 方法接近理论上限 |
| **单 rep Action Verification 效果差** | UX 增強減弱 | Per-rep F1 > 0.80 就够了；利用 set-level 结构约束做后处理（majority vote）可提升到 95%+；这不是核心功能 |

### 关键决策点

**决策点 1：Phase 1a Smoke test 后**
- 如果 RF+Refiner 不是最佳 → 执行 Contingency Plan
- 如果 RF+Refiner 是最佳 → 继续完整 9-fold

**决策点 2：Phase 1a 完整结果后**
- 如果所有方法 F1 > 0.7 且 RF+Refiner 最佳 → 继续 Phase 2/3
- 如果 Peak Detection 与 RF+Refiner 差距 < 5% → 转向 Action Classification 为主要贡献
- 如果所有方法 F1 < 0.7 → 停止，修复数据

**决策点 3：Phase 3a Smoke test 后**
- 如果 per-rep Macro F1 > 0.80 → ✅ **足够**，set-level 后处理可提升到 95%+，继续
- 如果 per-rep Macro F1 0.70-0.80 → ⚠️ **可接受**，但需要确认 set-level majority vote 能提升到 90%+
- 如果 per-rep Macro F1 < 0.70 → 🛑 **不足**，需要改进特征或考虑 per-action 模型
- 如果 set-level majority vote < 0.90 → 即使 per-rep 达标，后处理效果不佳，需调整策略

**关键认知**：Action Classification 不需要追求完美 per-rep 准确率。因为可以利用 set-level 结构约束（一个 set 内所有 reps 属于同一动作）做后处理（majority vote / consecutive agreement），per-rep F1 > 0.80 就能在实际应用中得到很好的效果。

---

## 九、与论文章节的对应关系

| 论文章节 | 对应实验 | Deliverable |
|---------|---------|-------------|
| 3.1 Method: Rep Segmentation | Phase 1a | Table 2 |
| 3.2 Method: Phase Segmentation | Phase 2 | Table 4 |
| 3.3 Method: Action Classification | Phase 3 | Table 3 |
| 4.1 Experiment: Rep Baselines | Phase 1a | **Deliverable 2** |
| 4.2 Experiment: Modality Ablation | Phase 1b | Modality Table |
| 4.3 Experiment: Action Baselines | Phase 3 | **Deliverable 3** |
| 4.4 Experiment: Phase Baselines | Phase 2 | **Deliverable 4** |
| 4.5 End-to-End Evaluation | Phase 4a | - |
| 4.6 Qualitative Analysis | Phase 4b | **Deliverable 1** |

---

*文档版本: 2026-05-17 v3*
*状态: 已更新 — 主推方法从 "Causal RF + Refiner" 改为 "Per-Action Plain RF"*
*下一步: 跑完 SDTW + BiLSTM 7-fold，然后进入 Phase 1b / Phase 2 smoke test*
