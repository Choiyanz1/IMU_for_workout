# 重新规划进度：Phase Segmentation + Action Classification

**规划日期**: 2026-05-17
**当前状态**: Phase 1 (Rep Segmentation) 基线对比完成，Phase 3 (Action Classification) 尚未开始

---

## 一、当前已完成的工作

### Phase 1: Rep Segmentation（已完成）

| 任务 | 状态 | 关键结果 |
|------|------|----------|
| 基线对比（10 种方法）| ✅ | Per-Action RF 最佳：F1=0.850 |
| 窗口大小优化 | ✅ | w=100 (1.0s) 是关键突破 |
| 按动作类型分解 | ✅ | 8 种动作的性能差异分析 |
| Rep Count 指标 | ✅ | Exact Count=65.9%（部署短板） |
| 后处理评估（Duration Prior + Refiner）| ⏳ | yoru 单 subject 上 Guardrail 回退到 baseline |
| 特征重要性分析 | ✅ | ax_mean 最重要，velocity features 有潜力 |

### Phase 3: Action Classification（尚未开始）

| 任务 | 状态 | 说明 |
|------|------|------|
| 代码框架 | ✅ | evaluate_rep_action_rf.py, evaluate_rep_complete_action_classifier.py 已存在 |
| AutoGluon 配置 | ✅ | config.yaml 中有完整配置 |
| 历史 TCN 实验 | ✅ | artifacts/micro_macro_recognition/ 下多个实验 |
| 基线对比（8 种方法）| ❌ | 未执行 |
| Set-level 后处理 | ❌ | 未执行 |

---

## 二、新的优先级规划

### P0（立即执行）

#### 任务 1: Phase Segmentation 准确性提升

**目标**: 提高 Phase-level 的 Sample Macro F1（当前 0.509）

**方法**: 
1. **分析当前 Phase 误分类模式**
   - 从 per_action_plain_rf_7fold 结果中提取 confusion matrix
   - 识别哪些 phase 最容易被混淆（concentric ↔ eccentric？other ↔ active？）
   
2. **尝试改进策略**
   - 策略 A: 调整 min_phase_samples（当前=3，尝试 5/10）
   - 策略 B: 调整 smoothing_window（当前=15，尝试 10/20/25）
   - 策略 C: 使用更长的 trailing window（1.5s vs 1.0s）对 phase 分类的影响
   - 策略 D: 加入 velocity/jerk 特征（之前 velocity 测试显示 +0.84% F1 提升）

3. **评估指标**
   - Sample Macro F1（3-class: other/concentric/eccentric）
   - Transition MAE（当前 334ms，目标 <200ms）
   - Rep F1（确保不下降）

#### 任务 2: Action Classification Smoke Test

**目标**: 快速验证 Action Classification 的可行性

**Smoke Test 设计**:
- **3 subjects**: kevin, yushuan, yoru（覆盖不同体型/风格）
- **2 种方法**: Statistical + RF, AutoGluon (rich)
- **2 种评估**: Per-rep Macro F1, Set-level Majority Vote Accuracy
- **判断标准**: Per-rep F1 > 0.70（ roadmap 中 0.80 为理想目标，0.70 为最低可接受）

**数据准备**:
- 使用 GT rep boundaries（不是 predicted boundaries）
- 每个 rep 提取为一个独立样本
- 特征：rich features（stats + FFT + correlations + 前后段不对称性）

---

### P1（P0 完成后执行）

#### 任务 3: Action Classification 全量基线对比

**基于 roadmap 05-phase3-action-classification.md 的 8 种方法对比**:

| # | 方法 | 类型 | 特征 | 目标 |
|---|------|------|------|------|
| 1 | Statistical + SVM | 经典 ML | Hand-crafted (rich) | 轻量基线 |
| 2 | **Statistical + RF** | 经典 ML | Hand-crafted (rich) | **主推候选** |
| 3 | Statistical + LogReg | 经典 ML | Hand-crafted (rich) | 轻量基线 |
| 4 | **AutoGluon (rich)** | AutoML | Rich features | **自动搜索最佳** |
| 5 | 1D CNN (per-rep) | 深度学习 | Raw sequence | 验证 DL 是否有优势 |
| 6 | LSTM (per-rep) | 深度学习 | Raw sequence | 上限对比（可能不部署）|
| 7 | Transformer (per-rep) | 深度学习 | Raw sequence | 上限对比（可能不部署）|
| 8 | **Hybrid Classifier** | 混合 | Statistical + Confidence | **我们的方法** |

**评估协议**:
- 9-fold LOSO（与 Rep Segmentation 一致）
- 每个方法输出：Per-rep Macro F1 + Set-level Majority Vote Accuracy
- 混淆矩阵分析（哪两个动作最容易混淆？）

#### 任务 4: Set-level 后处理策略实现与评估

**方法**:
1. Set-level Majority Voting（最简单有效）
2. Confidence Thresholding + Fallback
3. Consecutive Agreement

**目标**: 展示「per-rep 80% → set-level 95%+」的提升效果

---

### P2（可选 / 延后）

#### 任务 5: Phase Segmentation —— Direct vs Separated 对比

**目标**: 比较两种 Phase Segmentation 策略的效果

**策略 A: Direct（当前方法）**
- 一个模型同时预测：rep boundaries + phase labels (concentric/eccentric/other)
- Per-Action RF 已经做到了这一点（3-class classification: other/concentric/eccentric）
- 然后通过后处理（pair_concentric_eccentric_reps）提取 reps

**策略 B: Separated（未来实验）**
- Step 1: 用 Rep Segmentation 模型切 rep boundaries（不关心 phase）
- Step 2: 在每个 rep 内部，用另一个模型切 concentric/eccentric transition
- 两个模型各司其职

**预期比较维度**:
| 维度 | Direct (当前) | Separated (未来) |
|------|---------------|------------------|
| Rep F1 | 0.850 | ？ |
| Transition MAE | 334ms | ？ |
| Phase Macro F1 | 0.509 | ？ |
| 模型复杂度 | 一个模型 | 两个模型 |
| 错误传播 | 无（统一优化）| Rep 切错 → Phase 一定错 |

**论文叙事**:
> "We compare two strategies for phase segmentation: (1) direct joint prediction of rep boundaries and phase labels, and (2) a cascaded approach where rep boundaries are detected first, followed by per-rep phase classification. The direct approach simplifies the pipeline and avoids error propagation from rep mis-segmentation to phase classification."

---

#### 任务 6: Phase Segmentation 与 Action Classification 的耦合分析

**延后原因**: 用户明确表示「耦合分析的部份可以延後做」

**内容**: 使用 Predicted Rep Boundaries（而非 GT）作为 Action Classification 的输入，评估端到端性能下降。

#### 任务 6: Phase Segmentation 后处理再评估

**延后原因**: 当前优先级让位给 Action Classification

**内容**: 在全量 7-fold 上评估 Duration Prior + Boundary Refiner 对 Rep Count 的提升。

---

## 三、执行顺序图

```
Week 1 (P0)
├── Day 1-2: Phase Segmentation 误分类模式分析
│   ├── 提取 confusion matrix
│   └── 识别主要误分类模式
├── Day 3-4: Phase Segmentation 改进实验
│   ├── 调整 min_phase_samples (3→5→10)
│   ├── 调整 smoothing_window (15→10→20→25)
│   └── 测试 velocity/jerk features
├── Day 5: Action Classification Smoke Test
│   ├── 数据准备（rep-level 特征提取）
│   ├── Statistical + RF (3 subjects)
│   └── AutoGluon (3 subjects)

Week 2 (P1)
├── Day 1-3: Action Classification 全量基线
│   ├── 跑 Statistical + RF (9-fold)
│   ├── 跑 AutoGluon (9-fold)
│   └── 跑 1D CNN (9-fold, 如果时间允许)
├── Day 4: Set-level 后处理
│   ├── Majority Voting
│   ├── Confidence Thresholding
│   └── 对比 per-rep vs set-level
└── Day 5: 结果分析与文档撰写

Week 3 (P2, 可选)
├── 耦合分析（延后）
├── Phase Segmentation 后处理全量评估（延后）
└── 论文写作
```

---

## 四、关键决策点

### 决策点 1: Phase Segmentation 改进是否有效？
- **如果** Sample Macro F1 能从 0.509 提升到 0.60+：✅ 采纳改进，更新部署模型
- **如果** 提升 <0.05：⚠️ 记录结果但不改部署模型，因为 Rep F1 已是主要指标

### 决策点 2: Action Classification Smoke Test 是否通过？
- **如果** Per-rep F1 > 0.80：✅ 直接进入全量 9-fold
- **如果** Per-rep F1 0.70-0.80：⚠️ 可接受，但需要依赖 set-level 后处理
- **如果** Per-rep F1 < 0.70：🛑 需要改进特征或模型

### 决策点 3: Action Classification 选哪个方法？
- **如果** RF 和 AutoGluon 差距 <0.05：选择 RF（更简单、可部署、可解释）
- **如果** AutoGluon 明显更好 (>0.05)：评估模型大小和推理速度
- **如果** 1D CNN > 0.85：考虑作为上限参考，但可能不部署

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Action Classification smoke test F1 < 0.70 | Phase 3 无法推进 | 提前准备：尝试更 rich 的特征、数据增强、或简化问题（coarse 4-class） |
| Phase Segmentation 改进导致 Rep F1 下降 | 得不偿失 | 每次改进都同时监控 Rep F1，确保不下降 |
| 实验时间超出预期 | 延误论文进度 | 严格控制 smoke test 范围（3 subjects, 2 methods），不追求完美基线 |
| AutoGluon 内存不足 | 无法完成实验 | 限制 memory_limit_gb=4，排除 NN_TORCH/FASTAI |

---

## 六、下一步行动

### 今天立即执行：
1. **Phase Segmentation 误分类分析**
   - 运行 `analyze_phase_confusion_matrix.py` 或类似脚本
   - 提取 8 个动作的 confusion matrix
   - 识别 top-3 最容易混淆的 phase 对

2. **Action Classification Smoke Test 准备**
   - 确认 `evaluate_rep_complete_action_classifier.py` 是否可用
   - 确认数据路径和特征提取逻辑
   - 运行 1 个 subject 的快速测试（验证脚本是否 work）

---

*文档版本: 2026-05-17 v1*
*关联文档: docs/experiments/roadmap/05-phase3-action-classification.md*
*关联结果: docs/experiments/results/03-rep-count-metrics.md, 04-baseline-comparison.md*
