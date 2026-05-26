# 项目架构理解：IMU 阻力训练识别系统（最终版 v3）

## 文档目的

本文档记录对 IMU_for_workout 项目系统架构的完整理解，包括数据流、模型职责、部署约束，以及各模块之间的依赖关系。这是后续所有实验设计的**基础前提**。

**核心约束**：
1. 所有方法必须考虑 LuckFox Pico Zero（64MB RAM, ARM Cortex-A7）部署可行性
2. **Action 在 Rep Segmentation 之前已知（用户选择或前导识别）**
3. 严格 subject-wise split，零 data leakage

**关键设计（用户确认）**：
> 「實際進入 rep segment 之前先做動作辨識，才根據辨識結果」
>
> 这意味着：**Action 类型在 Rep Segmentation 之前已经确定**，Rep Segmentation 使用 per-action 专用模型

---

## 一、系统核心架构：三阶段串行管道（Action 在 Rep 之前已知）

本系统不是端到端模型，而是**三个串行阶段**组成。但和常见的「先切 rep 再识别动作」不同，本系统的**Action 类型在 Rep Segmentation 之前已经确定**。

### 1.1 实际部署流程

```
用户开始一个 set（例如：「我要做 db_bench_press」）
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 0: Action Selection / Detection        │
│                                             │
│  方式A: 用户手动选择（手机App点选）          │
│  方式B: 系统通过前2-3 reps 快速识别          │
│                                             │
│  输出: action_type = db_bench_press（已知）  │
└─────────────────────────────────────────────┘
    │
    ▼ Action 已知 → 载入对应模型
    │
┌─────────────────────────────────────────────┐
│ Stage 1: Rep Segmentation (Per-Action)      │
│                                             │
│  输入: 整个 set 的连续 IMU 序列              │
│        + 已知的 action_type                  │
│                                             │
│  模型: Per-Action Causal RF + Refiner        │
│        （每个动作有自己专用的模型）          │
│                                             │
│  输出: 多个 rep 的 [start, end] 区间        │
│                                             │
│  问题类型: Temporal Detection（已知动作）    │
│  部署约束: 必须 causal（实时在线）           │
│  硬件: LuckFox Pico Zero                     │
└─────────────────────────────────────────────┘
    │
    ▼ 产出: Rep boundaries
    │
┌─────────────────────────────────────────────┐
│ Stage 2: Phase Segmentation                 │
│                                             │
│  输入: 单个 rep 的 IMU 序列                  │
│        + 已知的 action_type                  │
│                                             │
│  输出: concentric/eccentric transition       │
│                                             │
│  问题类型: Temporal Segmentation             │
│  部署约束: 无实时要求                        │
└─────────────────────────────────────────────┘
    │
    ▼ 产出: Phase ranges
    │
┌─────────────────────────────────────────────┐
│ Stage 3: Action Verification（可选）         │
│                                             │
│  输入: 单个 rep 的 IMU 序列                  │
│  输出: 验证该 rep 是否确实符合宣称的動作     │
│                                             │
│  用途: 检测用户是否「说一套做一套」          │
│        （例如：说要做 bench press 却做 squat）│
│                                             │
│  问题类型: Static Classification             │
│  部署约束: 无实时要求                        │
└─────────────────────────────────────────────┘
```

### 1.2 和常见 pipeline 的区别

| 维度 | 常见 Pipeline | 本系统 Pipeline |
|------|--------------|--------------|
| **Action 何时确定** | Rep 切完后才识别 | Rep 切之前就已知 |
| **Rep Segmentation 模型** | 通用模型（所有动作共用） | Per-action 模型（每个动作专用） |
| **Action Classification 角色** | 主要任务（识别未知动作） | 验证任务（确认是否做错动作） |
| **优势** | 不需要预先知道动作 | 更高精度（专用模型更了解动作特性） |

### 1.3 为什么 Action 先已知更好？

**关键原因**：
1. **不同动作的 rep 模式差异很大**：bench press 和 squat 的运动学特征完全不同
2. **专用模型更精确**：db_bench_press 的专用 RF 只需要学习 bench press 的 rep 模式，不需要考虑 squat
3. **简化问题**：Rep Segmentation 从「8-class detection」变成「binary detection（这个动作者的 rep 边界在哪）」
4. **实际场景合理**：用户通常知道自己要做什么动作（或通过手机选择）

---

## 二、关键认知

### 2.1 问题类型根本不同

| 维度 | Stage 1 (Rep, Per-Action) | Stage 2 (Phase) | Stage 3 (Action Verification) |
|------|--------------------------|-----------------|------------------------------|
| **问题类型** | Temporal Detection（已知动作） | Temporal Segmentation | Static Verification |
| **输出格式** | 多个时间区间 | 一个时间点 / 两个子区间 | 二元判断（符合/不符合） |
| **输入粒度** | 整个 set stream | 单个 rep | 单个 rep |
| **错误传播** | 是后续所有阶段的瓶颈 | 只影响当前 rep 内的 phase 分析 | 只影响用户体验（提醒做错动作） |
| **适合的 baseline** | Peak Detection, DTW, Per-action RF | Energy thresholding | Statistical feature + SVM/RF |
| **部署约束** | 必须实时、causal、低功耗 | 可离线 | 可离线、可选 |

### 2.2 严格依赖链（Action 先已知）

```
Action 已知 → Stage 1 使用 Per-Action 模型 → 更高精度
    │
    ▼
Stage 1 切错 → Stage 2 一定错（范围已经错了）
Stage 1 切错 → Stage 3 可能错（输入数据包含错误范围）
Stage 2 切错 → Stage 3 不受影响（Action Verification 不需要 phase 信息）
Stage 3 失败 → 提醒用户「可能做错动作」（不影响 Stage 1/2）
```

这意味着：
1. **Action 先已知是系统设计的关键假设**
2. **Stage 1 (Rep Segmentation) 仍然是整个 pipeline 的瓶颈和第一优先级**
3. **Stage 3 (Action Verification) 是可选的 UX 增强，不是核心功能**

### 2.3 为什么 "Phase-only TCN" 等模型不适合 Stage 1

之前的 DS-MS-TCN 试图在一个模型中同时做：
1.  dense frame-level phase classification（Stage 2 的任务）
2.  action classification（在旧设计中 Stage 3 的任务，现在是 Stage 0）
3.  通过 phase pairing 间接导出 rep boundaries（Stage 1 的任务）

这导致了**目标错位**：模型优化的 loss 是 frame-level phase classification accuracy，但实际需要的是 rep-level boundary quality。两者的优化目标不一致。

**正确的做法**：
- Stage 0: 确定 action_type（用户选择或前导识别）
- Stage 1: 直接优化 rep boundary detection（使用 per-action 模型）
- Stage 2: 在已知 rep 范围内找 phase transition
- Stage 3: 验证 reps 是否符合宣称的 action（可选）

---

## 三、每个阶段的实际部署流程

### Stage 1: Rep Segmentation（在线实时，Per-Action）

```
用户选择动作（或前导识别）
    │
    ▼ action_type = db_bench_press（已知）
    │
传感器 → 实时 IMU 流
    │
    ▼
载入 db_bench_press 专用 Causal RF + Refiner 模型
    │
    ▼
Per-Action Phase Detector（只检测 bench press 的 rep 模式）
    │
    ▼
Phase Decoder（concentric → eccentric 配对）
    │
    ▼
Boundary Refiner（微调 start/end/transition）
    │
    ▼
输出: Rep boundaries（实时事件）
```

**关键设计**：
- 不是「通用 Rep 检测器」，而是「8 个专用 Rep 检测器」
- 根据已知的 action_type 路由到对应模型
- 每个模型只需要学习一种动作的 rep 模式

**部署约束**：
- 必须 causal：每个输出只依赖过去的数据
- 必须实时：延迟 < 100ms（@100Hz）
- 必须低功耗：适合 LuckFox Pico Zero（64MB RAM, ARM Cortex-A7）
- **内存需求**：8 个 per-action 模型 × ~200 KB = ~1.6 MB（可接受）

**板子部署路径**：

**路径 A：纯信号处理（Peak Detection）**
- 代码量：< 50 行 C
- 内存：~1 KB（一个通用模型）
- 延迟：O(window_size)
- **适合所有动作，但精度较低**

**路径 B：Per-Action 树模型（Causal RF / RF+Refiner）**
- 模型大小：每个动作 RF (~100 trees × depth 10) ≈ 50-200 KB
- 总内存：8 个动作 × ~200 KB = ~1.6 MB
- 延迟：O(n_trees × depth) ≈ 1000 次比较/样本
- 手写决策树遍历在 C 中可行
- **精度更高，因为专用模型更了解动作特性**

**路径 C：NPU 加速（如果路径 B 在 CPU 上太慢）**
- PyTorch → ONNX → RKNN (INT8) → NPU 推理
- 需要验证 RKNN 对决策树的支持
- 若不支持，考虑 tiny MLP

### 为什么 Stage 1 选用 Random Forest？

我们系统性地评估了多种候选模型，最终选择 **Causal Random Forest** 作为 Stage 1 的主推方法。以下是详细的模型选择 rationale：

#### 候选模型对比

| 模型 | Rep F1 | 因果 | 可部署 | 模型大小 | 训练稳定性 | 结论 |
|------|--------|------|--------|----------|-----------|------|
| DS-MS-TCN | 0.53 | ✅ | ⚠️ | >500 KB | 差（F1 波动 0.78→0.53）| ❌ 已放弃 |
| Phase-only TCN | 0.69 | ✅ | ⚠️ | >500 KB | 中 | ❌ 精度不足 |
| BiLSTM | ~0.80 (est.) | ❌ | ❌ | >1 MB | 中 | ⚠️ 仅作理论上限 |
| Peak Detection | 0.76 | ✅ | ✅ | 0 KB | N/A | ⚠️ 通用性差 |
| SDTW | — | ✅ | ⚠️ | ~10 KB | N/A | ⚠️ 计算量过大 |
| **Causal RF** | **0.78** | **✅** | **✅** | **~200 KB** | **高** | **✅ 主推** |

#### RF 的六大优势

1. **因果性保证实时部署**：trailing window 设计确保每个预测只依赖过去数据，满足在线实时要求
2. **低延迟可接受**：1.0s window + 0.15s smoothing = 1.15s 总延迟，对于阻力训练 rep 检测完全可接受
3. **轻量适合嵌入式**：~200 KB 模型，8 个 per-action 模型共 ~1.6 MB，64MB RAM 轻松容纳
4. **纯 CPU 推理**：无需 GPU/NPU，ARM Cortex-A7 单核即可实时运行
5. **结果完全可复现**：固定 random seed 后，相同配置产生完全相同的结果（vs 深度学习的不稳定性）
6. **可解释性强**：可分析每个特征对 rep boundary 决策的贡献，便于调试和论文分析

#### 为什么没有选深度学习？

**不是因为深度学习不好，而是因为「对于这个任务，RF 已经足够好」**。

- BiLSTM 可能有 +0.02-0.03 F1 的提升，但**不可部署**（非因果、>1MB、需 GPU）
- TCN 虽然因果，但 Rep F1 (0.69) **低于** RF (0.78)，且训练不稳定
- 在「精度-延迟-可部署性」三角中，RF 取得了**最佳平衡**
- 论文叙事："我们的方法证明，对于结构化良好的时序检测任务（阻力训练 rep 切割），精心设计的经典 ML 方法可以达到甚至超越深度学习的性能，同时满足严格的实时部署约束"

### Stage 2: Phase Segmentation（离线/后处理）

```
已切割出的 rep 序列
    │
    ▼
Phase Transition Detector（在 rep 范围内找向心/离心切换点）
    │
    ▼
输出: 每个 rep 的 concentric range + eccentric range
```

**部署约束**：
- 无实时要求：rep 完成后才做
- 可以在手机端或板子空闲时做（利用 Stage 1 的间隙）
- 对计算资源要求不高

### Stage 3: Action Verification（离线/后处理，可选）

```
已切割出的 rep 序列
    │
    ▼
Feature Extraction（统计特征、时频特征等）
    │
    ▼
Action Verifier（Logistic Regression / RF / Hybrid）
    │
    ▼
输出: 该 rep 是否确实符合宣称的 action_type？
    │
    ▼
如果不符合 → 提醒用户「您可能在做错动作」
```

**角色转变**：
- **不是「识别未知动作」**，而是「验证是否做错动作」
- 用户宣称「我在做 db_bench_press」，系统验证「是的，这确实是 bench press」
- 如果验证失败 → 提醒用户可能选错了动作

**部署约束**：
- 无实时要求：rep 完成后才做
- 每 rep 只做一次分类，计算量极小
- 可以在手机端做
- **这不是核心功能，而是用户体验增强**

---

## 四、部署可行性评估矩阵

| 方法 | Stage | 因果？ | 模型大小 | 内存需求 | CPU 需求 | **板子可行？** |
|------|-------|--------|---------|---------|---------|-------------|
| Peak Detection | 1 | ✅ | 0 KB | ~1 KB | 极低 | ✅ **首选** |
| SDTW | 1 | ✅ | ~10 KB (templates) | ~100 KB | O(n²) | ⚠️ **需验证** |
| Causal RF | 1 | ✅ | ~200 KB | ~5 MB | 中等 | ✅ **可行** |
| Causal RF + Refiner | 1 | ✅ | ~250 KB | ~5 MB | 中等 | ✅ **可行（主推）** |
| BiLSTM | 1/2/3 | ❌ | >1 MB | >10 MB | 高 | ❌ **不可行** |
| TCN | 1/2/3 | ✅ | >500 KB | >10 MB | 高 | ❌ **已放弃** |
| Sliding-window RF | 1/2/3 | ❌ | ~200 KB | ~5 MB | 中等 | ❌ **不可行（非因果）** |
| Statistical + SVM/RF | 3 | N/A | ~100 KB | ~2 MB | 低 | ✅ **可行** |

---

## 五、数据流与模块对应关系

### 5.1 核心模块（Action 先已知架构）

| 模块 | 负责阶段 | 部署位置 | 当前状态 | 使用建议 |
|------|---------|---------|---------|---------|
| `scripts/evaluate_causal_rf.py` | Stage 1 (Rep, Per-Action) | 板子/手机 | ✅ 核心模块 | **需验证 causal 和 leakage-free** |
| `scripts/benchmark_per_action_rf_refiner.py` | Stage 1 (Rep, Per-Action) | 板子/手机 | ✅ 核心模块 | **需验证 per-action 无 leakage** |
| `scripts/train_rf_boundary_refiner.py` | Stage 1 (Refiner) | 板子/手机 | ✅ 核心模块 | **需验证 refiner 不 leakage** |
| `preprocessing/sdtw_rep_segmentation.py` | Stage 1 (Rep, Per-Action) | 板子（需验证） | ⚠️ 需修复 | **修复为 per-action SDTW** |
| `train/phase_segmentation.py` | Stage 2 (Phase) | 手机/离线 | ✅ 已有 | 待验证 |
| `train/action_classification.py` | Stage 3 (Verification) | 手机/离线 | ✅ 已有 | 待验证（角色变为 verification） |
| `train/hybrid_action_classifier.py` | Stage 3 (Verification) | 手机/离线 | ✅ 已有 | 待验证 |

### 5.2 不使用模块

| 模块 | 原因 |
|------|------|
| `scripts/compare_baselines.py` | 包含不可部署的 BiLSTM/Sliding RF（仅作为对照保留） |
| `train/micro_macro_recognition.py` | DS-MS-TCN 已放弃（目标错位） |

### 5.3 新架构的关键变化

| 旧理解 | 新理解 |
|--------|--------|
| Action Classification 是 Stage 3（主要任务） | Action Verification 是 Stage 3（可选 UX 增强） |
| Rep Segmentation 是通用模型 | Rep Segmentation 是 Per-Action 模型（8 个专用模型） |
| Action 在 Rep 之后识别 | Action 在 Rep 之前已知（用户选择或前导识别） |
| Action 错误影响 Rep 精度 | Action 先已知 → Rep 使用专用模型 → 更高精度 |

---

## 六、关键术语统一

为避免混淆，以下术语在本项目中统一使用：

| 术语 | 含义 | 英文对应 |
|------|------|---------|
| Rep Segmentation / Rep Cutting | 将一个 set 的连续 IMU 流切割成多个 rep 区间（使用 per-action 模型） | Repetition Detection |
| Phase Segmentation | 在单个 rep 内切割出 concentric 和 eccentric 两个 phase | Phase Transition Detection |
| Action Selection (Stage 0) | 用户选择或前导识别确定 action_type | Exercise Selection |
| Action Verification (Stage 3) | 验证 reps 是否符合宣称的 action_type（可选） | Exercise Verification |
| Boundary Refiner | 对粗略检测的 rep 边界进行微调 | Boundary Refinement |
| Causal | 输出只依赖过去的数据，不能看未来 | Real-time / Online |
| Deployable | 可以在 LuckFox Pico Zero 上运行 | Edge-deployable |
| Per-Action Model | 每个动作有自己专用的模型（8 个模型） | Action-Specific Model |

---

## 七、数据格式说明

### 原始数据 CSV 列

```
Required columns:
- sensor_ts: 传感器时间戳
- ax, ay, az: 加速度计 (g)
- gx, gy, gz: 陀螺仪 (dps)
- action_type: 动作类型（如 db_bench_press）
- subject_id: 受试者 ID

Optional but present in current data:
- mx, my, mz: 磁力计
- phase: 相位标签（concentric/eccentric/inter_set_rest/none）
- rep, set: rep/set 编号
```

### 数据目录结构

```
datasets/raw_data/
├── {subject_id}/
│   ├── {session_id}/
│   │   ├── {action_type}/
│   │   │   ├── set0/
│   │   │   │   ├── rep0_*.csv
│   │   │   │   ├── rep1_*.csv
│   │   │   │   ├── ...
│   │   │   │   └── rest_after_set0/
│   │   │   │       └── rest*.csv
│   │   │   ├── set1/
│   │   │   ├── ...
│   │   └── big_rest/
```

**关键认知**：原始数据已经是**按 rep 切分好的**（每个 rep 一个 CSV）。这意味着：
- 对于 Stage 1 (Rep Segmentation) 的训练：需要把同一 set 的 rep CSV 拼接成 stream，模拟真实部署场景
- 对于 Stage 1 的评估：在拼接后的 set stream 上检测 rep，然后与拼接边界（即 rep0_end → rep1_start 的边界）比较
- 对于 Stage 2/3 的训练：直接使用单个 rep CSV 即可

---

*文档版本: 2026-05-16（修订版）*
*状态: 已确认（与用户共识达成）*
