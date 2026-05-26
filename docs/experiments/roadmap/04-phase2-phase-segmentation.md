# Phase 2: Phase Segmentation 详细设计（最终版）

## 文档目的

本文档是 Phase 2 的完整技术规格。Phase 2 的目标是在**已知 rep 范围**的前提下，确定最佳的 concentric/eccentric 分割方法。

**核心约束**：
1. 耦合分析（与 Rep Segmentation 的耦合）延后到后续改进阶段
2. 所有评估保持 subject-wise split

---

## 一、任务定义

### 1.1 问题描述

**输入**：单个 rep 的 IMU 序列（范围已锁定，长度约 200-400 samples @100Hz）
**输出**：一个 transition point（向心→离心的切换点，或离心→向心的切换点）

**关键认知**：
- 这不是在整个 stream 上做 dense labeling，而是在**已知区间内找一个时间点**
- 问题的本质是"在一个 rep 内，什么时候从 concentric 变成 eccentric？"
- 对于阻力训练，transition point 通常对应于"动作方向改变的时刻"（例如：卧推从向上推变为向下放）

### 1.2 与 Rep Segmentation 的根本区别

| 维度 | Rep Segmentation | Phase Segmentation |
|------|-----------------|-------------------|
| 问题类型 | Temporal Detection（找多个区间） | Temporal Segmentation（在区间内找一个点） |
| 输入粒度 | 整个 set stream | 单个 rep |
| 输出格式 | 多个 [start, end] | 一个 transition sample index |
| 错误传播 | 影响后续所有阶段 | 只影响当前 rep 内的 phase 分析 |
| 实时要求 | 必须 causal（在线） | 无实时要求（rep 完成后离线处理） |
| 部署约束 | 必须 LuckFox 实时 | 可在手机端或板子空闲时执行 |

---

## 二、当前已有实现

### 2.1 `train/phase_segmentation.py`

现有框架使用 AutoGluon 做 per-rep phase classification：
1. 从 rep CSV 中提取 sliding windows
2. 计算 rich features（stats, FFT, correlations）
3. 训练 AutoGluon 分类器（concentric / eccentric / rest）
4. 对 frame-level 预测做平滑
5. 找出 concentric→eccentric 的 transition

**问题**：当前实现是"在 rep 内做 frame-level classification"，但 GT phase 标签本身就包含在每个 CSV 的 `phase` 列中。这意味着如果 rep 的切割已经正确，phase 标签就是现成的。

### 2.2 关键问题：Phase Segmentation 的 baseline 是什么？

对于 resistance training，一个 rep 内的 concentric→eccentric transition 通常有明确的物理意义：
- **Concentric**：肌肉缩短，对抗重力或阻力向上/向前运动
- **Eccentric**：肌肉拉长，在重力或阻力作用下向下/向后运动

对于哑铃训练，transition 通常对应于：
- 速度方向改变的时刻
- 加速度从正变负（或反之）的时刻
- 合成加速度幅度的局部最小值或最大值

**最简单的 baseline**：
1. 计算合成加速度幅度 `acc_mag = sqrt(ax² + ay² + az²)`
2. 在 rep 范围内找到 `acc_mag` 的局部最小值
3. 这个最小值点作为 transition point

**理由**：在哑铃训练中，transition 通常对应于动作方向改变的"停顿点"，此时速度接近零，加速度幅度有一个特征性的极小值。

---

## 三、Phase 2a: Method Comparison（方法对比）

### 3.1 参与比较的方法

| # | 方法 | 类型 | 输入 | 原理 | 可部署？ |
|---|------|------|------|------|---------|
| 1 | **Acceleration Magnitude Minimum** | 信号处理 | acc_mag (1D) | 找 rep 内 acc_mag 的局部最小值 | ✅ |
| 2 | **Velocity Zero-Crossing** | 信号处理 | acc_mag (1D) | 积分得速度，找过零点 | ✅ |
| 3 | **Energy-based Threshold** | 信号处理 | 6-axis | 基于能量谷的分割 | ✅ |
| 4 | **Frame-level RF** | 经典 ML | 6-axis window features | 对每个 sample 做 concentric/eccentric 分类 | ✅ |
| 5 | **Our Method (AutoGluon)** | 经典 ML | rich features | 当前已有实现 | ✅ |

**注意**：Phase Segmentation 无实时约束，因此不需要像 Rep Segmentation 那样严格筛选因果方法。但为保持一致性，仍优先选择可部署方法。

### 3.2 评估指标

| 指标 | 定义 | 单位 |
|------|------|------|
| Transition MAE | 预测的 transition point 与 GT 的绝对误差均值 | ms |
| Transition Median Error | 误差的中位数 | ms |
| Phase Accuracy | 每个 sample 的 phase 预测正确率 | % |
| Phase IoU-F1@50 | 预测的 concentric/eccentric 区间与 GT 的 IoU | 0-1 |

### 3.3 评估协议

与 Rep Segmentation 相同：9-fold LOSO

**关键区别**：
- 对于 Phase Segmentation，输入是"已经切割好的单个 rep"
- 但在评估时，需要考虑两种不同的场景：

**场景 A：Ideal（完美 Rep Segmentation）**
- 使用 GT rep boundaries 作为输入
- 评估 Phase Segmentation 方法的"纯粹能力"

**场景 B：Realistic（有错误的 Rep Segmentation）**
- 使用 Phase 1 方法（如 RF+Refiner）预测的 rep boundaries 作为输入
- 评估"当 Rep 切得不完美时，Phase Segmentation 会受到多大影响"

**[延后]** 场景 B 属于耦合分析，用户要求延后到后续改进阶段。

**预期**：
- 场景 A 的结果展示 Phase Segmentation 方法的理论上限
- 场景 B 的结果展示端到端 pipeline 的实际效果（延后执行）

---

## 四、[延后] Phase 2b: 与 Rep Segmentation 的耦合分析

### 4.1 核心问题

"Rep Segmentation 的误差会如何影响 Phase Segmentation？"

### 4.2 分析维度

| 误差类型 | 对 Phase Segmentation 的影响 |
|---------|---------------------------|
| Rep start 偏早 | Phase Segmentation 的输入包含"前一个 rep 尾部"的 eccentric 数据 |
| Rep start 偏晚 | Phase Segmentation 的输入缺失"当前 rep 头部"的 concentric 数据 |
| Rep end 偏早 | Phase Segmentation 的输入缺失"当前 rep 尾部"的 eccentric 数据 |
| Rep end 偏晚 | Phase Segmentation 的输入包含"下一个 rep 头部"的 concentric 数据 |
| 漏检一个 rep | 该 rep 的 Phase Segmentation 完全缺失 |
| 多检一个 rep | 该"假 rep"的 Phase Segmentation 输入是噪声/休息 |

### 4.3 量化方法

对 Phase 1 中每个方法的每个 fold：
1. 记录所有 rep 的 predicted boundary vs GT boundary 的误差
2. 将边界有误差（>X ms）的 rep 分为一组，边界无误差（<X ms）的分为另一组
3. 分别计算两组的 Phase Segmentation Transition MAE
4. 比较：边界误差大的 rep，其 Phase Segmentation 质量是否显著下降？

### 4.4 状态

**延后执行**。用户明确表示：「耦合分析的部份可以延後做(那比較是後面改善的部分)」

---

## 五、与 Phase 1 的关系

```
Phase 1 (Rep Segmentation)
    │
    ├── 产出: Rep boundaries (predicted)
    │
    └── 作为 Phase 2 的输入（Realistic 场景）

Phase 2 (Phase Segmentation)
    │
    ├── 场景 A: 使用 GT Rep boundaries
    │   └── 评估 Phase Segmentation 的"纯粹能力"
    │
    └── [延后] 场景 B: 使用 Predicted Rep boundaries
        └── 评估端到端耦合效果
```

---

*文档版本: 2026-05-16（修订版）*
*状态: 设计阶段，等待 Phase 1 完成后执行*
