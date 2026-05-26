# Phase 4: 端到端整合与可视化详细设计（最终版）

## 文档目的

本文档是 Phase 4 的完整技术规格。Phase 4 的目标是在 Phase 1-3 全部完成后，验证整个三阶段串行 pipeline 的端到端效果，并生成论文所需的波形图。

**核心约束**：
1. 可视化是最后一步，必须在 Phase 1-3 完成后执行
2. 耦合分析延后，可视化主要展示各阶段独立效果
3. 所有方法必须考虑部署可行性

---

## 一、Phase 4a: 端到端系统验证

### 1.1 完整 Pipeline 流程

```
Raw IMU Stream (set-level)
    │
    ▼
┌─────────────────────────────┐
│ Stage 1: Rep Segmentation     │
│  - 输入: 6-axis IMU stream    │
│  - 模型: Causal RF + Refiner  │
│  - 输出: Rep boundaries     │
│    [(s1, e1), (s2, e2), ...]│
│  - 部署: LuckFox 实时        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Stage 2: Phase Segmentation   │
│  - 输入: 每个 rep 的 IMU     │
│  - 模型: [待 Phase 2 确定]   │
│  - 输出: Phase transition    │
│    point per rep              │
│  - 部署: 手机端/板子空闲      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Stage 3: Action Classif.    │
│  - 输入: 每个 rep 的 IMU     │
│  - 模型: [待 Phase 3 确定]   │
│  - 输出: Action label       │
│    per rep                    │
│  - 部署: 手机端/板子空闲      │
└─────────────────────────────┘
    │
    ▼
最终输出: 每个 rep 的
  - start/end sample
  - concentric range
  - eccentric range
  - action label
```

### 1.2 端到端评估指标

在端到端场景下，评估的不是单个阶段的质量，而是**整个 pipeline 的最终产出质量**：

| 指标 | 定义 | 说明 |
|------|------|------|
| End-to-End Rep F1 | 使用 predicted boundaries（非 GT）计算的 Rep F1 | 与 Phase 1 的 Rep F1 相同 |
| End-to-End Action Accuracy | 在 predicted rep 范围内分类正确的比例 | 受 Rep Segmentation 误差影响 |
| End-to-End Phase Accuracy | 在 predicted rep 范围内 phase 正确的比例 | 受 Rep Segmentation 误差影响 |
| Rep-Action Consistency | 同一个 set 内所有 reps 的 action label 一致的比例 | 衡量动作识别的稳定性 |

**[延后] 误差传播分析**：

端到端评估的核心价值在于揭示**误差传播**：

```
Rep Segmentation Error
    │
    ├──→ Phase Segmentation Input 脏数据
    │      └── Phase Transition MAE 增加
    │
    └──→ Action Classification Input 脏数据
           └── Action Accuracy 下降
```

**状态**：用户要求耦合分析延后，因此 Phase 4a 主要做"理想场景"的端到端验证（使用 GT boundaries 作为 Stage 2/3 的输入），展示各阶段独立工作时的最佳效果。

---

## 二、Phase 4b: 可视化生成

### 2.1 目标 Deliverable

**IMU切割rep、辨识动作与向心离心切割波形图**

### 2.2 可视化内容

每个波形图应包含以下元素：

#### 2.2.1 原始信号层

- **ACC 波形**：ax (蓝色), ay (绿色), az (红色)，或合成 acc_mag (深蓝色)
- **GYRO 波形**：gx (青色), gy (品红), gz (黄色)，或合成 gyro_mag (深青色)
- 放在两个 subplot 中（上: ACC, 下: GYRO）

#### 2.2.2 Rep 边界层

- **GT Rep 边界**：绿色垂直虚线（start）和绿色垂直虚线（end）
- **Predicted Rep 边界**：红色垂直虚线（start）和红色垂直虚线（end）
- 当 GT 和 Pred 重叠时，显示为橙色（表示匹配成功）

#### 2.2.3 Phase 色块层

- **GT Concentric**：浅橙色半透明矩形
- **GT Eccentric**：浅蓝色半透明矩形
- **Predicted Concentric**：深橙色半透明矩形
- **Predicted Eccentric**：深蓝色半透明矩形
- 当 GT 和 Pred 的 phase 重叠时，颜色加深（表示匹配）

#### 2.2.4 Action 标签层

- 每个 rep 上方显示 GT action label（绿色文字）
- 每个 rep 上方显示 Predicted action label（红色文字）
- 当匹配时，显示为黑色粗体

#### 2.2.5 辅助信息

- X 轴：时间（秒）或样本索引
- Y 轴：信号幅度（已标准化）
- 标题：Subject / Action / Set / Stream ID
- 图例：清晰标注每种颜色/线型的含义

### 2.3 可视化格式

#### 格式 A: 静态 SVG

- 适合插入论文 PDF
- 高质量矢量图
- 可调整大小不失真
- 每个图展示一个 set stream

#### 格式 B: 交互式 HTML

- 适合补充材料 / 在线展示
- 鼠标悬停显示详细信息（sample index, 信号值, phase label）
- 可缩放、平移
- 基于 D3.js 或 matplotlib 的 mpld3

### 2.4 代表性样本选择标准

波形图不能随机选择，必须展示"有代表性"的案例。选择标准：

#### 必须包含的案例类型

| 类型 | 选择标准 | 目的 |
|------|---------|------|
| **成功案例** | Rep F1 > 0.9, Action Acc = 100% | 展示系统正常工作的样子 |
| **边界误差案例** | Rep F1 0.5-0.8, 有明显 start/end offset | 展示边界误差的典型模式 |
| **漏检案例** | 有 zero-TP 或 high FN | 展示系统最难处理的场景 |
| **多检案例** | 有 high FP, over-segmentation | 展示 false positive 的模式 |
| **Phase 错误案例** | Rep 切对了但 phase 切错了 | 展示 Phase Segmentation 的独立失败模式 |
| **Action 错误案例** | Rep 和 Phase 都对了但 Action 错了 | 展示 Action Classification 的独立失败模式 |

#### 跨动作覆盖

至少选择 3-4 个不同动作：
- 1 个"简单"动作（如 db_biceps_curl，规律性强）
- 1 个"中等"动作（如 db_bench_press）
- 1 个"困难"动作（如 db_rdl，phase collapse 风险高）

#### 跨 subject 覆盖

至少选择 2-3 个不同 held-out subjects：
- 1 个训练数据多的 subject
- 1 个训练数据少或风格独特的 subject

### 2.5 生成流程

```python
# 伪代码
for subject in selected_subjects:
    for action in selected_actions:
        for set_dir in subject/action/:
            # 1. 加载 GT data（拼接 rep CSVs）
            gt_stream = load_and_concat(set_dir)
            
            # 2. 运行 Stage 1: Rep Segmentation
            pred_reps = causal_rf_refiner.predict(gt_stream)
            
            # 3. 运行 Stage 2: Phase Segmentation
            for rep in pred_reps:
                pred_phase = phase_segmenter.predict(rep)
            
            # 4. 运行 Stage 3: Action Classification
            for rep in pred_reps:
                pred_action = action_classifier.predict(rep)
            
            # 5. 生成可视化
            fig = create_figure(
                stream=gt_stream,
                gt_reps=gt_reps,
                pred_reps=pred_reps,
                gt_phases=gt_phases,
                pred_phases=pred_phases,
                gt_actions=gt_actions,
                pred_actions=pred_actions,
            )
            
            # 6. 保存
            fig.save(f"viz/{subject}_{action}_{set}.svg")
            fig.save_html(f"viz/{subject}_{action}_{set}.html")
```

---

## 三、与 Phase 1-3 的关系

```
Phase 1 ──→ 提供 Rep Segmentation 模型和结果
    │
Phase 2 ──→ 提供 Phase Segmentation 模型和结果
    │
Phase 3 ──→ 提供 Action Classification 模型和结果
    │
    ▼
Phase 4a ──→ 整合所有阶段，验证端到端效果
    │
Phase 4b ──→ 基于端到端结果生成可视化
```

---

## 四、时间估算

| 任务 | 预估时间 | 依赖 |
|------|---------|------|
| 端到端 pipeline 脚本编写 | 2-3 小时 | Phase 1-3 完成 |
| 代表性样本选择 | 1 小时 | Phase 1-3 结果 |
| SVG 生成（6-8 个样本） | 2-3 小时 | 脚本编写完成 |
| HTML 生成（可选） | 2-3 小时 | SVG 完成后 |
| 人工检查和标注 | 1-2 小时 | 所有图生成后 |
| **合计** | **8-12 小时** | |

---

## 五、论文中的位置

| 论文章节 | Phase 4 内容 |
|---------|-------------|
| 4.5 End-to-End Evaluation | 端到端指标表格（理想场景） |
| 4.6 Qualitative Analysis | 波形图及分析 |
| Figure X | 成功案例波形图 |
| Figure Y | 失败案例分析波形图 |
| Supplementary | 完整 HTML 交互式可视化 |

---

*文档版本: 2026-05-16（修订版）*
*状态: 设计阶段，等待 Phase 1-3 完成后执行*
