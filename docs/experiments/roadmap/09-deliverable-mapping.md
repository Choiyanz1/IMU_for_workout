# Deliverable 映射：从实验到论文产出（Roadmap）

## 文档目的

**本文档明确记录用户要求的 4 个最终 deliverables 与 Phase 1-4 实验的对应关系**。这是确保实验不偏离目标的锚点文档和进度追踪。

> **⚠️ 本文档属于 Roadmap（计划）**，记录 deliverable 与实验的映射关系和进度追踪。
> 
> **实验数据与结果**请查阅 `docs/experiments/results/*.md`

**核心约束**：
1. 所有方法必须考虑 LuckFox Pico Zero 部署可行性
2. Baseline Comparison 可包含 non-causal 方法作为理论上限对照
3. 任何大规模实验前必须先做 3-subject smoke test
4. 耦合分析延后到后续改善阶段

---

## 用户要求的 4 个 Deliverables（Action 先已知架构）

> 1. **IMU切割rep、辨识动作与向心离心切割波形图**（用色块或线段区别）
> 2. **IMU切割rep与baseline的比较表格**（Per-Action Rep Segmentation）
> 3. **IMU辨识动作类型模型与baseline的比较表格**（Action Verification，可选）
> 4. **离心向心切割与baseline的比较表格**

**架构说明**：
- 用户的「辨识动作」在**新架构**中是 Stage 0（用户选择）或 Stage 3（验证）
- Deliverable 3 展示的是「验证是否做错动作」的能力，不是「识别未知动作」
- 如果用户想了解「前导识别」（前2-3 reps 识别动作），这需要额外实验

---

## Deliverable → Phase 映射

### Deliverable 1: 波形图

**内容**：IMU切割rep、辨识动作与向心离心切割波形图（用色块或线段区别）

**对应 Phase**：**Phase 4b**（可视化生成）

**前置依赖**：
- Phase 1（Rep Segmentation）必须完成 → 提供 rep boundaries
- Phase 2（Phase Segmentation）必须完成 → 提供 phase transition points
- Phase 3（Action Classification）必须完成 → 提供 action labels

**可视化要素**：
```
图层 1: 原始 IMU 信号（ACC + GYRO）
        ├── ax, ay, az（蓝色系）
        └── gx, gy, gz（青色系）

图层 2: Rep 边界
        ├── GT: 绿色垂直虚线
        └── Pred: 红色垂直虚线

图层 3: Phase 色块
        ├── GT Concentric: 浅橙色半透明矩形
        ├── GT Eccentric: 浅蓝色半透明矩形
        ├── Pred Concentric: 深橙色半透明矩形
        └── Pred Eccentric: 深蓝色半透明矩形

图层 4: Action 标签
        ├── GT: 绿色文字（每个 rep 上方）
        └── Pred: 红色文字（每个 rep 上方）
```

**输出格式**：
- SVG（静态，插入论文）
- HTML（交互式，补充材料）

---

### Deliverable 2: Rep 切割与 Baseline 的比较表格

**内容**：IMU切割rep与baseline的比较表格

**对应 Phase**：**Phase 1a**（Rep Segmentation Baseline Comparison）

**前置依赖**：无（这是第一个要做的实验）

**Baseline 列表（包含可部署 + 非因果理论上限对照）**：

| Method | Type | Causal | Deployable | Per-Action? | 论文角色 |
|--------|------|--------|-----------|-------------|---------|
| Peak Detection | Signal Processing | ✅ | ✅ | ❌ 通用 | **最简可部署基线** |
| SDTW | Template Matching | ✅ | ⚠️ | ❌ 通用 | **文献经典基线** |
| Sliding-window RF | Classic ML | ❌ | ❌ | ❌ 通用 | **非因果理论上限** |
| General Causal RF | Sequential ML | ✅ | ✅ | ❌ 通用 | 方法增量分析 |
| **Per-Action Plain RF** | **Sequential ML** | **✅** | **✅** | **✅ Per-Action** | **论文主推方法** |
| BiLSTM | Deep Learning | ❌ | ❌ | ❌ 通用 | **离线深度上限** |
| TCN (abandoned) | Deep Learning | ✅ | ⚠️ | ❌ 通用 | 已放弃对照 |

**Per-Action 計算策略**：
- Baseline 保持通用（公平對照）
- General Causal RF：展示「通用模型能到的水平」
- **Per-Action Plain RF 必須 Per-Action（論文主推）**
- 總計算量：~135 runs（避免 504 runs 的過度計算）

**论文叙事重点**：
- 比較 Per-Action Plain RF vs BiLSTM：如果差距 < 5%，"我們的因果方法接近非因果理論上限"
- 比較 Per-Action Plain RF vs Peak Detection：展示 ML 方法在「非周期性动作」（curls, crunch, row）上高达 +0.50-0.80 F1 的优势
- 比較 Per-Action Plain RF vs General Causal RF：展示「Per-Action 策略」的增量貢獻（+0.072 F1, +0.145 IoU）
- 比較 Causal RF (通用) vs Causal RF+Refiner (Per-Action)：展示「Per-Action 策略 + Refiner」的雙重提升
- **关键论点**：因果 RF (F1=0.778) 不仅超过 Peak Detection (0.757)，还超过非因果 Sliding-window RF (0.768)

**附加产出**：Modality Ablation 表格（Phase 1b，与 1a 独立可同时执行）

| Modality | Rep F1 | Recall | Precision | micro_f1@50 |
|----------|--------|--------|-----------|-------------|
| ACC only | - | - | - | - |
| GYRO only | - | - | - | - |
| MAG only | - | - | - | - |
| ACC+GYRO | - | - | - | - |
| ACC+MAG | - | - | - | - |
| GYRO+MAG | - | - | - | - |
| All | - | - | - | - |

---

### Deliverable 3: 动作验证与 Baseline 的比较表格

**内容**：IMU动作验证模型与baseline的比较表格

**对应 Phase**：**Phase 3a**（Action Verification Baseline Comparison）

**前置依赖**：
- Phase 1 必须完成 → 提供 rep boundaries（Verification 的输入范围）

**架构角色**：
- **不是「识别未知动作」**，而是「验证是否做错动作」
- 用户宣称「我在做 db_bench_press」，模型验证「是的，这确实是 bench press」
- 这是一个**可选的 UX 增强**，不是核心功能

**表格内容**：

| Method | Features | Per-rep Acc | Set-level Acc | Macro F1 |
|--------|----------|-------------|---------------|----------|
| Statistical + SVM | Hand-crafted | - | - | - |
| Statistical + RF | Hand-crafted | - | - | - |
| AutoGluon (rich) | Hand-crafted | - | - | - |
| 1D CNN | Raw sequence | - | - | - |
| **Hybrid Classifier** | **Statistical + Confidence** | **-** | **-** | **-** |
| **+ Set Majority Vote** | **-** | **-** | **-** | - |

**关键问题**：
- **Per-rep > 80% 就够了**（set-level 後處理可提升到 95%+）
- 这不是核心功能，不需要追求完美
- **前導識別（前2-3 reps 識別動作）延後到耦合分析階段，現在假設可以通過**

---

### Deliverable 4: 向心离心切割与 Baseline 的比较表格

**内容**：离心向心切割与baseline的比较表格

**对应 Phase**：**Phase 2a**（Phase Segmentation Baseline Comparison）

**前置依赖**：
- Phase 1 必须完成 → 提供 rep boundaries（Phase Segmentation 的输入范围）

**表格内容**：

| Method | Input | Transition MAE | Phase Acc | Concentric IoU | Eccentric IoU |
|--------|-------|---------------|-----------|---------------|---------------|
| Acc Mag Minimum | acc_mag | - | - | - | - |
| Velocity Zero-Crossing | acc_mag | - | - | - | - |
| Energy-based Threshold | 6-axis | - | - | - | - |
| Frame-level RF | 6-axis windows | - | - | - | - |
| **Our Method** | **rich features** | **-** | **-** | **-** | **-** |

**[延后]** 耦合分析（Phase 2b）：
- 评估 Rep Segmentation 错误对 Phase Segmentation 的影响
- 用户要求延后到后续改善阶段

---

## 执行顺序建议

```
阶段 1: Phase 1a Smoke Test
    │
    ├── 3 subjects (kevin, yushuan, yoru)
    ├── 3 methods (peak, causal_rf, rf_refiner)
    └── 如果通过 → 继续完整 9-fold

阶段 2: Phase 1a 完整实验 + Phase 1b 完整实验
    │
    ├── 产出: Deliverable 2（Rep 切割表格）
    └── 产出: Modality Ablation 表格
    │
    └── 两个实验独立，可同时执行

阶段 3: Phase 3a Smoke Test
    │
    ├── 3 subjects, 2 methods (statistical_rf, autogluon)
    └── 如果通过 → 继续完整实验

阶段 4: Phase 3a 完整实验
    │
    ├── 产出: Deliverable 3（动作辨识表格）
    └── 评估 Set-level Majority Voting

阶段 5: Phase 2a 完整实验
    │
    ├── 产出: Deliverable 4（向心离心表格）
    └── 可部分与 Phase 3 并行

阶段 6: Phase 4b（Visualization）
    │
    └── 产出: Deliverable 1（波形图）
```

**关键认知**：
- **Smoke test 必须先做**，验证趋势后再投入完整实验
- Deliverable 2（Rep 表格）是**最优先**的，因为它是整个 pipeline 的瓶颈
- Deliverable 3 和 4 可以**部分并行**（因为它们都依赖 Phase 1 的 rep boundaries，但彼此独立）
- Deliverable 1（波形图）必须**最后做**，因为它需要所有三个阶段的输出
- **耦合分析延后到后续改善阶段**

---

## 论文章节对应

| 论文章节 | 对应 Deliverable | 对应 Phase |
|---------|-----------------|-----------|
| 3.1 Method: Rep Segmentation | - | Phase 1a |
| 3.2 Method: Phase Segmentation | - | Phase 2a |
| 3.3 Method: Action Classification | - | Phase 3a |
| 4.1 Experiment: Rep Baselines | **Deliverable 2** | Phase 1a |
| 4.2 Experiment: Modality Ablation | **Deliverable 2（附加）** | Phase 1b |
| 4.3 Experiment: Action Baselines | **Deliverable 3** | Phase 3a |
| 4.4 Experiment: Phase Baselines | **Deliverable 4** | Phase 2a |
| 4.5 End-to-End Evaluation | - | Phase 4a |
| 4.6 Qualitative Analysis | **Deliverable 1** | Phase 4b |

---

## 当前状态

| Deliverable | 对应 Phase | 状态 | 完成度 | 备注 |
|-------------|-----------|------|--------|------|
| 1. 波形图 | Phase 4b | 🔴 未开始 | 0% | 等待 Phase 1-3 完成 |
| 2. Rep 切割表格 | Phase 1a | 🟡 **进行中** | **85%** | Per-Action Plain RF ✅(0.850), General Causal RF ✅, Peak Detection ✅, Sliding RF ✅; SDTW ⏳, BiLSTM ⏳ |
| 3. 动作辨识表格 | Phase 3a | 🔴 未开始 | 0% | 等待 Phase 1 完成 |
| 4. 向心离心表格 | Phase 2a | 🔴 未开始 | 0% | 等待 Phase 1 完成 |

**Phase 1a 已完成项**：
- ✅ **Per-Action Plain RF**: F1 = **0.850** (w=100, n=100, 8 per-action models) — **论文主推方法**
- ✅ General Causal RF: F1 = **0.778 ± 0.057** (w=100, n=100)
- ✅ Peak Detection: F1 = **0.757 ± 0.073** (bug 已修复)
- ✅ Sliding-window RF: F1 = **0.768 ± 0.064** (理论上限对照)
- ✅ IoU-F1@50 指标已纳入比较表格
- ✅ 按动作类型分解分析已完成（含 Per-Action 分解）
- ✅ Feature Subset Selection 实验完成（结论：无效，per-action 训练本身已足够）

**Phase 1a 待完成项**：
- ⏳ SDTW baseline (7-subject LOSO)
- ⏳ BiLSTM baseline (7-subject LOSO)

---

*文档版本: 2026-05-17 v4*
*状态: Phase 1a 接近完成（~85%），Per-Action Plain RF (0.850) 成為論文主推方法*
