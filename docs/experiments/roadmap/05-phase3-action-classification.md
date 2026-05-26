# Phase 3: Action Verification 详细设计（最终版 v3）— 当前执行中 🔴

## 文档目的

本文档是 Phase 3 的完整技术规格。在**新架构**中，Stage 3 的角色从「识别未知动作」转变为「验证是否做错动作」。

**核心约束**：
1. 耦合分析延后到后续改进阶段
2. 所有评估保持 subject-wise split
3. Action Verification 无实时约束，可在手机端或板子空闲时执行
4. **关键认知**：这是一个可选的 UX 增强功能，不是核心功能

**架构角色转变**：

| | 旧架构 | 新架构（用户确认） |
|---|--------|------------------|
| **Stage 0** | 无 | Action Selection（用户选择或前导识别） |
| **Stage 1** | Rep Segmentation（通用模型） | Rep Segmentation（Per-Action 模型） |
| **Stage 2** | Phase Segmentation | Phase Segmentation |
| **Stage 3** | Action Classification（识别未知动作） | **Action Verification（验证是否做错动作）** |

**用户场景**：
```
用户：「我要做 db_bench_press」（在手机上选择）
    ↓
系统载入 db_bench_press 专用 Rep Segmentation 模型
    ↓
用户开始做动作 → 系统检测 reps
    ↓
每个 rep 完成后 → 系统验证「这确实是 bench press 吗？」
    ↓
如果验证失败 → 提醒「您可能在做 squat，要切换动作吗？」
```

---

## 一、任务定义

### 1.1 问题描述

**输入**：单个 rep 的 IMU 序列（范围已锁定，长度约 200-400 samples @100Hz）
**输出**：一个离散类别标签（8 种阻力训练动作之一）

**关键认知**：
- 这不是时序检测问题，而是**静态分类问题**
- 输入是一个完整的"rep 样本"，输出是一个类别
- 问题的本质是"这个 rep 的动作模式属于哪一类？"
- **无实时约束**：rep 完成后离线处理，可在手机端执行

### 1.2 核心洞察：Set-level 结构约束

用户的观察：「後期還可以後處理（連續兩個 rep 才判斷之類的）」

**这是极其重要的 insight**：

在實際使用場景中：
```
使用者做一個 set（例如 db_bench_press）
    ├── rep 1: 模型預測 db_bench_press (confidence 0.92)
    ├── rep 2: 模型預測 db_shoulder_press (confidence 0.45)  ← 錯誤！
    ├── rep 3: 模型預測 db_bench_press (confidence 0.88)
    ├── rep 4: 模型預測 db_bench_press (confidence 0.91)
    └── ...
```

**結構約束**：一個 set 內的所有 reps **必然屬於同一動作**。

**後處理策略**：
1. **Set-level Majority Voting**：整個 set 的所有 reps 投票，取最多票的動作
2. **Consecutive Agreement**：只有連續 2-3 個 reps 預測一致時才接受
3. **Confidence Thresholding**：低 confidence 的預測不採信，fallback 到鄰居
4. **Temporal Smoothing (Viterbi/HMM)**：利用動作轉換機率（set 之間轉換，set 內不轉換）

**結論**：
- **Per-rep F1 >= 0.80 就足夠了**（不需要追求完美）
- 因為後處理可以把 set-level accuracy 提升到 95%+
- 論文的重點應該是展示「即使有 per-rep 誤差，set-level 後處理能穩定輸出」

---

## 二、当前已有实现

### 2.1 `train/action_classification.py`

使用 AutoGluon 框架，支持多种 feature mode：
- `stats`: per-channel summary statistics（30 features for 6-axis）
- `flatten`: raw window flattened
- `rich`: stats + FFT + correlations + magnitude features（150+ features）

### 2.2 `train/hybrid_action_classifier.py`

实现 Hybrid Action Classifier：
1. 训练 Logistic Regression + Random Forest 在 per-rep rich features
2. `predict_segment()`: 单个 rep 分类
3. `hybrid_label()`: 当 macro confidence < 0.7 时 fallback 到 classifier

**使用旧脚本前验证要求**：
- [ ] 确认特征提取不泄露 test subject 统计信息
- [ ] 确认 z-score 只从 train subjects 计算
- [ ] 确认 AutoGluon 的内层 tuning 不跨越 subject 边界

---

## 三、Phase 3a: Method Comparison（方法对比）

### 3.1 参与比较的方法

| # | 方法 | 类型 | 特征 | 原理 | 可部署？ |
|---|------|------|------|------|---------|
| 1 | **Statistical + SVM** | 经典 ML | Hand-crafted (150+ dims) | 统计特征 + SVM 分类器 | ✅ |
| 2 | **Statistical + RF** | 经典 ML | Hand-crafted (150+ dims) | 统计特征 + Random Forest | ✅ |
| 3 | **Statistical + Logistic Regression** | 经典 ML | Hand-crafted (150+ dims) | 统计特征 + LogReg | ✅ |
| 4 | **AutoGluon (rich)** | AutoML | Rich features | AutoGluon 自动搜索最佳模型 | ✅ |
| 5 | **1D CNN (per-rep)** | 深度学习 | Raw sequence | 1D CNN 直接学习时序模式 | ⚠️ 需验证模型大小 |
| 6 | **LSTM (per-rep)** | 深度学习 | Raw sequence | LSTM 序列模型 | ❌ 模型太大 |
| 7 | **Transformer (per-rep)** | 深度学习 | Raw sequence | Self-attention 时序模型 | ❌ 模型太大 |
| 8 | **Hybrid Classifier** | 混合 | Statistical + Confidence | 我们的方法：confidence-based routing | ✅ |

**部署说明**：
- Action Classification 无实时约束，对模型大小要求较宽松
- 但仍需考虑：如果要在板子上执行，模型应 < 1MB
- LSTM/Transformer 模型通常 > 1MB，可能不适合板子
- 优先选择轻量方法（SVM/RF/LogReg/Hybrid）

### 3.2 特征工程细节

对于 resistance training，per-rep 的关键区分特征：

**时域特征**：
- 每个轴的 mean, std, min, max, median, skewness, kurtosis
- 加速度合成幅度的统计量
- 速度（积分）的统计量
- 零穿越率
- **前半段 vs 后半段的不对称性**（concentric 和 eccentric 的差异是区分动作的关键）

**频域特征**：
- 每个轴的 FFT energy, entropy, dominant frequency
- Top-K 频率分量的 magnitude
- Power spectral density bands

**跨轴特征**：
- 轴间 Pearson correlation
- 主成分方向（PCA on 6-axis）
- 加速度-陀螺仪耦合特征

**时序特征**：
- 前半段 vs 后半段的统计差异（concentric vs eccentric 的不对称性）
- **这是最关键的特征**：不同动作的 concentric/eccentric 模式差异很大

### 3.3 评估指标

| 指标 | 定义 |
|------|------|
| Per-rep Accuracy | 正确分类的 rep 比例 |
| Per-rep Macro F1 | 每个 action 的 F1 的未加权平均 |
| Set-level Accuracy | 經過後處理（majority vote）後，set 級別的正確率 |
| Confusion Matrix | 哪兩個動作最容易被混淆 |

### 3.4 评估协议

9-fold LOSO（与 Rep Segmentation 一致）

**关键区别**：
- Action Classification 的输入是单个 rep CSV
- 不需要拼接 set streams
- 每个 rep 是一个独立的分类样本

**数据泄露风险**：
- 必须确保 train 和 test 没有 subject overlap（LOSO 已保证）
- 特征标准化（z-score）只能基于 train reps

---

## 四、后处理策略：Set-level 结构约束

### 4.1 为什么后处理有效？

**核心原因**：一個 set 內的所有 reps **必然屬於同一動作**。

這是一個非常強的**領域約束**（domain constraint），因為：
- 使用者不會在一個 set 內換動作
- 一個 set 就是「做 10 下啞鈴卧推」或「做 12 下啞鈴彎舉」
- 因此所有 reps 的 GT label 都相同

### 4.2 后处理方法

#### 方法 A：Set-level Majority Voting（最簡單、最有效）

**算法**：
```
對每個 set:
    1. 對 set 內每個 rep 做獨立分類，得到 softmax 概率
    2. 對所有 reps 的 softmax 概率取平均
    3. 選擇平均概率最高的動作作為 set-level label
    4. 該 set 內所有 reps 都使用這個 label
```

**公式**：
```
set_action = argmax_a (1/N) * Σ_i P(action=a | rep_i)
```

**優勢**：
- 不改變模型，只是後處理
- 利用了數據集的結構約束
- 可以糾正單個 rep 的誤分類
- 簡單、可解釋、可部署

**效果預估**：
- 如果 per-rep accuracy = 80%，一個 set 有 10 reps
- 設每個 rep 獨立錯誤概率 = 20%
- Majority vote 後錯誤概率大幅降低（假設錯誤分散在不同類別）
- Set-level accuracy 可達 95%+

#### 方法 B：Consecutive Agreement（連續兩個 rep 才判斷）

**算法**：
```
對每個 set:
    對 rep_i 和 rep_{i+1}:
        如果預測相同 → 接受這個預測
        如果預測不同 → 標記為"不確定"，使用後續 rep 確認
    最終使用最多 consecutive agreement 的動作
```

**優勢**：
- 更保守，減少單點誤差
- 適合實時顯示（每做一個 rep 更新預測）

**劣勢**：
- 需要至少 2 個 reps 才能確定動作
- 對於只有 1-2 個 reps 的 set 無法使用

#### 方法 C：Confidence Thresholding + Fallback

**算法**：
```
對每個 rep:
    如果最高 confidence > 0.9 → 直接接受
    如果 0.7 < confidence < 0.9 → 暫時保留，等待鄰居確認
    如果 confidence < 0.7 → 標記為"不確定"
    最後使用 set-level majority vote 決定
```

**優勢**：
- 高 confidence 時反應快
- 低 confidence 時不亂猜

### 4.3 後處理的效果預估

| Per-rep Accuracy | Set-level Majority Vote | 提升 |
|-----------------|------------------------|------|
| 70% | ~85% | +15% |
| 75% | ~90% | +15% |
| **80%** | **~95%** | **+15%** |
| 85% | ~97% | +12% |
| 90% | ~99% | +9% |

**結論**：
- **Per-rep 80% 就足夠了**，後處理可以提升到 95%
- 這對實際應用來說已經非常可用
- 論文中應該同時報告 per-rep 和 set-level 的 accuracy

---

## 五、建议的 Action Classification 实验流程

```
Step 1: Smoke test（3 subjects: kevin, yushuan, yoru）
    ├── 方法: Statistical + RF, AutoGluon (rich)
    ├── 評估:
    │   ├── Per-rep Macro F1
    │   └── Set-level Majority Vote Accuracy
    └── 判斷:
        ├── Per-rep F1 > 0.80: ✅ 足夠了，繼續完整實驗
        ├── Per-rep F1 0.70-0.80: ⚠️ 可接受，但後處理更重要
        └── Per-rep F1 < 0.70: 🛑 需要改進特徵或模型

Step 2: 如果 smoke test 通過，跑完整 9-fold
    ├── 評估 per-rep accuracy
    ├── 評估 set-level majority vote accuracy
    └── 評估 confusion matrix（哪兩個動作最容易混淆？）

Step 3: 分析混淆矩陣
    ├── 哪些動作最容易被混淆？
    ├── 是否某些動作確實無法從單 rep 區分？
    └── 是否需要 per-action 模型（而非通用模型）？
```

---

## 六、[延后] Phase 3b: 与 Rep Segmentation 的耦合分析

### 6.1 核心问题

"Rep Segmentation 的误差会如何影响 Action Classification？"

### 6.2 分析维度

| 误差类型 | 对 Action Classification 的影响 |
|---------|---------------------------|
| Rep start 偏早 | 输入包含前一 rep 的尾部数据（可能不同动作） |
| Rep start 偏晚 | 输入缺失当前 rep 的头部数据（可能丢失关键特征） |
| Rep end 偏早 | 输入缺失当前 rep 的尾部数据 |
| Rep end 偏晚 | 输入包含下一 rep 的头部数据（可能不同动作） |
| 漏检一个 rep | 该 rep 完全未被分类 |
| 多检一个 rep | "假 rep"的输入是噪声/休息，可能被误分类 |

### 6.3 状态

**延后执行**。用户明确表示：「耦合分析的部份可以延後做(那比較是後面改善的部分)」

---

## 七、与 Phase 1 的关系

```
Phase 1 (Rep Segmentation)
    │
    ├── 产出: Rep boundaries (predicted)
    │
    └── 作为 Phase 3 的输入（Realistic 场景）

Phase 3 (Action Classification)
    │
    ├── 场景 A: 使用 GT Rep boundaries
    │   └── 评估 Action Classification 的"纯粹能力"
    │
    └── [延后] 场景 B: 使用 Predicted Rep boundaries
        └── 评估端到端耦合效果
```

---

*文档版本: 2026-05-16 v2*
*状态: 设计阶段，等待 Phase 1 完成后执行*
