# 数据集描述与来源

## 文档目的

本文档记录项目使用的数据集的完整信息，包括数据来源、收集方式、数据格式、清洗过程，以及最终用于实验的数据分布。论文中的 Dataset 章节应直接引用此文件。

---

## 一、数据来源

### 1.1 数据收集方式

**传感器**：OpenZen 可穿戴 IMU（采样率约 100 Hz）
**传感器安装位置**：右手腕（单一位置）
**轴定义**：
- ax, ay, az：加速度（X, Y, Z 三轴）
- gx, gy, gz：陀螺仪（X, Y, Z 三轴）
- mx, my, mz：磁力计（可选，部分数据缺失）

**数据格式**：每行一个时间戳 + 6-axis（或 9-axis）读数

### 1.2 动作类型

共 8 种阻力训练动作：

| # | 动作 | 类型 | 说明 |
|---|------|------|------|
| 1 | db_bench_press | 上身大动作 | 哑铃卧推，胸部主导 |
| 2 | db_biceps_curl | 手臂孤立动作 | 哑铃二头弯举，肘关节屈伸 |
| 3 | db_rdl | 全身大动作 | 哑铃罗马尼亚硬拉，髋部主导 |
| 4 | db_shoulder_press | 上身大动作 | 哑铃肩推，垂直推举 |
| 5 | db_squat | 全身大动作 | 哑铃深蹲，下肢主导 |
| 6 | db_triceps_curl | 手臂孤立动作 | 哑铃三头弯举，肘关节伸展 |
| 7 | db_weighted_crunch | 核心动作 | 负重卷腹，腹部主导 |
| 8 | one_arm_db_row | 单侧动作 | 单臂哑铃划船，背部主导 |

### 1.3 标签体系

**Phase 标签**（micro-level）：
- `concentric`：向心收缩（肌肉缩短，如弯举「举起」阶段）
- `eccentric`：离心收缩（肌肉拉长，如弯举「放下」阶段）
- `inter_set_rest`：组间休息
- `none` / `other`：其他未标注时段

**Rep 定义**：一个完整的 `concentric → eccentric` 配对构成一个 rep。

---

## 二、数据清洗

### 2.1 原始数据规模

- **总 subjects**：9 人
- **总 CSV 文件**：3,903 个
- **总动作类型**：8 种
- **采样率**：~100 Hz（经 audit 确认）

### 2.2 清洗步骤

**步骤 1：排除异常 subjects**
- **tsenyu**：数据质量差，动作不规范，标签错误率高
- **ziho**：持续性数据质量问题，经多次清洗仍不达标
- **最终保留 subjects**：7 人（haoyu, hsianshun, kevin, thomas, yoru, yushuan, yanz）

**步骤 2：删除非标准 reps**
- 识别标准：concentric/eccentric 比例异常（如 concentric 主导 > 70%）
- 原因：非标准 reps 通常表示「借力」或「动量辅助」，不属于标准阻力训练
- **清洗结果**：删除约 5-10% 的异常 rep 文件

**步骤 3：数据格式统一**
- 确保所有 CSV 包含 `phase` 列
- 确保 `sensor_ts` 或等效时间戳存在
- 统一列名：`ax, ay, az, gx, gy, gz`（6-axis）

### 2.3 最终数据集

| 指标 | 数值 |
|------|------|
| **Subjects** | 7 |
| **Actions** | 8 |
| **Streams** | 226（set-level merged streams）|
| **Total reps** | ~2,693 |
| **Median rep duration** | 2.5–3.0 seconds |
| **Duration range** | 1.5–4.5 seconds |
| **Concentric/Eccentric ratio** | 45:55 至 58:42（健康范围）|

### 2.4 数据分布

| Subject | Streams | Reps | 备注 |
|---------|---------|------|------|
| haoyu | 24 | 287 | 标准动作，表现最佳 |
| hsianshun | 25 | 288 | 中等难度 |
| kevin | 34 | 420 | 数据量大，部分 session 复杂 |
| thomas | 47 | 598 | 数据量最大，多 session |
| yoru | 25 | 274 | 中等难度 |
| yushuan | 24 | 280 | 历史 holdout 测试对象 |
| yanz | 47 | 546 | 最难泛化，跨 subject 差异大 |

---

## 三、实验协议

### 3.1 评估协议：严格 LOSO

**Leave-One-Subject-Out (LOSO)**：
- 每个 fold 中，1 个 subject 作为 test，其余 6 个作为 train
- 共 7 folds（对应 7 subjects）
- **零 data leakage**：test subject 的任何数据（包括 statistics）不出现在训练过程中

### 3.2 超参数固定策略

**严格不 inner-tune**：
- 所有方法的超参数在实验开始前固定
- 不根据 test subject 的表现调整参数
- 确保结果反映真实的跨 subject 泛化能力

### 3.3 指标定义

**Rep-level 指标**：
- **Precision**：检测 reps 中正确的比例（IoU ≥ 0.5）
- **Recall**：GT reps 中被检测到的比例
- **Rep F1**：Precision 和 Recall 的调和平均
- **Exact-count ratio**：rep 数量完全正确的 stream 比例

**Boundary 指标**：
- **Start MAE**：匹配 rep 的 start 误差（ms）
- **End MAE**：匹配 rep 的 end 误差（ms）
- **Transition MAE**：concentric→eccentric 切换点误差（ms）
- **IoU-F1@50**：sample-level phase IoU-F1@50%

**Stream-level 诊断**：
- Zero-TP streams：完全未检测到任何 rep 的 stream 数
- Under-segmented streams：检测 rep 数 < 50% GT 的 stream 数
- Over-segmented streams：检测 rep 数 > 150% GT 的 stream 数

---

## 四、数据伦理与隐私

- 所有数据来自知情同意的志愿者
- 数据仅用于学术研究，不包含个人身份信息
- 传感器数据为匿名化运动学信号

---

*文档版本: 2026-05-16 v1*
*用途: 论文 Dataset 章节*
