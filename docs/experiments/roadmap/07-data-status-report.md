# 数据状态报告（最终版）

## 文档目的

本文档记录截至 2026-05-16 的数据集完整状态，包括 subjects、actions、文件分布、数据质量等。这是所有实验设计的**数据基础**。

**更新说明**：
- rep0 文件：用户确认不是问题，关键文件已清理
- 所有实验保持 strict subject-wise split

---

## 一、数据集总体概况

| 项目 | 数值 |
|------|------|
| **Subjects** | 9 |
| **Actions** | 8 + rest |
| **总 CSV 文件数** | 3,903 |
| **平均每个 Subject 文件数** | ~434 |
| **磁力计可用性** | 100%（所有文件均含 mx/my/mz） |
| **Phase 标签可用性** | 100%（所有文件均含 phase 列） |
| **NaN 检出率** | 0%（抽样 20 个文件无 NaN） |

---

## 二、Subject 分布

| Subject | CSV 文件数 | 会话数 | 备注 |
|---------|-----------|--------|------|
| haoyu | 315 | 1 (haoyu0512workout) | |
| hsianshun | 321 | 1 (hsianshun0514workout) | |
| kevin | 619 | 1 (kevin) | 文件数最多 |
| thomas | 660 | 3 (thomas, thomas_2, thomas0506workout) | 会话数最多 |
| tsenyu | 326 | 1 (tsenyu0515workout) | 新增 |
| yanz | 703 | 2 (1000, yanz0510workout) | |
| yoru | 321 | 1 (yoru0511workout) | |
| yushuan | 311 | 1 (yushuan0513workout) | |
| ziho | 327 | 1 (ziho0512workout) | |

---

## 三、Action 分布

每个 Subject 均包含以下 9 个目录（8 个动作 + big_rest）：

1. `db_bench_press` — 哑铃卧推
2. `db_biceps_curl` — 哑铃二头弯举
3. `db_rdl` — 哑铃罗马尼亚硬拉
4. `db_shoulder_press` — 哑铃肩推
5. `db_squat` — 哑铃深蹲
6. `db_triceps_curl` — 哑铃三头弯举
7. `db_weighted_crunch` — 哑铃负重卷腹
8. `one_arm_db_row` — 单臂哑铃划船
9. `big_rest` — 组间大休息

### 数据组织方式

每个 action 目录下包含多个 `set*` 子目录：
```
{subject}/{session}/{action}/
├── set0/
│   ├── rep0_*.csv
│   ├── rep1_*.csv
│   ├── ...
│   └── rest_after_set0/
│       └── rest*.csv
├── set1/
│   ├── rep0_*.csv
│   ├── ...
│   └── rest_after_set1/
│       └── rest*.csv
└── ...
```

**关键认知**：
- 原始数据已经是按 rep 切分好的（每个 rep 一个 CSV）
- Rep CSV 的平均长度约 200-400 samples（2-4 秒 @100Hz）
- Rest CSV 的长度远大于 rep CSV（可达 5000+ samples）
- Phase 标签已存在于每个 CSV 的 `phase` 列中

---

## 四、CSV 列说明

### 标准列（所有文件均包含）

| 列名 | 含义 | 单位 | 用途 |
|------|------|------|------|
| `sensor_ts` | 传感器时间戳 | 秒或毫秒 | 时间对齐、采样率推断 |
| `ax`, `ay`, `az` | 加速度计 X/Y/Z | g | Stage 1/2/3 输入 |
| `gx`, `gy`, `gz` | 陀螺仪 X/Y/Z | dps | Stage 1/2/3 输入 |
| `mx`, `my`, `mz` | 磁力计 X/Y/Z | μT | Stage 1/2/3 输入（可选） |
| `phase` | 相位标签 | categorical | Stage 2 GT, Stage 1 训练监督 |
| `action_type` | 动作类型 | categorical | Stage 3 GT, 数据组织 |
| `subject_id` | 受试者 ID | categorical | 数据组织、LOSO 分割 |

### 可选列

| 列名 | 含义 | 备注 |
|------|------|------|
| `pc_time` | PC 接收时间 | 可用作备用时间戳 |
| `serial_num` | 数据包序列号 | 可用于丢包检测 |
| `host_ts` | 主机时间戳 | |
| `rpe` | 自感用力度 | 如有可分析运动强度 |
| `weight_kg` | 负重重量 | 如有可分析负荷影响 |
| `ppg_a` ~ `ppg_e` | 光电容积脉搏波 | 如有可用于心率分析（非本项目重点） |

---

## 五、数据质量检查

### 5.1 已确认的良好状态

| 检查项 | 结果 | 样本数 |
|--------|------|--------|
| 所有文件含磁力计 | ✅ 是 | 20/20 |
| 所有文件含 phase 标签 | ✅ 是 | 20/20 |
| 无 NaN 值 | ✅ 无 | 20/20 |
| 平均 rep 长度 | ~290 samples | 20 |
| rep 长度范围 | 202 - 476 samples | 20 |
| 采样率 | 约 100 Hz（推断） | 20 |

### 5.2 已知数据问题

| 问题 | 状态 | 影响 | 处理建议 |
|------|------|------|---------|
| **307 个 rep0 文件** | ✅ 用户已确认不是问题 | 关键文件已清理 | 无需额外处理 |
| **Rest 文件远大于 rep 文件** | ⚠️ 正常 | z-score 统计可能被 rest 数据主导 | 评估时只计算包含 reps 的 streams；z-score 可只基于 active segments |
| **Folder 结构有历史遗留** | ⚠️ 正常 | `kevin/kevin/...`, `thomas/thomas_2/...` | 数据加载器已适配，不影响实验 |

### 5.3 磁力计数据特性

根据抽样分析（`kevin/one_arm_db_row/set2/rep1_175206.csv`）：

| 轴 | 均值 | 标准差 | 变异系数 (CV) |
|------|------|--------|--------------|
| ax | -1.07 | 0.11 | 0.115 |
| gx | 2.10 | 12.10 | 10.165 |
| mx | 40.95 | 5.48 | 0.134 |

**关键发现**：
- mx 的 CV 极低（0.134），说明磁力计信号动态变化很小
- mx 的 z-score 范围仅 (-1.88, 1.52)，远小于 acc/gyro 的 ±3~5
- mx-my 相关性高达 -0.916（高度冗余）
- **结论**：磁力计可能不适合作为 rep segmentation 的主要信号源

---

## 六、数据使用策略

### 6.1 训练/测试分割

**严格 LOSO**：
- 每个 fold 用 8 个 subjects 训练，1 个 subject 测试
- z-score 统计只能从 train subjects 计算
- 不允许任何 cross-subject 信息泄露
- **黄金法则**：test subject 的数据在任何训练/校准/阈值估计中都不能出现

### 6.2 Stream 构建

**对于 Rep Segmentation（Stage 1）**：
- 将同一 set 的所有 rep CSV 按自然顺序拼接
- 拼接时保留原始时间戳的连续性（或假设均匀采样）
- GT rep boundaries 由拼接位置确定（rep0_end → rep1_start）
- **关键**：评估必须在拼接后的 set stream 上进行，不能只在单个 rep CSV 上

**对于 Phase/Action（Stage 2/3）**：
- 直接使用单个 rep CSV
- 无需拼接

### 6.3 Exclude 规则

根据 `config.yaml`：
- `*whole_session*`: 跳过（与 individual rep CSVs 重复）
- `*_w`: 跳过（损坏的 sets）
- `*rest_after*`: 跳过（休息期间不包含 active reps）

---

## 七、数据 Summary

```
IMU_for_workout Dataset (2026-05-16)
├── 9 Subjects
│   ├── haoyu (315 files)
│   ├── hsianshun (321 files)
│   ├── kevin (619 files) ⭐ 最大
│   ├── thomas (660 files) ⭐ 最多会话
│   ├── tsenyu (326 files) ⭐ 新增
│   ├── yanz (703 files)
│   ├── yoru (321 files)
│   ├── yushuan (311 files)
│   └── ziho (327 files)
│
├── 8 Actions (all subjects have all actions)
│   ├── db_bench_press
│   ├── db_biceps_curl
│   ├── db_rdl
│   ├── db_shoulder_press
│   ├── db_squat
│   ├── db_triceps_curl
│   ├── db_weighted_crunch
│   └── one_arm_db_row
│
├── 3,903 Total CSV files
│   ├── ~80% are rep files (rep0-rep23)
│   └── ~20% are rest files
│
├── Data Quality
│   ├── 100% have magnetometer (mx, my, mz)
│   ├── 100% have phase labels
│   ├── 0% NaN (sampled)
│   └── ~100 Hz sampling rate
│
└── Known Issues
    ├── rep0 files: 用户确认已处理，无需额外清理
    └── Rest files are much longer than rep files
```

---

*文档版本: 2026-05-16（修订版）*
*状态: 已完成数据审计，等待实验执行*
