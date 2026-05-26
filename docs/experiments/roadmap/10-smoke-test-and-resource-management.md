# Smoke Test 与资源管理规范

## 文档目的

本文档记录所有实验的**Smoke Test 强制流程**和**电脑资源使用规范**。这是确保"实验不跑偏、不浪费资源、可实时监控"的操作手册。

**核心原则**：
1. **任何大规模实验前必须先做 Smoke Test**
2. **善用电脑资源，但不要过度设计**
3. **绝对禁止未经 smoke test 直接开大规模训练**
4. **每完成一个 fold/方法就保存结果并报告进度，不要等全部跑完**
5. **可cache的部分必须cache，避免重复计算和重复I/O**

---

## 一、Smoke Test 强制规范

### 1.1 什么是 Smoke Test

Smoke Test 是一种**快速、低成本的验证方法**，在投入完整实验资源之前，先在小规模数据上验证：
- 脚本是否能正确运行
- 趋势是否正确（我们的方法是否优于 baseline）
- 是否存在 data leakage 或设计缺陷

### 1.2 Smoke Test 的必要性

**为什么不能直接跑完整实验？**
- 9-fold LOSO × 7 methods = 63 runs，可能需要 4-6 小时
- 如果脚本有 bug 或设计缺陷，63 runs 的结果全部作废
- 如果趋势不对（如 Peak Detection 比我们好），继续跑完整实验是浪费时间

**Smoke Test 的价值**：
- 30-60 分钟快速验证趋势
- 提前发现脚本问题
- 避免浪费数小时的计算资源

### 1.3 Smoke Test 标准流程

#### Step 1: 脚本验证（~15 分钟）

在跑任何数据之前，先人工检查脚本设计：

```bash
# 检查清单
python scripts/verify_script_design.py \
    --scripts evaluate_causal_rf,benchmark_per_action_rf_refiner,sdtw_rep_segmentation \
    --checks causal,leakage-free,set-level-evaluation
```

**人工检查项**：
- [ ] trailing window 是否只用过去样本？
- [ ] z-score 是否只从 train subjects 计算？
- [ ] 评估是否在 set-level merged streams 上？
- [ ] 随机种子是否固定？
- [ ] 是否存在 test subject 信息泄露？

#### Step 2: 3-Subject 快速验证（~30-60 分钟）

**选择的 subjects**：
- `kevin`：数据最多（619 files），代表"大数据量 subject"
- `yushuan`：数据最少（311 files），代表"小数据量 subject"
- `yoru`：中等数据量（321 files），代表"典型 subject"

**选择的原因**：
- 覆盖数据量的极端情况
- 快速验证方法在不同 data size 下的稳定性
- 如果在这 3 个 subjects 上趋势正确，大概率在全部 9 个上也正确

**执行命令**：

```bash
# Phase 1a Smoke test
python scripts/run_rep_baseline_comparison.py \
    --methods peak,causal_rf,causal_rf_refiner \
    --subjects kevin,yushuan,yoru \
    --actions db_bench_press,db_biceps_curl,db_rdl,db_shoulder_press,db_squat,db_triceps_curl,db_weighted_crunch,one_arm_db_row \
    --output artifacts/baseline_comparison/smoke_test

# Phase 1b Smoke test
python scripts/run_modality_ablation.py \
    --method causal_rf_refiner \
    --modalities A,AG,AGM \
    --subjects kevin,yushuan,yoru \
    --output artifacts/modality_ablation/smoke_test

# Phase 3 Smoke test
python scripts/run_action_classification.py \
    --methods statistical_rf,autogluon \
    --subjects kevin,yushuan,yoru \
    --output artifacts/action_classification/smoke_test
```

**注意**：
- 只跑 3 个 subjects = 3 folds（而不是 9 folds）
- 只跑最关键的方法（而不是全部方法）
- 目的是"验证趋势"，不是"得到最终数字"

#### Step 3: 趋势判断

**通过标准**（必须全部满足）：

| 检查项 | 通过标准 | 失败处理 |
|--------|---------|---------|
| RF+Refiner > Causal RF (plain) | Rep F1 差距 > 3% | 检查 refiner 是否实现正确 |
| Causal RF > Peak Detection | Rep F1 差距 > 5% | 可能數據規律性太強，執行 Contingency Plan A |
| 所有方法 Rep F1 > 0.65 | 最低标准 | 數據可能有問題，停止修復 |
| BiLSTM - RF+Refiner < 10% | 非因果上限与我们的方法差距不大 | 如果差距 > 15%，可能需要更强的模型 |
| No Zero-TP streams | 至少能检测到一些 reps | 检查脚本逻辑 |

**Phase 3 (Action Verification) 的通过标准**：

| 检查项 | 通过标准 | 失败处理 |
|--------|---------|---------|
| Per-rep Macro F1 > 0.80 | 可接受（後處理可提升到 95%+） | 嘗試更豐富的特徵或 AutoGluon |
| Per-rep Macro F1 0.70-0.80 | 可接受，但後處理更重要 | 確認 set-level majority vote 能提升到 90%+ |
| Per-rep Macro F1 < 0.70 | 不足夠 | 需要改進特徵或考慮 per-action 模型 |
| Set-level Majority Vote > 0.90 | 實際可用標準 | 如果達不到，降低 Action Verification 在論文中的權重 |

**关键认知**：
- Action Verification 的 per-rep F1 > 0.80 就足夠了，因為可以利用 set-level 結構約束做後處理
- **前導識別（前2-3 reps 識別動作）延後到耦合分析階段，現在假設可以通過**

**不通过的后果**：
- **任何一项不通过 → 停止完整实验**
- 分析原因 → 修复脚本或数据 → 重新 smoke test

#### Step 4: 完整实验（仅在 smoke test 通过后）

```bash
# Phase 1a 完整实验
python scripts/run_rep_baseline_comparison.py \
    --methods peak,dtw,sliding_rf,causal_rf,causal_rf_refiner,bilstm,tcn \
    --subjects haoyu,hsianshun,kevin,thomas,tsenyu,yanz,yoru,yushuan,ziho \
    --output artifacts/baseline_comparison/rep_segmentation_final
```

---

## 二、资源管理规范（防浪費設計）

### 2.1 核心原則

**絕對不允許**：
- ❌ 全部跑完才保存結果（如果中途崩潰，幾小時白費）
- ❌ 每個 fold 重新讀取所有 CSV（重複 I/O 浪費時間）
- ❌ 重新計算已經算過的特徵（重複計算浪費 CPU）
- ❌ 一口氣跑完所有方法才看結果（如果第3個方法就有問題，後面4個都白費）

**必須做到**：
- ✅ 每完成一個 fold 立即保存 + 報告進度
- ✅ 數據預加載（一次讀入，全程內存訪問）
- ✅ 特徵緩存（避免重複計算 z-score、統計量）
- ✅ 斷點續跑（支持從上次中斷處繼續）
- ✅ 實時進度報告（每完成一個方法/一個 fold 打印結果）

### 2.2 數據預加載策略（一次讀入，不再讀磁碟）

**問題**：每個 fold 重新讀取 CSV 是最大瓶頸
- 3,903 個 CSV，每個 fold 都要遍歷一次
- 磁碟 I/O 比內存訪問慢 1000 倍

**解決方案**：

```python
# 推薦：實驗開始時一次性加載所有 CSV 到記憶體
def load_all_data_to_memory(data_dir):
    """
    一次性加載所有 CSV，後續所有 fold 從記憶體讀取
    總數據量：3,903 CSV × ~300 samples × 12 columns ≈ 14 MB（很小！）
    """
    cache = {}
    for csv_path in tqdm(Path(data_dir).rglob('*.csv'), desc="Loading CSVs"):
        # 只加載需要的列，節省記憶體
        cache[csv_path] = pd.read_csv(csv_path, usecols=[
            'sensor_ts', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'phase', 'action_type'
        ])
    return cache

# 使用方式
data_cache = load_all_data_to_memory('datasets/raw_data')
# 後續所有 fold 直接從 data_cache 讀取，不再碰磁碟
```

**效果**：
- 第一次加載：~30 秒（一次性）
- 後續每個 fold：~0 秒（直接內存訪問）
- 總記憶體佔用：~20-50 MB（微不足道）

**禁止做法**：

```python
# ❌ 禁止：每個 fold 都重新遍歷文件系統
def run_fold(fold, data_dir):
    for csv_path in Path(data_dir).rglob('*.csv'):  # 重複遍歷！
        df = pd.read_csv(csv_path)  # 重複讀取！
        # ...
```

### 2.3 特徵緩存策略（避免重複計算）

**問題**：z-score 和特徵計算是 CPU 密集操作
- 每個 fold 的 train set 都要計算 z-score stats
- 每個 sample 都要提取 trailing window features

**解決方案**：

```python
# 推薦：緩存 z-score 統計量和特徵
def compute_and_cache_features(data_cache, output_dir):
    """
    一次性計算所有特徵並保存到磁碟
    後續實驗直接讀取緩存，不再重複計算
    """
    cache_file = Path(output_dir) / 'feature_cache.pkl'
    
    if cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        return pickle.load(open(cache_file, 'rb'))
    
    # 計算所有特徵（一次性）
    features = {}
    for path, df in tqdm(data_cache.items(), desc="Computing features"):
        features[path] = extract_features(df)  # z-score, trailing window, etc.
    
    # 保存緩存
    pickle.dump(features, open(cache_file, 'wb'))
    print(f"Features cached to {cache_file}")
    return features
```

**效果**：
- 第一次：~2-3 分鐘（一次性計算）
- 後續所有 fold：~0 秒（直接讀取緩存）
- 如果數據沒變，緩存可以跨實驗複用

### 2.4 中間結果保存（每完成一個 fold 就保存）

**核心原則**：不要等全部跑完！

```python
# 推薦：每完成一個 fold 立即保存 + 報告進度
import json
from datetime import datetime

def run_experiment(methods, folds, output_dir):
    """支持斷點續跑的實驗框架"""
    
    # 加載已有的進度（斷點續跑）
    progress_file = Path(output_dir) / 'progress.json'
    completed = set()
    if progress_file.exists():
        progress = json.load(open(progress_file))
        completed = set(progress['completed_runs'])
        print(f"Resuming from previous run. Completed: {len(completed)} runs")
    
    for method in methods:
        for fold in folds:
            run_key = f"{method}_{fold}"
            
            if run_key in completed:
                print(f"Skipping {run_key} (already completed)")
                continue
            
            # 執行這個 fold
            print(f"[{datetime.now()}] Running {run_key}...")
            result = run_single_fold(method, fold)
            
            # 立即保存結果
            result_file = Path(output_dir) / f"{run_key}.json"
            json.dump(result, open(result_file, 'w'))
            print(f"[{datetime.now()}] {run_key} completed. Rep F1: {result['rep_f1']:.3f}")
            
            # 更新進度文件
            completed.add(run_key)
            json.dump({'completed_runs': list(completed)}, open(progress_file, 'w'))
            
            # 每完成 3 個 runs 打印一次匯總
            if len(completed) % 3 == 0:
                print_summary(output_dir)
```

**效果**：
- 如果實驗中途崩潰，已完成的 fold 結果不會丟失
- 重新啟動時自動跳過已完成的 fold
- 隨時可以查看當前結果（不需要等全部跑完）

### 2.5 實時進度報告（每完成一個方法就打印匯總）

```python
def print_summary(output_dir):
    """打印當前已完成的結果匯總"""
    results = []
    for result_file in Path(output_dir).glob('*.json'):
        if result_file.name == 'progress.json':
            continue
        result = json.load(open(result_file))
        results.append(result)
    
    # 按方法分組
    from collections import defaultdict
    by_method = defaultdict(list)
    for r in results:
        by_method[r['method']].append(r['rep_f1'])
    
    print("\n" + "="*60)
    print("CURRENT PROGRESS SUMMARY")
    print("="*60)
    print(f"Completed: {len(results)} / {total_expected} runs")
    print(f"Methods completed: {list(by_method.keys())}")
    print()
    print("Per-method Rep F1 (so far):")
    for method, f1s in sorted(by_method.items()):
        print(f"  {method:25s}: {np.mean(f1s):.3f} (n={len(f1s)})")
    print("="*60 + "\n")
```

**效果**：
- 每完成一個 fold 都能看到最新趨勢
- 不需要等全部跑完就能判斷是否繼續
- 如果某個方法明顯有問題，可以立即停止

### 2.6 斷點續跑（Resume from Crash）

**設計要求**：

```python
# 實驗腳本必須支持 --resume 參數
python scripts/run_rep_baseline_comparison.py \
    --methods peak,causal_rf,causal_rf_refiner \
    --subjects haoyu,hsianshun,kevin,thomas,tsenyu,yanz,yoru,yushuan,ziho \
    --output artifacts/baseline_comparison/rep_segmentation_final \
    --resume  # 如果已有進度，從中斷處繼續
```

**實現邏輯**：
1. 檢查 `progress.json` 是否存在
2. 如果存在，讀取已完成的 runs
3. 跳過已完成的 runs，只執行剩餘的
4. 如果不存在，從頭開始

### 2.7 監控指標

在实验运行期间，建议打开系统监控（Windows: Task Manager, Linux: htop）：

| 指标 | 安全范围 | 警告范围 | 危险范围 |
|------|---------|---------|---------|
| CPU 使用率 | 60-80% | 80-95% | > 95%（持续） |
| 記憶體使用率 | 50-70% | 70-85% | > 85% |
| 磁碟 I/O | < 50% | 50-80% | > 80% |

**如果发现进入危险范围**：
1. 立即降低并行度（减少正在运行的进程数）
2. 检查是否有内存泄漏
3. 如果系统已经卡死，强制终止部分进程

---

## 三、完整實驗設計模板

```python
# scripts/run_rep_baseline_comparison.py

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

class ExperimentRunner:
    """
    支持以下特性的實驗框架：
    1. 數據預加載（避免重複 I/O）
    2. 特徵緩存（避免重複計算）
    3. 中間結果保存（每個 fold 完成後立即保存）
    4. 斷點續跑（支持 --resume）
    5. 實時進度報告（每個 fold 完成後打印）
    """
    
    def __init__(self, output_dir, resume=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加載已有進度
        self.progress_file = self.output_dir / 'progress.json'
        self.completed = self._load_progress() if resume else set()
        
        # 預加載數據（一次性）
        self.data_cache = self._preload_data()
        
        # 計算並緩存特徵（一次性）
        self.feature_cache = self._compute_features()
    
    def _preload_data(self):
        """一次性加載所有 CSV 到記憶體"""
        cache_file = self.output_dir / 'data_cache.pkl'
        if cache_file.exists():
            print("Loading pre-cached data...")
            return pickle.load(open(cache_file, 'rb'))
        
        print("Preloading all CSV files to memory...")
        cache = {}
        for csv_path in tqdm(Path('datasets/raw_data').rglob('*.csv')):
            cache[csv_path] = pd.read_csv(csv_path)
        
        pickle.dump(cache, open(cache_file, 'wb'))
        print(f"Data cached ({len(cache)} files)")
        return cache
    
    def _compute_features(self):
        """一次性計算所有特徵並緩存"""
        cache_file = self.output_dir / 'feature_cache.pkl'
        if cache_file.exists():
            print("Loading pre-cached features...")
            return pickle.load(open(cache_file, 'rb'))
        
        print("Computing features (this may take 2-3 minutes)...")
        features = {}
        for path, df in tqdm(self.data_cache.items()):
            features[path] = self.extract_features(df)
        
        pickle.dump(features, open(cache_file, 'wb'))
        print("Features cached")
        return features
    
    def run(self, methods, folds):
        """執行實驗，支持斷點續跑"""
        total = len(methods) * len(folds)
        print(f"Total runs: {total}, Already completed: {len(self.completed)}")
        
        for method in methods:
            for fold in folds:
                run_key = f"{method}_{fold}"
                
                if run_key in self.completed:
                    print(f"[SKIP] {run_key} already completed")
                    continue
                
                # 執行單個 fold
                print(f"[{datetime.now()}] Starting {run_key}...")
                result = self.run_single(method, fold)
                
                # 立即保存結果
                self.save_result(run_key, result)
                self.completed.add(run_key)
                self._save_progress()
                
                # 打印進度
                print(f"[{datetime.now()}] {run_key} done. Rep F1={result['rep_f1']:.3f}")
                
                # 每 3 個 runs 打印匯總
                if len(self.completed) % 3 == 0:
                    self.print_summary()
        
        # 最終匯總
        self.print_summary(final=True)
    
    def save_result(self, run_key, result):
        """保存單個 fold 的結果"""
        result_file = self.output_dir / f"{run_key}.json"
        json.dump(result, open(result_file, 'w'))
    
    def print_summary(self, final=False):
        """打印當前進度匯總"""
        # 收集所有結果
        results = []
        for f in self.output_dir.glob('*_fold_*.json'):
            results.append(json.load(open(f)))
        
        if not results:
            return
        
        # 按方法分組
        by_method = {}
        for r in results:
            m = r['method']
            by_method.setdefault(m, []).append(r['rep_f1'])
        
        print("\n" + "="*70)
        print(f"{'FINAL' if final else 'INTERMEDIATE'} SUMMARY")
        print("="*70)
        print(f"Completed: {len(results)} runs")
        
        for method, f1s in sorted(by_method.items()):
            print(f"  {method:25s}: mean={np.mean(f1s):.3f}  std={np.std(f1s):.3f}  n={len(f1s)}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', required=True)
    parser.add_argument('--subjects', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    args = parser.parse_args()
    
    methods = args.methods.split(',')
    subjects = args.subjects.split(',')
    
    # 生成 folds (LOSO)
    folds = []
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        folds.append((test_subject, train_subjects))
    
    runner = ExperimentRunner(args.output, resume=args.resume)
    runner.run(methods, folds)

if __name__ == '__main__':
    main()
```

---

## 四、禁止事项（Red Lines）

以下行为**绝对禁止**：

| # | 禁止行为 | 原因 | 违规后果 |
|---|---------|------|---------|
| 1 | 未经 smoke test 直接跑完整 9-fold | 可能浪费数小时在错误的方向上 | 已完成的实验结果作废 |
| 2 | 未验证旧脚本就直接使用 | 可能有 data leakage 或设计缺陷 | 所有结果不可信 |
| 3 | 同时开启超过 10 个并行进程 | 系统卡死，所有进程变慢 | 实验时间反而更长 |
| 4 | 不固定随机种子 | 结果不可复现 | 无法判断改进是否真实 |
| 5 | 使用 test subject 的数据计算 z-score | Data leakage | 结果过于乐观，不可信 |
| 6 | 在 smoke test 不通过的情况下继续 | 可能方向错误 | 浪费更多时间 |
| 7 | **全部跑完才保存结果** | **中途崩溃则全部丢失** | **数小时白費** |
| 8 | **每个 fold 重新读取所有 CSV** | **重复 I/O 浪费大量时间** | **實驗時間翻倍** |
| 9 | **不打印中间进度** | **无法及时发现问题** | **全部跑完才知道有 bug** |

---

## 五、Smoke Test 记录模板

每次 smoke test 后，必须填写以下记录：

```markdown
## Smoke Test Record: [日期]

### 实验信息
- Phase: [1a/1b/2/3]
- Subjects: [kevin, yushuan, yoru]
- Methods: [peak, causal_rf, causal_rf_refiner]
- 执行人: [你的名字]

### 脚本验证
- [ ] trailing window 只用过去样本
- [ ] z-score 只从 train subjects 计算
- [ ] 评估在 set-level streams 上
- [ ] 随机种子已固定
- [ ] 數據預加載正確（沒有重複 I/O）
- [ ] 中間結果保存正常（每個 fold 後都有 .json 文件）

### 结果
| Method | Rep F1 | Recall | Precision | 备注 |
|--------|--------|--------|-----------|------|
| Peak Detection | 0.XX | 0.XX | 0.XX | |
| Causal RF | 0.XX | 0.XX | 0.XX | |
| RF+Refiner | 0.XX | 0.XX | 0.XX | |

### 趋势判断
- [ ] RF+Refiner > Causal RF (差距 > 3%)
- [ ] Causal RF > Peak Detection (差距 > 5%)
- [ ] 所有方法 F1 > 0.65
- [ ] 无 Zero-TP streams

### 性能檢查
- [ ] 每個 fold 完成後立即保存了 .json 文件
- [ ] 進度報告正常打印
- [ ] 沒有重複讀取 CSV 的警告

### 结论
- [ ] 通过 → 继续完整 9-fold
- [ ] 不通过 → 原因：[描述] → 修复后重新 smoke test

### 下一步行动
[具体行动计划]
```

---

## 六、常见问题

### Q1: Smoke test 通过了，但完整实验结果不同？

**可能原因**：
- 3 subjects 不能代表全部 9 subjects 的分布
- 某个 held-out subject 特别难（风格独特、数据少）
- 随机波动（9-fold 的 std 可能较大）

**应对**：
- 这是正常的，smoke test 的目的是"验证趋势"不是"精确预测"
- 如果完整实验结果与 smoke test 趋势相反，检查是否有 bug

### Q2: 我可以跳过 smoke test 吗？

**不可以。** 这是强制要求。即使时间紧迫，smoke test 的 30-60 分钟能避免浪费数小时。

### Q3: 如果 smoke test 不通过，但我真的很想继续完整实验？

**必须停止。** 如果趋势不对，继续跑完整实验只是浪费更多时间和资源。先分析原因，修复后再重新 smoke test。

### Q4: 并行度怎么调？

**原则**：从保守开始，逐步增加。
1. 先跑 1 个 fold，观察资源使用率
2. 增加到 2 个并行，观察是否变慢
3. 继续增加到 3-5 个，直到 CPU 使用率稳定在 70-80%
4. 如果增加并行度后每个 fold 的速度反而下降（thrashing），说明并行度太高了

### Q5: 实验跑到一半崩潰了怎麼辦？

**應對**：
1. 檢查 `output_dir` 中已有多少 `.json` 結果文件
2. 重新執行相同命令，加上 `--resume` 參數
3. 腳本會自動跳過已完成的 fold，只跑剩餘的

---

*文档版本: 2026-05-16 v2*
*状态: 强制执行*
*适用范围: 所有 Phase 1-4 实验*
