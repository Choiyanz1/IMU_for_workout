# 模態消融實驗設計：傳感器組合對阻力訓練重複檢測的貢獻分析

## 1. 研究問題

在可穿戴 IMU 阻力訓練重複檢測中，加速度計（ACC）、陀螺儀（GYRO）與磁力計（MAG）三種模態的獨立與協同貢獻尚未被系統量化。本實驗旨在回答：

1. **獨立貢獻**：單一模態（ACC/GYRO/MAG）能否達到可用的檢測精度？
2. **協同效應**：多模態融合是否通過互補性降低漏檢率，還是僅引入冗餘？
3. **動作依賴性**：不同阻力訓練動作對模態的依賴是否一致？
4. **邊界質量 vs. 檢測召回**：犧牲模態多樣性是否會以漏掉特定 reps 為代價換取邊界精度？

## 2. 實驗設置

### 2.1 基線模型

統一使用當前驗證最強的 **Causal Random Forest Phase Detector + Boundary Refiner** 作為 backbone：

- **Phase detector**：窗口大小 50 samples（@100 Hz，即 0.5 s），stride 10，n_estimators=50，max_depth=15
- **Boundary refiner**：edge_window=20，獨立的 start/transition/end 回歸器
- **Phase decoder**：固定 `concentric → eccentric` 配對，max_phase_gap=3，min_phase_samples=3

**選擇理由**：該 pipeline 在 held-out `yushuan` 上達到 Rep F1=0.76（shared refiner）/ 0.78（per-action），是目前唯一能兼顧檢測率與邊界質量的可部署路徑。使用統一 backbone 可確保所有模態差異僅來自輸入信號本身。

### 2.2 數據

- **數據集**：清洗後的完整 9-subject 數據（`haoyu`, `hsianshun`, `kevin`, `thomas`, `tsenyu`, `yanz`, `yoru`, `yushuan`, `ziho`）
- **動作**：8 種阻力訓練動作（`db_bench_press`, `db_biceps_curl`, `db_rdl`, `db_shoulder_press`, `db_squat`, `db_triceps_curl`, `db_weighted_crunch`, `one_arm_db_row`）
- **總樣本**：3,903 個 rep/set/rest CSV 文件
- **特徵列**：所有實驗均統一進行 z-score 標準化（基於訓練 fold）

### 2.3 模態組合（7 組）

| 代號 | 模態 | 軸 | 維度 |
|------|------|---|------|
| A | ACC only | ax, ay, az | 3 |
| G | GYRO only | gx, gy, gz | 3 |
| M | MAG only | mx, my, mz | 3 |
| AG | ACC + GYRO | ax..gz | 6 |
| AM | ACC + MAG | ax..az, mx..mz | 6 |
| GM | GYRO + MAG | gx..gz, mx..mz | 6 |
| AGM | All | ax..mz | 9 |

**排除 `action_type` 作為輸入**：為確保公平，所有模態組合均不預設動作身份。模型必須從 raw IMU 直接學習 phase 分類，而非利用動作標籤作弊。

### 2.4 評估協議：嚴格 LOSO

為避免數據洩漏並獲得統計可信的結論，採用 **9-fold Leave-One-Subject-Out (LOSO)**：

```
Fold 1: test=haoyu,    train=hsianshun,kevin,thomas,tsenyu,yanz,yoru,yushuan,ziho
Fold 2: test=hsianshun, train=haoyu,kevin,thomas,tsenyu,yanz,yoru,yushuan,ziho
...
Fold 9: test=ziho,      train=haoyu,hsianshun,kevin,thomas,tsenyu,yanz,yoru,yushuan
```

- 每個 fold 內，z-score 統計僅從 train subjects 計算，應用於 test subject
- 不做 inner fold tuning（避免搜索偏差）
- 所有超參數固定為默認值（見 2.1）

### 2.5 報告指標

**主要指標（Primary）**：
| 指標 | 定義 | 優先級 |
|------|------|--------|
| Rep F1 | 重複級 F1-score（IoU≥0.5 視為匹配） | 最高 |
| Rep Recall | 真實 reps 中被成功檢測的比例 | 最高 |
| Rep Precision | 檢測 reps 中為真實的比例 | 高 |
| micro_f1@50 | 樣本級 micro phase IoU-F1@50% | 高 |

**次要指標（Secondary）**：
| 指標 | 定義 |
|------|------|
| Start MAE (ms) | 檢測起點與 GT 的平均絕對誤差 |
| End MAE (ms) | 檢測終點與 GT 的平均絕對誤差 |
| Transition MAE (ms) | 向心-離心轉換點誤差 |
| Exact-count ratio | rep 數完全正確的 stream 比例 |
| Zero-TP streams | 完全未檢測到任何 rep 的 stream 數 |
| Under-segmented streams | 檢測 rep 數 < 50% GT 的 stream 數 |

## 3. 互補性分析（核心創新）

為回答「多模態融合是否通過互補性降低漏檢」，本實驗設計以下 **cross-modality rep-level analysis**：

### 3.1 Union/Intersection 統計

對每一個 held-out stream（set-level），計算：

- `only_A`：被 A（ACC only）檢測到，但 G（GYRO only）未檢測到的 reps
- `only_G`：被 G 檢測到，但 A 未檢測到的 reps
- `both_AG`：被兩者都檢測到的 reps
- `missed_both`：兩者都未檢測到的 reps

對 AG（ACC+GYRO）組合，驗證其檢測集合是否接近 `only_A ∪ only_G ∪ both_AG`。

### 3.2 Per-Action Modality Profile

對每個動作，統計：
- 哪種單一模態的 Recall 最高？
- 多模態融合後 Recall 的提升是否來自「拯救」了單一模態漏掉的 reps？
- 是否存在「模態失效動作」（例如：某動作在 ACC only 下 zero-TP，但在 AG 下正常）？

### 3.3 信息冗餘度量

計算各模態在 RF 中的特徵重要性（Gini importance），比較：
- ACC 三軸的重要性總和
- GYRO 三軸的重要性總和
- MAG 三軸的重要性總和

若 MAG 的總重要性 < 5%，則支持「MAG 冗餘/低信噪比」的假設。

## 4. 公平性保證

為確保本消融實驗可被論文審閱接受，採取以下控制：

1. **模型鎖定**：所有模態使用完全相同的 RF 架構、超參數、解碼邏輯
2. **訓練/測試鎖定**：嚴格 LOSO，z-score 不跨 fold
3. **無 tuning**：不使用 per-action 或 per-fold 超參數搜索，避免搜索偏差
4. **完整指標**：同時報告 Recall（漏檢）與 micro_f1@50（邊界），避免單一指標誤導
5. **重現性**：所有 fold 的結果獨立保存，支持事後交叉驗證

## 5. 預期貢獻

### 5.1 對論文的貢獻

1. **首次系統量化**：在 9-subject 8-action 阻力訓練數據集上，首次以嚴格 LOSO 報告 ACC/GYRO/MAG 的獨立與協同貢獻
2. **互補性證據**：若 AG 的 Recall > max(A, G) 的 Recall，則提供多模態必要性的統計證據
3. **冗餘識別**：若 AGM 相對於 AG 無顯著提升，則支持「6-axis 已足夠」的部署建議
4. **動作依賴圖譜**：生成每個動作的最佳模態 profile，支持未來的 adaptive sensor selection

### 5.2 對部署的貢獻

- 若 A（ACC only）的整體 Recall 與 AG 差距 < 5%，則可考慮在電池受限場景下僅使用 3-axis 加速度計
- 若某些動作（如 `db_rdl`）必須依賴 GYRO，則在動作識別後啟動對應模態（而非始終開啟全部 9 軸）

## 6. 執行計劃

### Phase 1：單一模態基準（A, G, M）
- 跑 9-fold LOSO × 3 modalities = 27 runs
- 預期時間：每 run ~5-10 min，共 ~2-4 hours

### Phase 2：雙模態與全模態（AG, AM, GM, AGM）
- 跑 9-fold LOSO × 4 modalities = 36 runs
- 預期時間：~3-5 hours

### Phase 3：互補性分析
- 對 Phase 1 的 A 與 G 結果，做 set-level union/intersection 統計
- 生成 per-action 的模態 profile heatmap

### Phase 4：綜合報告
- 輸出統一格式的 `modality_ablation_summary.json`
- 生成 LaTeX-ready 表格（含 mean ± std across 9 folds）
- 標註統計顯著性（paired t-test 或 Wilcoxon signed-rank test，AG vs A）

## 7. 已知限制與風險

| 風險 | 緩解措施 |
|------|---------|
| LOSO 63 runs 計算時間長 | 使用 `scripts/benchmark_per_action_rf_refiner.py` 的 outer-fold 並行機制（已實現） |
| `rest` 文件遠長於 `rep` 文件，可能扭曲 z-score | 所有模態統一處理，且評估只計算包含 reps 的 streams |
| rep0 文件可能未完全清理 | 實驗前運行 `scripts/clean_rep0_phase_contamination.py` 做最終確認 |
| MAG 低信噪比導致 M 組合過弱 | 這本身就是預期發現之一；若 M 確實過弱，將在論文中明確報告 |

## 8. 與先前實驗的關鍵區別

| 維度 | 先前實驗（2026-05-14） | 本實驗設計 |
|------|----------------------|-----------|
| 評估協議 | 單一 held-out subject（yushuan） | 9-fold LOSO |
| 模態公平性 | Per-action tuned，模態與窗口同時搜索 | 固定超參數，只變模態 |
| 動作假設 | Per-action（已知動作身份） | Shared（動作未知，統一模型） |
| 互補性分析 | 無 | Union/Intersection + Per-action profile |
| 統計嚴謹性 | 單點估計 | Mean ± std + 顯著性檢驗 |
| 論文可用性 | 工程調參記錄 | 消融實驗規範格式 |

---

*設計日期：2026-05-16*
*目標輸出：論文 Table X（Modality Ablation） + Figure Y（Per-action Modality Profile）*
