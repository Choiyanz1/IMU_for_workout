# 報告與論文規劃

## 1. 目前系統總目標

目前整體系統想要達成的線上流程是：

```text
IMU 串流
  -> 短前綴（輕量辨識或 default）決定用哪個 rep cutter
  -> action-conditioned rep segmentation
  -> 每完成一個 rep：
       提取其 rich per-rep features
       -> Rep-Complete Action Classifier -> 穩定動作標籤
  -> rep count
  -> 向心 / 離心切分
```

關鍵區別：**動作辨識不是靠短前綴完成，而是靠每組完整切好的 rep 做 rep-level 分類**。
前綴只需要夠讓系統選對 rep cutter 即可，精確的最終動作標籤來自 rep-complete classifier。

具體作法：

- `train/hybrid_action_classifier.py` 中的 `HybridActionClassifier`
- 對每組已切好的 rep 擷取豐富特徵（duration、訊號統計、頻域特徵等）
- 訓練一個 sklearn classifier（logistic regression / RF 擇優）
- 可與 macro stage 做 confidence-based hybrid 融合

但以目前進度來看，**近期主線應先收斂在**：

- rep segmentation / rep count
- rep-complete action recognition
- 系統部署可行性

### 關於疲勞預測

疲勞模組目前**不應視為這一版主流程的核心結果**。

原因：

- 後續疲勞預測預期還會加入 `PPG` 訊號
- 所以疲勞預測本質上會是 **IMU + PPG 的多模態下游任務**
- 它依賴前面更穩定的：
  - rep segmentation
  - rep-level phase / feature extraction
  - rep-complete action classification

因此目前建議的敘事是：

- **主流程**：rep segmentation → rep-complete action recognition → rep count
- **延伸模組**：向心/離心切分、每 rep 變化分析、疲勞預測

## 2. 建議先做的流程

### Phase A：凍結一個可報告的主線版本

先不要同時追太多分支，先固定一條能講得清楚、能反覆重現的主線。

目前我建議的主線：

- 動作辨識：之後從 baseline 比較中選一個穩定模型
- rep segmentation：`causal RF + boundary refiner`
- 模態：預設 full 6-axis `ax, ay, az, gx, gy, gz`
- RF trailing window：`50`
- refiner edge window：`20`

原因：

- 這條線現在證據最多
- guarded modality 測試顯示，很多情況最後會回到 full 6-axis baseline
- direct event RF probe 目前效果不好，不適合取代主線

### Phase B：選出 rep-complete action classifier

這一步優先度很高，但與 rep cutting 不是先後關係，可以並行。

需要完成：

- 在已知 ground-truth rep 邊界上，比較不同 rep-complete classifier
  - Logistic Regression
  - Random Forest
  - XGBoost / CatBoost / LightGBM
- 分析 worst-case 類別與 confusion matrix
- 分析 hybrid（macro + classifier）的改善幅度
- 推論成本與部署可行性判讀

### Phase C：固定 action-conditioned rep segmentation 設計

目前應先固定：

- 短前綴負責輕量辨識以選擇 rep cutter（可接受有限準度）
- 每個動作各自一套 RF + boundary refiner
- full 6-axis default
- fixed `window=50`, `edge=20`
- 最終動作標籤來自 rep-complete classifier

### Phase D：只做高價值消融

建議消融不要太散，先做這幾類：

1. 動作辨識模型比較
2. RF rep-cutting 架構比較
3. modality / guardrail 比較
4. direct-rep 負向對照（目前已經有 quick probe）

### Phase E：整理論文與報告材料

當主線固定後，把所有輸出整理成：

- 表格
- 圖
- per-action breakdown
- 失敗案例
- 部署可行性說明

## 3. 報告一定要有的內容

### 3.1 問題定義與系統目標

需要說清楚：

- 為什麼動作辨識是 per-rep complete 層級的任務，不是 per-prefix 層級
- 為什麼 rep segmentation 不強依賴前綴動作辨識（可用 default cutter fallback）
- 為什麼 rep count 是主要目標
- 為什麼 fatigue 目前不列為主結果

目前已經有：

- `docs/specs/system.md`
- `docs/specs/model.md`

### 3.2 系統架構圖

至少要兩張：

1. **整體系統圖**
   - IMU stream -> short prefix cutter switch -> action-conditioned rep cutter -> per-rep feature extraction -> rep-complete action classifier -> rep count / phase split
2. **目前主線 rep-cutting 圖**
   - full 6-axis -> z-score -> causal RF phase detector -> phase pairing -> boundary refiner -> refined reps

目前已經有：

- `docs/specs/model.md` 中的主線 rep-cutting 文字架構圖

目前已補的正式 SVG 圖：

- `docs/specs/assets/system_architecture.svg`
  - 圖 1：整體系統架構圖（svg）
- `docs/specs/assets/rf_rep_cutting_architecture.svg`
  - 圖 2：目前主線 rep-cutting 架構圖（svg）
- 以上兩張已直接嵌入 `docs/specs/reporting_plan_assets.html`

### 3.3 模型 / baseline 比較

要有一張清楚的 baseline 比較表。

建議內容：

- 模型名稱
- 是否 causal
- 是否可部署
- rep F1
- precision
- recall
- micro_f1@50
- 備註

目前已經有：

- `docs/experiments/2026-05-14-model-architecture-gap-analysis.md`
- `docs/experiments/2026-05-14-heldout-yushuan-rep-cutting-results.md`

目前已補的 placeholder：

- `docs/specs/reporting_plan_assets.html`
  - 表 1：模型 / baseline 比較表

### 3.4 目前主線模型效果

這一張是目前最重要的主結果表。

目前已經有可直接引用的 guarded held-out `yoru` 結果：

- 檔案：
  - `artifacts/baseline_comparison/modality_count_guardrail_yoru_v1/results.json`

可直接放的數字：

| 指標 | 數值 |
|---|---:|
| Precision | 0.8648 |
| Recall | 0.8869 |
| Rep F1 | 0.8757 |
| micro_f1@50 | 0.8126 |
| exact-count streams | 19 / 25 |
| over-segmented streams | 6 / 25 |
| under-segmented streams | 0 / 25 |

### 3.5 Modality 與 guardrail 消融

要說清楚：

- 自由 modality tuning 為什麼不穩
- 為什麼單模態容易造成缺 rep
- 為什麼 count-aware / recall-aware guardrail 有必要

目前已經有：

- `artifacts/baseline_comparison/smoke_modality_guardrail_v1/results.json`
- `artifacts/baseline_comparison/smoke_modality_count_guardrail_v1/results.json`
- `artifacts/baseline_comparison/modality_count_guardrail_yoru_v1/yoru/best_config_per_action.csv`

目前已補的 placeholder：

- `docs/specs/reporting_plan_assets.html`
  - 表 2：modality / guardrail 消融表

### 3.6 Direct-rep 負向對照

這一段很值得放，因為它支持目前主線設計的合理性。

目前已經有：

- `docs/experiments/2026-05-15-direct-event-rf-probe.md`
- `artifacts/baseline_comparison/direct_event_rf_yoru_probe_v2/results.json`
- `artifacts/baseline_comparison/direct_event_rf_yoru_probe_v3_triceps/results.json`

目前判讀：

- naive direct event RF 不夠強
- phase-first RF 目前仍然更穩

目前已補的 placeholder：

- `docs/specs/reporting_plan_assets.html`
  - 表 3：phase-first vs direct-rep quick probe 比較表

### 3.7 Rep-Complete 動作辨識結果

這是目前**最重要的缺口之一**。

需要放的是：

- 在已知 ground-truth rep 邊界（或現有 rep cutter 結果）上，比較不同 rep-level classifier
- held-out subject 的 per-action 準確率
- confusion matrix
- hybrid（macro + classifier）的改善幅度
- 最終選定的 classifier 與 feature set

目前狀態：

- 已有 `train/hybrid_action_classifier.py` 與 streaming 整合
- 但尚未整理成一份乾淨、可直接放報告的完整比較表

目前已補的 placeholder：

- `docs/specs/reporting_plan_assets.html`
  - 表 4：動作辨識模型比較表
  - 表 5：prefix 長度 vs 準確率表
  - 圖 3：confusion matrix placeholder

### 3.8 End-to-End 系統展示

這張不是一定要最完整 benchmark，但至少要有 demo flow。

建議展示：

- 輸入 stream
- rep cutter 輸出（顯示切成 reps）
- 每組 rep 的 rep-complete classifier 結果
- 最終 rep count + action label
- 顯示某一條 stream 的成功案例

目前已補的 placeholder：

- `docs/specs/reporting_plan_assets.html`
  - 圖 4：end-to-end demo panel placeholder

### 3.9 部署可行性

你現在特別關心這件事，報告一定要獨立一節。

需要說清楚：

- 大小
- 記憶體
- 延遲
- 精度
- 現在能不能直接上 Luckfox Pico Zero

目前正確結論：

- RF + boundary-refiner 是目前最強 offline rep-cutting 參考模型
- 但它目前還不是板端 ready
- 原因不是先準度，而是：
  - 還沒有 ONNX / RKNN / board runtime 路徑
  - 還沒有 Luckfox 上的延遲 / 記憶體實測
  - 目前主線仍是 Python / sklearn benchmark

目前已經有：

- `docs/specs/model.md`
- `docs/specs/system.md`
- `docs/dev-log.md` 裡 DS-MS-TCN / Luckfox 既有說明

目前已補的 placeholder：

- `docs/specs/reporting_plan_assets.html`
  - 表 6：部署可行性摘要表

## 4. 論文需要有的內容

論文大致上需要這些模組：

1. 問題定義 / 任務設定
2. 系統架構圖
3. 方法設計
4. baseline 比較
5. 消融實驗
6. per-action breakdown
7. 失敗案例分析
8. 部署限制與未來工作

### 建議的論文主線敘事

建議目前用這個主線：

1. 一般 sequence model 與 direct sparse-event RF 都有試，但目前最穩的是 phase-first RF + boundary refinement
2. 自由 modality tuning 很容易傷害 rep completeness
3. 把 count-aware / recall-aware guardrail 加進 selection 後，穩定答案經常回到 full 6-axis baseline
4. 由於 rep 邊界不依賴前綴動作辨識，動作辨識被設計成 per-rep complete 層級
5. 在已知 rep 邊界上，rep-complete classifier 可達到高準確率，並與 macro stage 融合成 hybrid

## 5. 現在還缺什麼一定要跑

### 最高優先

1. Rep-complete action classifier held-out benchmark
   - 在 ground-truth rep 邊界上比較不同 model + feature set
2. 至少再補 1 到 2 個 guarded RF held-out subject
   - 先 `yushuan`
   - 再看 `haoyu`
3. 最終 per-action comparison table
4. Hybrid（macro + rep-complete classifier）對比分析

### 次高優先

1. end-to-end system demo（串接 rep cutter → rep-complete classifier）
2. 部署 prototype / latency 試驗

### 目前不建議搶先做

1. fatigue prediction 主結果
2. 大範圍 direct-rep 新分支搜尋
3. 沒有明確故事的廣泛 ablation
4. prefix 動作辨識優化（目前只需要夠選 cutter 即可）

## 6. 現在最應該如何安排時間

### 先做

1. Rep-complete action classifier 比較與選型
2. 固定目前最穩的 action-conditioned rep cutter + rep-complete classifier
3. 整理 baseline / ablation / guarded result 表格

### 再做

1. 再補一到兩個 held-out subject
2. 補 end-to-end demo（串接完整 pipeline）
3. 補 hybrid classifier 改善分析

### 最後才做

1. fatigue + PPG 整合
2. 更長線的新模型分支

## 7. 相關文件

- 系統與評估規格：
  - `docs/specs/system.md`
  - `docs/specs/model.md`
  - `docs/specs/metrics.md`
- 報告 / 論文材料樣板：
  - `docs/specs/reporting_plan_assets.html`
