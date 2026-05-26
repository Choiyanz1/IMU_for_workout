# New C Pipeline: Independent Active Detector + 2-Class Phase Segmentation

## 設計動機

**Baseline A (現有)** 把 other / concentric / eccentric 放在同一個 3-class 模型中，存在以下問題：
- `other` 是「非運動狀態」，`concentric/eccentric` 是「運動內部 phase」，語意層級不同
- 模型同時要學「是不是在運動」和「是向心還是離心」，任務互相干擾
- 導致 over-segmentation（把靜止誤判為動作）和 transition 不準確

**New C Pipeline** 的核心思想：
> **把「active detection」和「phase segmentation」拆開，各司其職。**

## Architecture

```
Baseline A (現有, 保留):
Raw IMU
  → Sliding Window (1.0s)
  → 3-class RF: other / concentric / eccentric
  → Smoothing
  → Rep Parser (pair C/E)
  → Action Classification (per-rep)
  → Set-level Majority Voting

New C (新增, 主測試):
Raw IMU
  → Sliding Window (0.5-1.5s)
  → Independent Active Detector: active / not-active
  → Active Segment State Control (hysteresis)
  → Active Segment Extraction
  → 2-class Phase Segmentation: concentric / eccentric
  → Phase Smoothing (MA + majority vote + min duration)
  → Rep Parser / State Machine (C/E alternation)
  → Rep Count + Rep Boundary + C/E Intervals
  → Action Classification (per-rep)
  → Set-level Majority Voting
```

## 模組規格

### 1. Independent Active Detector

**任務**：判斷當前 window 是否處於有效健身動作區間

**Input**：
- 6-axis IMU window (0.5-1.5s, ~50-150 samples @100Hz)
- Rich features (stats + FFT + correlation + magnitude)

**Output**：
- `active`：明顯且連續的運動模式，可能屬於健身動作
- `not_active`：靜止、準備、調整姿勢、休息

**Training target**：
- `other` → `not_active`
- `concentric` / `eccentric` → `active`
- 注意：這只是訓練標籤來源，推論時不參考 C/E 預測

**Model options**：
- RF (100 trees, depth 15) — 快速 baseline
- LogReg — 更輕量
- 1D-CNN — 如果需要時序特徵

### 2. Active Segment State Control

**任務**：避免單一 window 的抖動導致 segment 頻繁進出

**State Machine**：
```
IDLE → (連續 N 個 active) → ACTIVE
ACTIVE → (連續 M 個 not_active) → IDLE

參數：
- N = min_active_windows (預設 3)
- M = min_not_active_windows (預設 3)
- active_threshold = 0.5 (probability threshold)
```

**特性**：
- 在 ACTIVE 狀態中，短暫的 not_active 不會立刻中斷
- 避免在動作轉折點或低速離心時被切斷

### 3. 2-Class Phase Segmentation

**任務**：在 active segment 內部，判斷每個 sample 是 concentric 還是 eccentric

**Input**：
- Active segment 內的連續 IMU sequence

**Output**：
- P(concentric), P(eccentric) per sample

**Model options**：
- RF with sliding window (簡單 baseline)
- 1D-CNN (淺層)
- BiLSTM
- TCN

**關鍵差異**：
- 沒有 `other` class！模型只需區分 C/E
- 訓練時只使用 ground-truth active 區間的樣本

### 4. Phase Smoothing

**方法**：
1. Moving average on probability (window=5-15 samples)
2. Majority vote on short windows (window=3-5 samples)
3. Minimum phase duration constraint (min_phase_samples=3-10)

**目的**：
- 避免 C/E/C/E 快速抖動
- 確保 phase 有足夠長度才計為有效

### 5. Rep Parser / State Machine

**輸入**：平滑後的 C/E phase sequence

**規則**：
```
一個完整 rep = eccentric segment → concentric segment
（或根據動作類型支援反向順序）

輸出：
- rep_start：eccentric 開始
- transition：eccentric 結束 / concentric 開始
- rep_end：concentric 結束
```

### 6. Action Classification (保留)

- 以 Rep Parser 輸出的完整 rep 為單位
- 提取 per-rep rich features
- LogReg / RF / AutoGluon
- Set-level majority voting

## Evaluation Metrics

| Metric | 定義 |
|--------|------|
| Active Detection F1 | active 區間的 detection F1 |
| Phase Macro F1 | concentric / eccentric 的 macro F1（在 active 區間內計算）|
| Transition MAE | predicted vs GT transition point 的平均絕對誤差 (ms) |
| Rep IoU-F1@50 | Rep boundary 的 IoU-F1@50% overlap |
| Rep Count Error | predicted count - GT count（每 set）|
| Exact Count Accuracy | predicted count == GT count 的 set 比例 |
| Over-segmentation Count | predicted reps > GT reps 的 set 數 |
| Under-segmentation Count | predicted reps < GT reps 的 set 數 |
| Action Classification Accuracy | per-rep 正確率 |
| Set-level Action Accuracy | majority vote 後的 set 正確率 |

## Implementation Plan

### Phase 1: Active Detector
- [ ] 實現 active/not-active 標籤轉換
- [ ] 訓練 RF baseline (per-action 或 global)
- [ ] 評估 active detection F1

### Phase 2: State Control + Segment Extraction
- [ ] 實現 hysteresis state machine
- [ ] 提取 active segments
- [ ] 評估 segment 覆蓋率

### Phase 3: 2-Class Phase Segmentation
- [ ] 從 active segments 提取 C/E 訓練資料
- [ ] 訓練 2-class RF baseline
- [ ] 評估 phase macro F1 + transition MAE

### Phase 4: Smoothing + Rep Parser
- [ ] Phase smoothing (MA + majority vote + min duration)
- [ ] Rep Parser (C/E alternation)
- [ ] 評估 rep metrics

### Phase 5: Integration + Comparison
- [ ] 整合完整的 New C pipeline
- [ ] 與 Baseline A 比較所有 metrics
- [ ] 輸出詳細報告

## 預期效益

1. **Active Detector 專注**：不需要區分 C/E，只需判斷「有沒有在動」，理論上更穩定
2. **Phase Model 簡化**：2-class 比 3-class 簡單，減少混淆
3. **減少 Over-segmentation**：State control 避免靜止區間被誤判為動作
4. **更準確的 Transition**：Phase model 專注於 C/E 邊界，不受 other 干擾

## 風險

1. **延遲增加**：Active Detector + State Control 需要連續多個 window 確認，可能有 1-2 秒延遲
2. **Active Segment 切錯**：如果 active detector 漏掉開頭或結尾，後續 phase segmentation 一定錯
3. **兩階段錯誤傳播**：active detector 的錯誤會傳播到 phase model
