# 2026-05-10 App 顯示穩定化測試

## 目的

針對實際 app 使用需求，測試一個比 sample-level action 顯示更穩定的輸出方案：

1. rep 數量只在 completed rep event 時增加
2. action 不逐 sample 切換
3. action 改為等最近幾個 completed reps 有足夠一致性後才鎖定

本次使用的規則：

- `display_vote_window = 3`
- `display_min_reps = 2`
- `display_min_fraction = 0.67`
- `display_min_confidence = 0.0`

## 實作方式

新的 app-friendly 顯示狀態機是事件驅動，不是 sample 驅動：

```text
連續 IMU
  -> online predictor
  -> micro phase
  -> OnlineRepDecoder
  -> completed rep event
  -> display_rep_count += 1
  -> 用最近幾個 completed rep 的 action 多數決更新 display_action
```

這樣可以避免 app 上的動作名稱每幾個 sample 就跳來跳去。

## 測試模型

測了兩條 streaming 主線：

1. 原本較實用的 streaming baseline
   - `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn`
2. 新的 dual-head 模型
   - `artifacts/micro_macro_recognition/exp_dual_head/tcn`

測試資料：

- `kevin/db_weighted_crunch/set0`
- `kevin/db_rdl/set1`

## 結果一：Baseline streaming

### `kevin/db_weighted_crunch/set0`

- output:
  - `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step_smooth15_display`
- streaming summary:
  - online rep F1: `0.6923`
  - final display rep count: `16`
  - final display action: `one_arm_db_row`
  - expected display action: `db_weighted_crunch`
  - final display action correct: `false`

穩定性比較：

- raw sample-level `online_macro_label` switches: `75`
- app-facing `display_action` switches: `2`

解讀：

- rep 計數的事件式更新可以穩定工作。
- action 顯示從 sample 級別大幅穩定下來。
- 但這條 stream 的最終動作仍然穩定地鎖到錯誤類別，代表「穩定」和「正確」還不是同一件事。

### `kevin/db_rdl/set1`

- output:
  - `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/streaming_eval/kevin_db_rdl_set1_step_smooth15_display`
- streaming summary:
  - online rep F1: `0.3333`
  - final display rep count: `12`
  - final display action: `db_weighted_crunch`
  - expected display action: `db_rdl`
  - final display action correct: `false`

穩定性比較：

- raw sample-level `online_macro_label` switches: `33`
- app-facing `display_action` switches: `1`

解讀：

- 顯示確實更穩定。
- 但在較難的 `db_rdl` stream 上，基礎 action 預測本身仍然不夠好，所以最後會穩定地顯示錯誤類別。

## 結果二：Dual-head streaming

### `kevin/db_weighted_crunch/set0`

- output:
  - `artifacts/micro_macro_recognition/exp_dual_head/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step_display`
- streaming summary:
  - online rep F1: `0.7143`
  - final display rep count: `18`
  - final display action: `one_arm_db_row`
  - expected display action: `db_weighted_crunch`
  - final display action correct: `false`

穩定性比較：

- raw sample-level `online_macro_label` switches: `34`
- app-facing `display_action` switches: `1`

解讀：

- dual-head 讓 action sample-level 抖動比 baseline 小一些。
- 但在這條 `weighted_crunch` stream 上，最後仍然穩定地鎖到錯誤類別。

### `kevin/db_rdl/set1`

- output:
  - `artifacts/micro_macro_recognition/exp_dual_head/tcn/streaming_eval/kevin_db_rdl_set1_step_display`
- streaming summary:
  - online rep F1: `0.2609`
  - final display rep count: `17`
  - final display action: `db_rdl`
  - expected display action: `db_rdl`
  - final display action correct: `true`

穩定性比較：

- raw sample-level `online_macro_label` switches: `2`
- app-facing `display_action` switches: `1`

解讀：

- dual-head 對 `db_rdl` 的 action 語意幫助明顯。
- 這條 stream 的 action 顯示已經可以又穩又對。
- 但 rep 偵測本身仍然偏弱，所以作為完整 app 方案還不能只看 action。

## 整體結論

這個 app-friendly 顯示狀態機是有效的，因為：

1. `rep_count` 可以自然地只在 rep 完成時增加
2. `display_action` 的切換次數比 sample-level `online_macro_label` 少很多
3. UI 上的呈現會比直接顯示每 sample 預測穩定得多

但也要清楚區分兩件事：

- **穩定顯示**：這個方案已經能做到
- **穩定且正確顯示**：仍然受底層 model 的 action quality 限制

目前最實際的判斷是：

1. rep 計數可以走事件式更新，這是目前最可行的 app 路線
2. action 顯示不應逐 sample 更新，應該使用延後鎖定策略
3. 對某些動作（例如 dual-head 下的 `db_rdl`）已經能穩定且正確
4. 對另外一些動作（特別是 `db_weighted_crunch`）仍然會穩定地顯示錯誤類別

## 建議下一步

如果目標是讓 app 真的可用，最合理的下一步是：

1. rep 繼續使用事件式更新
2. action 顯示繼續使用延後鎖定
3. 把「動作來源」從 sample-level macro aggregation 逐步換成更好的 rep-level action source
   - 例如 dual-head 語意分支
   - 或 rep-complete classifier / hybrid routing
