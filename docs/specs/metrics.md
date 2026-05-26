# 指標說明

這份文件統一說明本專案常見指標的意義，避免不同報告、實驗、streaming replay、app 顯示時對數字的解讀不一致。

## 先看哪一類指標

本專案大致有四類指標：

1. **rep-level 指標**
   - 用來看重複動作切割有沒有成功
   - 最接近實際 rep counting 體驗
2. **sample-level 指標**
   - 用來看每個 sample 的分類是否正確
   - 適合做模型診斷，但不一定直接等於 app 體驗
3. **segment IoU-F1 指標**
   - 用來看整段 phase / action segment 是否切得完整、乾淨
   - 對 over-segmentation 很敏感
4. **streaming / runtime 指標**
   - 用來看即時性、吞吐量、延遲與 app 顯示穩定度

## 目前專案的主次順序

目前這個專案評估重點要用下面這個順序解讀：

1. **rep 邊界切割品質**
   - 最重要的是每一下 rep 的開始、結束、向心/離心轉折有沒有切準
2. **rep count**
   - 數量是主要指標，但它是 rep 邊界品質的直接下游結果
3. **動作分類**
   - 只有在 rep 已經被切出來之後，動作 identity 才有意義

所以如果看到：

- `rep_action_accuracy` 很高
- 但 `rep recall` 很低

不能說模型已經夠好，因為這通常代表：

- 已成功匹配到的 reps 分類得不錯
- 但大量 reps 根本沒有被正確切出來

## Rep-Level 指標

### `precision`

預測出的 reps 有多少是真的。

```text
precision = TP / (TP + FP)
```

高 precision 代表：

- 比較不容易亂多算
- app 上比較不會平白無故多跳 rep

### `recall`

真實 reps 有多少被抓到。

```text
recall = TP / (TP + FN)
```

高 recall 代表：

- 比較不容易漏算 rep
- app 上比較不會明明做了動作卻沒加數量

### `f1`

`precision` 和 `recall` 的平衡指標。

```text
f1 = 2 * precision * recall / (precision + recall)
```

這通常是 rep segmentation 的最重要總指標之一。

### `n_pred` / `n_true` / `tp` / `fp` / `fn`

- `n_pred`：模型預測出的 reps 數量
- `n_true`：真實 reps 數量
- `tp`：成功對上的 reps
- `fp`：多預測的 reps
- `fn`：漏掉的 reps

對 app 來說：

- `fp` 多代表容易多算
- `fn` 多代表容易漏算

### `count_diff`

這是單一 stream 中：

```text
count_diff = n_pred - n_true
```

解讀：

- `0`：數量剛好
- `> 0`：多算，通常對應 over-segmentation
- `< 0`：漏算，通常對應 under-segmentation 或 phase collapse

### `exact-count streams`

有多少條 stream 的預測 rep 數量和真實數量完全相同。

這個指標很適合看 app 角度的「整段使用體驗」是否接近可用。

### `over-segmented streams`

有多少條 stream 出現：

```text
n_pred > n_true
```

通常代表：

- 一個真 rep 被切成多個小 rep
- phase sequence 雖然還能形成合法 `concentric -> eccentric`，但切得太碎

### `under-segmented streams`

有多少條 stream 出現：

```text
n_pred < n_true
```

通常代表：

- 漏掉整個 rep
- phase 預測偏成單一 phase
- 或大量有效 rep 根本沒有成功配對

### `zero-TP streams`

有多少條 stream 完全沒有成功對上的 rep。

這個指標很重要，因為它直接反映：

- 系統是否在某些動作族上幾乎整段失效
- 問題是不是集中在少數災難型 stream，而不只是平均表現不好

### `start_mae_ms`

預測 rep 起點和真實起點的平均誤差，單位毫秒。

越小越好。

### `end_mae_ms`

預測 rep 終點和真實終點的平均誤差，單位毫秒。

越小越好。

### `transition_mae_ms`

預測向心/離心切換點和真實切換點的平均誤差，單位毫秒。

越小越好。

### `rep_action_accuracy`

在已成功匹配到的 reps 中，動作類別判對的比例。

這表示：

- rep 邊界已經對上後
- 動作 identity 有多準

它不包含漏掉的 reps，也不直接反映 sample-level action 穩定度。

## Sample-Level 指標

### `micro_sample_accuracy`

每個 sample 的 micro label 預測正確率。

目前 micro label 常見是：

- `other`
- `concentric`
- `eccentric`

### `micro_sample_macro_f1`

micro label 的 sample-level macro F1。

它會對每一類分別算 F1，再平均，所以比單純 accuracy 更能看出少數類別是否被忽略。

### `macro_sample_accuracy`

每個 sample 的 macro/action label 預測正確率。

### `macro_sample_macro_f1`

macro/action label 的 sample-level macro F1。

注意：

- sample-level 指標高，不代表 rep segmentation 一定高
- sample-level 指標低，也不代表 app 就一定不能用

因為 app 更關心的是 rep event 和穩定顯示，而不是每個 sample 的 label 是否瞬間正確。

## Segment IoU-F1 指標

### `micro_f1_at_10` / `micro_f1_at_25` / `micro_f1_at_50`

這是 micro segment 的 IoU-F1 指標。

意思是：

- 先把連續相同標籤合成一段 segment
- 預測段和真實段必須：
  - 類別相同
  - IoU 大於門檻
- 才算匹配成功

例如 `micro_f1_at_50` 表示 IoU 門檻是 `0.50`。

這類指標對下列問題很敏感：

- segment 被切碎
- phase 邊界晃動
- 一段動作中間被插進 `other`

所以它很適合衡量 phase segmentation 品質。

### `macro_f1_at_10` / `macro_f1_at_25` / `macro_f1_at_50`

和上面一樣，只是對象換成 macro/action segments。

### `micro_semantic_f1_at_50`

這是 action-aware micro label 的 segment IoU-F1，例如：

- `db_rdl::concentric`
- `db_weighted_crunch::eccentric`

它比單純 phase-level `micro_f1_at_50` 更難，因為不只要切對 phase，還要切對是哪個動作的 phase。

## Edit Score

### `micro_edit`

micro segments 的 normalized edit score。

它反映的是整段 label sequence 的順序有多接近真實答案。

高 `micro_edit` 代表：

- 整體序列順序比較合理
- 比較不會反覆出現破碎的小片段

### `macro_edit`

macro/action segments 的 normalized edit score。

## Streaming / Runtime 指標

### `throughput_samples_per_second`

每秒能處理多少 sample。

### `real_time_factor`

推論速度和資料真實速度的比值。

```text
real_time_factor = throughput / sample_rate
```

解讀：

- `> 1.0`：比真實速度快，可以即時跑
- `< 1.0`：比真實速度慢，無法即時跟上

### `buffer_size`

線上推論時使用的 rolling buffer 大小（sample 數）。

### `buffer_seconds`

把 `buffer_size` 換算成秒數後的長度。

### `online_rep_emit_delay_ms`

當一個 rep 真正結束後，到系統發出 completed rep event 的平均延遲，單位毫秒。

這對 app 很重要，因為它會影響：

- rep 數字什麼時候加一
- 使用者覺得系統是不是「有跟上」

### `online_rep_emit_delay_ms_p95`

emit delay 的 95 百分位數。

它可以看出是否偶爾有很長延遲的情況。

## App 顯示穩定化指標

### `display_rep_count`

給 app 用的 rep 數量顯示欄位。

它不是每個 sample 都更新，而是：

- 只有在 completed rep event 發生時才加一

### `display_action`

給 app 用的穩定動作顯示欄位。

它不是每個 sample 都跟著 `online_macro_label` 變，而是根據最近幾個 completed reps 的結果延後鎖定。

### `display_action_confidence`

目前顯示動作的聚合信心分數。

### `display_action_locked`

表示目前顯示動作是否已經進入「鎖定」狀態。

### `display_action_switches`

app-facing `display_action` 在整段 replay 中切換了幾次。

通常越少越穩定。

### `final_display_action`

這段 stream 最後顯示在 app 上的動作名稱。

### `expected_display_action`

依照 ground truth 主動作推得的期望顯示動作。

### `final_display_action_correct`

最後顯示的動作是否和期望動作一致。

這個指標很適合快速判斷：

- app 最後顯示得穩不穩
- 穩定下來之後到底是對還是錯

## 如何解讀不同使用情境

### 如果你要做離線模型比較

優先看：

1. `start_mae_ms` / `end_mae_ms` / `transition_mae_ms`
2. `rep f1`
3. `rep precision / recall`
4. `count_diff`、`exact-count streams`、`over/under-segmented streams`
5. `micro_f1_at_50`
6. `rep_action_accuracy`

### 如果你要做即時 rep counting

優先看：

1. `start_mae_ms` / `end_mae_ms`
2. `rep precision / recall / f1`
3. `count_diff`、`exact-count streams`、`over/under-segmented streams`
4. `online_rep_emit_delay_ms`
5. `real_time_factor`
6. `display_rep_count` 是否跟完成事件同步

### 如果你要做 app 動作顯示

優先看：

1. `display_action_switches`
2. `final_display_action_correct`
3. `rep_action_accuracy`
4. `macro_sample_*` 只作為輔助診斷

## 簡短結論

- 想看「rep 邊界切得準不準」：先看 `start_mae_ms`、`end_mae_ms`、`transition_mae_ms`，再看 `rep f1`
- 想看「數量準不準」：看 `count_diff`、`exact-count streams`、`over/under-segmented streams`
- 想看「即時能不能跑」：看 `real_time_factor`
- 想看「app 會不會亂跳」：看 `display_action_switches`
- 想看「最後 app 顯示對不對」：看 `final_display_action_correct`
