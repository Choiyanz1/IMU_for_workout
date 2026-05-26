# Configs

根目錄的 `config.yaml` 仍然是預設的一體式設定檔。

如果你想把不同任務的實驗設定分開管理，可以使用這些副本：

- `base.yaml`：目前共用的預設設定。
- `action_classification.yaml`：動作分類實驗。
- `phase_segmentation.yaml`：phase segmentation 實驗。
- `rep_segmentation.yaml`：SDTW repetition segmentation 實驗。
- `micro_macro_recognition.yaml`：DS-MS-TCN 風格的 micro/macro 實驗，
  目前使用固定順序 `concentric -> eccentric` 做 rep segmentation 與 action recognition。
  可用 `micro_macro.micro_source` 設定 Stage 1 來源：
  - `both`
  - `tcn`
  - `dtw`

資源設定預設為自動：

- `device: auto`：優先使用 CUDA，其次 MPS，最後 CPU。
- `num_workers: auto`
- `pin_memory: auto`

資料切分採 subject-wise split，以 subject folder 為單位。DTW 的搜尋參數可在
`micro_macro.dtw` 下調整。

訓練腳本不要求特定檔名，只要用 `--config` 指定你要的設定檔即可。

大多數訓練/評估入口現在都會在模型專屬 artifact 旁邊輸出標準化比較檔：

- `report.md`
- `metrics/summary.json`
- `metadata/run_manifest.json`
- `metadata/config_snapshot.yaml`

你可以用下面指令把它們彙整成一份比較表：

```bash
python -m evaluation.compare_runs --root artifacts
```

如果你想一次跑多個可比較模型，也可以用：

```bash
python -m evaluation.model_suite --models ds_ms_tcn --mode sets
```

這會在共用資料集上執行指定模型，並同時保存：

- 各模型自己的 artifact
- 一份共享的 comparison table
