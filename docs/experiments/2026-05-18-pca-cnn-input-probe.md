# PCA Input Probe for 1D Causal CNN

Date: 2026-05-18

## Question

Does replacing the raw 6-axis IMU input with PCA-transformed channels improve the phase-only 1D Causal CNN?

## Method

- Script: `scripts/new_c_pipeline/test_pca_input.py`
- Fold: held-out `kevin` only, for speed.
- Data: light-weight sessions excluded.
- Model: phase-only 1D Causal CNN, hidden size 32, 5 epochs, CPU.
- Decoder: MA25 + Viterbi penalty 0.3.
- PCA fitting: training subjects only; fit `StandardScaler` on flattened training active samples, fit PCA on standardized samples, transform each time step, then z-score PCA components from training segments.

## Results

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| Raw 6-axis | 0.7443 | 0.458 | 1.67 | 0.6566 | 13 / 0 |
| PCA-3 | 0.7941 | 0.417 | 1.38 | 0.6624 | 13 / 1 |
| PCA-4 | 0.8484 | 0.667 | 0.54 | 0.7005 | 6 / 2 |
| PCA-5 | 0.7426 | 0.375 | 1.33 | 0.6311 | 13 / 2 |
| PCA-6 | 0.7042 | 0.458 | 1.83 | 0.6353 | 13 / 0 |

PCA cumulative explained variance on the training fold:

| Components | Cumulative Variance |
|------------|---------------------|
| 1 | 0.237 |
| 2 | 0.450 |
| 3 | 0.638 |
| 4 | 0.801 |
| 5 | 0.902 |
| 6 | 1.000 |

## Interpretation

PCA95 requires all 6 components, so it is only a rotation and does not reduce the input. Fixed PCA-4 is promising on this single fold because it improves both Rep F1 and Exact Count while sharply reducing over-counts.

## Status

This initial single-fold probe was followed by a fast all-subject LOSO probe.

## Fast 9-Fold Follow-Up

- Script: `scripts/new_c_pipeline/pca4_cnn_9fold.py`
- Command: `python -u scripts\new_c_pipeline\pca4_cnn_9fold.py --epochs 5 --hidden 32 --output artifacts\cnn_variant_comparison\pca4_cnn_9fold_fast.json`
- Output: `artifacts/cnn_variant_comparison/pca4_cnn_9fold_fast.json`

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| Raw 6-axis | 0.8634 | 0.559 | 1.37 | 0.7477 | 89 / 8 |
| PCA-4 | 0.8673 | 0.577 | 1.00 | 0.7470 | 78 / 15 |

PCA-4 is a strict win in the fast LOSO probe, but it should still be tested with the formal 20-epoch hidden=64 protocol before replacing the raw 6-axis CNN input.

## Formal GPU 9-Fold Follow-Up

- Script: `scripts/new_c_pipeline/pca4_cnn_9fold.py`
- Command: `python -u scripts\new_c_pipeline\pca4_cnn_9fold.py --epochs 20 --hidden 64 --output artifacts\cnn_variant_comparison\pca4_cnn_9fold_gpu_h64e20.json`
- Device: CUDA GPU
- Output: `artifacts/cnn_variant_comparison/pca4_cnn_9fold_gpu_h64e20.json`

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| Raw 6-axis | 0.8513 | 0.514 | 1.68 | 0.7575 | 99 / 8 |
| PCA-4 | 0.8453 | 0.514 | 1.68 | 0.7498 | 103 / 4 |

## Final Decision

Do not replace the raw 6-axis CNN input with PCA-4. The fast probe improvement did not survive the stronger hidden=64, 20-epoch GPU evaluation.

## PCA-1 Formal GPU Follow-Up

- Script: `scripts/new_c_pipeline/pca4_cnn_9fold.py`
- Command: `python -u scripts\new_c_pipeline\pca4_cnn_9fold.py --epochs 20 --hidden 64 --pca-components 1 --output artifacts\cnn_variant_comparison\pca1_cnn_9fold_gpu_h64e20.json`
- Device: CUDA GPU
- Output: `artifacts/cnn_variant_comparison/pca1_cnn_9fold_gpu_h64e20.json`

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| Raw 6-axis | 0.8616 | 0.509 | 1.61 | 0.7582 | 96 / 12 |
| PCA-1 | 0.6160 | 0.150 | 3.88 | 0.6003 | 175 / 12 |

PCA-1 is rejected. It discards too much information and causes severe over-counting.

## 9-Axis Raw IMU Formal GPU Follow-Up

- Script: `scripts/new_c_pipeline/axis_subset_cnn_9fold.py`
- Command: `python -u scripts\new_c_pipeline\axis_subset_cnn_9fold.py --epochs 20 --hidden 64 --subset-columns "ax,ay,az,gx,gy,gz,mx,my,mz" --subset-name "imu9" --output artifacts\cnn_variant_comparison\imu9_cnn_9fold_gpu_h64e20.json`
- Device: CUDA GPU
- Output: `artifacts/cnn_variant_comparison/imu9_cnn_9fold_gpu_h64e20.json`

| Input | Rep F1 | Exact Count | Count MAE | Phase F1 | Over / Under |
|-------|--------|-------------|-----------|----------|--------------|
| Raw 6-axis | 0.8535 | 0.464 | 1.75 | 0.7611 | 115 / 3 |
| 9-axis IMU | 0.7871 | 0.468 | 2.05 | 0.6903 | 98 / 19 |

9-axis raw IMU is rejected. Adding magnetometer channels hurts phase modeling and rep detection despite a tiny Exact Count increase.
