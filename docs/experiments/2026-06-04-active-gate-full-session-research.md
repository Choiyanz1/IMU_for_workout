# Active Gate Full-Session Research

## Goal
- Fix the blocker for complete realtime use: rest/full-session false positives.
- Separate active/rest gating from phase decoding so we can test gate ideas quickly.

## Literature Signals
- `Temporal Action Localization for Inertial-based HAR` reports that segment/timeline models improve NULL-class accuracy compared with fixed-window classification.
- `Exploring the Impact of the NULL Class on In-the-Wild HAR` emphasizes that NULL-heavy real-world activity recognition needs preprocessing for recall and postprocessing for precision.
- `An Autoencoder-based Approach for Recognizing Null Class...` treats NULL rejection as a distinct problem, not just another balanced class.
- `Window Size Impact in HAR` supports 1-2s windows as a reasonable speed/accuracy tradeoff, but not as a complete event detector.

## Implemented
- Added `scripts/evaluate_periodic_active_gate_loso.py`.
- Added `scripts/evaluate_active_cnn_full_timeline_loso.py` for a causal CNN active segmenter trained on `set + rest_tail` timelines and rest fragments.
- Added optional CNN+RF hybrid verification to `scripts/evaluate_active_cnn_full_timeline_loso.py`: the CNN proposes active segments, then a basic/periodic RF can reject weak segments.
- Compared active gate features under LOSO:
  - `basic`: existing statistical window features.
  - `periodic`: basic features plus magnitude/jerk autocorrelation, frequency-band power, dominant frequency, spectral entropy, and zero-crossing features.
- Added optional realtime evaluator mechanisms in `scripts/evaluate_realtime_soft_top5_pipeline.py`:
  - periodic active gate via `--active-gate-features periodic`;
  - active-masked fixed-lag parsing via `--fixed-lag-active-mask`;
  - active segment cleanup via `--min-active-segment-samples` and `--active-mask-bridge-samples`;
  - candidate set confirmation via `--min-confirmed-reps` and `--confirmed-set-gap-samples`.
  - event-level confirmation via `--event-confirm-min-reps`, which groups buffered fixed-lag reps into candidate events and releases the whole event retroactively after confirmation.

## Standalone Active Gate Results
Command:
```sh
python scripts/evaluate_periodic_active_gate_loso.py \
  --feature-modes basic,periodic \
  --n-estimators 100 \
  --window-samples 200 \
  --stride-samples 50 \
  --enter-threshold 0.7 \
  --exit-threshold 0.45 \
  --enter-hold-samples 50 \
  --exit-hold-samples 100 \
  --min-active-samples 200 \
  --bridge-gap-samples 50 \
  --rest-tail-seconds 20 \
  --output artifacts/action_recognition/active_gate_periodic_loso/summary_basic_periodic_e07_x045.json
```

| Gate | Set F1 | Set Recall | Rest False-Active/min | Rest-Tail Active Rate | Rest-Tail Segments |
|---|---:|---:|---:|---:|---:|
| basic RF | `0.875` | `0.836` | `1.84` | `0.088` | `154` |
| periodic RF | `0.895` | `0.856` | `1.49` | `0.086` | `156` |

Periodicity features help across subjects, but they do not solve rest-tail leakage. Rest-after-set motion can itself be periodic or rep-like.

Conservative variants:

| Gate | Set F1 | Set Recall | Rest False-Active/min | Rest-Tail Active Rate | Rest-Tail Segments |
|---|---:|---:|---:|---:|---:|
| periodic, min active 400 | `0.875` | `0.836` | `0.91` | `0.078` | `142` |
| periodic, threshold 0.8, min active 300 | `0.851` | `0.805` | `0.84` | `0.075` | `140` |

More conservative gates reduce rest false-active time, but the recall/precision tradeoff is still not enough for deployment.

## Full-Timeline Active CNN
The RF gate appears limited by fixed-window features, so a causal CNN active segmenter was trained directly on full timelines. Training streams are held-out-subject LOSO and include rest fragments plus each train set appended with up to `20s` of matching `rest_after_set`.

Command:
```sh
python scripts/evaluate_active_cnn_full_timeline_loso.py \
  --epochs 2 \
  --hidden 16 \
  --batch-size 64 \
  --max-pos-weight 1.0 \
  --train-rest-tail-seconds 20 \
  --rest-tail-seconds 20 \
  --enter-threshold 0.7 \
  --exit-threshold 0.45 \
  --output artifacts/action_recognition/active_cnn_full_timeline_loso/summary_e2_h16_pos1_thr07.json
```

| Gate | Set F1 | Set Recall | Rest False-Active/min | Appended False-Active/min | Rest-Tail Segments |
|---|---:|---:|---:|---:|---:|
| periodic RF, threshold 0.7 | `0.895` | `0.856` | `1.49` | n/a | `156` |
| active CNN, threshold 0.7 | `0.905` | `0.879` | `2.12` | `1.40` | `117` |
| active CNN, threshold 0.75 | `0.845` | lower | `1.73` | n/a | n/a |
| active CNN + soft periodic verifier | `0.866` | `0.836` | `1.54` | `1.17` | `103` |

The CNN is better at preserving full workout sets and reduces rest-tail segment count, but it is still too permissive on rest-only streams. Raising the threshold hurts set recall too much. A soft periodic RF verifier improves false-active time and tail segments (`117 -> 103`), but gives up too much set recall to be the primary solution.

Negative smoke: increasing capacity/training to e5/h32 on one fold made the gate more conservative but did not improve the tradeoff. At threshold `0.7/0.45`, set F1 dropped to `0.701` with rest false-active/min `1.07`; lowering to `0.6/0.35` only recovered set F1 to `0.738` with rest false-active/min `1.54`.

## Full Pipeline Candidate Confirmation
Command:
```sh
python scripts/evaluate_realtime_soft_top5_pipeline.py \
  --epochs 5 \
  --hidden 64 \
  --action-n-estimators 50 \
  --active-gate-features periodic \
  --active-threshold 0.7 \
  --fixed-lag-active-mask \
  --min-confirmed-reps 3 \
  --confirmed-set-gap-samples 300 \
  --phase-step-samples 10 \
  --fixed-lag-samples 100 \
  --merge-threshold-scales 0.8 \
  --rest-tail-seconds 20 \
  --output artifacts/action_recognition/realtime_soft_top5/summary_full_realtime_active07_confirmed3_e5_h64.json
```

| Setup | Rep F1 | Exact | Count MAE | Phase IoU | C/E MAE | Rest False Reps | Rest-Tail Overlap Reps |
|---|---:|---:|---:|---:|---:|---:|---:|
| active mask only, threshold 0.7 | `0.651` | `0.205` | `2.90` | `0.489` | `1.423` | `346` | `131` |
| + min confirmed reps 3 | `0.651` | `0.191` | `3.72` | `0.489` | `1.133` | `79` | `110` |
| + event confirm, min reps 2 | `0.654` | `0.195` | `3.32` | `0.489` | `1.304` | `145` | `112` |
| periodic active gate | `0.712` | `0.245` | `2.64` | `0.538` | `0.965` | `254` | `149` |
| periodic + event min2, gap300 | `0.707` | `0.264` | `2.91` | `0.529` | `0.945` | `116` | `136` |
| periodic + event min2, gap1000 | `0.708` | `0.255` | `2.65` | `0.531` | `0.902` | `133` | `145` |
| periodic + event gap1000 + action lock | `0.695` | `0.255` | `3.14` | `0.532` | `0.934` | `78` | `137` |

Candidate confirmation is useful for suppressing short rest false positives, but it hurts count stability and does little for rest-tail overlap. The event-level version is closer to the deployment behavior we want: reps are buffered first, then all reps in a confirmed event are released retroactively. With `min_reps=2`, it is a better count/false-positive tradeoff than the old hard `min_confirmed_reps=3`, but it still does not solve rest-tail leakage.

The current best full-session tradeoff is `periodic active gate + event min2 + gap1000`. The wider event gap better matches user behavior: one set may contain long inter-rep pauses and should still be one candidate event. It preserves periodic-gate Count MAE (`2.64 -> 2.65`) while cutting rest-only false reps almost in half (`254 -> 133`). Adding action-lock evidence is a useful safety mode (`78` false reps), but it under-counts too much (`MAE 3.14`) for the primary mode.

Command:
```sh
python scripts/evaluate_realtime_soft_top5_pipeline.py \
  --epochs 5 \
  --hidden 64 \
  --action-n-estimators 50 \
  --active-threshold 0.7 \
  --fixed-lag-active-mask \
  --phase-step-samples 10 \
  --fixed-lag-samples 100 \
  --merge-threshold-scales 0.8 \
  --rest-tail-seconds 20 \
  --event-confirm-min-reps 2 \
  --event-confirm-gap-samples 1000 \
  --output artifacts/action_recognition/realtime_soft_top5/summary_full_realtime_periodic_event_min2_gap1000_e5_h64.json
```

Negative follow-ups:
- `active-threshold=0.8` reduced false positives but hurt true-set recall and Count MAE too much.
- Per-rep action-active evidence trimmed rest-tail reps but also removed too many true reps; the action-active head is not precise enough at rep-level timing.
- Post-event cooldown suppressed tail reps, but it also killed real reps when one set was split into multiple candidate groups.

## Rest-Tail Metric Split
The original `rest_overlap_reps` metric counted every rep with `end_idx > rest_start`. Debugging showed this mixes two different cases:

| Case | Meaning | Deployment interpretation |
|---|---|---|
| boundary crossing | `rep.start < rest_start < rep.end` | Usually last-set rep whose end was delayed; can be backdated to the set. |
| new rest rep | `rep.start >= rest_start` | True rest-tail false positive. |
| post-grace rest rep | `rep.start >= rest_start + grace` | Strict false positive after set-closing grace. |

Two-fold debug run on `_tsenyu_temp` and `_ziho_temp` with `periodic + event min2 + gap1000`:

| Metric | Count |
|---|---:|
| old rest overlap reps | `34` |
| boundary crossing reps | `23` |
| new rest reps | `11` |
| post-grace rest reps, 2s grace | `3` |

Most apparent rest-tail overlap is not a new rest rep. It is the final detected rep crossing the artificial set/rest boundary. For realtime use, this should be handled with set-closing grace and backdating, not by making the active gate stricter.

Full 9-fold strict-rest rerun:

Command:
```sh
python scripts/evaluate_realtime_soft_top5_pipeline.py \
  --epochs 5 \
  --hidden 64 \
  --action-n-estimators 50 \
  --active-gate-features periodic \
  --active-threshold 0.7 \
  --fixed-lag-active-mask \
  --phase-step-samples 10 \
  --fixed-lag-samples 100 \
  --merge-threshold-scales 0.8 \
  --rest-tail-seconds 20 \
  --rest-tail-grace-samples 200 \
  --event-confirm-min-reps 2 \
  --event-confirm-gap-samples 1000 \
  --output artifacts/action_recognition/realtime_soft_top5/summary_full_realtime_periodic_event_min2_gap1000_strictrest_e5_h64.json
```

| Metric | Value |
|---|---:|
| Rep F1 | `0.706` |
| Exact Count | `0.259` |
| Count MAE | `2.65` |
| Phase IoU-F1@50 | `0.531` |
| C/E MAE | `0.930` |
| Rest-only false reps | `137` |
| Old rest-tail overlap reps | `146` |
| Boundary-crossing reps | `104` |
| New rest reps | `42` |
| Post-grace rest reps, 2s grace | `8` |

Thus, only `42 / 146` old overlap reps are true new-rest reps, and only `8 / 146` start after a 2s set-closing grace. The main remaining full-session issue is rest-only false bursts, not set-tail false reps.

## Workout-Set Quality Analysis
Ignoring rest-only safety and focusing only on held-out workout sets, the current mainline still under-counts:

| Variant | Rep F1 | Count MAE | Bias | Exact | Within-1 |
|---|---:|---:|---:|---:|---:|
| cropped active-set fixed-lag soft x0.8 | higher baseline | `1.34`-`1.43` range | under-counting | better | better |
| full-session periodic + event gap1000 | `0.706` | `2.65` | `-2.28` reps/set | `0.259` | `0.550` |

The full-session degradation versus the cropped-active rows is concentrated:

| Action | Full MAE | Full Bias | Full Rep F1 | MAE Increase vs Cropped |
|---|---:|---:|---:|---:|
| `db_squat` | `4.74` | `-4.74` | `0.531` | `+2.48` |
| `db_rdl` | `4.71` | `-4.71` | `0.509` | `+2.75` |
| `one_arm_db_row` | `3.14` | `-3.00` | `0.605` | `+1.71` |
| `db_weighted_crunch` | `2.22` | `-1.93` | `0.649` | `+1.48` |
| `db_biceps_curl` | `2.07` | `-2.07` | `0.731` | `+1.00` |
| `db_bench_press` | `1.68` | `+0.11` | `0.817` | `-0.07` |
| `db_shoulder_press` | `1.37` | `-1.07` | `0.739` | `+0.41` |
| `db_triceps_curl` | `1.22` | `-0.78` | `0.719` | `+0.63` |

Worst subjects are similarly concentrated:

| Subject | Full MAE | Bias | Main issue |
|---|---:|---:|---|
| `hsianshun` | `5.24` | `-4.84` | severe under-count, especially RDL/squat/row |
| `_tsenyu_temp` | `4.20` | `-4.20` | severe under-count, especially RDL/squat/row |
| `yushuan` | `3.21` | `-1.54` | squat failures plus bench over-count cases |
| `_ziho_temp` | `2.40` | `-2.00` | RDL failures, otherwise mixed |

Severe under-count streams (`pred_count <= 0.5 * gt_count`) are `32 / 220` streams but contribute `311 / 583` absolute count error, or `53.3%` of total Count MAE. They are concentrated in `db_rdl` (`10`), `db_squat` (`9`), `one_arm_db_row` (`5`), and `db_biceps_curl` (`5`).

Event confirmation is not the main cause: comparing periodic active baseline to periodic + event gap1000 changed only `44 / 220` streams and had near-zero net effect on absolute error (`-2` total abs delta). The degradation comes mostly from full-session active gating/streaming context and phase confidence loss versus cropped active-set decoding.

Phase quality strongly predicts count error:

| Phase Macro F1 Bucket | Streams | Count MAE | Bias | Rep F1 | Zero-Pred Streams |
|---|---:|---:|---:|---:|---:|
| `<0.40` | `31` | `6.06` | `-5.87` | `0.130` | `13` |
| `0.40-0.55` | `26` | `4.04` | `-3.42` | `0.356` | `1` |
| `0.55-0.70` | `56` | `3.09` | `-2.48` | `0.639` | `1` |
| `0.70-0.85` | `73` | `1.36` | `-1.03` | `0.874` | `0` |
| `>0.85` | `34` | `0.53` | `-0.47` | `0.969` | `0` |

Interpretation: fix workout-set quality by improving streaming active coverage and phase stability for low-phase-quality action/subject combinations, not by tightening event confirmation.

### Biceps Curl Active-Mask Regression Check
`db_biceps_curl` looked suspicious because cropped active-set decoding is very strong once the yanz outlier is removed, but full-session metrics were still much worse. A targeted check showed this is mostly an active-mask regression, not evidence that biceps curl is intrinsically poor under bounded-latency phase decoding.

Full 9-fold, fixed-lag soft x0.8:

| Split | n | GT | Pred | Count MAE | Bias | Exact | Within-1 | Rep F1 | Phase Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cropped active-set, excluding yanz | `25` | `282` | `276` | `0.24` | `-0.24` | `0.880` | `0.920` | `0.976` | `0.858` |
| full-session periodic + event gap1000, excluding yanz | `25` | `282` | `256` | `1.04` | `-1.04` | `0.520` | `0.840` | `0.819` | `0.766` |

Comparing full-session variants showed event confirmation is not the cause. For biceps excluding yanz, periodic active alone and periodic + event min2 gap1000 were effectively identical (`MAE 1.04`, `Rep F1 0.819`). Narrow event grouping (`gap300`) was worse (`MAE 1.76`) because it split true sets, but the chosen `gap1000` did not drop biceps reps.

Minimal hsianshun biceps ablation from existing 9-fold artifacts found the first large regression before event confirmation:

| Increment | GT | Pred | Count MAE | Bias | Rep F1 | Phase Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| cropped-like legacy `threshold=0.5`, no mask/event | `27` | `24` | `1.00` | `-1.00` | `0.902` | `0.766` |
| legacy `threshold=0.7`, soft online | `27` | `6` | `7.00` | `-7.00` | `0.218` | `0.446` |
| legacy `threshold=0.7` + confirmed3/fixed-soft | `27` | `7` | `6.67` | `-6.67` | `0.133` | `0.518` |
| periodic `threshold=0.7`, no event/fixed-soft | `27` | `12` | `5.00` | `-5.00` | `0.390` | `0.491` |
| periodic `threshold=0.7` + event gap1000/fixed-soft | `27` | `12` | `5.00` | `-5.00` | `0.390` | `0.489` |

Interpretation from the minimal ablation: the earliest big biceps failure is the move to a stricter full-session active gate around `threshold=0.7`, not event confirmation. Periodic features are better than the legacy `threshold=0.7` gate but still under-cover biceps compared with the cropped-like `threshold=0.5` baseline. Active-mask bridging is a follow-up recovery mechanism, not the first root cause.

The debug run added per-stream diagnostics to `scripts/evaluate_realtime_soft_top5_pipeline.py`: active coverage, active segment count, pre/post event rep counts, and compact event-confirmation summaries. On selected non-yanz biceps folds (`_tsenyu_temp`, `hsianshun`, `kevin`), all biceps errors had `dropped_reps=0`; the reps were already missing before event confirmation. The worst case, `hsianshun/db_biceps_curl/set0`, had only `8.3%` active coverage and produced only `2/10` reps.

Active-mask bridging smoke on the same selected folds:

| Setting | Biceps n | GT | Pred | Count MAE | Bias | Exact | Within-1 | Rep F1 | Phase Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| periodic `threshold=0.7`, no bridge | `9` | `97` | `75` | `2.44` | `-2.44` | `0.111` | `0.556` | `0.537` | `0.602` |
| periodic `threshold=0.7`, `bridge=300` | `9` | `97` | `85` | `1.56` | `-1.33` | `0.222` | `0.667` | `0.872` | `0.765` |
| periodic `threshold=0.6`, `bridge=300` | `9` | `97` | `90` | `1.22` | `-0.78` | `0.444` | `0.667` | `0.896` | `0.787` |

Interpretation: the full-session biceps regression is mostly caused by stricter active gating under-covering and then fragmenting smooth/low-amplitude biceps motion. Bridging short inactive gaps recovers much of the lost biceps phase quality and rep matching, but the simpler root-cause ablation says we should first revisit the active threshold/hysteresis policy before treating bridge as the main fix. This needs a full 9-fold check because lower thresholds or bridge settings can increase rest-only false positives, but biceps should not be treated as a fundamental phase-model failure.

## Interpretation
- The current gate fails because it detects motion, not confirmed workout-set structure.
- Periodicity helps but rest-after-set can contain periodic-looking transitions.
- A temporal CNN learns set continuity better than RF windows, but still needs a precision verifier for rest-only streams.
- CNN+periodic hybrid verification is useful as a secondary precision filter, but it trades away too much set recall when used as a hard segment rejector.
- Event-level confirmation should share the action-lock delay: early reps are buffered and then released retroactively once the event is confirmed. Current min-rep confirmation improves false positives but still under-counts and leaves rest-tail leakage.
- Wider rep-gap event grouping (`1000` samples) is important. Narrow event grouping (`300` samples) splits real sets too often and causes under-counting.
- The rest-tail blocker should be measured with `new_rest_reps` or `post_grace_rest_reps`, not raw `rest_overlap_reps`; many overlaps are boundary-crossing final reps that can be backdated.
- Post-hoc set confirmation removes many one-off false reps, but rest-tail leakage often contains enough reps to pass confirmation.
- A correct full-session solution needs an event-level active segmenter trained on `set + rest_after_set` timelines, not only window labels.

## Next Hypotheses
- H1: Train a temporal active segmenter on full timelines with labels `exercise_active`, `transition_motion`, and `rest_static`, then collapse to active/non-active for deployment.
- H2: Add phase-CNN uncertainty/entropy and rep regularity as confirmation features, because false rest reps should have weaker sustained C/E alternation quality than real sets.
- H3: Use action lock as a set-level verifier after candidate set onset, not as a per-window gate. Per-window action-active gating was too harsh.
- H4: For deployment, use a candidate buffer: do not display reps until a set is confirmed, then release buffered reps; if the candidate expires, drop it.
- H5: Replace hard CNN+RF segment rejection with softer candidate scoring. Candidate active starts from CNN, but final set confirmation should combine periodic/RF, phase-quality evidence, and action confidence rather than immediately deleting weak segments.
