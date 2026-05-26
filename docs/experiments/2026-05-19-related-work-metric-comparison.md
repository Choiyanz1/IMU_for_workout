# Related Work Metric Comparison for IMU Workout Rep-Structured Analysis

Date: 2026-05-19

## Goal

Compare related exercise recognition / repetition counting / repetition segmentation systems against our current IMU pipeline using the target metrics requested for this project:

- Count MAE
- Rep IoU-F1@50
- Phase IoU-F1@50
- C/E ratio MAE

## Our Current Reference Point

| Method | Sensor / Input | Real-time oriented | Count MAE | Rep IoU-F1@50 | Phase IoU-F1@50 | C/E ratio MAE | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Raw6 1D Causal CNN + MA25/Viterbi | single 6-axis IMU stream | Yes | 1.505 | 0.860 | 0.720 | 0.670 | 9-fold LOSO, 220 held-out streams |
| Raw6 1D Causal CNN + selective `top5_p5` merge | single 6-axis IMU stream | Yes | **0.973** | **0.874** | 0.716 | **0.601** | Best current core pipeline; phase labels unchanged by merge |

Important caveat: current core pipeline assumes action context for per-action active gating and selective duration merge. The action-recognition branch and rest-aware active detector still need integrated streaming validation before full-session deployment claims.

## Literature Comparison Table

`NR` means the paper did not report that metric. `N/A` means the task does not produce that output. Metrics are not always directly comparable because datasets, sensors, split protocols, and MAE definitions differ.

| Work | Task & data | Method / model | Real-time? | Count MAE | Rep IoU-F1@50 | Phase IoU-F1@50 | C/E ratio MAE | Comparison to ours |
|---|---|---|---:|---:|---:|---:|---:|---|
| Morris et al. 2014, RecoFit, CHI | Wearable repetitive exercise detection, recognition, counting | Handcrafted features + segmentation/counting pipeline | Oriented to wearable use | NR in searched metadata | NR | N/A | N/A | Classic baseline for find/recognize/count, but does not report rep IoU or phase metrics |
| Mortazavi et al. 2014, BSN | Smartwatch best-axis exercise recognition/counting | Axis selection + signal processing / ML | Yes | NR | NR | N/A | N/A | Useful simple heuristic baseline; narrower output than ours |
| Soro et al. 2019, Sensors | 10 complex full-body exercises, smartwatch | End-to-end deep model / 1D CNN style | Not clearly streaming | NR; reports ±1 count in 91% sets | NR | N/A | N/A | Strong recognition/counting baseline; no rep boundary or C/E phase output |
| Prabhu et al. 2020, Sensors / INSIGHT-LME | Rehab/local muscular endurance recognition + count | CNN / AlexNet-style and time-series CNN vs traditional ML/peak detection | Remote setting oriented | NR; reports ±1 count in 90% sets | NR | N/A | N/A | Comparable for recognition/counting, but does not report segmentation quality or C/E ratio |
| Strömbäck et al. 2020, MM-Fit | Multimodal exercise logging, activity segmentation, recognition, rep counting | CNN-based multimodal learning across smartwatch/phone/earbud/video/skeleton | Not board-focused | NR in paper-search metadata; Hsu cites baseline normalized MAE 0.67 on MM-Fit | NR | N/A | N/A | Multimodal stronger context; our advantage is single-IMU rep-structured C/E output |
| Ishii et al. 2020, ExerSense | Real-time exercise segmentation, classification, counting | Correlation / template-style IMU algorithm | Yes | NR in metadata | Reports exercise-period segmentation P/R/F1, not rep IoU-F1 | N/A | N/A | Strong practical baseline for segmentation/classification/counting from a single IMU |
| Nishino et al. 2022, WeakCounterF | Few-shot weakly supervised wearable repetition counting | Attention model with weak count labels and target-user adaptation | Not primarily board-focused | NR in metadata | N/A; count-only weak supervision | N/A | N/A | Useful comparison for low-label-counting setup, but not boundary/C-E focused |
| Hsu et al. 2021, Viewpoint-Invariant Exercise Repetition Counting | Skeleton-based counting on UI-PRMD and MM-Fit | Self-similarity + spectrogram / frequency integration | Yes, CPU ~0.001s/input | 0.06 MAE as reported on MM-Fit and UI-PRMD; definition differs from absolute set-count MAE | N/A; counting only | N/A | N/A | Excellent count-only result, but no rep boundaries or phase/C/E outputs |
| Abedi et al. 2023, CRV / arXiv | Rehab repetition segmentation/counting from skeletal joints | LSTM + 1D CNN, density map / binary / count outputs | Not explicitly edge | KIMORE 0.531; UI-PRMD 0.58; IntelliRehabDS 0.72 absolute count MAE | Reports mean segment IoU, not IoU-F1: KIMORE 0.689, UI-PRMD 0.82, IntelliRehabDS 0.68 | N/A | N/A | Strongest related segmentation/counting paper, but skeleton/depth-camera modality and no C/E ratio |
| Shang et al. 2024, DS-MS-TCN Otago | Single waist IMU OEP micro/macro exercise recognition | Dual-scale multi-stage TCN with micro labels | Not explicitly edge; seq-to-seq | N/A; recognition not count | Closest metric: macro segment IoU-F1@50 per activity 0.633-0.923, all >0.60 | N/A for C/E; micro/macro exercise labels instead | N/A | Strong related evidence for micro-label seq-to-seq IMU segmentation; task is rehab exercise recognition, not resistance C/E rep analysis |
| Lee et al. 2024, Information Fusion | Real-time exercise repetition counting | Multimodal IMU + respiration fusion | Yes | NR in available metadata | NR | N/A | N/A | Relevant real-time counting baseline; uses extra modality and does not expose C/E ratio metrics in metadata |
| Lim & Lee 2024, Few-shot IMU repetition counting | IMU counting for unseen exercises | Siamese / triplet-loss few-shot peak-vs-nonpeak detection | Real-time candidate | Does not report absolute MAE; reports 86.8% probability of counting at least 10 reps accurately, error bins | N/A; counting only | N/A | N/A | Strong generalization story for unseen exercises; narrower output than ours |
| Oberhofer et al. 2021, StrengthControl | Apple Watch strength exercise recognition/counting/1RM | Commercial smartwatch app validation | Yes | NR; count accurate for squat/deadlift, poor for bench press | NR | N/A | N/A | Good applied strength-training reference; no rep boundary/phase metrics |
| King et al. 2025, AI hypertrophy coaching | Single wrist IMU rep segmentation + near-failure/RIR | ResNet segmentation + LSTM/context classifier, edge deployment | Yes | NR | Segmentation F1 0.83, not explicitly IoU-F1 | N/A | N/A | Very close deployment story; focuses near-failure feedback rather than C/E ratio |

## Metadata and Metric Extraction Status

This pass verified core bibliographic metadata using CrossRef, arXiv, Semantic Scholar, or OpenAlex. Some detailed metric values still require checking against the full paper tables before camera-ready use.

| Work | Metadata status | Metric extraction status |
|---|---|---|
| Morris et al. 2014 | CrossRef verified, DOI `10.1145/2556288.2557116` | Need full-text check for count-error definition |
| Mortazavi et al. 2014 | CrossRef verified, DOI `10.1109/BSN.2014.21` | Need full-text check for per-exercise counting results |
| Soro et al. 2019 | CrossRef abstract verified, DOI `10.3390/s19030714` | ±1 in 91% sets verified from abstract; no MAE in abstract |
| Prabhu et al. 2020 | CrossRef abstract verified, DOI `10.3390/s20174791` | F1 97.18% and ±1 in 90% sets verified from abstract; no MAE in abstract |
| Strömbäck et al. 2020 | CrossRef abstract verified, DOI `10.1145/3432701` | Activity recognition values verified from abstract; repetition-count MAE needs full-text check |
| Ishii et al. 2020 | CrossRef verified, DOI `10.1007/978-981-15-8944-7_15`; Sensors extension DOI `10.3390/s21010091` | Segmentation P/R/F1 values come from prior project literature notes and need full-text table check |
| Nishino et al. 2022 | CrossRef abstract verified, DOI `10.3389/fcomp.2022.925108` | Detailed count metrics unavailable from CrossRef metadata |
| Hsu et al. 2021 | arXiv verified, DOI `10.48550/arXiv.2107.13760` | MAE 0.06 and OBOA 0.94/0.95 verified from abstract; MAE is not directly comparable to absolute set-count MAE without definition check |
| Abedi et al. 2023 | CrossRef verified, DOI `10.1109/CRV60082.2023.00044` | Reported MAE and mean segment IoU values need full-text table check |
| Shang et al. 2024 | CrossRef verified, DOI `10.1109/JBHI.2024.3455426` | Segment IoU-F1 range needs full-text table check |
| Lee et al. 2024 | CrossRef verified, DOI `10.1016/j.inffus.2023.102153` | Detailed count metrics unavailable from CrossRef metadata |
| Lim and Lee 2024 | arXiv/Semantic Scholar verified, DOI `10.48550/arXiv.2410.00407` | 86.8% probability for counting at least 10 reps accurately verified from abstract |
| Oberhofer et al. 2021 | CrossRef abstract verified, DOI `10.3390/sports9090118` | Recognition/count findings verified from abstract; no MAE or boundary metrics in abstract |
| King et al. 2025 | Semantic Scholar/OpenAlex verified, DOI `10.1109/ICMLA66185.2025.00179` | Segmentation F1 0.83, near-failure F1 0.82, 1.6 Hz simulated real-time rate, and edge latency values verified from abstract |

## Expanded Common-Metric Comparison

The table below uses metrics that are more commonly reported in exercise-recognition/counting literature. This is useful for the related-work section because many papers do not report our stricter phase-aware metrics.

| Work / method | Modality | Subjects / exercises | Recognition metric | Count exact / OBOA / within-1 | Count MAE | Segmentation metric | Real-time / latency | Fairness note |
|---|---|---|---|---|---:|---|---|---|
| **Ours: raw6 CNN + `top5_p5`** | single 6-axis IMU | 9-fold LOSO, 220 streams, 8 actions | Current action recognition not integrated | Exact **59.1%**, within-1 **75.5%** | **0.973 reps/set** | Rep P/R/F1 **88.8/86.0/87.4%**; Phase acc **76.1%**; Phase macro F1 **75.6%** | Causal 3s window; streaming-style replay demonstrated | Best current structured pipeline; assumes action context |
| **Ours: raw6 CNN + count calibration** | single 6-axis IMU | Same 220 streams | Current action recognition not integrated | Exact 57.3%, within-1 **85.9%** | **0.759 reps/set** | Same raw boundaries; calibration does not change rep F1 | Tiny post-hoc display layer | Best count-only display result, but not a boundary solution |
| **Ours: raw6 CNN raw decoder** | single 6-axis IMU | Same 220 streams | Current action recognition not integrated | Exact 51.8%, within-1 67.3% | 1.505 reps/set | Rep P/R/F1 81.4/91.2/86.0%; Phase acc 76.4%; Phase macro F1 75.9% | Causal 3s window | Useful no-merge baseline |
| **Ours: rest-aware active detector probe** | single 6-axis IMU | held-out `yushuan`, 4 set+rest snippets | Active/rest detection, not exercise ID | N/A | N/A | Active F1 **97.3%** at threshold 0.6; false-active rest 2.62s/20s | 1.0s RF window, 0.1s stride | Preliminary full-session gate, not full 9-fold yet |
| RecoFit, Morris et al. 2014 | body-worn IMU | repetitive exercises | NR in verified metadata | Commonly cited: 93% within ±1 rep | NR | exercise-period segmentation P/R >95% commonly cited | Wearable use oriented | Closest classic conceptual pipeline; needs full-text metric check |
| ExerSense, Ishii et al. 2020 | IMU wearables | 5 exercise types, multiple device positions | Classification included; exact value needs full-text check | NR in verified metadata | NR | Segmentation P/R/F1 97.9/93.9/95.9% from project notes | Explicit real-time algorithm | Very comparable as segmentation/classification/counting pipeline, but metrics are exercise-period not rep/C-E phase |
| Soro et al. 2019 | smartwatch IMU | 10 CrossFit-style exercises | Accuracy **99.96%** | **91% within ±1 rep** | NR | NR | Not clearly board-streaming | Strong recognition/counting reference, no rep-boundary metric |
| Prabhu et al. 2020 | wearable IMU | local muscular endurance rehab exercises | F1 **97.18%** | **90% within ±1 rep** | NR | NR | Remote rehab oriented | Good CNN-vs-classical reference, but count-oriented |
| MM-Fit, Strömbäck et al. 2020 | smartwatch/phone/earbud/video/skeleton | multimodal workout dataset | 96% all modalities; 94% smartwatch-only | NR in verified metadata | Full-text check needed; project notes mention smartwatch gyro MAE 0.34 reps/set | Activity segmentation supported; rep IoU not in metadata | Not board-focused | Strong benchmark but multimodal and task framing differs |
| Hsu et al. 2021 | video skeleton | UI-PRMD, MM-Fit | N/A | OBOA 0.94 on MM-Fit, 0.95 on UI-PRMD | 0.06 as reported | Counting only | CPU ~0.001s/input reported | Strong count-only baseline, but skeleton modality and MAE definition differ |
| Abedi et al. 2023 | skeletal joints | KIMORE, UI-PRMD, IntelliRehabDS | N/A | NR | 0.531/0.58/0.72 absolute count MAE | mean segment IoU 0.689/0.82/0.68 | Not explicitly edge | Strong segmentation/counting baseline, but camera/skeleton rather than IMU |
| DS-MS-TCN, Shang et al. 2024 | single waist IMU | Otago Exercise Program | Recognition/segmentation task | N/A | N/A | macro segment IoU-F1@50 0.633-0.923 by activity | seq-to-seq, not explicitly edge | Closest action-segmentation metric style, but not rep counting/C-E |
| Lee et al. 2024 | smart-earbud IMU + respiration | 30 exercise types | NR in metadata | NR in metadata | NR in metadata | N/A | Explicit real-time counting | Strong venue and real-time count reference; extra respiration modality |
| Lim and Lee 2024 | IMU | 28 exercises, unseen-exercise counting | peak vs non-peak few-shot classification | 86.8% probability of accurately counting >=10 reps | Does not report absolute MAE in abstract | N/A | Real-time candidate | Strong generalization reference, narrower count-only output |
| Oberhofer et al. 2021 | Apple Watch | squat/deadlift/bench press | exercise recognition 88.4% of sets | count accurate for squat/deadlift, poor for bench press | NR | NR | App-based real-time analysis | Applied strength-training validation, no boundary metrics |
| King et al. 2025 | single wrist 6-axis IMU | 13 participants, preacher curls to failure | near-failure classifier F1 0.82 | NR | NR | rep segmentation F1 **0.83** | 1.6 Hz simulated real-time; 112 ms Raspberry Pi 5, 23.5 ms iPhone 16 | Very close edge story, but single exercise and no C/E phase ratio |

### Best Metrics To Use In A Paper Table

Use these as the main cross-paper columns because they appear most often:

- **Recognition accuracy/F1**: useful for Soro, Prabhu, MM-Fit, Oberhofer, King; mark ours as pending unless the action branch is integrated.
- **Within-1 / OBOA**: the cleanest count-comparison column; ours `top5_p5` is 75.5%, count calibration is 85.9%.
- **Absolute Count MAE**: use where available; ours `top5_p5` is 0.973 reps/set, display calibration is 0.759 reps/set, Abedi reports 0.531-0.72 on skeleton rehab datasets.
- **Segmentation F1 / IoU**: use as a loose grouping column, but label the unit carefully: exercise-period segmentation, rep segmentation, action segment IoU-F1, and phase segment IoU-F1 are not interchangeable.
- **Real-time / latency**: useful because our contribution is deployment-oriented; King et al. and Hsu et al. report concrete runtime numbers, while ours needs an embedded timing benchmark.
- **Sensor burden**: single 6-axis IMU is an advantage versus skeleton/video or IMU+respiration, even if some count-only methods report lower MAE.

## What This Comparison Shows

1. **Count MAE alone is not enough for our story.** Some skeleton/video methods report very low counting errors, especially Hsu et al. on MM-Fit/UI-PRMD, but the metric definition and modality differ from our absolute set-count MAE. They also do not produce rep boundaries or C/E phase structure.

2. **Rep segmentation is reported in fewer papers.** Abedi et al. report segment IoU and MAE-F for rep boundaries, and DS-MS-TCN reports segment IoU-F1@50 for exercise segments. These are the closest metrics to our Rep IoU-F1@50.

3. **Phase IoU-F1@50 and C/E ratio MAE are mostly absent from prior work.** Existing systems usually report recognition accuracy/F1, count accuracy, OBOA, count MAE, or segment IoU. They rarely evaluate concentric/eccentric phase segmentation or per-rep phase balance.

4. **Our differentiator is rep-structured feedback, not raw count SOTA.** The strongest story is not that our count MAE is lower than all prior work. It is that we provide a single-IMU, causal, deployment-oriented pipeline that outputs count plus rep boundaries plus C/E phase structure.

5. **Our closest methodological neighbors are DS-MS-TCN and Abedi et al.** DS-MS-TCN supports the idea of micro-label seq-to-seq IMU segmentation and IoU-F1 evaluation. Abedi et al. supports density/sequence learning for rep segmentation and counting. Our contribution can be framed as bringing this rep-structured idea into resistance training with C/E phase metrics and low-latency IMU inference.

## Recommended Comparison Plan for the Paper

### Primary Baselines To Implement On Our Dataset

| Baseline | Why it matters | Expected reported metrics |
|---|---|---|
| Peak detection / threshold heuristic | Classical counting baseline, easy to explain | Count MAE, Rep IoU-F1@50 if boundaries are emitted |
| DTW / template matching | Common repetition segmentation/counting baseline | Count MAE, Rep IoU-F1@50 |
| Random Forest window classifier | Strong lightweight non-neural baseline | Count MAE, Rep IoU-F1@50, maybe Phase IoU-F1@50 |
| DS-MS-TCN-style multi-stage TCN | Closest seq-to-seq wearable segmentation literature | Count MAE, Rep IoU-F1@50, Phase IoU-F1@50, C/E ratio MAE |
| DeepConvLSTM | Standard wearable HAR deep baseline | Same four metrics if adapted to phase labels |
| Non-causal 1D CNN / TCN | Tests cost of causal constraint | Same four metrics |
| Transformer/HART-like model | Modern sensor sequence baseline | Same four metrics + latency/model size |

### Metrics To Emphasize

Use a two-tier evaluation:

- **Standard comparability metrics:** count MAE, exact count, within-1 count, Rep IoU-F1@50.
- **Our contribution metrics:** Phase IoU-F1@50, C/E ratio MAE, streaming latency, model size, rest false-active rate.

## Suggested Positioning Sentence

Most prior wearable exercise systems optimize exercise recognition or final repetition count, while rep-level temporal structure is either not reported or limited to start/end segmentation. Our system targets resistance-training feedback as a structured streaming segmentation problem: it reports live count, rep boundaries, and concentric/eccentric phase balance from a single 6-axis IMU, evaluated with count error, rep IoU-F1, phase IoU-F1, and C/E ratio MAE.

## References

- Morris et al. RecoFit: using a wearable sensor to find, recognize, and count repetitive exercises. CHI 2014. DOI: `10.1145/2556288.2557116`.
- Mortazavi et al. Determining the Single Best Axis for Exercise Repetition Recognition and Counting on SmartWatches. BSN 2014. DOI: `10.1109/BSN.2014.21`.
- Soro et al. Recognition and Repetition Counting for Complex Physical Exercises with Deep Learning. Sensors 2019. DOI: `10.3390/s19030714`.
- Prabhu et al. Recognition and Repetition Counting for Local Muscular Endurance Exercises in Exercise-Based Rehabilitation. Sensors 2020. DOI: `10.3390/s20174791`.
- Strömbäck et al. MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging across Sensing Devices. IMWUT 2020. DOI: `10.1145/3432701`.
- Ishii et al. ExerSense: Real-Time Physical Exercise Segmentation, Classification, and Counting Algorithm Using an IMU Sensor. Smart Innovation, Systems and Technologies 2020. DOI: `10.1007/978-981-15-8944-7_15`.
- Ishii et al. ExerSense: Physical Exercise Recognition and Counting Algorithm from Wearables Robust to Positioning. Sensors 2020. DOI: `10.3390/s21010091`.
- Nishino et al. Few-Shot and Weakly Supervised Repetition Counting With Body-Worn Accelerometers. Frontiers in Computer Science 2022. DOI: `10.3389/fcomp.2022.925108`.
- Hsu et al. Viewpoint-Invariant Exercise Repetition Counting. arXiv 2021. URL: `https://arxiv.org/abs/2107.13760`.
- Abedi et al. Rehabilitation Exercise Repetition Segmentation and Counting Using Skeletal Body Joints. CRV 2023 / arXiv. DOI: `10.1109/CRV60082.2023.00044`.
- Shang et al. DS-MS-TCN: Otago Exercises Recognition With a Dual-Scale Multi-Stage Temporal Convolutional Network. IEEE JBHI 2024. DOI: `10.1109/JBHI.2024.3455426`.
- Lee et al. Multimodal sensor fusion models for real-time exercise repetition counting with IMU sensors and respiration data. Information Fusion 2024. DOI: `10.1016/j.inffus.2023.102153`.
- Lim and Lee. Intelligent Repetition Counting for Unseen Exercises: A Few-Shot Learning Approach with Sensor Signals. arXiv 2024. URL: `https://arxiv.org/abs/2410.00407`.
- Oberhofer et al. Validation of a Smartwatch-Based Workout Analysis Application in Exercise Recognition, Repetition Count and Prediction of 1RM. Sports 2021. DOI: `10.3390/sports9090118`.
- King et al. Rep Smarter, Not Harder: AI Hypertrophy Coaching with Wearable Sensors and Edge Neural Networks. ICMLA 2025. DOI: `10.1109/ICMLA66185.2025.00179`.
