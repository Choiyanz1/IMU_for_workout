# 2026-05-13 - Autoresearch Literature Review For Rep Segmentation Models

## Scope

This note reviews published models that are relevant to the repo's current main
question:

- how to improve **rep cutting quality** for resistance-training IMU data,
- under **single-IMU** and preferably **causal / deployable** constraints.

The user asked specifically for models that can achieve high-quality repetition
segmentation or counting, preferably from strong venues, and for a conclusion on
which is best suited to this project.

## Autoresearch Workflow Note

The `autoresearch` skill loader failed in this environment, so the skill's
`SKILL.md` was read directly and the bootstrap process was followed manually:

1. multi-source literature search
2. shortlist formation
3. method/metric/fit comparison
4. structured notes written to `literature/`

## Search Sources

- CrossRef
- Google Scholar
- direct publisher pages when abstracts/highlights were accessible

## Core Comparison Rule

The papers below are **not** directly comparable as a single leaderboard because
they use different:

- exercises
- sensor placements
- subject split rules
- modalities
- counting vs segmentation metrics

So the ranking here optimizes for **fit to our task**, not only for raw reported
performance or venue prestige.

## Shortlist

### 1. RecoFit

- Paper: Morris et al., `RecoFit: using a wearable sensor to find, recognize, and count repetitive exercises`
- Venue: CHI 2014
- DOI: `10.1145/2556288.2557116`

Why it matters:

- Probably the closest classic reference to our actual problem setup.
- Explicitly decomposes the task into:
  - segmentation
  - recognition
  - counting

Commonly cited performance:

- exercise-period segmentation precision / recall above `95%`
- `93%` of sets counted within `+-1 rep`

Why it fits us:

- segmentation-first
- single wearable sensor
- directly about repetitive workout tracking

Why it is not a full solution copy:

- older pipeline
- not phase-aware
- not obviously optimized for our boundary MAE metrics

### 2. MM-Fit

- Paper: Stromback et al., `MM-Fit`
- Venue: PACM IMWUT 2020
- DOI: `10.1145/3432701`

Why it matters:

- Strongest top-tier wearable exercise logging benchmark in the shortlist.
- Unseen-subject protocol is much closer to the discipline we want.

Accessible reported details:

- recognition up to `96%` across modalities
- `94%` using smartwatch only
- Google Scholar snippet exposes repetition counting baseline MAE of `0.34 reps/set` for smartwatch gyroscope

Why it fits us:

- wearable exercise logging
- smartwatch-only baseline is relevant
- strong benchmark protocol reference

Why it is not the single best architectural fit:

- more set-level counting emphasis than exact rep-boundary timing

### 3. Lee et al. (Information Fusion 2024)

- Paper: `Multimodal sensor fusion models for real-time exercise repetition counting with IMU sensors and respiration data`
- Venue: Information Fusion 2024
- DOI: `10.1016/j.inffus.2023.102153`

Why it matters:

- Strongest raw venue in this shortlist.
- Broadest exercise coverage among the directly relevant papers.

Accessible reported details:

- multimodal deep model
- smart-earbud IMU + respiration audio
- `30` exercise types
- fusion outperforms sensor-only counting

Why it fits us:

- real-time repetition counting
- strong evidence that additional modalities help

Why it is **not** the best current fit:

- hardware mismatch: we do not currently assume audio/respiration on the board path
- optimizes count more directly than explicit boundary timing

### 4. ExerSense

- Paper: Ishii et al., `ExerSense: Real-Time Physical Exercise Segmentation, Classification, and Counting Algorithm Using an IMU Sensor`
- Venue: Springer chapter 2020
- DOI: `10.1007/978-981-15-8944-7_15`

Reported results:

- segmentation precision `97.9%`
- segmentation recall `93.9%`
- segmentation F1 `95.9%`

Why it matters:

- very explicit segmentation-first design
- simple peak proposal + weighted DTW template matching

Why it fits us:

- highly deployable classical baseline
- useful if explicit rep proposals beat dense neural decoding

Why it is not the main reference:

- simpler exercises than our dumbbell resistance set
- weaker venue and narrower domain match than RecoFit / MM-Fit

### 5. Soro et al. (Sensors 2019)

- Paper: `Recognition and Repetition Counting for Complex Physical Exercises with Deep Learning`
- DOI: `10.3390/s19030714`

Reported results:

- recognition accuracy `99.96%`
- `91%` of sets counted within `+-1 rep`

Why it matters:

- direct deep-learning workout counting reference

Why it is secondary for us:

- still mainly count-oriented, not boundary-first

### 6. Prabhu et al. (Sensors 2020)

- Paper: `Recognition and Repetition Counting for Local Muscular Endurance Exercises in Exercise-Based Rehabilitation`
- DOI: `10.3390/s20174791`

Reported results:

- recognition F1 `97.18%`
- `90%` of sets within `+-1 rep`

Why it matters:

- clear classical-vs-CNN comparison

Why it is secondary for us:

- rehab exercise domain
- count-oriented headline metric

### 7. WeakCounterF

- Paper: `Few-Shot and Weakly Supervised Repetition Counting With Body-Worn Accelerometers`
- DOI: `10.3389/fcomp.2022.925108`

Why it matters:

- lowers annotation cost by using only weak count labels per segment
- target-user adaptation is explicit

Why it is not the main next move:

- our current bottleneck is not label cost first; it is boundary collapse
- this line is stronger for count supervision than precise boundary timing

## Which One Is Best?

### Best raw venue

- `Information Fusion 2024`

### Best top-tier wearable benchmark reference

- `MM-Fit` (PACM IMWUT 2020)

### Best direct fit for **our exact repo objective**

- `RecoFit`

Reason:

- It is the cleanest match to a segmentation-first wearable exercise pipeline.
- It is closer to our boundary-first evaluation than pure count-regression work.
- It does not require extra modalities.

## Final Recommendation For This Repo

The best outside reference to prioritize is **not** the multimodal Information
Fusion model, even though it has the strongest venue.

The best practical external direction is:

1. use `RecoFit` as the main conceptual reference for segmentation -> recognition -> counting
2. use `MM-Fit` as the benchmark/reporting reference for unseen-subject wearable evaluation
3. use `ExerSense` as the simplest classical explicit-proposal baseline to reproduce quickly
4. defer multimodal fusion (`Information Fusion 2024`) until single-IMU boundary quality is no longer the main bottleneck

## Concrete Research Direction Suggested By This Review

For this repo, the strongest next literature-grounded experiment is:

1. keep the current **phase-only causal TCN** branch as the learned dense phase model
2. add a **RecoFit/ExerSense-inspired explicit rep proposal baseline**
3. compare both on the repo's real priority metrics:
   - `start_mae_ms`
   - `end_mae_ms`
   - `transition_mae_ms`
   - `rep precision / recall / f1`
   - exact-count / over / under / zero-TP streams

In other words:

- literature does **not** currently justify jumping first to more complex multimodal counting systems
- literature **does** justify adding a strong segmentation-first external baseline against the current DS-MS-TCN family
