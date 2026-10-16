# V14-tiny: 24-hour single-seed test (architecture and continual learning)

Status: **pre-registered, not yet run** (written 15 Oct 2026 from 02:20 IST). Design: `../v14_Architecture.md`.
Decisions and deviations: `DEVIATIONS.md`. Advisor consultations: `ADVISOR_LOG.md`.

## 1. What runs

V14-tiny is V13-tiny-v2 with V14's next-concept target (an EMA twin; the target view tau = [a; b], local half and the
global change beyond its per-family linear local part; one head for horizons 1-4; document-centred SIGReg plus the
uncentred test on b), shrunk for throughput at the user's request: E2 R = 1, a 4-layer decoder, a 2-layer E3
(DEVIATIONS item 1). 6,807,755 trained parameters; the twin's 3,479,720 exist only in training.

Timeline (fixed now): `run_base_v14.sh` waits for V13-tiny's post pipeline (`../v13_tiny/runs/post/ALL_DONE`), runs two
60-step smoke tests that decide activation checkpointing (DEVIATIONS item 7), then trains for 11 hours of elapsed
training time (V13-tiny's budget). `post_base_v14.sh` then runs the final base evaluation, `diagnostics_v14.py`, the
continual stage (sequential LoRA, V14's §9.4 default, replay; 100 steps per phase on V13-tiny's cached rows), the
continual matrix and the tables. Everything must finish within 24 hours of the base run's start.

## 2. Pre-registered readings (v14_Architecture.md §11, copied unchanged; written before any V14 result)

| # | Question | Pass | Computed by |
|---|---|---|---|
| V1 | Does the head beat copy-current and the marginal? | On the global half at horizon 1, the head's accuracy beats both baselines by at least 10 points: persistence (the current group's code) and the family marginal (each segment's most frequent word in that family on training rows); in at least 3 of the 4 reasoning families, at the final checkpoint | `diagnostics_v14.py` on fresh dev-time rows |
| V2 | Does the target need the concept path? | The u_g probe beats the local-only probe (L_enc at row 4g+3) by at least 2 points on the global half in at least 2 families | same |
| V3 | Does the concept module compute? | §5.6's per-half probe test on the global half: u_g beats p_g by at least 5 points; reported per family, pass or fail, beside V13-tiny's result in the same family | same |
| V4 | No slow-feature collapse | chi of the twin's normalised global half at least 0.5 at the final checkpoint, in every family. The uncentred test on b is on from step 0, so a pass shows the barrier held, not that b is naturally fast; the evidence about slowness is V1 against the marginal and the action-sensitivity read | same |
| V5 | Are codes stable? | Code reassignment on fixed inputs between snap_010 and snap_050 is lower in V14 than in V13-tiny; snap_030-snap_070 reported beside it, not compared | same, on both runs' snapshots |
| V6 | What does it cost? | Text NLL at matched training time within 0.02 nats of V13-tiny's; the Reasoning score reported against V13-tiny and the baseline. **Now a joint effect of the target redesign and the shrink** (DEVIATIONS item 1) | `final_eval.py` and the training logs |
| V7 | Continual learning | §9.4's table, with Dataset_v13 §10.3's thresholds; against V13-tiny's matrix a joint effect as V6 | `cl_eval.py`, `make_report.py` |

A V14 that passes V1-V4 and fails V6 has a better concept path at a price. One that fails V1 or V4 has not fixed what it
was built to fix. Every result is one seed: "observed, seed-unreplicated".

## 3. Throughput (the user's second goal)

Filled from the smoke tests and the run's own log: tokens/s (median per 50-step interval and the average over the
11 hours), seconds per step, the wait for rows, memory, against V13-tiny's 3,234 (launch benchmark) and 2,471 (11-hour
average). Contended benchmark before launch (15 Oct 02:00-02:13, while V13-tiny's continual stage held the GPU;
relative only): median 3.29 s/step for V14-tiny against 3.60 for V13-tiny under the same conditions, best step 2.36
against 2.94; peak memory 5.3 against 5.4 GB with checkpointing.

## 4. Base run

(to be filled)

## 5. Readings V1-V6

(to be filled)

## 6. Continual stage (V7)

(to be filled)
