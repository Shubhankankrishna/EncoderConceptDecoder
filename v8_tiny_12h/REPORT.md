# V8 full-exposure 12-hour pilot

Full-exposure diagnostic pilot; low score thresholds did not stop training

Status: **completed_extended_pilot**. One independent training seed; this is a pilot, not seed confirmation or a scaling test.

Policy: diagnostic thresholds; no competence-based stopping. Scores and confidence thresholds are evidence readouts; running a stage does not mean that it succeeded.

The user authorized continuation after the compiled decoding parity failure. Eager static decoding passed the same tolerance on trained weights; training compilation and the completed alpha-zero checkpoint were preserved. The original clock/deadline were retained. The failed source, preflights, lock and report are archived under `spec/recovery_2026-10-06`. This is an execution revision, not an additional training seed.

A later development-result export failed on a NumPy boolean. Competence and coverage flags now return native Python booleans with unchanged numerical criteria. Both completed 24,000-update foundations are reused with checksum and training-provenance checks; the completed alpha-zero development result retains its earlier evaluation provenance. The failed NCP evaluation is rerun. The source, failure and unfinalized output are archived under `spec/recovery_json_2026-10-06`; the original deadline is unchanged.

## Foundation evidence

| Arm | Parameters | Standard dev | Lookup dev | Competence |
|---|---:|---:|---:|---|
| A_alpha0 | 2,448,640 | 77.2% | 100.0% | False |
| A_delayed_ncp | 2,448,640 | 80.5% | 100.0% | True |
| A_flat | 2,450,048 | 73.7% | 99.9% | False |

## Training exposure

- A_alpha0: 24,000 updates, 576,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_alpha0/train_summary.json).
- A_delayed_ncp: 24,000 updates, 576,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_delayed_ncp/train_summary.json).
- A_flat: 24,000 updates, 576,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_flat/train_summary.json).
- B_refit_dense: 6,000 updates, 144,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/B_refit_dense/train_summary.json).
- B_refit_recurrent: 6,000 updates, 144,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/B_refit_recurrent/train_summary.json).
- C_exploration: 6,000 updates, 144,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/C_exploration/train_summary.json).
- D_lsl_phase1: 2,000 updates, 0 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_lsl_phase1/train_summary.json).
- D_lsl_phase2: 2,000 updates, 0 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_lsl_phase2/train_summary.json).
- D_lsl_phase3: 2,000 updates, 0 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_lsl_phase3/train_summary.json).

## Acceptance and component results

- B_refit_dense: retention-qualified=False, coverage-qualified=not applicable, research-only=True.
- B_refit_recurrent: retention-qualified=False, coverage-qualified=not applicable, research-only=True.
- C_exploration: retention-qualified=True, coverage-qualified=True, research-only=False.
- D_lsl_phase1: evidence-qualified=False, skill change={'nll_delta': -0.013094942539400911, 'interval': [-0.018744404799677986, -0.00775569763166565], 'installation_interval': [0.001277343957765183, 0.005271817384314877]}; continued as an exploratory trajectory.
- D_lsl_phase2: evidence-qualified=False, skill change={'delta': 0.251953125, 'interval': [0.2013941216169769, 0.297653607900514], 'gains': 129, 'losses': 0, 'n': 512, 'method': 'Bonferroni exact binomial bounds for gain/loss probabilities; valid at zero discordance', 'installation': {'delta': 0.00390625, 'interval': [-0.03280647735848343, 0.040531944328384736], 'gains': 17, 'losses': 15, 'n': 512, 'method': 'Bonferroni exact binomial bounds for gain/loss probabilities; valid at zero discordance'}}; continued as an exploratory trajectory.
- D_lsl_phase3: evidence-qualified=False, skill change={'delta': 0.001953125, 'interval': [-0.008497557194133837, 0.012397457103173068], 'gains': 1, 'losses': 0, 'n': 512, 'method': 'Bonferroni exact binomial bounds for gain/loss probabilities; valid at zero discordance', 'installation': {'delta': -0.33203125, 'interval': [-0.38076064787153463, -0.27725503965177556], 'gains': 0, 'losses': 170, 'n': 512, 'method': 'Bonferroni exact binomial bounds for gain/loss probabilities; valid at zero discordance'}}; continued as an exploratory trajectory.
- C_exploration/coverage.json: True; [evidence](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/C_exploration/coverage.json).
- runs/D_always_open_summary.json: see phase decisions; [evidence](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_always_open_summary.json).
- runs/D_lsl_summary.json: see phase decisions; [evidence](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_lsl_summary.json).
- E_search/summary.json: partial_time_budget; [evidence](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/E_search/summary.json).

## Partial and skipped work

- D_always_open_phase1: Final-test reserve; no score-based stop.
- D_flat_lsl: No full-trajectory time allocation remains.
- B_no_teacher_control: Complete update budget will not fit; foundation exposure never silently reduced.

## Sealed endpoint measurements

- A_alpha0_sealed_test: completed; standard 1567/2048, interval [0.7462913359320973, 0.7829893233173673]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_alpha0_sealed_test/summary.json).
- A_delayed_ncp_sealed_test: completed; standard 1646/2048, interval [0.7859469507325959, 0.8203377086655016]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_delayed_ncp_sealed_test/summary.json).
- A_flat_sealed_test: completed; standard 1510/2048, interval [0.7178126407846765, 0.7559081702562807]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_flat_sealed_test/summary.json).
- B_refit_dense_sealed_test: completed; standard 1654/2048, interval [0.7899761378748931, 0.8241063949559773]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/B_refit_dense_sealed_test/summary.json).
- B_refit_recurrent_sealed_test: completed; standard 1657/2048, interval [0.7914877721193345, 0.8255189632488258]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/B_refit_recurrent_sealed_test/summary.json).
- C_exploration_sealed_test: completed; standard 1657/2048, interval [0.7914877721193345, 0.8255189632488258]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/C_exploration_sealed_test/summary.json).
- D_lsl_phase1_sealed_test: completed; standard 1659/2048, interval [0.792495739719343, 0.8264604640070105]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_lsl_phase1_sealed_test/summary.json).
- D_lsl_phase2_sealed_test: completed; standard 0/2048, interval [0.0, 0.0018722006050613785]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_lsl_phase2_sealed_test/summary.json).
- D_lsl_phase3_sealed_test: completed; standard 0/2048, interval [0.0, 0.0018722006050613785]; [all task/text results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_lsl_phase3_sealed_test/summary.json).

The 12-hour pilot intentionally does not complete the full draft matrix: additional independent training seeds, exact fresh V6 retraining, enlarged hard-context training, reverse continual order, IRC/CRC, alternate NCP coefficients, canonical-BFS label comparison, full Z ablation grid, concept-only continual banks, post-continual search confirmation, and long-context/LLM scaling remain untested. Missing stages are not successful results. Development thresholds do not establish superiority; sealed paired results and intervals must support any claim.

Raw decisions, failures, checkpoint provenance and per-item predictions are retained under `runs/`. The earlier stopped run remains in `v8_tiny`. No early low-score gate terminated this revision.
