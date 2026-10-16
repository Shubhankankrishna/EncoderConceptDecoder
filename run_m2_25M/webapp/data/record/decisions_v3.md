# HRM Thinker v3 — implementation decisions

Spec: `../draft_v3.md` (r2). Advisor calls: `ADVISOR_LOG.md`. Tests: `runs/tests/`.

## V3-001 · Owner's instruction (2026-09-29 ≈ 21:30)
Implement draft_v3.md from scratch in a new folder; stop v2 after its 24 h evaluation (no more v2 training); delete v1 and
v2 checkpoints except 1–2 best; test v3, then train it from scratch for 48 h with the 24 h (milestone) evaluation;
optimise for throughput, including a torch.compile check; ≥ 40 advisor calls.
Reading of "48 h with 24 h evaluation": the draft's 48 h training run (time-based cosine) with the out-of-process full
evaluation at the 24 h milestone (training resumes afterwards), plus the final evaluation at 48 h.
v2 was stopped by a PAUSE file set while its 24 h evaluation ran (the launcher checks PAUSE before resuming), so the
evaluation completed and training did not resume.

## V3-002 · Code layout
`code/` holds byte-identical copies of v1/v2 modules (hrm_encoder, hrm_decoder, concept_core_v2, data, probes, tasks,
compat, speed) and the v3 files: `model.py` (EncoderV3, ConceptCoreV3, HRMThinkerV3, forward, param_table,
param_groups), `train.py`, `evaluate.py` (ported), `final_eval.py` (ported), `speed_v3.py`, `test_v3.py`,
`test_speed.py`, `mps_check.py`, scripts (run_job, start, pause, resume, battery_guard, v3_check, v3_watch) and
`V3_MONITOR_POLICY.md`. Module names follow v3_design/v3_ref.py so a state_dict transfers.
- One Engram gather feeds both reads (review C4); read 2 receives the rows (no aliased table, review C5).
- Compute parameters 18,986,871; stored 101,116,663; the component ledger reconciles with the unique parameters (asserted).

## V3-003 · Acceptance harness (`runs/tests/test_v3.txt`): 49/49 PASS, exit 0
- Bit-parity with v3_ref on all four schedules (same weights; torch.equal on logits and u).
- Repairs C1–C7: isolated L_scons graph with a detach-removed mutant; untouched-init vs perturbed gradient tests with the
  expected temporarily-zero branches listed; reuse parity on gradients and one AdamW step (dg 1.4e-16, dp 9.7e-14);
  one-gather hook count; ledger reconciliation; non-zero exit; table-only and shared-only perturbations; gain-free readouts.
- Reach 63 (encoder L with read 2 and 6-head F layers) and 70 (decoder token path); causality, document isolation,
  row-permutation invariance; batched shortcut == separate trajectories (0.0 in float64).
- Target masks (padded and partial chunks), empty-mask finite zero, full-batch denominator == one big batch under unequal
  valid counts (247 vs 78 targets).
- Mutants caught: future-token leak, crossed document boundary, removed teacher detach, detached reused prefix,
  duplicated optimizer registration, table in the decay group.

## V3-004 · Trainer
- Main M = 3 + shortcut sampled per micro-batch (1: 0.5, 2a: 0.25, 2b: 0.25) from its own Generator (checkpointed); the
  shortcut runs batched with the main passes (rows [main; shortcut]); 2a reuses the main S1.
- Full-batch denominators: all 16 micro-batches are fetched first; each micro-batch backpropagates Σ λ·sum / N_total
  (no /accum). Loader prefetch = accum + 4.
- Codebook: k-means++ from complete real chunks (≥ 4 × 128 candidates, accumulate-until fallback); restarts every 100
  optimizer steps from complete chunks, with Adam-moment reset of the restarted rows (unit-tested).
- Per-step LR trace (lr_trace.npy + checkpoint). Resume state: model, AdamW, cursor, RNGs, trajectory RNG, elapsed,
  LR trace, codebook hits.
- Resume exactness: in `--deterministic` mode (1 CPU thread), an uninterrupted 4-step run, a second identical run, and a
  2 + 2 resumed run are bit-identical (model and Adam moments). Without it, fp32 CPU runs differ at 2e-7 between two
  identical runs (threaded reductions), and the time-based LR differs by elapsed time — both expected.
- Carried: V-011 footprint re-exec guard (tested with --debug_restart_step), v1 in-loop memory guard, divergence guard,
  heartbeat, dashboard file (`run_m2_25M/webapp/data/v3_live.json`).
- Evals: hourly quick eval + step-triggered at all of v1's eval steps (185 … 4050); quick eval adds ncp at M = 1 and the
  loop geometry; the 24 h milestone runs `final_eval.py --split val` (test sealed until 48 h, draft r2 §7.4); the 48 h
  final runs `--split test` (test + OOD). Launcher path tested on CPU: milestone → evaluation → resume → final → done.
- Arms available: --no_shortcut, --no_table, --shared_ff, --core_ff, --main_sched (A1 / A3 / T3 / own-992 smokes pass).

## V3-005 · Speed paths (CPU float64 parity, `test_speed.py` 6/6)
- Block-local sliding-window attention for window-8 layers (encoder S, decoder confined): exact up to 1e-14 (forward) /
  8e-16 (grad) at block 16/32. Scores T·(block + 7) instead of T².
- Micro-step-wide doc-shift mask memo (strides in the key; never across steps): bit-identical.
- torch.compile: dynamo traces the core layer, encoder S (with and without block attention) and F layers as ONE graph
  each with zero graph breaks (CPU explain). MPS A/B pending.

## V3-006 · Throughput on the M2 (MPS fp32; charger connected for every measurement, 56–76 %)
**MPS parity** (`mps_check.py`, full config, B 1 × T 1024): max |Δ logits| 4.5e-6 (block 0) / 3.9e-6 (block 64), max
gradient relative error 1.5e-5 / 2.1e-5 → PASS.
**Profile** (`profile_step.py`, one micro-step B 2, medians of 6): encoder fwd 202 ms (17.5 %), core fwd 176 ms (15.2 %),
head + decoder fwd 81 ms (7.0 %), losses 24 ms, backward 655 ms (56.7 %); micro-step 1,154 ms; optimizer step 151 ms;
estimate 1,761 tok/s; driver 6.75 GiB. Block-local attention 64: micro-step 1,062 ms, est. 1,912 tok/s, driver 5.75 GiB.
Quick eval (full, 128 rows / items): 87 s.
**Sequential 8-step runs are unusable (thermal drift on the fanless M2 Air):** base 2,059 then 1,478 tok/s (−28 %) three
minutes apart; block64 1,671 / 1,798; block128 1,962. Kept only as evidence.
**Interleaved A/B** (`ab_interleave.py`: one process, configs alternate every optimizer step, ABBA; 12 measured steps each;
rule fixed in advance: adopt if ≥ 5 % paired and IQRs disjoint):

| comparison | base median | variant median | paired Δ tok/s | IQR base / variant | verdict |
|---|---|---|---|---|---|
| block64 vs base | 16.75 s (1,956) | 15.68 s (2,089) | **+6.1 %** | 16.34–17.21 / 15.62–15.83 | **adopted** |
| block128 vs base | 17.48 s (1,874) | 16.82 s (1,948) | +4.9 % | 17.37–18.58 / 16.64–16.91 | not needed (64 better) |
| memo off vs on | 18.22 s (1,799) | 17.67 s (1,854) | +0.9 % | overlap | noise; memo kept (bit-identical) |
| B4 × 8 (± block64) | — | — | — | — | **rejected**: warm-up unfinished after 7.8 min, swap 4.6 → 6.4 GB (thrash) |

Base drifted 1,956 → 1,874 → 1,799 across the series (thermal), which is why only paired ratios are used.
**Swap baseline for the run:** ≈ 6.2 GB of the 7.2 GB swapfile is in use before launch (v2's evaluation + the B4 thrash; no
reboot possible). G4 "flat" = no rise above that baseline; the monitor flag is baseline + 2 GB (policy updated).

### V3-006b · torch.compile (inductor on MPS, torch 2.10)
- First A/B (block 0 both arms, memo off both, `ab_interleave.py memo0,compcore`): compiled core layers **+18.0 %**
  (paired 0.847; IQR 16.12–19.06 vs 13.56–15.45); 4 specialisations (3 recompiles: S requires_grad, 2 vs 4 rows); warm-up
  107 s. Swap drifted down 3.9 → 3.5 GB (the swapfile shrank to 5 GB).
- **Compiler bug found by `compile_check.py`**: logits match (3e-6) and every gradient matches to ≤ 2e-6 except
  `core.layers.*.attn.k_norm.gamma`, whose compiled gradient is ≈ 8.5× too large (2.84e-2 vs 3.32e-3; eager vs eager
  repeat 3.5e-9). Same for the encoder F layers under `--all`. Not the GQA repeat_interleave (expand/reshape: same error),
  not the transposed layout (norm before transpose: same error). **Workaround** (`speed_v3.compile_layer`): the k_norm
  call runs eagerly (`torch.compiler.disable`, one graph break per attention). Then all three shortcut schedules pass:
  worst gradient rel 6e-6 (core) / 2e-5 (core + enc + dec), table 1.6e-6, shared.w2 1.2e-6.
- Wiring: compiled only in training (eval runs eager: `final_eval` / `evaluate` need no compile parity); `--compile` is a
  restored run setting (it survives pause / resume / milestone); the memo must be off with compile (`--step_memo 0`), its
  dict would be a guard input.
- **Combined A/B** (`block64, block64compcore, block64compall`, memo off in all arms, 10 measured steps): block64 16.71 s
  (1,961 tok/s); + compiled core 14.45 s (**+17.7 %**); + compiled core, encoder and decoder layers 10.54 s (3,108 tok/s,
  **+54.4 %**, paired 0.648, IQR 10.17–11.07 vs 15.65–17.80). 25 recompiles, all before the warm-up ended (122 s). Driver
  ≤ 7.03 GiB. **Adopted: `--block_attn 64 --compile all --step_memo 0`.**
- Consequences: ≈ 3,100 tok/s is 2.4× the draft's estimate midpoint (§2.1's "about v1's speed" does not hold for this
  implementation). 48 h ≈ 15,500 steps ≈ 508M slots ≈ 1.8 epochs (draft: 0.75–1.03); matched data vs v1 holds only in
  epoch 0 (steps ≤ ≈ 8,556; epoch 1 is reshuffled by seed + epoch). v1's step-triggered evals end at 4,050 (≈ 12 h); the
  rest is hourly. Every start pays ≈ 2 min of compilation (monitor allowance 6 min).
- Swap baseline correction: after the A/Bs the swapfile shrank; swap is ≈ 2.0 GB of 3 GB now (the 6.2 GB above was the
  peak after the B4 thrash). The launch baseline is recorded at G4.

## V3-007 · Launch (2026-09-30 01:09)
`bash code/start.sh --block_attn 64 --compile all --step_memo 0 --est_total_steps 16130` → runs/v3_48h. Flags verified in
config.json; battery guard (7 %) and the 8766 dashboard (copy of v2's webapp, HRM_MAIN=v3_48h) started.
- G3 (scratch, warm-up 20, 60 steps, launch flags): NTP 8.99 → 4.66, gn 1.3 → 0.38, 3,000–3,350 tok/s; audit at 0/30/60:
  dec_plan g_short = 0, core_table cos 0.00 / 0.14 / 0.00, shortcut / main gradient 0.04–0.23 on the core.
- Eval smoke on MPS (`final_eval --split val --limit 2`, all 7 parts, 123 s) found and fixed two MPS-only crashes that
  would have stopped the 24 h milestone: the audit's float64 reduction (now on CPU) and `aten::mode` in fit_probe (CPU).
- G4 (100 steps, launch flags): mean 3,227 tok/s (steps 30–100); footprint 6.68 → 7.15 GiB over 90 steps (≈ 1.8 GiB/h →
  first V-011 re-exec expected ≈ 7 h in); swap baseline 2.6 GB (flat, 2.59 → 2.51); MPS driver 5.86 GiB flat; OPENED at
  100: table 0.61, shared w2 1.06, W_v1 1.63, W_v2 1.71, nothing still zero.
- est_total_steps = 45.5 h × 3,600 × 3,227 / 32,768 = 16,130 (≈ 1.8 epochs; matched data vs v1 through step ≈ 8,556).
- Expected: milestone at 24 h of training time ≈ 2 h (val); final ≈ 3–4 h (test + OOD, then best); run ends ≈ 54 h after
  launch (≈ 2026-10-02 07:00). Monitor: cron V2 check replaced by the V3 check (13, 43 past each hour), watcher armed.

## V3-008 · First evaluations (running notes)
- Step 185 (01:44, 85 s, eager eval on the compiled model — the l.training routing works; step 200 window 3,038 tok/s
  → the eval-time subtraction works): val 5.434 (TS 4.672 / WK 6.196) vs v1 5.089 (4.235 / 5.943) at the same step:
  **+0.35 behind v1** (TS +0.44, WK +0.25); v2 was 6.33 at step 138 and 5.14 at 267 (v1 4.56 at 265). recall1 0,
  bind 0.039, codes_min 0.66 (v1 0.76), VE 0.49 (v1 0.52), CRC G share 0.822, angle23 1.12°, rho3 0.998. Both runs are
  in the step-indexed warm-up (identical LR), so the gap is not an LR artefact. Not a gate (first gate 1,217); watch
  265 / 365 / 554. Advisor overloaded at this point; call deferred to the next eval.
- Throughput windows: 3,214 (1–25), 2,795, 2,861, 2,862 (26–100, browser work on the same GPU), 2,939, 3,011, 3,151
  (browser idle), 3,038 (after the eval). No thermal warning recorded.
