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

## V3-009 · Gate step 1,217 (2026-09-30 04:54, eval 79 s)
- **E2 pass** (both languages within v1 + 0.10 at the same step): TS 2.502 vs v1 2.408 (+0.094), WK 4.136 vs 4.045
  (+0.091); mean val 3.319 vs 3.227 (+0.092). Margin to the line is only 0.006–0.009. Gap trend (mean, v3 − v1):
  +0.35 (185) → +0.26 (265) → +0.24 (365) → +0.17 (554) → +0.13 (766) → +0.10 (1,009) → +0.09 (1,217).
- **Recall onset is under way and routed through the plan**: recall1 0.148 with plan-swap 0.000 (v1 0.336 here; v2 0.008
  at step 1,177), so "no v2-like delay" is already answered. Observation, not a verdict: v1 went 0.117 → 0.336 over
  1,009 → 1,217, v3 0.031 → 0.148 over the same steps: same onset shape ≈ 200 steps later; E1 (≥ 0.25 by 1,792) looks
  likely. Numbers to watch: recall1 at 1,402 and 1,593.
- E5 (report): CRC G share 0.848 (≥ 0.5). Codebook: codes_min 0.96, VE 0.625 (v1 0.628). Eval angle23 1.71°, rho3 0.995.
- **Diagnostic: ncp − ncp_M1 ≈ 0 at every eval** (3.399 / 3.399 here; 0.000–0.002 since step 185). Verified read-only
  that it is not an evaluation bug: evaluate.py `codebook_health` runs `m.core` separately with schedule "3"
  (3 passes) and "1" (one (0, 1) pass, SCHED in model.py), and the two differ in the 4th decimal. So one pass currently
  predicts the next concept as well as three: loop depth is not yet paying off on NCP (the shortcut / L_scons design
  makes M = 1 usable, but M = 3 should eventually beat it). Design signal to re-read at 1,792 and at the 24 h milestone;
  not a gate.
- Throughput 3,262–3,370 tok/s since the charger returned (04:11) and macOS's XProtect scan finished; one window at
  2,844 during that scan. Earlier dips traced to macOS background work (mediaanalysisd / VTDecoder at 03:12, after an
  iCloud sync), not to the run.

## V3-010 · Gate step 1,792 (2026-09-30 06:33, eval 80 s)
- **E1 pass**: recall1 0.594 (bar 0.25; v1 0.820 here; v2 0.047 at 1,837); plan-swap 0.016, so recall goes through the
  plan. Recall lag vs v1 ≈ 200 steps and holding, not growing (v1 0.602 at 1,593; v3 0.594 at 1,792).
- **E2 pass, second consecutive gate**: TS 2.320 vs v1 2.228 (+0.092), WK 3.922 vs 3.835 (+0.087); mean 3.121 vs
  3.032 (+0.09). **The gap has plateaued**: it closed +0.35 → +0.09 by step 1,217 and has stayed +0.09 at 1,217 / 1,402 /
  1,593 / 1,792. The third reading at 2,376 is a live question (amber at worst; fail needs > +0.20 at two gates).
- Harder probes: v3 carry3 0.000, multi4 0.000, hop 0.008; v1 at 1,792 was also ≈ 0 (carry3 0.008, multi4 0, hop 0).
  No lag on these; v1's first real movement was at 2,376 (carry3 0.125, hop 0.047), which is what to read v3 against.
  At 128 items per family, a change of 0.008 is one item.
- ncp − ncp_M1 = −0.002 (3.269 / 3.271): three passes edging one, as at every eval since 1,402.
- E5 (report): G share 0.723, stable after falling 0.848 (1,217) → 0.776 → 0.718. Codebook codes_min 0.99, VE 0.643
  (v1 0.639). Eval angle23 2.55°, rho3 0.992. Throughput 3,234–3,292 tok/s.

## V3-011 · Gate step 2,376 (2026-09-30 08:13, eval 78 s) — all pre-registered early gates done
- **E2 pass, third gate**: TS 2.214 vs v1 2.125 (+0.089), WK 3.796 vs 3.700 (+0.096; margin to the +0.10 line 0.004);
  mean 3.005 vs 2.913 (+0.092). Summary of the early gates: E1 pass (1,792), E2 pass at 1,217 / 1,792 / 2,376, E3 held
  (gn 0.30–0.31 after warm-up), E4 held (≈ 3,300 tok/s vs 1,290), E5 reported.
- **The language gap plateaued, not narrowing**: mean +0.09 at all six evals 1,217 → 2,376 (it had closed +0.35 → +0.09
  over 185 → 1,217). No more E2 gates; the dashboard's "Keeping pace with v1" card still applies the 0.10 line at every
  eval, so a WK reading of +0.101 at 2,559 would show amber: expected behaviour, not a gate fail.
- **Recall**: recall1 0.773 (v1 0.922; v2 0.219 at 2,394); plan-swap 0.000. Absolute gap to v1 −0.23 (1,792) → −0.18
  (1,992) → −0.22 (2,190) → −0.15 (2,376): narrowing.
- **carry3 is the first probe where v3 trails something v1 had started**: v1 0.008 → 0.070 → 0.125 at 1,792 / 2,190 /
  2,376 (16 of 128 items); v3 0.000 at all three. hop runs the other way (v3 0.062, v1 0.047). Watch carry3 at 2,559 / 2,764.
- ncp − ncp_M1 = −0.003 (3.229 / 3.232): three passes edging one, widening from −0.002. G share 0.804 (report).
  VE 0.646 (v1 0.648), codes_min 0.99. Eval angle23 2.89°, rho3 0.992.
- Machine: swap 0.9 → 1.1 → 1.4 GB over the last hour (flag 4.6 GB); footprint 13 GiB, planned re-exec at 24 GiB ≈ 8 h out.

## V3-012 · FLAG: macOS update staged with automatic install ON (2026-09-30 08:47)
- Disk free fell 27 → 16 GB in 30 min (08:17 → 08:47). Cause: macOS software update preparing in the background —
  local snapshot `com.apple.os.update-MSUPrepareUpdate`, `softwareupdated` running, recommended update macOS 27.0.1
  (+ Command Line Tools 27.0). The run folder itself is 2.3 GB (checkpoints hard-linked); swap added one 1 GB file.
  Checkpoints present when the snapshot was taken stay pinned by it (bounded, not a growing leak). Free space should
  level off when staging ends; the 5 GB disk flag is the trigger to watch. Throughput dip 2,966 / 2,925 at steps
  2,425–2,450 matches the download.
- Risk: /Library/Preferences/com.apple.SoftwareUpdate has AutomaticallyInstallMacOSUpdates = 1, and this Mac has a
  history of "install tonight" updates (FirstInstallTonightDateDictionary). Expected window: overnight, roughly
  02:00–04:00 IST on 2026-10-01, when idle on AC — inside the run (milestone ≈ 01:09, run end ≈ 07:00 on 2 Oct).
- Nothing relaunches the run after a reboot (no LaunchAgent or cron entry for it). Recovery is manual:
  `cd run_v3_19M && bash code/resume.sh`. Cost of a restart ≈ training since the last `ckpt_last` (written at every
  eval, ≈ hourly) + ≈ 2 min compile; the 48 h budget is training-time based.
- Owner action (a system setting, so owner only): System Settings → General → Software Update → ⓘ next to Automatic
  Updates → turn off "Install macOS updates"; if a "Restart tonight" notice appears, choose "Remind me later". Owner
  notified by push at 08:5x. No action taken by the monitor (not in the allowed list).

## V3-013 · Interim assessment at 9.5 h (step 3,200 of ≈ 16,090; 2026-09-30 10:45)
- **Verdict: healthy, and ahead of v1 in the frame the 48 h budget is spent in.**
- Same step (same tokens): val +0.09 behind v1 at every eval since 1,217 (0.088–0.095; step 3,121: TS +0.084, WK +0.097).
  All early gates passed (E1, E2 ×3, E3, E4; E5 reported). Recall level with v1 since 2,764 (0.969 vs 0.984 at 3,121;
  plan-swap ≤ 0.016 throughout). carry3 started ≈ 700 steps after v1's but climbs fast (0.289 vs 0.375); hop level;
  multi4 first nonzero at 3,121. Codebook identical to v1 (VE 0.647 vs 0.648). v2 at the same steps: +0.3 behind v1.
- Same wall-clock hours: v3 ahead of v1 by 0.13–0.29 nats at every hour (9.3 h: 2.906 vs 3.036) and on recall
  (0.969 vs 0.800), from 1.9× the tokens per hour (median 3,281 tok/s vs ≈ 1,700).
- Reading, not fact: the constant, language-independent +0.09 at matched tokens most likely reflects capacity
  (19M compute params vs v1's 25M, −24 %); v2 (also 19M) sits +0.3 behind. The ablation arms, not this run, confirm it.
- Uncertain: depth advantage is small (ncp_M1 − ncp +0.002–0.003, positive since 1,402); whether the loop is worth its
  cost is for the controlled arms. Shared-FFN usage unmeasured (W₂ size 0 → 13.1); post-run checkpoint probe.
  Watch the per-language deltas after the matched-data boundary (step ≈ 8,556).
- Expectation for the 24 h milestone: v3 at step ≈ 7,500 with LR mid-cosine (≈ 3.3e-4) will read above v1's annealed
  final (2.754 at 4,050); that is not a failure. The verdict is the 48 h final on the sealed test split.
- Run risks: macOS auto-install still ON (can restart the Mac tonight); planned memory re-exec ≈ 21:00; charger dropped
  3× (each reconnect ≈ 30 min at −10 % from macOS XProtect scans). Throughput median 3,281, gn max 0.39 after warm-up,
  no guard events, footprint 6.9 → 15.3 GiB, swap 2.8 GB.
- 2026-09-30 10:49 · **Closed**: AutomaticallyInstallMacOSUpdates now reads 0 (owner turned automatic macOS install off). The staged update and its snapshot remain, but it will no longer install and restart by itself.

## V3-014 · End of the step-matched comparison (step 4,050; 2026-09-30 13:19, 12.2 h of training)
- v1's main run ended at step 4,050, so this is the last evaluation compared with v1 at the same tokens. v3: val 2.813
  (TS 2.032, WK 3.594) vs v1's final 2.754 (1.981 / 3.528): **+0.059** (TS +0.051, WK +0.066). v1 reached this at the end of
  its annealed 24 h schedule; v3 is at LR 5.2e-4, mid-cosine, after 12.2 h.
- Probes at step 4,050: recall1 0.992 (v1 0.984), carry3 0.617 (v1 0.594), multi4 0.086 (v1 0.055), hop 0.102 (v1 0.086);
  plan swap 0.000. v3 matches or leads v1 on every probe at the same tokens.
- Gap history (mean, same tokens): +0.35 (185) → +0.09 (1,217–2,376 plateau) → +0.07 (3,669–3,853) → +0.06 (4,039–4,050).
- From here: evaluations are hourly; v1 has no same-token reference (its continuation used unseen rows with an LR re-warm).
  The next comparison points are the 24 h milestone (≈ 01:10, val split) and the 48 h final (test split) against v1's final.

## V3-015 · Reference numbers for the 24 h and 48 h readings (set in advance)
- The 48 h final is on the **test** split. v1's test-split final (run_m2_25M/runs/main/eval_final.json, lang.intact):
  TinyStories 1.9768, WikiText 3.6290, **mean 2.803**. Compare v3's final to 2.803, not to v1's val-split 2.754.
- The 24 h milestone is on the **val** split at LR ≈ 3.3e-4 (mid-cosine): compare to v1's val numbers and read it as
  under-reporting the model. Log-linear trend of v3's evals 1,593–3,853 at near-constant LR puts val ≈ 2.5–2.6 at step
  ≈ 8,000 (≈ 0.2 below v1's annealed val final at matched wall-clock). No projection is made for 16,000: it crosses the
  epoch boundary (step ≈ 8,557, repeated data) and the anneal, two effects of unknown size in opposite directions.
- Milestones: step ≈ 8,000 = 24 h (switch-off tests, NTP by passes); step ≈ 8,557 = first epoch boundary in this project;
  step ≈ 16,000 = 48 h final (annealed, test split) — the only reading of the model's own final capability. Whether the
  loop / shared FFN are worth their cost is not decidable from this run (controlled arms).

## V3-016 · v4 hypotheses, pre-registered before the 24 h milestone (2026-09-30 18:40)
Evidence at 17 h: same-token gap to v1 constant +0.06–0.09 (language-independent) with 19.0M vs 25M compute params;
v3 at 1.9× v1's tokens/hour; recall 1.0, carry3 0.84, multi4 0.30 (v1 best 0.156); hop ≈ 0.46 success per link; depth
advantage on NCP +0.004 nats; v1 R1 ≈ R4 on language (0.001); NCP flat on training batches at high LR; shared-FFN W₂ 0 → 14.6
(share not logged); Engram gate 0.76–0.83; process footprint +0.9 GiB/h with MPS driver memory flat.
**Decidable now**
- Width: spend part of the throughput dividend on width. Params scale ≈ d²: d 288 ≈ 24M (v1's size, ≈ 1.5× v1's speed);
  d 320 ≈ 30M (≈ 1.2× v1's speed). Hypothesis: closes the constant same-token gap (capacity). Test at matched tokens.
- Engineering: find the Python-side memory growth (footprint rises while mps_driver_gib is flat: compile caches, logging
  buffers or retained tensors), so the run needs no 24 GiB re-exec.
**Gated on the milestone (fixed-checkpoint switch-offs are out-of-distribution: a large cost proves use, a small one only suggests non-use; the arms confirm)**
| milestone reading | v4 change |
|---|---|
| c_off costs ≈ 0 (< 0.02 nats) | drop c_pred from the plan; NCP stays auxiliary or goes |
| c_off costs > 0.02 | keep c_pred; revisit the 0.1 NCP weight |
| R1 − R3 next-token delta < 0.02 | M = 1: one 4-layer pass, compute moved to width |
| R1 − R3 > 0.02 and hop drops at R1 | keep M = 3 and the (t, Δ) table |
| shared_ffn_off costs < 0.02 | remove the shared FFN; parameters back to own FFNs (param-matched arm design) |
| engram_read2_off costs < 0.02 | one Engram read (v1's design) |
| table_off costs < 0.02 | remove the table only together with M = 1 (at M = 3 it breaks the fixed point) |
**New mechanism to test (not a subtraction):** hop fails at re-querying, not retrieval (recall 1.0; ≈ 0.46 per link; v1's bridge
entity decodable 2.6× at pass 1 and flat across passes). v4 research direction: a pass whose query into C is built from what
the previous pass retrieved (pointer-following bias). Metrics: hop by depth and by pass, bridge-by-pass probe, OOD hop 5–6.
**Keep:** encoder / concept pipeline (VE, codebook, recall, carry3, multi4 at or above v1), the pre-registered gate design,
and nothing changes on the running v3.

## V3-017 · Dashboard file resets on every process restart (2026-09-30 21:50)
`write_dash` builds `history` / `evals` of `v3_live.json` from in-process lists (`dash`, `evals_dash`), so the memory
re-exec at step 6,875 (21:27) restarted them; the milestone relaunch and any crash resume will do the same. The run logs
(`train_log.jsonl`, `log.jsonl`) are append-only and complete, and the :8766 server reads them directly (unaffected).
Fixed client-side in the :8765 v3 Live tab (webapp DESIGN #27); nothing on the running trainer changed.
**For v4:** seed `dash` / `evals_dash` from the jsonl logs at startup (or on `--resume`).

## V3-018 · Owner-approved CPU experiment slowed v3 for 3.65 h (2026-10-01 14:25–18:04)
The owner approved a CPU-only tiny-v4 experiment (`v4_tiny/`), accepting a slower v3. With 2–3 arms on the P-cores v3 ran
at ≈ −30% (heartbeats 1,563–2,472 vs a 2,907–3,229 reference; full series in `v4_tiny/REPORT.md`, engineering record).
v3's cosine is time-based and ends at 48 h, so ≈ 13–14M tokens (≈ 400–420 steps, ≈ 2.5% of the run) are not recovered.
There was no guard event (`GUARDS.jsonl` unchanged), and the arithmetic is unaffected (deterministic batches).
Hourly val kept improving (2.513 → 2.497). Clean recovery: 2,900 tok/s at 18:10. Read the 48 h final with this in mind.

## V3-019 · Owner-approved CPU experiment (v4 vs LLM stack) slowed v3 for its last 3.5 h (2026-10-01 22:44 – 2026-10-02 02:13)
The owner approved a second CPU-only tiny experiment (`v4_tiny/VS_LLM_REPORT.md`: v4 vs a 3 : 1 SWA : full LLM stack on
text and gridworld-pinpad), accepting a slower v3 and stating "don't stop the training". Two experiment processes ran
beside v3 (the owner was told −20–25%, then corrected to ≈ −30% at 23:50). Measured: mean heartbeat 2,264 tok/s from
22:44 to the 48 h stop (33 heartbeats) vs 3,234 tok/s in the clean window 18:10–22:40 (63 heartbeats), ≈ −30% for 3.5 h,
≈ 12M tokens (≈ 370 steps, ≈ 2.4% of the run) not recovered under the time-based cosine. No guard event was caused by
the experiment (the 00:30 memory re-exec is v3's own 24 GiB rule; GUARDS.jsonl 2 lines). Hourly val kept improving
(2.478 → 2.473; in-run final 2.471). The final evaluation shared the CPU with the last arms and analyses and took ≈ 3 h
instead of ≈ 1 h; its arithmetic is unaffected. Together with V3-018, ≈ 25M tokens (≈ 5%) of the 48 h run were lost to
the two owner-approved experiments; read the 48 h final with this in mind.
