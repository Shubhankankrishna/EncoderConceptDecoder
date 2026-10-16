# HRM Thinker v2 (19M) — decisions

Spec: `../run_m2_25M/model_v2/MODEL_V2_19M.md` (architecture, budget, review). This file records what the implementation
and the run actually do, and why. Advisor log: `ADVISOR_LOG.md`.

## V-001 · Scope (user request 2026-09-28 evening)
Implement v2 from scratch in `run_v2_19M/`, run ONE 48 h pre-training with a full evaluation at 24 h and at 48 h, stop
v1 (the continuation was not improving enough to justify the machine), free disk by deleting older/worse v1
checkpoints, smoke-test, launch, keep the 7 % battery brake.

## V-002 · The 24 h evaluation is a planned pause, the schedule never restarts (v1 lesson D-041)
v1's continuation started a NEW 24 h schedule on a trained model: re-warm to 6e-4 → validation rose ~+0.1 and took
~1,500 steps to recover. v2 defines one schedule over the full 48 h at launch. At 24 h of training time the trainer saves
and exits with code 5; `run_job.sh` runs `final_eval.py` on `ckpt_last.pt` (→ `runs/v2_48h/eval_24h.json`, out of
process, not counted in the training clock) and resumes. LR, cursor, RNG, AdamW state and elapsed time continue exactly.

## V-003 · LR schedule on training TIME, not steps
v1 mis-estimated total steps twice (D-021, D-033) and spliced the cosine. v2: warm-up 300 steps from 0, then cosine from
6e-4 to 0.1 × 6e-4 with progress = elapsed training time / 48 h (hourly evals included; paused time and the 24 h
final_eval excluded). The schedule therefore ends exactly at 48 h of training whatever the throughput. Peak 6e-4, floor
0.1, warm-up 300 are v1's evidenced values (grad norm ≤ 0.38, no guard events at 6e-4). WSD was considered (predicted
~0.05 better in D-041 addendum 3) but gives an un-annealed 24 h model and an untested path → not adopted.
Wall clock ≈ 48 h + the 24 h evaluation (~1–1.5 h) + the 48 h evaluation (~2–3 h).

## V-004 · Data: v1's corpus, reshuffled every epoch
Tokenizer, PQ init, probes and every v1 comparison are tied to `run_m2_25M/data` (264.4M train tokens, 273,797 rows;
language 45 %, symbolic 22 %, retrieval 13 %, code 10 %, long-range probes 10 %). Rebuilding would break the comparison and
does not address the measured bottleneck (core retrieval). 48 h ≈ 0.8–0.9 epoch at the measured speed, so repetition is
not an issue. Fixed a v1 defect anyway: `ShardStream` replayed epoch 1's exact order in epoch 2; v2 reshuffles with
seed + epoch (epoch 0 is identical to v1's order).

## V-005 · Implementation choices
- Model exactly as the spec (row C): encoder v1 design at d 256 (d_ff 704, Engram 8 × 208,333 × 48), D6 core (4 layers,
  16 experts × 512, top-2, per-layer routers, LN_h carry, capacity 1.25 in training / exact in evaluation), confined
  decoder (W_u 256 → 256), NextLat on out_norm(D) (d 1, ψ 512, λ_h = λ_KL = 1). Compute params 18,833,529 (= budget
  + 16 per-head temperatures).
- Per-window backward with `c_hat`/`tok` leaves; one backward into the encoder per micro-batch.
- Loss = NTP + 0.1 · mean_r NCP(u_r) + 0.1 · consistency(final window) + VQ + 0.01 LB + 0.001 z + NextLat-h + NextLat-KL.
- Evaluation R 6; R sweep 1–16 in final_eval.
- Exact CPU tests (float64): `runs/tests/test_v2.txt` — MoE capacity == dense, per-window backward == one graph (5e-15),
  identity at init, causality + document isolation, param count, NextLat head frozen, windows layout. Trainer smoke on CPU:
  milestone exit 5 → resume, pause exit 4 → resume with all args restored, final_eval all parts.

## V-006 · Throughput (M2, 16 GB, 8-core GPU; measured, see ADVISOR_LOG)
Profiled one micro-batch (B 2, K 10): first 926 tok/s in the trainer. Fixes with identical results:
- MoE top-2 via two `max` ops instead of `topk` (3.7 ms → 0.17 ms per call on MPS; identical indices and values).
- NCP head distance via ‖q‖² − 2 q·e + ‖e‖² (no (B, M, 48, 128, 8) difference tensor) and no c_pred on non-final passes.
→ 1,311 tok/s in the profiler (B 2). B 4: 1,408 (2× work took 1.86× time → GPU-bound, not launch-bound);
B 8 swapped (53 tok/s); bf16 autocast slower (1,102 at B 4). No silent CPU fallbacks (MPS fallback env unset).
Then, A/B back-to-back (the fanless Air throttles: repeated baseline runs differ by ±5 %):
- X~-branch of the core's gated reads computed once per window (X~ is constant within a window) and the encoder's
  doc-shift masks memoised inside one forward (`code/speed.py`): +6 % (1,303 → 1,381 tok/s, profiler). T2 still 4e-15.
- Capacity factor 1.5 instead of 1.25: −6 % → kept 1.25 and the dropped fraction is monitored (flag if > 5 % after step 300).
- Windowed (unfold) attention for the S layers: exact (1e-6) and faster forward, but slower backward → not used.
- torch.compile (inductor on MPS, new in 2.10): ceiling ≈ the ~10 % launch overhead, untested numerics → not used (D-041 lesson).
Trainer, 20 steps: B 2 → 1,293–1,334 tok/s, allocated 1.6 GiB, driver 7.9 GiB, swap flat (1.2–1.4 GB);
B 4 → 1,017–1,022 tok/s, driver 11.7 GiB, swap rising (2.6 → 2.8 GB): the allocator-cache failure of v1's D-019.
Launch configuration: see V-007.

## V-007 · Launch configuration
micro_bs 2 × accum 16 × 1,024 = 32,768 tokens per optimizer step (as v1), fp32, capacity 1.25 in training, peak LR 6e-4,
warm-up 300, cosine to 6e-5 over 48 h of training time, K ~ U{8..12}, eval R 6, λ_h = λ_KL = 1, hourly quick eval (128 items),
keep 2 step checkpoints + last + best, milestone at 24 h. Expected ≈ 1,250–1,350 tok/s → ≈ 215–235M tokens in 48 h
(0.8–0.9 epoch), ≈ 6,600–7,100 optimizer steps (--est_total_steps 6900 for the dashboard until the speed is measured).
Hourly quick eval ≈ 160 s (v1 141 s) → ≈ 2.1 h of the 48 h. Wall clock ≈ 48 h training + ~1.5–2 h (24 h evaluation, R sweep
to 16 and 16-pass loop dynamics) + ~2–3 h (final, last + best). Capacity drops start at ~0.10 under warm-up LR; watched.

## V-008 · v1 stopped (2026-09-28 19:05)
v1 continuation paused at step 1879 (last eval step 1826, val 2.7287). D-043 cron and watcher removed, battery guards
killed. Deleted: main ckpt_step003853/004039, cont ckpt_step001642/001879 (+ its last link), screens S1–S3 checkpoints.
Kept: main final (step 4050 = last = best, the 24 h result of record) and cont best (step 1826). Disk 29 → 35 GB free.

## V-009 · Launch (2026-09-28 20:22)
Stability run first (scratch, warm-up 20, 60 steps at 6e-4): losses finite, NTP 8.75 → 5.06, gn 0.58–1.35 → 0.73,
routing divergence 0.08–0.15 (> 0), MoE dropped fraction 0.19 → 0.37 → 0.22 (falling; the router starts near uniform,
where the load-balance loss is weak), val 6.22 at step 60, eval 128 s, 1,334–1,431 tok/s.
Launched via code/start.sh at 20:22:02: run_job.sh 92203, trainer 92216 (nice 0), caffeinate -w 92203, battery guard 92209
(7 %), v2 dashboard 92227 (:8766, nice 19). First line: step 25, NTP 8.90, gn 0.90, drop 0.128, 1,378 tok/s. Monitoring:
session cron :13/:43 (code/V2_MONITOR_POLICY.md) + code/v2_watch.sh.

## V-010 · v2 live page = v1's live-page engine on v2 data (user request, 20:45)
The first v2 page (sparklines from data/v2_live.json) was replaced. `run_v2_19M/webapp` (a copy of the v1 dashboard,
served on :8766 with HRM_MAIN=v2_48h) runs v1's full Live training page on the v2 run, made v2-correct: server passes the
v2 fields (cons, lb, z, nl_h, nl_kl, drop, load_ent, route_div, K_mean, ncp_pass, engram_alpha, progress), budget from
config.json (48 h, `budget_h()`), an "evaluating" state while final_eval runs (24 h pause), a four-stage plan
(train 0→24 h · 24 h evaluation · train 24→48 h · final evaluation), `[steps projected` lines ignored for the "starting"
state, CORS for the 8765 page. telemetry.js: time-based progress and LR check, v2 speed thresholds (1,200/1,000), v1
reference lines on validation (2.754 / 2.729), KPI tiles + MoE drops and routing divergence, new sections "Concept module
(D6)" (NCP by pass, consistency, drops, load entropy, routing divergence, LB/z) and "NextLat" (next-h exempt from the
falling-loss verdict, KL, K), early-training grace for grad-norm and per-pass verdicts. The 8765 dashboard shows it inside
v2 → Live training (iframe ?embed=train, height and theme via postMessage) and its header pill shows the v2 run in v2 mode.
The two webapps diverge from here: run_m2_25M/webapp = v1 dashboard + v2 design pages; run_v2_19M/webapp = v2 live engine.
Only the :8766 server was restarted; the trainer was not touched.

## V-011 · Native memory leak in the trainer → footprint-based re-exec guard (2026-09-29 15:50)
Found at the 15:49 monitor check: trainer pid 92216 physical footprint 39.7 GiB (31 GiB compressed), growing ~2 GiB/h;
vmmap: "Malloc Nano" 116.5 GiB virtual / 25.9 GiB swapped (millions of tiny native allocations, i.e. outside torch's
allocator: mps_driver_gib flat at 7.7). Swap 1.4 → 17.7 GB over 19 h, disk 29 → 12 GB free (swap files; freed on exit).
Growth is linear between evaluations (not stepwise at evals), so it is a per-step leak in the MPS/Metal path, not the
eval-mode MoE shapes. Training itself healthy (val 3.2175 falling, recall1 0.219 vs plan swap 0.008).
Mitigation (identical numerics, only a process restart): train.py reads its own footprint (proc_pid_rusage
ri_phys_footprint, validated = vmmap's 39.7 GiB) at every log step; above --mem_restart_gib (24 GiB) it checkpoints,
logs a `memory_restart` guard event and os.execv's the same command with --resume ckpt_last.pt (same pid for run_job.sh,
no crash count consumed, HRM_REEXEC env prevents a forced-test loop). Tested on CPU (--debug_restart_step 4): checkpoint →
re-exec → "resumed ... step 4" → normal finish rc 0. The running process has the old code; the guard goes live when
train.py is next started — at the 24 h milestone resume (~4.5 h), which itself frees the leaked memory (runway at 0.9 GB/h
swap: ~7.5 GB free disk at 24 h, enough for a 1.2 GB checkpoint). Root cause not yet found (candidates: per-op Metal
objects from index_add / scatter / one_hot / repeat_interleave in the 40 layer-passes per step).

## V-012 · v2 ended by the owner (2026-09-29 23:13)
Owner decision: no more v2 training; v3 (draft_v3.md r2, `run_v3_19M/`) takes the GPU. At 21:33 a PAUSE file was set so the
launcher would not resume after the 24 h evaluation; at 23:13 the owner asked to stop the evaluation too (it was 2 h 48 m
in, on the OOD item suite). `final_eval` was stopped, the launcher exited ("24 h evaluation exited 143"), battery guard
and watchers stopped.
- Last training state: step 3,089, 24.05 h, milestone eval val 3.1074 (TS 2.306 / WK 3.908), recall1 0.594 (quick).
- Partial 24 h evaluation (test): lang + trunc in `eval_24h.json`; test items (log only): intact recall1 0.605, carry3
  0.000, multi4 0.000, hop 0.006, bind 0.070, synthetic macro 0.244; plan-swap recall1 0.008. OOD items, probes, health,
  gen not run.
- Checkpoints: kept `runs/v2_48h/ckpt_last.pt` (step 3,089; for a later evaluation: `final_eval.py --ckpt
  ../runs/v2_48h/ckpt_last.pt --parts items,probes,health,gen`). Deleted ckpt_best / step003088 (one file, 1.22 GB) and
  the step003089 link. v1: kept `main/ckpt_last.pt` (step 4,050, the 24 h result of record; was also main's best) and
  `cont/ckpt_best.pt` (lowest v1 val 2.7287); deleted the link names main/ckpt_best, main/ckpt_step004050,
  cont/init_from_main, cont/ckpt_step001826 (no space: hard links). Disk 25 → 26 GB free.
