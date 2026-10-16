# V14-tiny: decisions and deviations from v14_Architecture.md

Written 15 Oct 2026 from 02:16 IST, before any V14 training step on the real run. Items are added as they happen,
with the time from `date`. The design is `v14_Architecture.md` (repo root); this file records where the 24-hour test
differs from it and why.

1. (02:16, requested by the user at about 01:35) **A throughput-oriented shrink to 6-7M parameters.** The user asked
   for a 24-hour V14 test that starts as soon as V13-tiny's pipeline completes, at 6-7M parameters and the best
   throughput (V13-tiny averaged 2,471 tokens/s over its 11 hours against 3,234 in its launch benchmark). Changes
   from V13-tiny-v2, beyond V14's own (§1.5):
   - E2 runs **R = 1** loop instead of 2. Evidence: on V13-tiny's final checkpoint, exiting E2 after loop 1 changed
     answer NLL by at most 0.0005 and exact match by 0 on every task type (2,048 dev items; v14_Architecture §1.1 C4).
     E2's two loops were 562 of 1,546 ms of an eager forward (V13-tiny REPORT §4).
   - The decoder has **4 layers** instead of 6, and E3 **2** ([S', F]) instead of 4.
   - E1, the Engram, the concept module, the pools, d = 128 and the reach bounds are V13-tiny-v2's: R_L = 51 and
     R_D = 66 exactly (`tests_g0.py`; decoder depth does not compound, because every decoder layer's keys are L_enc;
     E2 and E3 write G only).
   - Trained parameters **6,807,755** (V13-tiny-v2: 7,744,181). The EMA twin stores 3,479,720 more during training
     only (no gradient; the Engram table is shared, not copied).
   **What this confounds.** v14_Architecture §11 made the model differ from V13-tiny only by §1.5's table, so that
   V14 against V13-tiny would isolate the target redesign. It no longer does. V1-V5 are read within V14's own run
   (against its own baselines) and stand. V6 (text NLL at matched training time against V13-tiny) and every V14 vs
   V13-tiny comparison, including V7's continual matrix, are now joint effects of the redesign and the shrink, and
   are reported as such. The baseline transformer the user planned for the 24 hours after V13 runs after V14.
2. (02:16) **Default-run configuration** from `v14_design/checks/checks.json` `rules.default_run` (pre-registered rules
   applied on 14 Oct 22:43): target view [a; b]; lambda_max 0.1; the uncentred SIGReg test on b on from step 0.
3. (02:16) **The codebook is updated, then assigns** (V13's order): each step the normaliser and the EMA k-means words
   are updated on the batch's valid targets (first groups of documents and padding excluded), then the batch's codes
   are assigned. The first update initialises the normaliser and draws the words from valid targets. Dead words
   (usage below 10% of uniform, against the EMA of rows per step) take the worst-fitted valid targets, distinct, in
   order of error.
4. (02:16) **SIGReg as §8.3 writes it**: the Epps-Pulley statistic without an N factor (a point mass gives 0.131 per
   direction, 2,000 Gaussian samples about 3e-4; `tests_v14.py`), the mean over M = 256 unit directions drawn per step
   from a generator seeded by the step, separately for a and b; the uncentred test on b reuses b's directions and all
   valid groups; the centred terms keep documents with at least 4 valid groups.
5. (02:16) **The gradient-share cap** (§8.2): every 500 steps once the lambda ramp has ended (time fraction >= 0.1),
   eager, on a fixed held batch of 2 rows from each reasoning family (seed 4242, curriculum fraction 0.5); the ratio
   ||grad_C lambda L_NCP|| / ||grad_C L_NTP|| over the concept module; two consecutive readings above 1 halve lambda's
   scale (floor 1/8). The same step logs §8.1's G-side balance (the three losses' gradient norms over E2, E3 and E2's
   exit).
6. (02:16) **Reads.** Live, every 50 steps: per-horizon head accuracy on each half against copy-current, the fraction
   of groups whose argmax tuple is identical at all horizons, the batch's per-segment code-usage perplexity (batches
   are single-family), the codebook's EMA usage perplexity and resets, the residue maps' status, tokens/s, seconds per
   step, the time spent waiting for rows, memory. The heavy reads of §6.3 (K3 probes per half, chi per half, the
   residue map's R2 per family and pooled, the cross-correlation of the halves, the action sensitivity, effective
   rank) and V1's two baselines are computed by `diagnostics_v14.py` on the 10% snapshots and the final checkpoint,
   written during the base run before any of its outputs exist.
7. (02:16) **Activation checkpointing by rule** (`choose_ckpt.py`, fixed now): two 60-step smoke tests on the idle
   machine; checkpointing off if that smoke finished, its peak MPS driver memory (steps >= 10) is at most 9.0 GB and
   its median s/step is at least 3% below the checkpointed smoke's; otherwise on. B = 8 x 1,024 as in V13-tiny.
8. (02:16) **Data workers: 5** (V13-tiny: 4), because the faster step needs more rows per second; the wait for rows is
   logged.
9. (02:16) **Continual stage** (§9). V13-tiny's cached incoming rows (`runs/cl/rows/`, copied byte for byte), 100
   steps per phase, the same protocol. Methods: sequential LoRA, replay (20%, 1,000 rows), and "v14", V14's continual
   default chosen by §9.4's table from V13-tiny's matrix (`cl_choice.py`, at the step's start). Latent anchoring is
   implemented (O2 and O3 need it); the bank's partition key is the segment-1 code of a_g (§9.2(f)). The tuple-level
   softmax and code gates are arms and are not run.
10. (02:16) **Not run in this 24-hour test**: every arm of v14_Architecture §11 items 5-6 (one arm after the default run
    needs a second 11-hour run): inverse dynamics, the successor head, the static teacher, the control without the
    uncentred test on b, beta_1 lowered, "V13 plus", N = 64, the cosine latent term, the statistic projectors, the
    branch head, soft targets, the fixed quantiser. One seed throughout.
11. (02:20, before V13-tiny's continual matrix exists) **§9.4's decision, made executable** (`cl_choice.py`). §9.4's
    rows can overlap and one case is missing, so the rule is fixed now: precedence O6 > O1 > O4 > O2 > O3; "helps"
    uses the paired interval V13's evaluation computes (S0 retention before the return phase; forgetting has no
    interval there, so its comparison is a point comparison); "continual advantage" compares mean acquisition gain
    (within 0.03 of replay's) and S0 retention before return (at least 0.05 above replay's); a method that helps,
    misses the advantage bar and is not below replay on final average maps to O1's default; O5 is reported and
    changes no default. In O2, V13's method gets 20% replay as one replay batch (next-token loss on the phase's adapter)
    after every fifth incoming batch, as the replay method does.
12. (02:33) **The gates' density ratio in matmul form** (`cl.py` `GMMGate.forward`; `logp` keeps the reference form).
    V13-tiny's continual stage showed the cost: phase S1 of its method took 1,031 s and S2 2,008 s, and every later
    phase adds a fitted gate. On V13-tiny's base with its after_S2 adapters (two fitted gates at 92 sites), one forward of
    an S2 batch took 3.51 s without adapters, 6.30 s with the two gated adapters and 2.68 s with the gates in matmul form
    (15 Oct 02:26, measured while V13-tiny's continual stage held the GPU, so these times are noisy; the comparison, not
    the numbers, is the evidence). The same density ratio, expanded about the mean of the gate's eight component means,
    changed 28 of 6,832,128 gate decisions against the reference form (4.1e-6; V13-tiny's after_S1 and after_S2 gates on
    batches of S1, S2, S3 and S6; largest difference in the log ratio 0.12). Every V14-tiny method uses this form, so
    V14's methods are compared with one gate computation. V13-tiny's pipeline was not changed: its flip rate is above the
    1e-6 bar set for changing a running pre-registered evaluation.
13. (02:46) **Diagnostics, fixed before any V14 result** (`diagnostics_v14.py`, `v5_v13.py`). Rows: 48 fresh dev-time
    rows per family (seed 99, as V13-tiny's diagnostics; text from held-out validation stories); "training rows" for
    the family marginal and the cross-pair table: 64 per family (training seed 1301 at curriculum fraction 1.0; the
    first rows of the training text stream). Targets are the twin's codes under the checkpoint's own maps, normaliser and
    codebook (the codes the head trained on). Checkpoints: snap_010, 030, 050, 070, 090 and final for every read;
    the action sensitivity (64 prefixes per family, K = 8, rows of 256 tokens, 4 prefixes per batch), the cross-pair
    rate and the local half's share of the head's signal on snap_050 and final only (cost). Interpretations: V1-V3 are
    decided over the four reasoning families (V1 "3 of the 4 reasoning families" as written; V2's "2 families" and
    V3's "per family" read the same four, as V13-tiny's probe test did); V4's "every family" is all five, with the
    reasoning-only result reported beside; V5 pools all five families, uses every valid group (V14) or every real
    group (V13-tiny, whose code is that of m_g), and compares snap_010-snap_050 only. Probes are V13-tiny's (MLP 256
    wide, 8 epochs, 80/20 split, seed 0) on all four segments, accuracy reported per half. V1 compares head and
    baselines on the pairs where copy-current is defined (tau_g exists). Effective rank uses the uncentred matrix
    (RankMe). T_glob's gain divides by its norm at step 0, rebuilt from the training seed.
14. (02:46) **§6.1's guard and era refit** (`era_v14.py`, a step added to `post_base_v14.sh` at 02:46 while it was
    only waiting). The guard is diagnostics_v14's pooled twin-versus-online agreement at the final checkpoint (at
    least 0.8 in each half). If it fails, the era is refitted before the continual stage on 24 batches per family of
    base-stream rows (training seed at curriculum fraction 1.0, unseen row indices; training text rows): P_f from
    fresh statistics of the online dl and dG; the normaliser exactly; the words by two passes of the training EMA
    k-means update from the frozen words; then the head alone for 500 steps (AdamW, lr 1e-3, clip 1.0, the training
    horizon loss on concept states cached once; the last 10% of batches held out for its before/after accuracy).
    `cl.py` loads `runs/base/era.pt` when it exists (the bank key, anchoring targets and the head); the base
    evaluation and the diagnostics read the checkpoint as trained.
15. (02:48) **Bug fix before launch: the codebook's counts at initialisation.** The 02:00 benchmark log showed 125 of
    128 words reset at the first update after initialisation: the initialising update set the EMA of rows per step to
    the batch's valid rows but left every word's EMA count at 1, so every word fell below 10% of uniform usage at the
    next update and was reseeded at the worst-fitted targets (segment 4's batch perplexity then sat near 1 for several
    steps). The initialising update now sets every count to uniform usage (rows / N). A G0 check was added (no reset
    after initialisation on Gaussian data; `tests_v14.py` all pass).
16. (13:52, 15 Oct) **Started at 13:49:45 on 15 Oct without V13-tiny's ALL_DONE.** V13-tiny's continual job had run
    11 h in phase S4 with a memory footprint of about 20 GB (7.0 GB of swap, 0.1 GB free), and the user reported the
    laptop was choppy. Given the choice, the user stopped V13-tiny's continual stage and started this run (V13 DEVIATIONS
    item 38). `run_base_v14.sh` took its "post_base.sh gone without ALL_DONE" branch 58 s after the stop. The 24 hours
    run from 13:49:45, to about 13:50 on 16 Oct. Consequence fixed in advance by item 11: V13-tiny's continual matrix
    will not exist, so `cl_choice.py` returns O6 and the continual stage runs the O6 default (base `v13`, no anchoring, no
    replay, 12 positive batches, rank 4). The smokes started with 2.9 GB of swap still in use; it was down to 0.9 GB at
    13:51 and was draining.
    Smokes (item 7's rule, steps >= 10): checkpointing off, median 1.689 s/step (4,650 tokens/s), peak 10.63 GB;
    checkpointing on, median 2.432 s/step (3,204 tokens/s), peak 5.13 GB. Off's peak is over the 9.0 GB ceiling, so the
    rule chose on (13:56:43) and training started then. The draining swap left no visible mark on smoke_off: after the
    first step the wait for rows was at most 0.03 s, and after step 3 no step took more than 2.7 s.
17. (14:12, 15 Oct) **Crash at step 200; training restarted from step 0 with the fix.** `train.py` crashed at 14:05:34,
    at step 200 (8.5 minutes into training), in `ResidueMaps.solve`: `self.A.double().cpu()` asks the MPS device for
    float64, which it does not have. The G0 tests run on CPU, and the 60-step smokes end before the first solve at step
    200, so nothing had exercised this line on MPS. Fixed by moving before converting (`.cpu().double()`, identical
    values). The launcher's restart 1 (14:06:04) had already loaded the unfixed code and would have crashed at the same
    step, so it was stopped at 14:06:58; restart 2 (14:07:28) loaded the fix. No `latest.pt` existed yet, so restart 2
    began from step 0. The 11-hour training budget runs from then (base training ends about 01:10 on 16 Oct); the
    24-hour window still ends about 13:50 on 16 Oct. About 11 minutes of training were lost; 4 of the launcher's 6
    restarts remain.
    The same audit (every float64 use in the V14 files) found two more defects of the same kind, fixed before either
    ran. (a) The resume path loads `latest.pt` with `map_location` set to MPS and then calls `torch.set_rng_state` on the
    CPU generator's state, which must be a CPU tensor. Every restart after the first `latest.pt` save would have failed
    at once. It now moves the state to the CPU first. (b) `diagnostics_v14.py` converted three MPS tensors to float64
    before moving them (lines 201, 299 and 308: the residue maps' R^2, the K-sample target spread and its control). They
    now move first. Every remaining float64 use acts on CPU tensors.
    `runs/base/log.jsonl` is appended across attempts, so it also holds the first attempt's steps 1-150 and restart 1's
    steps 1-3. `make_report.py` keeps the last record of each step, and restart 2's records replace these. Before the
    crash, the first attempt ran steps 50-150 at a median 2.30 s/step (3,439 tokens/s). The restarts log's "exited
    with 0" is a launcher artefact: `$?` is read after `$(date)` has run, so it always prints 0. The launcher is not
    edited while it runs.
18. (21:54, 15 Oct) **Memory-pressure maintenance prepared at the user's request to improve speed.** The running
    trainer's logged MPS driver allocation rose from 4.51 GB initially to 13.16 GB at step 8400 (peak 13.39 GB).
    Throughput fell sharply near step 8000, with data-loader waits still below 0.1 s per step; system swap exceeded
    8 GB. Driver allocation includes unused allocator cache and MPSGraph allocations, so these logs alone cannot
    distinguish cache accumulation from retained live tensors. `train.py` now prepares a cache-maintenance pass
    every 100 steps when driver allocation exceeds 8 GiB: synchronize, collect unreachable Python objects, release
    unused MPS allocator cache, and log before/after driver allocation plus live tensor allocation. Training records
    additionally include `mem_live_gb`. This changes memory management and timing, not the model, losses, batch size,
    curriculum, or optimizer. Syntax checked; effectiveness is unmeasured until a resumed process loads the change.
    The user selected a restart after the next fresh checkpoint. A one-shot guard verified the atomic checkpoint saved
    at 22:10:14 (step 8434), then terminated only PID 42131; the existing launcher resumed training as PID 81895
    from step 8434 and 8.04 elapsed hours. No saved progress was discarded. The first resumed record (step 8450,
    including warm-up) reports 1,970 tokens/s, 4.77 GiB driver memory, 0.21 GiB live tensor memory, and zero skipped
    updates. This establishes immediate recovery after restart, not long-term effectiveness of cache maintenance.
    The next full 50-step interval, ending at step 8500, reaches 3,424 tokens/s and 2.290 s/step, with 5.72 GiB
    driver allocation, 0.79 GiB live tensor allocation, and zero skipped updates. This is 3.14 times the last
    pre-restart logged throughput of 1,091 tokens/s; startup, power, and concurrent-process changes are confounds.
    The concurrent CPU plumbing test (PID 67338 and its three multiprocessing helpers) was lowered from nice 0 to
    nice 15 and given background scheduling. A 45-second pause/resume diagnostic produced no new training log;
    the test was resumed automatically, and PID 67338 had exited by 22:12. AC power was connected by 21:52.
19. (22:26, 15 Oct) **Where the memory growth came from, a cleanup in `cl.py`, and the CPU plumbing test's cost.**
    Written after item 18, in a session that had not seen item 18. A draft of this item, numbered 18, said the system
    killed training at 22:10 and planned a second deliberate restart. Both were wrong: the 22:10:15 stop was item 18's
    deliberate restart, and no further restart was made.
    - Restart 2 (from 14:07:28) logged MPS driver memory growing from 4.71 GB in training hour 0 to 13.39 GB in hour 6.
      Every rise of 0.1 GB or more but one (step 2050, +0.75 GB) came at a gradient-share read (`grad_share`, every 500
      steps from step 1500). Rises were about 0.9-1.5 GB at multiples of 1,000 and 0.1-0.5 GB at the odd 500s; memory
      was flat between reads. Median tokens/s by training hour: 3,049, 2,712, 2,830, 3,210, 3,086, 2,724, 1,422, 904.
      The last two hours ran in swap.
    - Restart 3's records add `mem_live_gb` (item 18). At step 8750 (22:24), driver memory was 5.18 GB with 0.75 GB of
      live tensors, so most of the driver allocation is cache. A CPU check (`runs/plumbing/gs_leak_check.py`: real
      snap_010, B 4, cyclic GC off) ran three reads, and RSS fell (632, 435, 391 MB). One `gc.collect()` then found
      2,443 unreachable objects, including 168 tensors (43 MB) and 44 frames. Reference cycles exist in the eager pass
      with non-reentrant checkpointing, but they hold about 14 MB per read. The rest is consistent with the MPS caching
      allocator keeping the eager read's blocks, whose sizes differ from the compiled step's.
    - Prepared, not loaded by the running trainer: `train.py` also runs `gc.collect(); torch.mps.empty_cache()` right
      after each gradient-share read. This only takes effect if the launcher restarts training; item 18's maintenance
      pass is what the running process uses. `cl.py` runs `gc.collect()` and `torch.mps.empty_cache()` at each phase
      boundary. It uses the same eager-with-checkpointing pattern, and V13-tiny's continual phases grew 7.1, 9.5,
      12.8 GB. `post_base_v14.sh` reads `cl.py` when that step starts. No computation changes.
    - The CPU plumbing test (ADVISOR_LOG row 4 had estimated about 20 minutes) ran 15:13:55-22:11:27, about 7 hours,
      at background QoS. It used up to 2.5 cores: cl_v14's gate fits took 1,270-1,710 s per phase on contended
      efficiency cores. It overlapped training hours 1-8. A Chrome tab using 1.6-2.2 cores (observed 14:17) also
      overlapped the early hours. Every step the test reached exited 0 on the real snap_010: final_eval,
      diagnostics_v14, era_v14 (`--force`), cl_choice (O6), and cl.py seq, v14 (refit era and final-checkpoint era) and
      replay through all six phases. cl_eval was stopped at 22:15, minutes after it started, and make_report was not
      reached; both rerun after base training. Outputs are in `runs/plumbing/`.
