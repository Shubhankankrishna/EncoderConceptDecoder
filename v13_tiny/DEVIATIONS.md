# V13-tiny-v2, 24-hour single-seed test: deviations and decisions (written at launch, 14 Oct 2026)

These are recorded before the first scored step. Tomorrow's baseline-transformer run inherits every data and
protocol item here unchanged (same frozen generators, tokenizer, ladders, rows, schedules and evaluation code), so the
two architectures differ only in the model.

## Protocol concessions (from the user's request: one seed, 24 hours, V13 first, baseline second)
1. **One seed** (1301). Every result is exploratory (Dataset_v13 §10.1). The challenge panels (extension rungs,
   new-parameter, composition) need three seeds before a claim is made; none is made here.
2. **Ladder endpoints are fixed a priori, not by reference pilots** (§7.4 wants a transformer and an LSTM pilot to set
   them). V13 runs first, so wide endpoints were chosen from the generator bounds: 10 rungs per task type, the first 8
   trained, the top 2 test-only (the extension part). The same frozen ladders are used for the baseline.
3. **Base budget 11 h of training time**, not 12 h, so that continual learning and evaluation fit in the 24 h.
4. **One sequence bucket** (1,024 tokens) for everything; §6.3 allows several buckets with an identical policy.
5. **The secondary knowledge-update stream (§8.2) is skipped** (§11.3 lists it first among cuts).
6. **No separate continual-calibration stream** (§8.3): continual hyperparameters are V10.3's measured values
   (rank 4, sigma, eta cap 0.1) and the replay share is §9's 20%; none is tuned on the scored stream.
7. **Answer-only track only**; no trace-supervised track.
8. **External anchors and panels (RE-ARC, ConceptARC, 1D-ARC, DeepMind maths) are not run.**

## Data-generator decisions
9. Grid: rules are geo (8 dihedral maps), trans (offsets in [-2, 2], no clipping), recol (a consistent map), sel
   (unique largest/smallest object, canvas kept); crop is the S2 primitive. Count, class filters and pattern
   extension are not implemented (no identifiable finite grammar was written for them in time).
   Identifiability is certified analytically per skeleton over the declared depth-2 grammar.
10. New-parameter panel: colour substitutions a -> b and three translation offsets held out of training by hash
    (manifest.json lists them). The dihedral maps and largest/smallest have too few values to hold any out.
11. Composition: of the 6 legal S0 grid pairs, 3 are trained, 1 is reserved for S1, 1 is the dev panel and 1 is the
    permanent test panel. Lists: filter+rsort is the permanent test composition.
12. State tracking uses a ladder on the number of events (2 ... 40), not on dependency-distance bins in tokens; the
    distance is recorded per example so the §5.7 tag can still be reported.
13. Numeric answers are LSD-first everywhere, including list outputs (the spec's default read literally).
14. Text: 320,000 deduplicated TinyStories training stories (first train shard), 10% of processed tokens.

## Model decisions (V13-tiny-v2 versus v13_NOTES.md's V13-tiny)
15. d = 128, 4 heads; E1 = 2 x [S, S, S, F] with w_E = 8 (§8.2's "more, shallower L layers"); E2 = [M, M, M, F],
    R = 2 (the user's maximum); E3 = [S', S', S', F] with window 16; n_C = 4; N_D = 6 (more blocks, as asked).
    7,744,181 parameters including the 1.05M Engram table and the tied 2,048 x 128 embedding.
16. Exact reach is R_L = 51 and R_D = 66, inside the design bound 53 / 68: Engram follows the first S layer, so its
    n-gram and that layer's window are parallel paths (tests_g0.py).
17. GDN-2: exact chunkwise form with a factorised decay and a log-depth scan between chunks; the per-step log decay
    is bounded below at -5 (alpha >= 0.0067) so every factor is finite in fp32. Matches the token recurrence to 1e-15.
18. The pooled L tap is Pool(sg[L_enc]) (the pool's own weights train), not sg[Pool(L_enc)] (which would freeze them).
19. The concept self-attention and the decoder's concept read have no key-side elementwise gain (query gain and
    temperature kept): the compiled backward of that gain produced non-finite values on MPS at step 1. A guard skips
    any update with a non-finite gradient and counts it in the log.
20. The decoder's null entry is scored with the unrotated query (no position), so the read is shift invariant.
21. Embedding init N(0, 1) with the tied head scaled by 1/sqrt(d).
22. Activation checkpointing (encoder layers and stack sublayers), batch 8 x 1,024, fp32, torch.compile:
    measured 3.2K tokens/s at 5.4 GB; batch 16 swapped (1.5K tokens/s) and checkpointing off cost 10.9 GB for +7%.

## Instrumentation not run during training (§12.7)
23. GDN-2 state norms per loop, gradient shares and code drift are measured afterwards on the 10% snapshots,
    not live (the compiled step has no hooks).

## Changes after launch (evaluation side only; training-path code unchanged)
24. (09:50) `tasks.py`/`evaluate.py`: the new-parameter panel now samples a held-out parameter value directly (it was
    rejection-sampled at about 0.3% acceptance), and the new-parameter and permanent-composition panels skip the
    semantic hash bucket (`want='any'`): their items cannot occur in training by construction (held-out values and
    never-trained pairs). Generation with `mode='train'` is byte-for-byte unchanged, so the training stream is the
    frozen one; `manifest.json` keeps the launch hashes and `manifest_eval_update.json` records the new ones.

## Continual-learning decisions (written before any continual run, 14 Oct 10:00)
25. Sites and optimiser, identical for all three methods: rank-4 LoRA on every projection of E2's four layers, the
    concept module and the decoder (E1, E3, Engram, embeddings and pool frozen); AdamW (0.9, 0.95), LR 2e-3 (V10.3's
    measured value), 10-step warmup, no weight decay, clip 1.0; eager fp32 on MPS with activation checkpointing (train
    mode only switches checkpointing on; the model has no dropout).
26. Incoming data: per phase 100 family-homogeneous batches of 8 rows (cycling over the families whose generator
    changes in that phase, rungs uniform over the trained rungs 0-7, seed 7001), identical rows in identical order for
    every method (equal incoming exposure, not equal compute: v13 runs a wake pass and a sleep pass per batch).
    S2's state-tracking primitive (nested containers, container transfer) is not implemented, so S2 changes grid
    (crop), arithmetic (multiply, divide) and planning (key/door) only.
27. Replay: a reservoir of 1,000 rows, initialised with base-stream rows regenerated at fraction k/k_end from the base
    run's row counters (an approximation of rows actually seen) plus text rows, then reservoir-sampled over every
    incoming row; every 5th step adds one replay batch of a single family (the §9 20% share). Not tuned.
28. v13 wake runs on the current consolidated model (earlier gated adapters plus the current phase adapter, base
    weights frozen) and writes the RBF bank with V10.2's rule (value = -eta dL/du, eta capped so the largest value has
    RMS 0.1, sigma 0.3, partition by the segment-1 PQ code of m_g); the bank holds one 4-batch chunk (8,192 rows) and
    is reset after each chunk. Sleep distils teacher = current model + bank into the phase adapter (KL on scored tokens,
    one epoch over the chunk, gate open).
29. v13 gates: per adapted matrix, hard gate log p_pos(x) > log p_neg(x) with diagonal 4-component GMMs on the live
    input of that matrix (LSL's form; no adapter-off pass, no token-rate smoothing). Phi_pos: the phase's first 12
    batches; Phi_neg: 32 base-stream rows, 8 text rows and the first 3 batches of every earlier phase. E2's two loops
    share one gate fitted on inputs pooled over both loops. Up to 3,000 inputs per matrix per side.
30. The continual matrix A[i, j] uses teacher-forced exact match on 24 items per trained rung; planning is scored by
    teacher-forced match to the canonical BFS plan (a lower bound on success: any other optimal plan counts as a
    failure). Simulator-verified generation is used only in the base test.
31. The continual stage starts from the base run's final checkpoint (11 h); one seed; no calibration stream (item 6).
32. (10:30, before any continual timing existed) **Continual budget rule.** After training, a B = 8 probe (v13 on S1
    and S2 for 4 steps, seq on S1 for 5 steps; MPS plumbing and timing on the idle machine) feeds `cl_budget.py`:
    the steps per phase n are the largest n in [40, 100] for which seq, v13 and replay are predicted to finish by
    04:00 IST on 15 Oct, leaving time for the continual matrix and the report before the 24-hour deadline. The same n
    applies to all three methods. 100 is the intended value; the rule only shortens the stage if it is slower than
    expected. `CL_STEPS` overrides it (not used unless recorded here).
33. Incoming rows of each phase are generated once and cached (`runs/cl/rows/`), so the three methods see the same
    bytes; v13's Phi_neg rows are built only for v13.
34. Side jobs during the base run (panel building, CPU plumbing tests) ran at background QoS (`taskpolicy -b`), on the
    efficiency cores: at default priority, three panel workers had slowed training from 2.5 to 3.45 s/step (09:30-09:53).
    Base-run throughput was therefore below the launch benchmark for about 40 minutes; the report gives the actual
    token count.
35. (20:22) The continual-budget probe (item 32) did not run on an idle machine. Besides the cl_test panel build's
    four workers (background QoS, efficiency cores, running since 11:34), a single-process CPU profiler of the cl_test
    generator cells (background QoS, written for the V14 design session) ran from about 19:45 until it was stopped at
    20:21:41, overlapping the first six minutes of `probe_v13` (started 20:15:28). cl_budget.py reads only the probe's
    per-batch MPS step times (s_train, s_wake + s_sleep) and the gate fit time, all on default-QoS threads that the
    background jobs cannot preempt on the performance cores; any slowdown makes n smaller, not larger. Recorded, not
    corrected: the probe is not re-run. A network-bound paper download for the V14 design (curl at default QoS, one
    request per second) also began at about 20:36, overlapping the last minute of `probe_seq` (20:24:05-20:37:02).
36. (21:15, approved by the user) **cl_test built with a per-cell wall-time cap.** The uncapped build (started 11:34,
    four background-QoS workers) had not finished after 9 h 40 min. A profile of the cells (20-try calls) found
    phase-specific grid cells where one call took up to 220 s and returned nothing (S1 grid2 rung 2; S2 grid1 rungs 0
    and 2): each episode runs an identifiability search, and only 10% of instances fall in the test split, so such a
    cell could take hundreds of hours at the build's 200 inner and 720 outer tries. cl_eval.py builds the panel itself
    when the file is missing, so it would have hung. `build_cl_test_capped.py` rebuilds the same cells with the same
    seeds and the same sampling loop, so every item is identical to what the uncapped build would produce; a cell stops
    early only at 300 s of wall time (background QoS) and keeps the items it has. Items per cell are in
    `data/panels/cl_test_caps.json`; cells with fewer than 24 items are reported with their n, and a cell with none is
    reported as unmeasured. **Result** (finished 22:08:30): 7,193 items; 295 of the 312 cells full, 17 capped; 3,161 s of wall time. The
    cap is checked between tries, so a cell whose single try runs long overshoots it: the longest, (grid2, rung 2, id,
    S3c), ran 829 s for 2 items. `data/panels/cl_test.pkl` was in place about six hours before cl_eval needs it.
37. (23:39) **The continual stage will not finish by 04:00; projected end of cl_eval about 10:30-11:00 on 15 Oct.**
    Item 32's budget rule counts each method's per-batch training time (s_train, s_wake + s_sleep, from the probe),
    gates, loads and saving, and 15 minutes for generating the incoming rows of all phases (`cl_budget.py` line 30). Row
    generation is far slower: in cl_seq, S1's 800 rows took about 90 minutes (21:27:51-22:58) and S2's about 16
    (23:07-23:23), against 511 and 523 s of training. S2's rows (grid1, arith, plan) cost about 6 worker-seconds each,
    so S1's grid2 rows cost about 117 worker-seconds each (inferred, not timed per row; S3, which has every family,
    checks it at about 00:35). The rule chose n = 100 (its maximum) on a prediction that omitted this. Projection from
    these rates: cl_seq ends about 04:40 (S5 has about 270 grid2 rows), cl_v13 about 07:15 (rows cached), cl_replay
    about 09:15 (its 1,000-row base-stream reservoir is generated uncached), cl_eval about 10:30-11:00 (19 adapter
    states on the 7,193-item panel, about 6 minutes each from the diagnostics run's per-item rate). **Not corrected:**
    steps, tasks and `post_base.sh` are unchanged; REPORT §6 at 07:30 reports what has finished then, as the
    pre-registered "did not finish by the deadline" outcome, and the full matrix when cl_eval ends.
    **Correction (01:25, 15 Oct): the projection above was wrong.** The later phases' rows took minutes, not hours
    (S2 16.8, S3 5.9, S4 1.0, S5 19.5, S6 1.5 minutes, from the cache files' times), so the per-row grid2 cost inferred
    from S1 is not a property of grid2: the slow cells are phase-specific, as item 36 found for the test panel (S1 grid2
    rung 2; S2 grid1 rungs 0 and 2), and S1's rows also overlapped CPU side jobs at background QoS (the cl_test build
    until 22:08, the V14 checks until 22:43). cl_seq ended at 00:36:32 (188.6 minutes) and cl_v13's S1 took 17.5
    minutes. Revised projection: cl_v13 about 02:20, cl_replay about 03:45, cl_eval about 05:40, so the stage should
    finish before 07:30. Nothing in the run was changed, before or after this correction.
    **Second correction (06:54, 15 Oct): the revised projection was wrong as well.** It assumed every phase of V13's
    method would take about S1's 17.5 minutes. Each phase adds a fitted gate (two 4-component diagonal Gaussian mixtures
    evaluated on every adapted projection's input), and the phases grew: S1 1,031 s at 7.1 GB MPS driver memory, S2
    2,009 s at 9.5 GB, S3 4,386 s at 12.8 GB, with the machine in swap from S3 on. S4 started at 02:40:38 and was still
    running at 06:53. REPORT §6 was filled at the deadline with the continual reading marked "did not finish". Nothing in
    the run was changed; a stop-and-resume with an algebraically identical gate computation was prepared at 02:52 and
    awaits the user's decision (it would be item 38).
38. (13:48, 15 Oct, decided by the user) **The continual stage was stopped in S4; V13-tiny has no continual-learning
    matrix.** At 13:44 the `cl_v13` process's memory footprint was about 20 GB on the 16 GB machine (system: 9.4 GB
    wired, 3.8 GB compressed, 7.0 GB of swap used, 0.1 GB free), it had used 3 h 25 min of CPU time in 13 h 08 min, and
    S4 had run 11 h 04 min (S3 took 73 minutes). The user asked why the laptop was choppy. Offered three choices (stop and
    resume S4-S6 with the matmul gate form prepared at 02:52; stop and start V14-tiny now; leave it running), the user
    chose to stop and start V14-tiny. `post_base.sh` (pid 45911, with its `caffeinate`) and then `cl_v13` (pid 84190, with
    its five data workers) were killed at 13:48:47; free memory rose to 12 GB within 11 s. `cl_replay`, `cl_eval` and
    `make_report` did not run, so reading 5 stays "did not finish" (REPORT §6.6). Kept: `runs/cl/seq/` (all six phases),
    `runs/cl/v13/after_S1.pt`, `after_S2.pt`, `after_S3.pt` and `log.jsonl`, and the cached phase rows. The prepared
    stop-and-resume script is saved, never run, as `cl_resume_item38.py` (cl.py plus `--resume` from the latest
    `after_*.pt`, a `runs/cl/HOLD` wait skipped by `--ignore-hold`, and the two-matmul form of `GMMGate.forward`, which
    changed 28 of 6,832,128 gate decisions on the after_S1/after_S2 gates). Finishing the stage after V14-tiny would
    also need that gate form in `cl_eval.py`, which imports `cl.py`'s `GMMGate`.
