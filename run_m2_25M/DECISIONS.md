# DECISIONS — HRM THINKER-25M run on the MacBook Air M2

Every choice the PROMPT leaves open, every deviation, and every incident, in the order they were made.
Format: id · when (IST) · decision · why · evidence file.

---

## Setup and exact tests

**D-001 · 2026-09-27 02:00 · `hrm_encoder.py --mps` fails as shipped; ran the identical smoke test through a rename shim.**
`EncoderLayer` defines a helper `_apply(self, L, G, y, s)`, which shadows `torch.nn.Module._apply(fn)`; `Module.to(device)`,
`.float()`, `.double()` call `child._apply(fn)` and raise `TypeError: EncoderLayer._apply() missing 3 required positional
arguments`. This is a name collision, not a numerical or architectural problem, so it is not a §10 stop condition.
Fix without editing the reference files: `code/compat.py` renames the helper to `_write_streams` and rebinds `forward`
to it (same statements). Verified bit-identical to the original forward in float64 (tok, c_hat, G), then the original
`hrm_encoder.mps_smoke_test()` body passes: MPS vs CPU max abs diff tok 3.22e-6, c_hat 2.15e-6 (< 1e-3); 0.68 GiB after
backward at B = 2, T = 1024. Evidence: `runs/tests/T_encoder_mps.txt` (the failure), `runs/tests/T_encoder_mps_via_compat.txt`.
Every script in `code/` imports `compat` through `model.py`. The five reference files in `code/` are byte-identical to
the originals.

**D-002 · 02:00 · Python environment.** `python3 -m venv --system-site-packages .venv` reusing the installed PyTorch 2.10.0
(MPS build), tokenizers 0.22.1, datasets 4.5.0, numpy 1.26.4, matplotlib 3.9.2, scikit-learn 1.5.1. No pip installs were
needed. `ENV.txt`.

**D-003 · 02:05 · "Compute parameters" convention.** The PROMPT's 25.05M is encoder non-embedding (15.511M) + concept
module without the codebook (2.010M) + decoder (7.532M); the tied embedding (3.146M) and the codebook (0.049M) are lookups
and are counted only in "stored" (78.579M). With that convention every row of §2.2 matches the live model within 0.31%
(`runs/tests/config_check.txt`). Training cost from the live shapes: 28.0M MACs/token → 168 MFLOP/token (PROMPT: 166).

**D-004 · 02:06 · P5 (new test) design.** P1–P3 rerun with the full §2.1 configuration (d = 384, 6/2 heads × 64, d_ff 1056,
Engram 131,071 × 48, core S = 48, N = 128, confined decoder L = 4, w = 8) at V = 96, T = 512, float64, all weights
randomised (+0.2·N(0,1)), R = 4. All PASS (`runs/tests/P5_d384.txt`). `test_p5.py --ckpt` reruns it on trained weights (§7.1).

## Architecture choices fixed by code or PROMPT (logged because a spec said otherwise)

**D-005 · Scratchpad initial state S⁰ = 0.** The concept spec's review amendment A1 (learned s₀ with ‖s₀‖ ≈ √d_c) was
marked "recommended; needs your approval". `concept_core_v2.py` and PROMPT §2.1 both use S⁰ = 0 (precedence: code > spec).
Not adopted. Its gate item (pass-1 : pass-4 gradient ratio, consecutive-step angle) is still measured in §7 telemetry.

**D-006 · NextLat off, XM latent off, stack decoder only as screen S2, GradCuit not used** (PROMPT §2.4).

**D-007 · Codebook init and restarts under gradient accumulation.** k-means++ init (`ConceptVocab.init_from`) from an
encoder-only `no_grad` forward on the first real training batch, before the first training forward.
`ConceptVocab.step` counts micro-batches, so `restart_dead`'s own `step % 100` check would fire irregularly; the
training loop keeps its own optimizer-step counter and calls a copy of `restart_dead` without that check every 100
optimizer steps, **after** `optimizer.step()` (never between forward and backward). The concepts it re-seeds from are
those of the last micro-batch of that step.

**D-008 · Gradient clipping.** PROMPT §6.3: global norm 1.0 over all parameters (as `e2e.py`). The concept spec §9 R5 warns
that the VQ gradient enters the global norm and can shrink every other gradient. Followed the PROMPT; the codebook's own
gradient norm and the global norm without it are logged every 50 steps, so the effect is measured rather than argued.

## Data (built 2026-09-27 02:05–02:11; `data/MANIFEST.json`, `data/build.log`, `code/build_data.py`, `code/probes.py`)

**D-009 · Raw language text reused from D-MIX.** `~/Desktop/HRM THINKER/experiments/HRM_CC_DUALIRC_vs_NANBEIGE42_M2_10_20M_20260923/dataset/raw/`
holds the HF parquet files of `roneneldan/TinyStories` (train shard 0 + validation) and `Salesforce/wikitext`
`wikitext-103-raw-v1` (train shard 0 + validation + test), with SHA-256 sums (copied into MANIFEST). These are the
PROMPT's datasets, so nothing was downloaded. Budget met from shard 0 alone (TinyStories 52.65M, WikiText 64.35M tokens;
`budget_met: true` for both). D-MIX generators (`tasks.py`) copied unchanged; tokenizer, packing and shards rebuilt here.

**D-010 · FineWeb-Edu not used; `tool_traj` not used.** The eval suites are TinyStories/WikiText only, the local text
meets the budget, and WikiText articles (~4.9k tokens) already supply long documents. `tool_traj` is a D-MIX family that
is not in the PROMPT §4.3 mixture. MANIFEST: `fineweb_edu_used: false`, `tool_traj_used: false`.

**D-011 · Probe evidence floor = 128 tokens (PROMPT: > 90).** Chosen above the S2 stack decoder's token-path reach
(63 + 4·10 = 103), so no probe answer can leak through S2's token path either. ID gaps 128–420 tokens (measured
evidence distance min 145, median ≈ 300); OOD gaps 500–820 (min ≈ 520). Asserted per item in the generator and again in
`check_data.py` (`runs/data_checks.json`). Evidence = last value token of the fact (recall/carry/multi4), the latest
chain fact (lr_hop), the latest binding of x (lr_bind; other variables are rebound closer, x never).

**D-012 · Probe details.** Names = 3 distinct syllables from the D-MIX ID syllable set (3,360 names); OOD names from a
disjoint 16-syllable set that never appears in training (also disjoint from D-MIX's held-out alphabet). Values = 125
English words that encode to exactly one token with a leading space under the final tokenizer (all 125 candidates
passed), so chance = 1/125 per value token; carry3/multi4 values are exactly 3 tokens (asserted). lr_bind answers are
digit tokens only (the preceding space is in the prompt). Documents are assembled from segments encoded once and reused,
so a name/value has identical tokens in the fact and the answer. lr_multi4 queries come in random order; earlier answers
lie within 70 tokens of later queries, so a core-less model can exclude seen values: effective chance per query is
slightly above 1/125 (reported). lr_hop: chain length h+1..h+2 so the answer is never the chain's end (no "terminal node"
shortcut); training uses h ∈ {1..4} (1-hop as an easy rung), evaluation ID h ∈ {2,3,4}, OOD h ∈ {5,6}.
lr_bind: x bound 1–3 times (OOD 4–6).

**D-013 · Canonical ids for special tokens.** `build_canonical_map` decodes with `skip_special_tokens=False`, so `<pad>`
and `<eos>` keep distinct canonical ids instead of both mapping to the empty string. 6,572 canonical ids for V = 8,192.

**D-014 · Language share inside the language group = D-MIX's 45% TinyStories / 55% WikiText.** Realised training
mixture: language 44.3%, symbolic 22.3%, code 10.1%, retrieval 13.0%, long-range 10.4% of 264.37M non-pad tokens
(273,797 rows × 1024; 5.7% padding). No example exceeded 1024 tokens.

**D-015 · Evaluation items: duplicates of training documents dropped, then re-sampled to full size.** With 2.79M
generated training documents, tiny-space families (expr_eval, sort_trace, brackets, arith_transform, …) collide with
training often. Every candidate identical (SHA-1 of its token ids) to a training document is dropped and counted
(`MANIFEST eval.<split>.<family>.dropped_train_duplicates`, e.g. expr_eval val 854 dropped); fresh candidates are drawn
until 200 items (512 per probe family) remain, and duplicates inside one eval set are removed. Consequence: in tiny-space
families the eval items are items never seen verbatim in training, i.e. slightly rarer than the training distribution.
The packed val/test/maintenance shards (mixture rows) report `synthetic_docs_identical_to_train` (≈ 21k of ≈ 100k).

**D-016 · Language eval rows and context.** 256 packed rows each (T = 1024) of TinyStories-val (first half of the HF
validation split), WikiText-valid (hourly eval), TinyStories-test (second half) and WikiText-test (final eval). Per row
the same-document context of the scored last-16 positions is stored (`*_ctx16.npy`): WikiText rows have ≥ 512 tokens of
context in 231/234 of 256 rows, TinyStories rows almost never (median ≈ 100–116), so the truncation curve is reported
overall and restricted to rows with ≥ 512 tokens of context (advisor recommendation).

## Training loop (`code/train.py`, smoke-tested on CPU)

**D-017 · Resume verified.** CPU smoke test (d = 64): 30 steps straight vs 20 steps + resume to 30. Losses agree to
~1e-7 at every logged step; the same 1e-7 difference is already present at step 5, before any resume (multithreaded CPU
reduction order), so resume adds no divergence. The checkpoint holds model, optimizer, step, cursor (rows consumed),
all RNG states (python, numpy, torch CPU, MPS), LR scale, micro-batch/accumulation, schedule length and elapsed time.
Codebook buffers (`hits`, `step`) are in the model state dict.

**D-018 · Codebook init batch.** k-means++ init uses the concepts of the whole first optimizer batch (accum × micro_bs
rows; 8 × 4 × 256 = 8,192 concepts at the main configuration), computed with an encoder-only no-grad forward in eval mode;
the same rows are then trained on (the data cursor is rewound).

## Calibration (MPS, 2026-09-27 02:26–02:33; `runs/calibration.json`, `runs/calibration.log`, `runs/calibration_B4_watermark.txt`)

**D-019 · Micro-batch B = 2, accumulation 16 (32,768 tokens per optimizer step).** 10 warm-up + 30 timed micro-steps each,
real optimizer steps, R ~ U{1..4}:

| B | tok/s | peak allocated | % of recommended (11.84 GiB) | driver | swap Δ |
|---|---:|---:|---:|---:|---:|
| 4 | 1,533 | 7.51 GiB | 63% | 14.26 GiB | **+3,017 MB** |
| 2 | 1,365 | 4.57 GiB | 39% | 8.61 GiB | −32 MB |
| 1 | 1,463 | 3.70 GiB | 31% | 5.76 GiB | 0 MB |

B = 4 meets the 75% allocation rule but swapped (the allocator's cache drove the driver to 14.3 GiB on a 16 GB machine);
retried with the allocator capped at the recommended maximum (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`) it ran out of
memory (8.38 GiB allocated + 3.30 GiB other). So B = 2 is the largest B without swap (PROMPT §6.1). The per-B speeds
are noisy (each 30-micro-step window holds a different number of optimizer steps, and B = 2 ran while the machine was
still paging after the B = 4 test), so B = 1 appearing faster is not trusted; the main run's step count is derived from
S1's measured seconds per step and evaluation time (D-021), not from these windows. Speed is well above the 400 tok/s
stop threshold.

**D-020 · Loss implementation.** NTP uses `F.cross_entropy(..., ignore_index=-100)` with masked targets set to −100
(bit-identical mean to boolean-mask indexing, verified in float64) to avoid a host sync and a logits copy per micro-batch;
loss values are read from the device once per optimizer step.

**D-022 · Screens run to a fixed step count.** S1–S3: `--max_steps 180 --total_steps 180` (≈ 72 min at 24 s/step; the
PROMPT's 75 minutes, fixed in steps so §7.6's comparison at a common step count is exact), cosine to 10% over those 180
steps, warm-up scaled to 16 steps (300 × 180/3419, same schedule shape as the main run), `--hours 1.5` as a safety stop,
evaluation at 60 minutes and at the end (validation language + all five probes at 256 items). The main run gets an extra
evaluation at step 180.

**D-023 · Crash before the first checkpoint.** `run_job.sh` restarts from scratch with the same seed if train.py dies
before its first checkpoint; if that happens because of a non-finite loss, the operator restarts with `--lr_scale 0.5`
instead of letting the launcher retry identically.

**D-024 · Incident: non-finite loss at step 0 on MPS (fixed before any training).** The first S1 launch stopped at step 0
(NTP = NaN in one micro-batch, gradient norm inf) and the launcher retried once identically before being stopped.
Diagnosis: every micro-batch was finite in isolation, with and without gradients; the difference was
`tensor.to("mps", non_blocking=True)` on unpinned CPU tensors produced by the prefetch thread, whose memory can be
released before the asynchronous copy completes. Removed `non_blocking` in `train.py` and `evaluate.py`; three steps then
ran cleanly (loss 12.49 → 12.42). The failed attempt's run folder was deleted (no step was taken).

**D-025 · Screens resized to 280 steps (supersedes the 180 of D-022).** With the copy fixed, the loop runs at
≈ 2,000–2,170 tok/s (≈ 15.6 s per 32k-token step), faster than the calibration window (taken while the machine paged).
75 minutes ≈ 280 steps, so screens run `--max_steps 280 --total_steps 280`, warm-up 16 steps (≈ 300 × 280/5,300, the main
run's expected length), evaluation at 60 minutes and at the end. Main gets `--extra_eval_steps 280`.

## Screens and chaining

**D-026 · S1 stops on its time budget; S2/S3 stop at S1's actual final step (amends D-025).** Steady speed in S1 is
≈ 1,520–1,590 tok/s (≈ 21 s per 32k-token step; the 2,000+ tok/s of the first 3 debug steps did not persist), so S1's
`--hours 1.6` safety expires near step 260–265, before 280. S1 was not restarted. S2 and S3 run with
`--max_steps N --total_steps 280` where N = S1's final step (read from `runs/screens/S1/log.jsonl`), same cosine shape
and warm-up, so all three stop at the same step; main gets `--extra_eval_steps N` (PROMPT §7.6 common-step comparison).

**D-027 · Divergence-guard window fixed before the main run.** The guard needed ≥ 10 smoothed-loss points in the last hour,
but one point is logged per 50 steps (≈ 18 min), so it could never fire. Now ≥ 2 points (the guard compares the NTP EMA
with the minimum over the last hour). S1 ran with the old condition (it never approached a divergence).

**D-028 · S1 learnability check (PROMPT §6.2): PASS.** Train NTP 9.05 → 5.99 (step 50) → 3.82 (100) → 3.44 (150);
validation mean 4.54 nats at step 167 (TinyStories 3.63, WikiText 5.45). Codebook: 100% of codes used in every segment
over the step-100 restart window (39.3M assignments; 0 dead codes; this restart-window number is the G-C1 measure),
79% in the 4k-concept eval sample (telemetry only). NCP CE 3.75 < code-frequency entropy 4.11 (G-C2 already met).
VE 0.545. Probes at chance (5.5M tokens). Hourly quick eval = 141 s (3.9% of an hour). Swap flat, MPS driver 6.3 GiB.

**D-029 · Operator gate before main.** `code/chain.sh` computes the main step count after S3 and waits up to 20 min for
`runs/GO_MAIN` (or an override in `runs/MAIN_STEPS_OVERRIDE`) so the number can be reviewed; it proceeds on its own if no
one answers. The best-checkpoint evaluation runs interventions on the probe families only (`--probe_only_interv`) to keep
the post-training testing near the PROMPT's 2–3 h per checkpoint.

**D-030 · S1 result (`runs/screens/S1/log.jsonl`, final).** Stopped on its 1.6 h budget at step 265 (8.68M tokens).
Validation language 4.341 nats (TinyStories 3.425, WikiText 5.257); probes (val, 256/family) at chance
(recall1 0.008, bind 0.031, others 0); codebook: VE 0.547, NCP CE 3.707 vs code entropy 4.171. Final eval with
256-item probes took 271 s. S2 and S3 therefore stop at step 265 (cosine over 280). S2 started 04:21:58.

**D-031 · S2 result and screen gate (`runs/screens/S2/log.jsonl`).** Stack decoder, step 265: validation language
4.352 nats (TinyStories 3.437, WikiText 5.265) vs S1 (confined) 4.341 (3.425, 5.257): confined − stack = −0.011 nats,
inside ±0.02 → screen gate PASS (single seed, 8.7M tokens). Probes at chance in both. S3 started 06:06:52.

**D-032 · Screen verdicts (single seed, 8.7M tokens, all at step 265; `runs/screens/S*/log.jsonl`).**
S1 confined 4.341 · S2 stack 4.352 · S3 R = 1 4.336 nats (validation language mean). Confined − stack = −0.011 → screen gate
PASS. S3 − S1 = −0.005: tied; the recurrence shows neither cost nor gain at this token count (the F7 question stays open
for the main run). All three: probes at chance, VE 0.54–0.55, NCP CE below code entropy, 100% codes used in every restart
window, no guard events.

**D-021 · Main-run step count = 3,600 (≈ 118M tokens), overriding the computed 3,971.** `main_steps.py` used S1's
steady-state median (1,574 tok/s → 20.8 s/step, eval 141 s/h) and gave 3,971. S1's end-to-end wall rate, including
prefetch, restarts, evaluations and checkpoints, was ≈ 22.8 s/step (265 steps in 1.68 h); at that rate 3,971 steps plus
24 hourly evals would need ≈ 26 h and the 24 h budget would cut the cosine at ≈ 93%. 3,600 × 22.8 s + 24 × 156 s ≈ 23.8 h,
so the schedule completes. Cosine over 3,600 steps, warm-up 300, peak LR 6e-4, 32,768 tokens/step. Numbers:
`runs/main_steps.json`, `runs/MAIN_STEPS_OVERRIDE`. Power: AC, battery 100%; disk 33 GB free; swap 2.59 GB and falling.

## D-033 — Main run extended from 3,600 to 4,050 steps (amends D-021)
- Measured main-run speed: 19.4–20.2 s/step including hourly evals (step 554 at 3.11 h elapsed = 20.2 s/step; 1,650–1,815 tok/s
  between evals). The 3,600 figure came from 22.8 s/step derived from S1, which was inflated by the screen's 256-item final eval
  and the k-means++ init, both absent from the main loop. At 3,600 steps the run would have ended ~3.4 h inside the 24 h budget.
- 4,050 steps × 32,768 = 132.7M tokens; projected ≈ 22.7 h, leaving ~1.3 h margin for restarts. The 24 h wall budget still caps it.
- Procedure (2026-09-27 10:56): hourly eval at step 554 written ckpt_step000554.pt; killed chain.sh, run_job.sh, trainer; set
  ck["total_steps"] and ck["args"]["total_steps"] to 4050 in ckpt_last.pt (code/extend_ckpt.py, atomic replace); ckpt_step000554.pt
  kept untouched as backup; relaunched the chain tail as code/chain2.sh. Resume line verified: step 554, cursor 17728, lr_scale 1.0,
  micro_bs 2 × accum 16, "cosine over 4050", elapsed 3.11 h. Effect on LR: at step 554 the LR multiplier moves from 0.987 to 0.990 (≈0.3%)
  — a one-time 0.3% upward step in LR at step 554, then the decay is stretched over 450 more steps (spliced schedule; noted in REPORT).
- Val at step 554: 3.718 nats (TS 2.831, WK 4.604), already below S1's 4.341; codes_min 0.91 (eval window), VE 0.593.

## D-034 — Pause/resume for the main run (user must travel with the laptop, ~12:00–12:30 on 2026-09-27)
- train.py: at each step boundary, if `<run_dir>/PAUSE` exists → save a checkpoint (best link untouched, no eval) and exit
  with code 4. Elapsed training time is stored in the checkpoint, so paused wall time does not count against the 24 h budget;
  the finish time shifts later by the pause length. LR schedule, data cursor, RNG and optimizer state resume exactly.
- New scripts (running chain2.sh/run_job.sh left untouched on disk because bash reads scripts lazily): run_job2.sh (exit 4 →
  no auto-resume), chain3.sh (chain2 tail; on pause exits before final_eval), pause.sh, resume.sh, swap_at_ckpt.sh.
- The trainer running at the time had loaded the old train.py, so swap_at_ckpt.sh waits for the next hourly save (~12:00),
  then stops chain2/run_job/trainer and relaunches via chain3.sh on the pause-aware code (~1 min cost). If PAUSE already
  exists at that moment it just stops after the save.
- Tested on CPU (--small, scratchpad run dir): pause at step 12 → rc 4, ckpt_step000012 written, launcher did not retry;
  resume → "step 12 ... cursor 12", paused again at step 19 with rc 4.
- Usage: `bash code/pause.sh` (wait for "SAFE TO CLOSE LID"), later `bash code/resume.sh`.
- Behaviour change: chain3.sh stops (logs "main failed") if run_job2 exhausts its retries, instead of running final_eval
  regardless as chain2.sh did. Check runs/chain.log first thing after the run.

## D-035 — Battery guard (user runs on battery at times)
- code/battery_guard.sh checks `pmset -g batt` every 60 s; if discharging at <= 15% while the main trainer runs and no PAUSE
  file exists, it runs code/pause.sh (checkpoint at the current step, exit 4) and posts a macOS notification. It keeps watching
  but never resumes by itself; resume.sh starts the guard if it is not running.
- Dry-run tests (DRY=1, fake readings): 14% discharging → would pause; 9% on AC charging → no action; real 80% → no action.

## D-036 — Architecture + live telemetry web app (webapp/)
- Local, read-only dashboard: `nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > webapp/server.log 2>&1 &`,
  then open http://127.0.0.1:8765 (bound to 127.0.0.1 only). Stdlib-only server (~20 MB RSS, nice 19), never imports torch,
  never writes under runs/; log files are read incrementally by byte offset and the JSON is rebuilt only when they change;
  OS probes (pmset, sysctl, pgrep, ls) at most every 20 s; the page polls every 30 s (every 2 min when hidden).
- Tabs: Overview (param/MAC breakdown), 2D architecture (9 SVG figures, every number from webapp/data/arch.json, which is
  generated from code/model.py → build_model() on the meta device), 3D architecture (Three.js 0.169 vendored in
  webapp/vendor/ so it works offline), Live training (losses, validation, probes intact vs plan swap, codebook health,
  optimisation, memory, evals table, checkpoints, guards, logs), Screens & gates (S1–S3 vs main@265, gate status).
- GPU contention with MPS training: the 3D scene is built only when its tab is first opened, renders on demand (orbit
  change / click), pixel ratio 1, no shadows; the data-flow animation is off by default and stops when the tab is hidden.

## D-037 — Live dashboard layer and per-step heartbeat
- Server: a sampler thread every 5 s (ioreg GPU "Device Utilization %", trainer ps CPU/RSS, swap, battery every 30 s,
  log-file signatures) feeds /api/live; the page polls it every 5 s and refetches /api/telemetry as soon as a log file
  changes. ~20 ms of CPU per sample. Heartbeats are copied to webapp/data/live_steps.jsonl (never under runs/).
- train.py: `--heartbeat 1` (default) writes <run_dir>/live.json once per optimizer step (step, time, loss/ntp/ncp/vq,
  grad norm, lr, R, EMA) via tmp + os.replace, inside try/except so it can never stop training; no extra GPU sync (the
  values are already on the CPU). log_every is unchanged (it also feeds the divergence guard). The running trainer loaded
  the old file, so the heartbeat activates at the trainer's next restart. CPU test (--small, scratchpad): live.json
  updates every step; pause → rc 4, resume → correct step, heartbeat continues.
- UI redesign for readability: story-first overview (three parts, how the concept stream is verified, token journey),
  "Right now" panel (estimated step until the heartbeat is active), health lights with "what it measures / bad would be",
  per-chart "Measures / Good" guides, read-first boxes on every 2D figure, a 9-stop guided 3D tour, Guide & glossary tab.

## D-038 — Chart range selection, per-chart verdicts, live-training page redesign
- Every chart (webapp/js/charts.js · LineChart) supports drag-to-select: shaded range, both points marked, and a table of start → end, change (end − start) and % change per series; a click clears it.
- Every telemetry chart carries a verdict pill: green Healthy / amber Watch / red Problem / grey no verdict, with the reason in words. Rules live in webapp/js/telemetry.js (attachStatus / attachLiveStatus), e.g. training losses must fall over the last 300 steps, validation must fall between evaluations, recall1 must beat plan swap, codebook ≥ 95 % used and VE ≥ 0.6, lr on the planned cosine, grad norm < 0.8, ≥ 1,700 tok/s, swap < 3 GB, learned τ drift < 30 % over 3 evaluations. Thresholds are dashboard heuristics, not PROMPT gates.
- The Live training tab was rebuilt: hero (step, progress with evaluation ticks, ETA, overall verdict listing anything amber/red with jump links), six KPI tiles with sparklines, a sticky section nav whose dots show each section's worst status, per-section status counts, three-column grids for secondary charts.
- Read-only as before: no change to training, the server or any log. Backups of the previous files: scratchpad pre_redesign/.

## D-039 — Whole-dashboard redesign (design system v4)
- webapp/css/style.css rewritten as one organised stylesheet (tokens → base → app bar → primitives → components → charts → per-page), replacing the v1–v3 override layers; every class used by the HTML/JS was checked against it.
- App bar: single row with icon tabs in a segmented control, live pill and theme toggle; wraps to two rows below 1,180 px. Hash routing now follows #tab changes; tab clicks scroll to the top.
- Overview: hero with live status card (step, four key metrics, health verdict), stat strip, three-part cards, an evidence panel with recall1 vs plan-swap bars, health summary, a 7-step journey stepper, Figure 1, and the parameter/setup/compute cards.
- How it works: sticky figure sidebar with scroll tracking; numbered figure cards (diagrams stay white "paper" figures). 3D: page header with controls help, tour bar, glass toolbar and panels (scene colours unchanged).
- Screens & gates: arm cards with Δ vs S1, a verdict panel, and gate cards coloured pass / not yet / final evaluation with a summary. Guide: glossary search plus category filters, colour-coded categories, vertical run timeline.
- Read-only as before; no change to training or the server. Backup of the previous webapp files: scratchpad pre_revamp/.

## D-040 — 3D model v2 (webapp/js/view3d.js)
- Rebuilt on Three.js r169 add-ons, vendored locally in webapp/vendor/jsm/ (same version as vendor/three.module.min.js, no network at runtime): RoundedBoxGeometry, RoomEnvironment image-based lighting (PMREM), soft PCF shadows on a shadow-catcher floor, Khronos Neutral tone mapping, EffectComposer with 4× MSAA + OutlinePass (hover / selection) + OutputPass, CSS2DRenderer labels, damped OrbitControls.
- Clarity: three coloured floor plates numbered 1–3 with titles (click to focus); focus modes dim everything outside the chosen module; flow tubes carry direction chevrons (scroll only when "Animate flow" is on); concepts drawn as spheres; all 6,144 codes as instanced gems; labels switch to key-only when zoomed out.
- Side panel (not an overlay): overview + colour key, per-part details with share of compute parameters and a link to the matching 2D figure, module summaries, and the 9-step tour (progress, autoplay, ← → / Esc keys). Hover tooltip; double-click zooms to a part; reset and PNG export.
- Theme-aware (light/dark) scene. Still renders on demand only (camera tween, damping, explode or the opt-in flow animation), so it does not compete with training for the GPU when idle.

## D-041 — Continuation: +24 h of training after the main run (runs/cont)
- Asked by the user after step ~2,500. The live run is not stretched: its schedule and 24 h budget live in the running
  process (a change needs a restart), a naive stretch to ~8,100 steps would jump the LR from 2.3e-4 to ~4.8e-4, and the
  finished 24 h run with its final evaluation stays the result of record for REPORT.md.
- Instead code/chain4.sh waits for chain3's "chain done" (main + final eval + P5 + plots) and trains runs/cont from
  main's final checkpoint (hard link runs/cont/init_from_main.pt): weights, AdamW moments, data cursor and RNG carried;
  step, clock and best reset. Schedule: warm-up 100 steps from 0.2 × 3e-4 = 6e-5 (main's final LR) to 3e-4 (half the
  original peak), cosine to 0.2 × 3e-4 = 6e-5 over 4,050 steps (re-warm + re-decay, cf. Ibrahim et al. 2024, continual
  pre-training). Same mix; the cursor continues so only unseen rows are used (≈ 0.47 → 0.94 epoch). Final eval on its end.
- train.py gains --init_from, --lr_min_frac (0.1), --warmup_start_frac (0.0), --step_offset (0); defaults reproduce the
  old schedule bit for bit (0 mismatches over 4,300 steps for main and screen args). --resume wins over --init_from; the
  k-means codebook init is skipped on --init_from; the continuation's schedule fields are stored in its checkpoints.
  CPU smoke test (small config): no k-means re-init (codebook moved 0.04% in 2 steps), cursor 6 → 10 → 14, injected
  lr_scale 0.5 reset to 1.0, schedule fields survive a resume, Adam state carried. Backup: runs/train.py.bak_pre_cont.
- pause.sh / resume.sh detect the active run (unchanged behaviour for main); battery_guard2.sh RUN watches any run.
  chain4: single-instance pid lock, NO_CONT kill switch, refuses to start unless main stopped normally
  (max_steps/time_budget) and eval_final.json exists; dry-run tested (normal, kill switch, main failed, paused main,
  resume mode, double arm, stale lock). Armed 22:14 at nice 0 (a first launch from zsh ran at nice 5 via BG_NICE and was
  replaced before anything started).
- Throughput: no configuration change is safe for the experiment. B = 4 swaps on 16 GB (D-019); B = 1 changes the
  per-micro-batch codebook statistics used by dead-code re-seeding; mixed precision would change the numerics halfway
  through. The continuation is a fresh process (clean allocator); speed otherwise depends on keeping the GPU free of
  other work (browser/3D) and on heat (fan-less MacBook Air).
D-041 dashboard: server.py/telemetry.js continuation support (tested on port 8799 with fake runs); live server not restarted per the user's instruction.
- D-041 addendum (dashboard): server `plan` (stage 1 main → final evaluation → done; stage 2 not armed / armed / starting /
  running / paused / evaluating / done / cancelled / not started) and a two-stage strip on Live training and Overview; the
  page derives the same plan from the chain log if an older server is running. Continuation rows are stitched at step +
  step_offset with a "continuation" marker; the LR verdict follows the continuation schedule; loss verdicts are capped at
  amber for the first ~600 continuation steps (re-warm bump). Tested with node on live telemetry and on fake runs; the
  dashboard server was restarted at the user's request (training untouched: trainer pid 78431, chain3, chain4 unchanged).

## D-042 — 2D figures v2 ("How it works")
- Style through the shared helpers in webapp/js/svg.js, geometry frozen (no coordinate, box size or font size changed):
  palette tints aligned with the 3D colours, a soft vertical gradient per palette key (defined once per SVG with the
  per-SVG id prefix, like the arrowheads), 1.1 px strokes, radius 8, swept arrowheads with round caps/joins, frames as
  solid soft panels with the title in a pill. No SVG filters (paint cost on this GPU-shared Mac).
- Elements are tagged (boxes data-k = palette key, arrows data-c = L / G / plan / loss / ink) so webapp/js/figtools.js
  can add, per figure: a key-idea line, highlight chips generated from the tags actually present (focus mode: others
  dim to 0.16), click-to-pin details (tooltip text + source file, Esc clears), zoom / pan through the viewBox
  (buttons, ⌘/Ctrl + scroll, drag when zoomed), fullscreen, SVG / PNG export, "View in 3D" (opens the matching 3D focus
  view), previous / next. The read-first keys render as tiles; the sidebar shows numbered figures and tracks scrolling.
- Verified with a scripted audit before and after: 0 text-box overlaps in all 8 SVG figures (unchanged), every marker
  and gradient reference resolves, no console errors; filter, pin, Esc, zoom / reset and export exercised by script.
  Backups: scratchpad svg.js.pre_2d, diagrams.js.pre_2d (diagrams.js unchanged). Training untouched.

## D-043 — Overnight monitoring (user asleep through the main → continuation transition)
- Audit before arming: chain4 alive (pid 17830, nice 0, parent launchd), trainer untouched (pid 78431), run_job2.sh
  executable, venv present, train.py compiles, cursor after the continuation 259,200 < 273,797 rows (no repeats),
  31 GB free. Found: the Mac sleeps after 1 min idle, and chain3's keep-awake ends at "chain done" up to 60 s before
  chain4 starts its own → added `caffeinate -ims -w <chain4 pid>` (idle sleep only; the lid still sleeps the Mac).
- Monitoring: a recurring session cron at :13 and :43 runs `bash code/opus_check.sh` (read-only snapshot) under
  code/OVERNIGHT_POLICY.md (observer; never stop/restart the trainer or edit running scripts; a short list of allowed
  actions on exact triggers: resume when paused and back on AC, re-arm a dead chain4, start the continuation if only
  main's final-eval file is missing, one resume after a continuation crash, restart the dashboard). A background
  `bash code/opus_watch.sh` exits on any chain/STOP/GUARD/Traceback/launcher event. No automatic kill of a stalled
  trainer (the user said never stop the training); a stall is only notified. Log: runs/opus_monitor.log.
- D-041 addendum 2 (2026-09-28 07:50, user's choice): continuation peak raised from 3e-4 to 6e-4 with a 300-step warm-up
  from 0.1 × peak (6e-5, continuous with the main run's end) and cosine to 0.1 × peak (6e-5): from step 300 on it is
  exactly the main run's schedule. Reason (main-run evaluations): validation gain per 100 steps per 1e-4 of LR stayed
  ≈ −0.006 to −0.009 from step 1,400 to 4,050 across an 8× LR range, i.e. the run was LR-limited, not data-limited; the
  continuation's LR budget Σ(LR) is ≈ 1.8× the 3e-4 plan. Stability at 6e-4 was shown by the main run (grad norm ≤ 0.38,
  no guard events). Expected cost: a larger temporary rise at the start (~+0.1 val, recovered in ~500–800 steps). The
  user's "0 drop" was the final evaluation 11 steps after the 06:26 hourly one (−0.0003), not a stall. Procedure: NO_CONT
  kill switch → chain4 exited (07:52:30) → ARGS edited (backup runs/chain4.sh.bak_lr3e-4) → dry run → re-armed 07:52:56
  (pid 87964, nice 0, keep-awake re-added). Nothing was training; chain3's final evaluation ran undisturbed.
- D-041 addendum 3 (08:00, review requested by the user): schedule-aware loss model (Tissue et al. 2024,
  L = L0 + A·S1^−α − C·S2; S1 = Σ LR, S2 = momentum annealing area after warm-up) fitted to the 21 main-run
  validation points (steps ≥ 365): rmse 0.003; fitted on steps < 3000 it predicts every later eval within 0.004
  (λ = 0.99 / 0.995; λ = 0.998 discarded: hold-out error 0.011–0.015 and an implausible ~0 re-warm bump).
  Finding: ≈ 0.2 nats of the main run's late gain was annealing, so the earlier "constant gain per LR" argument was
  confounded. Predictions at step 8,100 (λ 0.99 / 0.995): 3e-4 peak 2.610 / 2.599; 4.5e-4 2.571 / 2.559; 6e-4 (chosen)
  2.540 / 2.526 → the choice stands (~0.07 lower than 3e-4). Cost: re-warm bump +0.15 / +0.10, back below 2.754 only after
  ~1,450–1,560 steps (≈ 17:00 Mon), not "2–3 evaluations" as first said; dashboard grace window widened to warm-up + 1,300
  steps. The bump size is the least certain number (the main run has no re-warm; extrapolated from the anneal term).
  WSD (6e-4 constant, decay over the last 1,000–1,500 steps) is predicted ~0.05 lower still, but needs a new, untested
  schedule path in train.py shortly before launch, gives uninformative interim evals and leaves an un-annealed model if
  interrupted → noted for a future run, not adopted. A 100-step warm-up changes the end by only ~0.005 → kept 300.

## D-044 — Results tab (final evaluation)
- webapp/server.py: `/api/final` returns runs/main/eval_final.json (whole file), p5_trained.txt and eval_best.json, the
  same for runs/cont once they exist, and the screens' P5 files. webapp/js/final.js renders a Results tab with a
  24 h / 48 h model switch (48 h enabled when runs/cont/eval_final.json exists; KPI tiles then show Δ vs 24 h).
- Sections, each against its reference: KPI tiles; five findings computed from the file (plan swap → chance on all
  probes; concept module off +0.578 / +0.455 nats; R1 vs R4 +0.001 nats; test → OOD drop; G-E1 0.63–0.70 vs 0.99);
  language interventions (intact / c_off / u_off / core_off / plan_swap, Δ vs intact); loop-depth sweeps (language and
  probes, zoomed axes); truncation curve with gain beyond 70 and wrong-plan cost; probe matrix (test / OOD, heat-shaded,
  all interventions and R); per-token partial credit; 21 synthetic families test vs OOD; linear probes vs majority
  baseline and gate; codebook, CRC mix, Engram α and loop dynamics (per-pass charts); final gate verdicts; generations
  (cut at the first <eos>, UTF-8 display repair); P5 on trained weights.
- G-C3 is shown as WITHIN NOISE, not PASS: hop R4 9.2% vs R1 8.6% is 3 items of 512 (z = 0.33).
  G-E3/G-D2 is PARTLY: plan swap → chance on every probe, but the wrong-plan cost (0.925 nats TinyStories) is far
  larger than the gain beyond 70 tokens (0.076). G-E1 and G-D3 NOT MET; G-E0/D0, G-E2, G-C1, G-C2 PASS.
- Dashboard server restarted by pid (training untouched). Note: a `grep "webapp/server.p[y]"` inside a command that
  also contained the server path matched the tool's own shell wrapper and ended that command; always exclude
  `zsh -c` lines or use an explicit pid.

## D-045 — Analysis tab (Claude reads the dashboard and writes the analysis)
- webapp/server.py builds a plain-text digest (~8 KB) of every page's data: architecture and parameter counts, screens,
  every main-run hourly evaluation, training health, the full final evaluation (language interventions, loop-depth
  sweeps, truncation, probe matrices test/OOD, 21 synthetic families, linear probes, codebook/loop health, a sample, P5),
  the gate verdicts and the continuation's status. `/api/analysis?run=1` sends it to Claude through the local Claude Code
  CLI in headless mode (`claude -p --tools "" --model opus --no-session-persistence`, prompt on stdin so no process
  name in any argv, run from an empty temp directory), with instructions to write a plain-language Markdown analysis
  using only digest numbers. The answer is cached in webapp/data/analysis.json; the page shows it with a contents
  sidebar and the exact digest Claude read, and a button to ask again (one job at a time; ~1–3 min).
- First analysis 09:46 (98 s, 1,733 words). Reads only files; the claude process uses CPU/network, not the GPU.
- D-035 addendum (2026-09-28 10:00, user's request): the continuation's battery guard now pauses at ≤ 7 % on battery
  (battery_guard2.sh default THRESH 15 → 7; old guard2 pid 98824 stopped first, file edited, new guard2 pid 4697 started
  via bash at nice 0; backup runs/battery_guard2.sh.bak_thresh15). Arithmetic: drain under training ≈ 0.5 %/min (89 → 73 %
  in 31 min), the guard polls every 60 s, a pause takes ≤ 25 s (end of step + ~10 s checkpoint) or ≤ ~3 min mid-evaluation
  → finished at ≈ 5–6 %. Caveat: macOS's low-battery reserve varies with battery age (446 cycles); if it forced sleep
  during the checkpoint write, the run would resume from the previous hourly checkpoint (≤ 1 h lost). 10 % would add
  ~6 min of cushion. battery_guard.sh (main, 15 %) left running untouched; it only watches runs/main, which is finished.
