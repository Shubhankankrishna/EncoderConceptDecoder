# PROMPT — Build, train for 24 hours, and test HRM THINKER-25M on the MacBook Air M2

*For the agent that will run this on the user's MacBook Air M2 (16 GB unified memory, 8-core GPU). Written 2026-09-27. Read the whole prompt before touching anything. The user expects a finished, tested model and a report, not a plan.*

---

## 0. Your job in one paragraph

Assemble the three designed modules — **encoder v1.0 (frozen)**, **concept module v2 (looped core + NCP)** and **decoder v1 (confined)** — into one NCP-style language model at width **d = 384 (≈25M compute parameters, ≈79M stored)**. Build the tokenizer, the training data and a fixed evaluation benchmark. Verify everything with the existing exact unit tests. Calibrate memory and speed on MPS, run three short learnability screens, then **train the main model for 24 hours of wall-clock time**, and finally run the full test matrix in §7 and write `REPORT.md`. Every number in the report must come from a file you produced.

### Ground rules

1. **Do not redesign.** The architecture is fixed by the specs in the folder. If something in them seems wrong, log it in `DECISIONS.md`, keep the spec's version, and mention it in the report. The only things you choose are listed as "you choose" below.
2. **Never edit the originals.** Copy code into a run folder (§1.3) and work there.
3. **Exact tests before training.** All unit tests in §3.3 must pass bit-exactly before the first training step.
4. **One MPS job at a time.** Two processes on the GPU caused allocator growth and swap in earlier runs.
5. **Everything resumable.** Atomic checkpoints, a resume flag, and logs you can re-plot from.
6. **Stop and ask the user** only in the situations listed in §10. Otherwise make the most reasonable choice, write it in `DECISIONS.md`, and continue.

---

## 1. Inputs

### 1.1 The folder

`~/Desktop/EncoderConceptDecoder/` contains:

```
HRM_THINKER_PAPERS.md        every paper behind the design and what it did (context only)
Encoder Design/              encoder spec, reference code, tests, harness
Concept Design/              concept-module conversation, test benches, results
Decoder Design/              decoder spec, reference code, end-to-end harness, results
(+ the paper PDFs)
```

First **inventory** it: `find ~/Desktop/EncoderConceptDecoder -type f | sort > run_m2_25M/INVENTORY.txt`. If `decoder_v1_bundle.zip` is present and the `.py` files are not, unzip it into the run folder. The files you need (names may carry an upload prefix such as `14d88e15-…_hrm_encoder.py`; match on the suffix):

| file | what it is | how you use it |
|---|---|---|
| `HRM_THINKER_ENCODER_v1_SPEC.md` | frozen encoder spec | source of truth for the encoder |
| `hrm_encoder.py` | encoder reference code + unit tests | import unchanged |
| `HRM_THINKER_DECODER_v1_SPEC.md` | decoder spec (derivation, tests, gates) | source of truth for the decoder |
| `hrm_decoder.py` | decoder reference code + tests D1–D5 | import unchanged |
| `concept_core_v2.py` | concept module v2: one looped FULL layer, PQ vocabulary, NCP cross-entropy | import unchanged |
| `e2e.py` | encoder → core → decoder wiring (`HRMThinker`), pipeline tests P1–P4, training/eval loop for synthetic tasks | reuse the wiring, the tests and the interventions |
| `harness.py` | the encoder harness's synthetic task (Markov filler) | used by `e2e.py` |
| `probe_multi.py` | linear probes on concepts and core outputs | reuse for §7.5 |
| `Conversation.rtf`, `concept_v2_testbench.py`, `exp_xm.py`, `exp_gradcuit.py`, `RESULTS*.txt` | concept-module history (v2, v2.1 XM, GradCuit) | context; v2.1's XM latent is **not** in the main run (§2.4) |
| `ARCHITECTURES_AND_PAPERS.md` | earlier programmes, failure ledger F1–F7, constraints C1–C9, dataset D-MIX | the "why" behind the gates |

### 1.2 Precedence

Code + its tests > spec > conversation. If the decoder spec and `hrm_decoder.py` disagree, the code wins and you log it.

### 1.3 Run folder

Create `~/Desktop/EncoderConceptDecoder/run_m2_25M/` with:

```
code/        copies of hrm_encoder.py, concept_core_v2.py, hrm_decoder.py, e2e.py, harness.py + your new files
data/        tokenizer, canonical-id table, packed shards, eval sets, MANIFEST.json
runs/        screens/ and main/ (logs, checkpoints, eval JSON)
REPORT.md    DECISIONS.md    INVENTORY.txt    ENV.txt
```

---

## 2. The model to assemble (HRM THINKER-25M)

### 2.1 Configuration

| module | setting | value |
|---|---|---|
| shared | vocabulary / row length / chunk | V = 8192 byte-level BPE, T = 1024, k = 4 |
| shared | width, heads | d = d_c = 384; 6 query heads, 2 KV heads, head dim 64; SwiGLU hidden 1056 |
| encoder v1.0 | everything else | exactly `EncoderConfig` defaults: pattern `SSSFSSSF`, SWA window 8, partial RoPE 16/64, NoPE + τ in FULL, q/k/v short convs, Engram after S1 (orders 2,3; 4 heads; 131,071 rows × 48), CRC depth-mix → concat-project pooling → sphere, gated residual with streams L/G, tied embedding |
| concept module v2 | `ConceptCoreV2(ecfg, S=48, N=128)` | one shared FULL layer looped R times, reads {X, S}, writes S; X = concepts (re-read every pass, gradient flows), S⁰ = 0; **R ~ U{1..4} per training step, R = 4 at evaluation**; product-quantised vocabulary 48 segments × 128 codes × 8 dims, k-means++ init from the first batch of real concepts, dead-code restart every 100 steps **after** `optimizer.step()`; NCP = cross-entropy on the next concept's code indices; u = zcRMSNorm(S), ĉ = Σπ·sg(e) |
| decoder v1 | `DecoderConfig(d_model=384, n_heads=6, n_kv_heads=2, head_dim=64, d_ff=1056, n_layers=4, window=8, layout="confined", plan_in="add")` | D⁰ = tok + P; queries from D, keys/values from `tok` only; plan read at j*(t) = ⌊(t+1)/4⌋ − 1 with offset embedding and learned null plan; reach 70 at any depth; tied head |
| loss | $\mathcal L = \mathcal L_{\text{NTP}} + 0.1\,\mathcal L_{\text{NCP}} + \mathcal L_{\text{VQ}}$ | NextLat **off** |

In code the wiring is `e2e.HRMThinker` with `configs()` replaced by the table above. Keep the rule that the decoder receives only `tok`, `u`, `c_pred`, `doc` and `E`.

### 2.2 Expected sizes (computed from the code; your instantiation must match within 1%)

| part | parameters |
|---|---|
| encoder, non-embedding | 15.51M |
| Engram table (lookup only) | 50.33M |
| token embedding (tied head) | 3.15M |
| concept module | 2.06M (codebook 0.049M) |
| decoder | 7.53M |
| **compute parameters** | **25.05M** |
| **stored total** | **78.58M** |

Training cost ≈ **166 MFLOP per token** (encoder 16M, core 1.3M, decoder 7.3M, head 3.1M multiply-accumulates per token, ×6). At the 0.1–0.3 TFLOP/s measured on this machine before, that is **≈600–1,800 tokens/s, ≈52–156M tokens in 24 h**.

### 2.3 Guarantees that must still hold (tested in §3.3, re-checked on the trained model in §7)

- The decoder reads only the encoder's L stream (`tok`), never G.
- Position t reads the core only at chunk j*(t); the null plan at the first 3 tokens of every document.
- The decoder's reach without the core is exactly 70 tokens; everything older reaches the prediction only through the concept module.
- No position reads another position's plan.
- The codebook learns only from VQ; NCP targets and VQ targets are stop-gradient encoder concepts.

### 2.4 What is deliberately not in the main run

- **XM latent (v2.1, K = 4)** and **GradCuit steering**: tested on the concept test bench only; they are not part of the main model. If time remains after §7, you may run the XM arm as an extra screen (§6.2, arm S4); never change the main run for it.
- **NextLat**: an optional arm; measured neutral-to-slightly-negative on the decoder test. Off.
- **The stack decoder** (`layout="stack"`): only as a control arm in the screens.

---

## 3. Environment and exact tests (≈1 h)

### 3.1 Setup

```bash
cd ~/Desktop/EncoderConceptDecoder/run_m2_25M
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy tokenizers datasets matplotlib scikit-learn
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -c "import torch;print(torch.__version__, torch.backends.mps.is_available())"
```

Write `ENV.txt`: macOS version, `sysctl -n machdep.cpu.brand_string hw.memsize`, Python and PyTorch versions, `torch.mps.recommended_max_memory()`, free disk. Keep the Mac on power with the lid open, and wrap every long job in `caffeinate -dimsu`. Do **not** use `torch.compile` on MPS. Train in float32. bfloat16 autocast is allowed only after a 200-step loss-parity check against float32; keep softmaxes, norms, the sphere projection and all losses in float32.

### 3.2 Configuration check

Instantiate the d = 384 model, print the table of §2.2 from the live model, and confirm `DecoderConfig.reach() == 70` and `EncoderConfig.reach() == 63`.

### 3.3 Exact tests (CPU, float64 — must all pass)

```bash
python code/hrm_encoder.py          # T1–T9: reach 63, causality, document isolation, init values
python code/hrm_decoder.py          # D1–D5: decoder reach, plan confinement, null plan, gradients
python code/e2e.py --test           # P1–P4: whole-pipeline causality, reach 70, documents, gradient routing
python code/hrm_encoder.py --mps    # CPU vs MPS forward agreement < 1e-3, memory report
```

Then add one new test of your own, **P5 (d = 384 config)**: rerun P1–P3 with the §2.1 configuration at small V and T in float64. Record every PASS line in the report. If anything fails, stop (§10).

---

## 4. Data (≈2–3 h, CPU)

### 4.1 Reuse first

The earlier programme built **D-MIX** (93.4M tokens: TinyStories, WikiText-103 and 22 synthetic families, byte-level BPE 8192). Look for it:

```bash
find ~ -name DATASET_DESIGN.md -path "*HRM_CC_DUALIRC*" 2>/dev/null
```

If found, reuse its **generators and design** (families, splits, leakage check), but rebuild the tokenizer and shards here so that everything below holds. If not found, implement the families listed in §4.3 from their descriptions in `ARCHITECTURES_AND_PAPERS.md` §1A.1.

### 4.2 Natural language (download)

With `datasets`: `roneneldan/TinyStories` and `Salesforce/wikitext` (`wikitext-103-raw-v1`). Optionally add a FineWeb-Edu sample (`HuggingFaceFW/fineweb-edu`, `sample-10BT`, streamed, ≤ 60M tokens) to make the language less toy-like; log whether you did. Use the datasets' own validation/test splits for the language evaluation, never training text.

### 4.3 Mixture (by training tokens)

| group | share | content |
|---|---|---|
| natural language | 45% | TinyStories, WikiText-103 (+ FineWeb-Edu if used) |
| symbolic / compositional | 22% | pointer_chase, graph_walk, state_machine, permutation, variable_binding, kv_updates, relational, arith_transform, stack_ops, nested_scopes, brackets, seq_transform |
| code / structured | 10% | py_exec, json_build, yaml_lookup, sort_trace, expr_eval |
| exact detail / retrieval | 13% | kv_retrieval, copy, needle, json_lookup |
| **long-range probes (new)** | 10% | the five families of §4.4 |

Build **≥ 250M training tokens** so the 24-hour run never repeats data (it will see at most ~160M).

### 4.4 Long-range probe families (the tests that isolate the concept route)

Every item puts its evidence **more than 90 tokens** before the answer, beyond the token path's 70-token reach (assert this in the generator). Fillers are natural-language sentences sampled from the training text, so the probes look like documents, not like isolated puzzles.

| family | layout (schematic) | what it tests |
|---|---|---|
| `lr_recall1` | fact "the code of NAME is V" … filler … "the code of NAME is ▢" (1-token value) | one long-range retrieval |
| `lr_carry3` | one fact with a 3-token value … filler … query | writing out a 3-token plan (decoder unrolling) |
| `lr_multi4` | four facts with 3-token values … filler … four queries | retrieval among several facts + unrolling (the decoder spec's gate G-D3) |
| `lr_hop` | chain "A points to B", "B points to C" (2–4 hops, facts in random order, all > 90 tokens before the query) | composition in the concept module |
| `lr_bind` | "let x = 7" … filler (other variables rebound) … "x + 2 =" | tracking the latest binding |

Use names built from syllables; hold out a disjoint syllable set and longer hop counts for the OOD split.

### 4.5 Tokenizer, packing and indices

1. Train a byte-level BPE with 8,192 tokens (HF `tokenizers`) on training text only, with digits split.
2. Build the canonical-id table of the encoder spec §5.8 (strip the leading space marker, NFKC, lowercase) and save it.
3. Pack rows of T = 1024: each document ends with EOS and is **padded to a multiple of 4** (padding takes the id of the document it pads); a `doc` id per position. The NTP loss ignores targets in another document and targets that are padding tokens; EOS is a real target.
4. Compute `engram_indices(canon[ids], doc, cfg)` in the data loader on CPU.
5. Store shards as `uint16` token arrays + `int32` doc ids (`np.memmap`).
6. Splits: train, validation (for the hourly evals), test (final evals), OOD, and a small maintenance split (codebook calibration; never trained on).
7. Leakage: SHA-1 every generated document; drop any eval item identical to a training document and count the drops.

Write `data/MANIFEST.json`: token counts per family and split, tokenizer hash, generator seeds, drop counts.

### 4.6 Data checks before training

- Decode 3 random rows per family and paste them into the report.
- Assert every probe's evidence-to-answer distance > 90 tokens.
- Assert no document length is odd modulo 4 after padding.

---

## 5. The evaluation benchmark (build it before training; freeze it)

### 5.1 Fixed evaluation sets

| suite | size | metric |
|---|---|---|
| language | 256 rows each of TinyStories-val and WikiText-test | next-token loss (nats), also bits per byte |
| synthetic families (§4.3) | 200 items per family, ID and OOD | teacher-forced greedy exact match on the answer span; macro average |
| long-range probes (§4.4) | 512 items per family, ID and OOD | exact match (all answer tokens), per-offset accuracy for 3-token values |

### 5.2 Interventions (run on the same sets)

| name | how | what it measures |
|---|---|---|
| core off | `core_off=True` (null plan everywhere) | upper bound on the core's share; off-distribution, so **not** a skill share for language loss |
| **plan swap** | replace `u` and `c_pred` with another row's (`roll(1, dim=0)`) | the core's real share: long-range probes must fall to chance |
| u off / ĉ off | `use_u=False` / `use_c=False` | which plan vector carries information |
| loop depth | evaluate with R = 1, 2, 3, 4, 6, 8 | is the recurrence used; does depth extrapolate on `lr_hop` |
| context truncation | score only the **last 16 positions** of each language test row; feed the model only the last r ∈ {72, 128, 256, 512, 1024} tokens of the row (r is a multiple of 4, so chunks stay aligned; the cut row is treated as a new document). At r = 72 every scored token has 56–71 tokens of context, i.e. about the token path's reach | the gain from context beyond ~70 tokens, loss(r = 72) − loss(r = 1024), is what only the concept route can deliver; compare it with the plan-swap cost measured on the same 16 positions (gate G-D2) |

### 5.3 Health telemetry (log every eval)

Codebook: codes used per segment (% of 128), perplexity, variance explained by the PQ reconstruction. NCP cross-entropy vs the entropy of the code frequencies. Encoder CRC mix weights (e0 / L / G), Engram W_V norm, FULL-layer temperatures τ_h. Gradient norm before clipping. Tokens/s. `torch.mps.current_allocated_memory()` and `driver_allocated_memory()`. Swap usage (`sysctl vm.swapusage`).

### 5.4 Gates (the report must give PASS / FAIL / NOT REACHED for each)

| gate | criterion |
|---|---|
| G-E0 / G-D0 | all exact tests pass (§3.3) |
| G-E1 | linear probe from each concept recovers its 4 tokens at ≥ 99% top-1 |
| G-E2 | PQ variance explained on concepts (report the number; ≥ 0.6 expected) |
| G-C1 | ≥ 95% of codes used in every segment |
| G-C2 | NCP cross-entropy below the code-frequency entropy |
| G-E3 / G-D2 | plan swap drops every long-range probe to chance; the language-loss cost of the swap is of the same order as the gain from context beyond ~70 tokens (truncation curve) |
| G-C3 | loop depth: R = 4 beats R = 1 on `lr_hop`; report the curve over R |
| G-D3 | `lr_multi4` learned (exact ≥ 0.5); if not, run the probes of §7.5 to say where it breaks |
| screen gate | the confined decoder is within ±0.02 nats of the stack decoder on language at equal steps (§6.2) |

---

## 6. Training protocol

### 6.1 Calibration (≤ 45 min)

1. Micro-batch search on MPS at T = 1024: try B = 4, then 2, then 1. Keep the largest B whose peak allocated memory is ≤ 75% of `torch.mps.recommended_max_memory()` and causes no swap. Measure tokens/s over 30 steps after 10 warm-up steps.
2. Choose gradient accumulation so each optimizer step sees **≈ 32k tokens** (e.g. B = 4 × T = 1024 × 8).
3. From the measured speed, compute the step count that fits 24 h, minus 5% for evaluations. The cosine schedule below uses that count.
4. If speed is below 400 tokens/s, see §10.

### 6.2 Learnability screens (≈ 3–4 h; run before the main run)

Each screen: same data and schedule shape, 75 minutes, one seed, evaluate on validation language + all five long-range probe families at the end.

| arm | change from the main model | question |
|---|---|---|
| S1 | none | does the full pipeline learn at all (loss falling, codebook healthy, plan used)? |
| S2 | decoder `layout="stack"` | confined vs NCP-style stack (screen gate, §5.4) |
| S3 | concept module fixed at R = 1 in training and evaluation | is the recurrence doing anything the one-pass core cannot? |
| S4 (only if time after §7) | XM latent K = 4 with relaxed winner-take-all (ε = 0.05), decoder reads a random k in training | does v2.1 change language loss (must be within 0.02 nats) or probes |

If S1 shows no learning (loss flat after 60 minutes, or all codebook codes collapse into < 10% use), stop and fix before the main run (§10).

### 6.3 Main run (24 h wall clock)

| hyperparameter | value |
|---|---|
| optimiser | AdamW, β = (0.9, 0.95), ε = 1e-8, weight decay 0.1 on matrices and zero-centred gains; **no** decay on embeddings, Engram table, conv kernels, log τ, biases, codebook, null plan, chunk offsets (use `hrm_decoder.param_groups`) |
| learning rate | peak **6e-4**, 300 warm-up steps, cosine to 10% over the step count from §6.1. (Recurrent cores are LR-sensitive, C8: never above 1e-3.) |
| batch | ≈ 32k tokens per optimizer step |
| clipping | global norm 1.0 (log the pre-clip norm) |
| loop passes | R ~ U{1..4} per optimizer step (same R for all micro-batches of that step) |
| codebook | k-means++ init from the first real batch **before the first forward pass**; dead-code restart every **100 optimizer steps**, called **after** `optimizer.step()`. Note: `ConceptVocab.step` counts forward passes (micro-batches), so with gradient accumulation `restart_dead`'s own `step % 100` check fires irregularly. Keep your own optimizer-step counter and call a copy of `restart_dead` without that check (log it in `DECISIONS.md`) |
| precision | float32 (bf16 only after the parity check of §3.1) |
| seed | 0 |

**Every 60 minutes:** save a checkpoint atomically (write `ckpt.tmp`, then rename; keep the last 3 plus the best by validation loss); run the quick eval (validation language, the five long-range probes at 128 items each, plan swap on `lr_recall1`, codebook health); append one JSON line to `runs/main/log.jsonl`. Log training loss, NCP loss, VQ loss, gradient norm and tokens/s every 50 steps.

**Resume:** `--resume runs/main/ckpt_last.pt` must restore model, optimizer, scheduler, codebook buffers, RNG states and the data cursor.

**Guards:** if loss becomes NaN or increases 3× over its 1-hour minimum, reload the last checkpoint, halve the learning rate for the rest of the run, and log it in `DECISIONS.md`. If memory exceeds 90% of the recommended maximum, save, halve the micro-batch (doubling accumulation) and resume.

---

## 7. Detailed testing after training (≈ 2–3 h)

Run everything on the **final** checkpoint and on the **best** checkpoint; report both when they differ.

1. **Re-run the exact tests** on the trained weights: causality (P1), core-off reach exactly 70 (P2), document isolation (P3). Trained weights must not change any guarantee.
2. **Full benchmark** of §5.1 (test split, ID and OOD).
3. **All interventions** of §5.2 on every suite.
4. **Learning curves:** language loss, probe exact match, plan-swap cost and codebook usage over the 24 hours. Plot them (matplotlib PNGs in `runs/main/figs/`).
5. **Probes** (`probe_multi.py`, extended): from concepts → each of the chunk's 4 tokens (G-E1); from core output u at a query chunk → the answer tokens, for each probe family; from u after pass 1, 2, 4 → the bridge entity on `lr_hop` (does each pass add a hop?).
6. **Screens vs main:** compare S1–S3 (and S4 if run) at their common step count.
7. **Generation sanity:** 10 greedy and 10 sampled (temperature 0.8) continuations of 128 tokens from TinyStories and WikiText prompts; paste them. There is no incremental (cached) inference code yet: recomputing the full forward pass for every new token is correct (the model is causal) and fast enough for 20 samples. Do not write a KV-cache implementation for this run.
8. **Throughput and memory:** tokens/s, peak memory, and total tokens seen.

---

## 8. Deliverables

`REPORT.md` with these sections, every number traceable to a file in `runs/`:

1. Summary: what was trained, for how long, on how many tokens, and the verdict per gate (§5.4).
2. Environment and exact-test results.
3. Data: manifest summary and sample rows.
4. Model: the parameter table from the live model; FLOPs per token; measured tokens/s.
5. Training: curves, incidents, every `DECISIONS.md` entry.
6. Results: benchmark tables (ID and OOD), intervention tables, probes, loop-depth curve, truncation curve.
7. Screens S1–S3 (S4).
8. What the results say about each module (encoder, concept module, decoder), stated against the failure ledger F1–F7: in particular, whether the concept route is the only long-range route (F6), and whether the recurrence is used (F7).
9. Limitations and the next experiment you would run.

Also deliver: `DECISIONS.md`, `data/MANIFEST.json`, `runs/**/log.jsonl`, the eval JSON files, the figures, and the final and best checkpoints.

---

## 9. Pitfalls (from the specs; each one silently breaks a guarantee)

1. Reading the encoder's G stream in the decoder, or any intermediate loop pass of the core.
2. Chunk-aligned plan shift instead of j*(t) = ⌊(t+1)/4⌋ − 1.
3. Keys/values of the decoder from D instead of `tok`; any convolution over D or the queries.
4. Documents not padded to multiples of 4; missing null plan at document starts; convolutions, n-grams or attention crossing documents.
5. Stop-gradient on the core's input (the core must consume the encoder's concepts with gradient); forgetting stop-gradient on NCP/VQ targets.
6. Editing the codebook in place between the forward and backward pass (init and restarts must be outside that window).
7. Weight-decaying log τ, the Engram table or the embeddings; zero-initialising both the Engram table and W_V.
8. Tolerance-based reach tests (use float64 and exact equality).
9. Boolean vs additive masks in `F.scaled_dot_product_attention` (boolean: True = attend).
10. Treating the null-plan "core off" loss change as the core's skill share (use the plan swap).
11. Learning rate above 1e-3 for the recurrent core (collapse, C8).
12. Two training processes on MPS at once.

---

## 10. Stop and ask the user when

- any exact test in §3.3 fails and the cause is not an obvious environment problem;
- the instantiated model differs from §2.2 by more than 1%;
- measured speed is below 400 tokens/s after trying B = 1–4 (the 24-hour run would see < 35M tokens; ask whether to shrink to d = 320 or extend the run);
- screen S1 shows no learning after two fixes you logged;
- a spec instruction is impossible on MPS (an operation missing even with the CPU fallback);
- data downloads fail (do not substitute other corpora without asking).

Otherwise decide, log it in `DECISIONS.md`, and continue. The expected total wall-clock time is about **33–36 hours**: setup and tests 1 h, data 2–3 h, calibration 0.75 h, screens 3–4 h, main run 24 h, testing 2–3 h.

---

## Appendix A. Wiring sketch (for orientation; `e2e.HRMThinker` is the reference)

```python
ecfg = EncoderConfig(vocab_size=8192, d_model=384, n_heads=6, n_kv_heads=2, head_dim=64, d_ff=1056)
dcfg = DecoderConfig(d_model=384, n_heads=6, n_kv_heads=2, head_dim=64, d_ff=1056,
                     n_layers=4, window=8, layout="confined", plan_in="add")
enc, core, dec = Encoder(ecfg), ConceptCoreV2(ecfg, S=48, N=128), Decoder(dcfg)

tok, c_hat, _ = enc(ids, doc, eng_idx)                       # (B,T,384), (B,T/4,384)
u, ncp_logits, c_pred = core(c_hat, doc[:, ::4], R)          # R ~ U{1..4} in training, 4 at eval
logits, aux = dec(tok, u, c_pred, doc, enc.embed.weight)     # (B,T,8192)

ntp = cross_entropy(logits[:, :-1], ids[:, 1:])   # mask targets in another document AND padding targets
ncp, vq = core.losses(c_hat, ncp_logits, doc[:, ::4])
loss = ntp + 0.1 * ncp + vq
loss.backward(); clip(1.0); opt.step(); sched.step(); core.vocab.restart_dead(c_hat)
```

`e2e.py`'s training loop adds a 2× loss on answer positions; that was a test device for the synthetic harness. **Do not use it in the main run** (plain NTP on every position).

---

## Appendix R. Review of this prompt (for the user; the agent may skip it)

The prompt was checked against the code it tells the agent to use, not against the conversation.

**Verified by running:**

| claim in the prompt | check | result |
|---|---|---|
| §2.2 parameter table and 166 MFLOP/token | instantiated the d = 384 model from `hrm_encoder.py`, `concept_core_v2.py`, `hrm_decoder.py` | 25.05M compute, 78.58M stored; reach 70 / 63 ✓ |
| Appendix A wiring, losses, `param_groups`, codebook init and restart, plan swap | ran two optimizer steps on CPU at d = 384 with two packed documents and masked cross-document targets | NTP 9.17 → 8.14, NCP 4.89, VQ 0 → 0.55, plan-swap forward OK ✓ |
| test commands in §3.3 | the same files pass D1–D5 and P1–P4 in this session | ✓ |

**Problems found and fixed in this review:**

1. **Codebook restarts under gradient accumulation.** `ConceptVocab.step` counts forward passes, so with 8 micro-batches per optimizer step, `restart_dead`'s `step % 100` check would fire only when the micro-batch count hits a multiple of 200, i.e. every 25 optimizer steps. §6.3 now tells the agent to keep its own optimizer-step counter.
2. **Padding targets were not masked** in the loss. §4.5 and Appendix A now mask cross-document targets and padding targets; EOS stays a target.
3. **The truncation curve was not implementable as written** (which positions are scored, and how chunk alignment is kept). §5.2 now scores the last 16 positions of each row with r ∈ {72, …, 1024} (multiples of 4), so r = 72 matches the token path's reach and the plan-swap cost is measured on the same positions.
4. **Generation had no code path.** §7 now says to recompute the full forward pass per token (correct by causality) and not to build a KV cache for this run.
5. **The time budget was under-added** (32 h); the phases sum to 33–36 h.

**Checked and left as they are:**

- Learning rate 6e-4: the encoder spec gives 6e-4 at d = 512 and 1e-3 at d ≤ 256; the recurrent core collapsed at 2e-3 (C8), so the lower value is the safe side at d = 384.
- 32k tokens per optimizer step: 1,600–4,800 optimizer steps in 24 h at the measured speed range; 300 warm-up steps is under 20% of the smallest case.
- The main run excludes XM, GradCuit, NextLat and the stack decoder: each is either untested at language scale or measured neutral; they appear only as screens.
- Stop conditions: < 400 tokens/s means < 35M tokens in 24 h, below what the gates need.

**Risks the prompt cannot remove:** the concept module's retrieval failed the 4-fact test at d = 128 (decoder spec §15.4), so G-D3 may be NOT REACHED; the screens are single-seed and 75 minutes; the throughput range (0.1–0.3 TFLOP/s) is from earlier runs and is re-measured in §6.1.
