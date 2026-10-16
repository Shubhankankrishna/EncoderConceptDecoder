# Deep audit — 11 October 2026

**The training infrastructure is functioning, but the experiment is not yet up to the requested standard for reasoning quality or clean math evaluation.** The telemetry backend is implemented and running. This audit does not change the active model, optimizer, tokenizer, dataset, or 24-hour deadline.

The review covers the V10 proposal/retrieval-correction interfaces, the active `v10_10M` implementation, data preparation and evaluation, prior implementation contracts, the live run, and a trained checkpoint. Earlier V9 findings remain context; this is not a claim that every historical experiment has been independently reproduced.

## Findings requiring attention

### P1 — Math data retains serialized byte literals

`prepare_data.py` decodes native `bytes`, but the upstream parquet schema is **string**, with values such as `"b'What is 2 + 2?\\n'"`. These pass through unchanged. All **239,904 training**, **4,037 development** and **3,963 sealed-test** math answer records retain these wrappers. Test inspection here was format/integrity inspection, not model scoring.

For the answer 10, the supervised target token sequence is:

```text
Ġb  '  1  0  \  n  '  <eos>
```

Only two of those eight tokens are the numerical answer. Thus low teacher-forced NLL can be dominated by predictable formatting. The same-checkpoint first-eight math probe obtained **71.875% teacher-forced token accuracy yet 0/8 exact answers**. Wrong generated answers included 12 instead of 10 and 10 instead of −49. This is a concrete evaluation confound; it is not evidence that the model has intentionally reward-hacked, and there is no reward-learning stage in this run.

Evidence: `math_serialization_audit.json`, `checkpoint_probe.json`, `live_summary.json`.

A safe normalizer is supplied in `../data_tools/clean_math.py`, with tests for native UTF-8 bytes, serialized byte literals, ordinary strings, apostrophes and invalid inputs. `../data_tools/prepare_v2.py` applies it before deduplication, partition hashing and tokenizer fitting, and writes only to a new destination:

```sh
/opt/anaconda3/bin/python -m v10_10M.data_tools.prepare_v2 --output /absolute/path/to/new_clean_corpus
```

This separate preparation entry point has unit-tested normalization; a complete new corpus build has **not** been executed. It rebuilds identities and a fresh tokenizer. It must belong to a **new training experiment**, with that corpus explicitly wired into training, not be substituted under the current checkpoint. The active trainer still points at the original `data/`; no clean-corpus training has been started. Retaining the current run preserves the already-authorized 24-hour experiment, but its math results must carry the formatting caveat.

### P2 — Routine math generation covers one module

The trainer's `evaluate.py` takes the first eight metadata rows. Preparation orders records by module, so all eight are `arithmetic__add_sub_multiple`; multiplication, algebra and sorting are absent from that metric. Two Sudoku examples per difficulty also provide very weak accuracy estimates.

A separate stratified CPU evaluation of checkpoint **17,607** sampled 16 development examples per math module and 16 per Sudoku family with seed 1927. It used ordinary greedy generation, prefix-only codes, exact answer/EOS checks and the full Sudoku verifier. It did not use solver assistance, output repair, or sealed-test model scoring.

| Evaluation | Correct / attempted |
|---|---:|
| Algebra, linear 1D | 0 / 16 |
| Arithmetic, addition/subtraction | 1 / 16 |
| Arithmetic, multiplication | 0 / 16 |
| Comparison, sorting | 1 / 16 |
| **Math total** | **2 / 64** |
| Sudoku, native easy sources | 0 / 16 |
| Sudoku, native hard sources | 0 / 16 |

These are early-checkpoint development diagnostics, not final 24-hour results or precise population estimates. The math audit still scores the current serialized-format targets; it is not a clean-data benchmark. Full responses, sampled identities and per-module counts are in `balanced_evaluation.json`, exposed as `probe_evaluation` events.

### P2 — Concept wiring works, useful contribution remains unproven

Fresh checkpoint probes show nonzero token-loss gradients through the pooler, core, PQ head and decoder fusion, as well as dense encoder and sparse Engram parameters. Concept parameters are not disconnected. Changing selected codes changes forecasts and logits. However, sampled concept ablation barely affects token loss:

| Family | NLL increase with concepts removed | NLL increase with Engram branch removed |
|---|---:|---:|
| Text | +0.000473 | +0.858704 |
| Math | −0.000132 | +1.911173 |
| Sudoku easy | +0.000150 | +0.390980 |
| Sudoku hard | +0.000940 | +0.546650 |

Each row is a small, separately drawn two-example development batch at step 17,607. They are diagnostics, not statistically established effects. The Engram ablation suggests the current checkpoint depends on that branch; it does not prove a table of factual knowledge or superiority to an equal-budget baseline.

Token-only concept-core gradient norms are roughly 0.0064–0.0089 in these probes, compared with decoder norms around 0.95–1.24. Auxiliary objectives contribute substantially to concept learning. The architecture's direct encoder/CRC path permits token prediction without strong use of concept guidance. This is a research concern to test, not a reason to remove the user-requested path silently. Learned fusion RMS is 0.0711; it is nonzero. No claim of complete concept collapse is justified by these checks.

### P2 — Gradient and continual-learning coverage is incomplete

The trainer records the dense preclip norm only on the final update of each 20-step log interval. Sparse Engram norms are checked/clipped but not logged. Historical per-layer or per-step gradients cannot be recovered. At the audit snapshot, about **88.7% of recorded dense norms** exceeded the clip threshold 1; this is a sampled clipping rate, not a measured rate across all optimizer updates. Recorded norms were finite (range about 0.354–6.219).

New CPU checkpoint probes supply component and layer norms under token-only, concept-only and total losses, including sparse Engram norms. Telemetry labels these `probe.*`; it does not misrepresent them as past live gradients.

RBF read/write, immutable snapshot, code partition, causality and frozen-backbone gradient-ownership contracts passed in prelaunch validation. The bank is intentionally empty in pretraining. Continual-learning improvement, retention and transfer have not been established, and no such claim should appear in the UI.

## Checks that passed

- **Architecture budget:** width 128; 5,465,890 dense parameters plus 20,000,320 Engram table parameters, as separately authorized.
- **Depth:** E1/E2/E3 = 4/4/4 distinct layers; E2 executes twice, yielding 16 encoder block applications. Concept core = four full group-causal attention layers visited twice; decoder = four layers. Encoder attention follows three local then one full causal block, ordinary 4× FFNs.
- **Interfaces:** eight tokens → two pooled slots; K=8 learned codes; pretraining temporary perturbation zero; retained recurrent states; soft PQ head; first legal concept forecast at row 7, four repeats per slot; direct encoder path and CRC; RBF correction after final concept state, before PQ.
- **Prelaunch contracts:** CPU/MPS causality, partial groups, padding isolation, actual layer applications, candidate history, variable boundaries, sparse gradients, memory-write ownership and strict complete-board verification. Existing artifacts are in `../research/`; these were not all rerun concurrently with the live GPU job.
- **New trained-checkpoint checks:** all parameter tensors finite; source hash matches; future changes caused **zero** earlier-logit change across all four families; full-vs-prefix differences ≤ 8.6e−6; all probed gradients finite. These ran on CPU with two threads, without optimizer steps.
- **Data structural checks:** existing hashes, offsets, answer boundaries, identity partitions and Sudoku target validity passed. Those checks did not detect the serialized text defect; this audit adds that missing semantic-format check. Exact deduplication is not proof of semantic or Sudoku-isomorphism separation beyond upstream guarantees.
- **Operational state:** trainer PID 52954 remained alive, log steps/tokens advanced, periodic checkpoint files continued updating, and no training exception was found. Source hash remains `3cbaf426badbfc45f723431838c4f5340cbf7ce2c99f1a8438ed8e3b7152a0e9`. The deadline remains **12 October 2026, 12:12:15 IST**.
- **Telemetry:** six tests passed; live log counts and metric values matched the API after collector restart, with no ingestion issues. Collector memory was approximately 30 MiB in the sampled host readings. This is a point observation, not a formal overhead benchmark.

The log snapshot for this report was around step **18,660**, **39.8 million input tokens**, approximately **2.9 hours** into the run. The checkpoint probes intentionally report their older stored step separately. The latest live state is available through the telemetry API.

## What to do with these results

Use the current run as an explicitly qualified pilot. A stronger follow-up needs clean math serialization, balanced per-module generation throughout training, and enough held-out complete-answer trials. Compare concept ablation and a matched baseline before claiming the bottleneck helps. Evaluate RBF adaptation and retention only after the backbone is frozen and base-task performance is established.

The next UI can be built directly against [the telemetry contract](../telemetry/README.md). It should keep training loss, complete-answer success, checkpoint probes, and data-quality alerts visible as different quantities. Improving loss alone must not be presented as solved reasoning.
