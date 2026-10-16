# Repository understanding: EncoderConceptDecoder

Reviewed 13 October 2026, approximately 17:30–17:50 IST. This is a snapshot of a changing research workspace. The V12 Phase-2 sweep was active during inspection. Existing source, training jobs, checkpoints and result files were left unchanged; this directory contains the review only.

## 1. What this project is

The project studies whether a language model benefits from doing part of its computation on a shorter sequence of continuous concept states, predicting what comes next at that level, and giving a token decoder access to those states. Later experiments ask whether the concept interface also supports latent control, search, fast corrective memory and continual learning.

It is an experiment workspace, rather than one packaged application. There is no `.git` directory at this root, no unified top-level training entry point, and no shared configuration that makes all versions interchangeable. Each version carries its own code, data assumptions, protocol, results and sometimes copied predecessors. An architecture document, executable implementation and trained experiment can be three different states of the same version.

The recurring research questions are separate:

1. Can relevant information reach the prediction through the concept path?
2. Does the trained model use example-specific concept information?
3. Does next-concept supervision improve that information or merely change its representation?
4. Does repeated computation improve the result enough to pay for itself?
5. Can a causal controller choose useful latent actions without seeing the future?
6. Can an update learn something new while protecting earlier behavior?

A causal path, a nonzero gradient, populated codewords, a lower auxiliary loss and better verified task success answer different questions. Much of the repository's history is about learning to measure those distinctions.

## 2. Inspection coverage and limits

The initial project inventory contained **6,179 files and 25,645,709,124 bytes**: about 25.65 decimal GB. This excludes the local virtual environment and Python bytecode. A later full filesystem pass separately recorded **861 environment files and 255 bytecode files**; the live workspace had gained four additional project files by that pass.

Coverage:

| Material | Inspection performed |
|---|---|
| All project paths | Recursive inventory, sizes, file types and snapshot classification |
| Python | All 847 files parsed with `ast.parse`; 475 distinct byte contents, 97,060 unique source lines indexed; imports, definitions, method locations and CLI arguments extracted |
| Markdown | 207 files, 156 distinct contents; text digests, headings and openings indexed; principal specifications and experimental reports read in depth |
| Text/logs | 4,823 text files hashed; 1,339 JSONL/log/output files streamed, totaling 1,361,884 lines; large JSON result files also parsed and summarized |
| NumPy | Headers of all 411 `.npy`/`.npz` files inspected; shapes and dtypes recorded without loading dataset bodies |
| Checkpoints | File size and ZIP archive metadata inspected where available; no large tensor payloads deserialized |
| PDFs | 66 paths, 53 distinct PDFs, 1,502 unique-paper pages in total; metadata and opening/closing pages screened; selected existing methodology page renders viewed |
| Other storage | All eight Parquet schemas/row counts and all three SQLite database schemas inspected read-only |
| Other documents | Encoder DOCX text extracted; both RTF design conversations converted to text for inspection; decoder ZIP contents inventoried |
| Images/UI | All 20 PNG/JPEG dimensions recorded; selected research charts and paper-method figures visually inspected; dashboard/server and visualization source structure reviewed |

This is **complete inventory and structural coverage**, with deeper semantic tracing of the architecture, objectives, data, trainers, evaluation, search, memory and operational code. It is not a claim that all 97,060 distinct Python lines, all log records or all 1,502 paper pages received an equally detailed manual review. Vendor Three.js files, compiled cache outputs and duplicate snapshots were classified rather than treated as new research implementations. Sealed dataset contents were not opened.

Every Python source parsed successfully. That establishes syntax readability, not runtime correctness. Existing test artifacts were inspected; the long test suites were not rerun alongside the active experiment. The one fresh runtime diagnostic was a small CPU parameter-ownership check described in §11.

The machine-readable indexes and complete source map are linked in §13.

## 3. Resolve the names first

| Name | Executable home | Meaning |
|---|---|---|
| Original encoder/concept/decoder | `Encoder Design/`, `Concept Design/`, `Decoder Design/` | Individually designed components and their isolated testbenches |
| Whole-model v1 / THINKER-25M | `run_m2_25M/code/` | Original assembled model; `ConceptCoreV2` is a component inside this v1 |
| Whole-model v2 / THINKER-19M | `run_v2_19M/code/` | D6 mixture-of-experts concept core plus training-only NextLat |
| Whole-model v3 | `run_v3_19M/code/` | Dense recurrent core, shortcut modulation and shared FFN; distinct from the earlier `proposed_v3.md` design |
| v4 / v4.1 | `v4_tiny/`, `v41_tiny/` | Revised pooling/PQ/recurrence, followed by decoder token-memory retrieval |
| v5 | `v5_tiny/` | NCP foundation → GRAM/XM replacement → transplant → metacontroller/internal RL → GradCuit |
| v6 / v6.1 | `v6_tiny/`, `draft_v6_1.md` | Deterministic/recurrent replacement and bounded latent repair; adaptive tree-search proposal |
| v7 | `v7_tiny/` | Larger independent foundation/refit/control/search queue and matched flat model |
| v8 | `v8_tiny/`, `v8_tiny_12h/` | Competence-first pilot, then a separate full-exposure revision with recorded recoveries |
| v9 | `v9_tiny/` | H-NCP hypotheses, grounded/imagined concept search, expert iteration and LSL |
| v10 | `v10/`, `v10_10M/` | RCDL design and its implemented pretraining system |
| v10.1 / v10.2 | `v10_1/`, `v10_2/` | Cross-module interfaces, forecast supervision and released concept-history retrieval |
| v10.3 | `v10_3_tiny/` | Mixture-likelihood exploration, full forecast map, codebook-decay fix and sleep experiments |
| v11 | `v11.md`, `v11_tiny/` | Code-free two-visit core and concept-anchored latent prediction bridge |
| v12 | `v12_Architecture.md`, `v12_tiny/` | Simpler E→C→D trunk, categorical NCP and a separately trained frozen-base metacontroller |

`v11_Papers/` and `V11_Paper/` are different folders. The latter is also an evidence bundle containing frozen V10.3 sources/results. A copied source under an audit or `source_snapshot/` describes the old run that captured it, not necessarily today's live implementation.

## 4. The original architecture and why its boundaries matter

Primary code: `run_m2_25M/code/model.py`, `run_v2_19M/code/model.py`, `run_v3_19M/code/model.py`, plus the shared `hrm_encoder.py`, `hrm_decoder.py` and concept primitives in those folders.

The original encoder has local and global streams, L and G. Sliding-window S layers read L and write both streams. Full-attention F layers read L/G and write G. This keeps global information out of the decoder-visible local token stream while allowing it into pooled concepts. Hashed n-gram Engram reads supply learned local-pattern features. Original pooling produces one continuous concept per four tokens.

The confined decoder reads fixed encoder token features for attention K/V and a private per-position plan. It does not attend to another position's private decoder state. Consequently stacking its layers does not repeatedly expand its token-memory reach. With the original encoder and convolution settings, the local encoder lag is 63 and the confined decoder's lag is 70.

Concept group g covers tokens `kg … kg+k−1`. Prediction row t predicts token t+1, so it may first read group g at `t = kg+k−1`. The latest available group is `floor((t+1)/k)−1`. This release rule is foundational throughout the project, including the later history readers.

The original plan combines the current continuous core output, predicted next concept and within-chunk offset. The continuous pooled state enters the core directly: the PQ alphabet does not force every bit of information through a discrete bottleneck.

The versions differ materially:

- **v1:** token/concept width 384, one shared concept layer with variable pass count, confined decoder.
- **v2:** token width 256, concept input width 384, four concept layers and shared routed experts; deeper recurrent processing and detached training windows; NextLat auxiliary on decoder states.
- **v3:** width 256, two reads of one shared Engram table, four dense concept layers, three main passes, `(t, Δ)` conditioning and training-only shortcut trajectories. Own and shared FFNs run in parallel. NTP supervises the main path; auxiliary NCP and consistency supervise shortcuts.
- **v4:** attention pooling, a coarse-plus-product-residual alphabet, revised loops and optional decoder Engram reads.
- **v4.1:** removes decoder Engram reads and pooler FFN; introduces a full causal token-memory R layer while keeping plans private. The retrieval layer releases pressure on the concept route to carry every distant detail.

Old headline “compute parameter” counts exclude substantial stored tables and embeddings. For example, the v3 ledger is about 18.99M headline parameters versus about 101.12M stored parameters, including an 80M Engram table. Those exclusions do not make storage, optimizer cost or tied-output projection free. Newer reports generally give dense, table and deployed counts separately.

## 5. The v5–v9 control/search lineage

### Foundation and replacement

The v5/v6/v8 family uses ordinary token encoder/decoder blocks, mean pooling at k=4, a concept core, a soft-PQ forecast and a continuous port. `System.context()` produces the frozen context; `System.plans()` selects the original or replacement core; `Foundation.decode()` shifts the legal plans back into token rows.

A replacement core can be dense or use two recurrent H/L blocks. It is trained against a frozen surrounding foundation. Bounded adapters write a low-dimensional control Z into the replacement. A deterministic zero-control path is a reference; prior and posterior latent paths have different information and deployment meaning.

The v5 result makes this distinction concrete: future-informed posterior Z can strongly steer the decoder, while a poorly calibrated prior can steer it toward the wrong choice. XM improves that prior's deployability. This is evidence for control authority, not evidence that the controller has discovered persistent subgoals. The later switch-learning/internal-RL experiment did not establish a useful persistent abstraction.

### Repair and AB allocation

v6–v8 keep token generation, replay likelihood, gradient-based latent repair and adaptive allocation separate. The verifier checks a realized answer against the public problem statement. Oracle fitting with a valid target trajectory is diagnostic and is not ordinary repair success. Time-matched and attempt-matched comparisons are recorded separately.

Incremental and fixed-size caches matter operationally. Cached decoding must agree with a full recomputation on trained weights, not only at initialization. The v8 full-exposure revision records a trained compiled-cache parity failure and recovery to eager static decoding, followed by a separate NumPy-boolean JSON export recovery. These are preserved as implementation revisions, not new independent training seeds.

### H-NCP and concept search in v9

Primary modules:

| File | Responsibility |
|---|---|
| `code/hncp.py` | K hypotheses at the final concept block; code enters query and residual while K/V come from the unperturbed stream; K=1 creates no code parameters |
| `code/train_s2.py` | Frozen-foundation refit; winner/soft-responsibility objectives, selector fitting, controlled closed-loop input substitution and distillation |
| `code/hgen.py` | Shared primitives for grounded context, concept prediction, chunk decoding, progress and STOP semantics |
| `code/csearch.py` | PUCT over K codes plus STOP; receding horizon and one-tree modes; grounded versus imagined children and immutable plan records |
| `code/train_s4.py` | Search-generated training records, value/terminal heads and expert iteration |
| `code/baselines.py` | Token-policy comparison including GRPO training |
| `code/lsl.py`, `code/s6.py` | Per-site/per-phase low-rank adapters and density gates for continual learning |
| `code/register_s*.py`, `code/stats.py` | Gate decisions, promotion, paired bootstrap/sign-flip/McNemar tests and multiplicity handling |
| `code/drive_q*.py` | Dependency-aware experiment queues and stage completion records |

Imagining a concept does not make it a verified environment state. Grounding decodes pending chunks under their stored plans, then re-encodes the realized tokens. A STOP at a grounded node can be verified; a STOP prediction at an imagined node depends on a trusted terminal head. Search logs count expansions, grounding/replay, rollouts and verifier calls.

S1–S6 scientific stages and Q1–Q6 execution blocks are different labels. In the saved workspace, Q1–Q4 are complete, while the original Q5/Q6/QF queues are stopped. A separate reduced Q5 pilot completed. Do not infer that S6 has no artifacts merely because the original Q6 queue stopped: Q2 already contains continual-learning work.

The K8 promotion is explicitly seed-dependent: both downstream S3 seeds derive from the passing S2 seed's K8 model. They are not independent foundation replications. The reduced Q5 comparison reports standard success 83.20% reference versus 85.16% EI, and hard10 33.98% versus 33.01%; GRPO collapsed to zero. This does not establish a credible equal-compute MCTS advantage.

## 6. The v10–v11 forecast/interface lineage

The dependency chain is important for edits:

```text
v10_10M primitives/config/data
    └── v10_1: EncoderConceptCRC, visit handling, chunk supervision, memory contracts
          └── v10_2: released-history ConceptReader + forecast-only readout
                └── v10_3_tiny: tiny config, full forecast map, mixture objective
                      └── v11_tiny: code-free core, Gaussian latent loss, bridge
```

`v10_10M/model.py` defines E1, a two-stream E2 executed twice, E3, 8→2 attention pooling, a two-visit concept core and decoder. The production reference has width 128, 12 distinct encoder layers/16 applications, four concept layers/two visits and four decoder layers. The sparse Engram table is on CPU while gathered features retain autograd across the device boundary.

**v10.1** adds pooled early/final encoder→concept CRC and a continuous concept→decoder route. It supplies per-visit latent supervision, a teacher-forced chunk head and candidate utility scoring. Winner-only recomputation avoids retaining all K candidate graphs; effective-batch reductions preserve each objective's denominator.

**v10.2 RF** separates a causal reader over every completed two-slot concept group from aligned forecast guidance. The forecast readout receives forecast features and offsets, with no preceding target tokens. It supervises both visits with 1/3 and 2/3 weights. A zero sentinel handles rows with no released memory. Its loss replaces latent/VQ/chunk regression terms with future-token CE, candidate utility and selector CE.

**v10.3** gives aligned forecasts a full learned map, excludes the PQ codebook from weight decay and replaces hard best-code selection with mixture likelihood under the selector prior. C0 is a concept-free control on the inherited backbone. Its sleep experiment is a separate post-training system.

**v11** removes strategy codes and selector from the core, keeps two visits, normalizes pooled targets with EMA buffers and trains Gaussian next-concept objectives. A training-only bridge initializes from concept state and rolls forward through eight observed tokens toward stopped decoder states, with a frozen-head KL. The bridge is concept-anchored and is not the later v12 one-step decoder NextLat implementation. A detached forecast probe is diagnostic, but can still affect optimization if its parameters share the trunk's clip budget.

The saved v11 study found and separated bridge/probe clipping artifacts. Its fair ordinary baseline is ORDE, which includes the same Engram table; ORD alone omits that table. The report's failed promotion gates and bridge costs are more informative than a successful wiring check.

## 7. Trace the latest implemented v12 forward pass

v12 is mostly independent of the v10 model classes. Its primary implementation is only about 205 lines in `v12_tiny/model.py`, backed by separate objective, data and controller modules.

For input IDs of shape `[B,T]`, width d=128 and k=4:

1. Tied embedding maps IDs to `[B,T,d]`.
2. Two causal encoder blocks produce h.
3. Each complete four-token group is mean pooled, RMS-normalized and projected, giving `p[B,G,d]`, `G=floor(T/4)`.
4. Two causal concept blocks produce normalized `u[B,G,d]`. The generic class supports repeated visits, but the completed main ECDe comparison uses one visit.
5. Each of two decoder blocks performs causal token self-attention, cross-attention to released concept history, then an MLP.
6. Final RMSNorm and the tied embedding matrix produce vocabulary logits.

The concept group is visible at token row `t ≥ 4g+3`. The decoder reads **u**, not the NCP forecast. The NCP head is an auxiliary readout, so the categorical forecast does not create an obligatory quantized channel into D.

Control points are either mid-concept (between concept blocks, or between visits for the looped variant) or the token-rate decoder input. The TF control can be inserted between ordinary transformer blocks.

Main arms in `train.py`:

| Arm | Meaning |
|---|---|
| TF6 / TF7 | Ordinary causal transformer; TF7 is the approximate parameter match |
| ECD | E→C→D topology, NTP only |
| ECDe / ECDd | ECD plus categorical NCP using encoder / decoder group targets |
| TFNL / ECDeNL | Token-rate decoder NextLat variants |
| ECDL2e | Two concept visits plus encoder-target NCP |

The matched main 8k snapshots exist for TF7, ECD and ECDe. This does not mean every arm in the dictionary was trained. The original ECD/ECDe manifests still request 12k steps, and `_8k` comparison directories retain copied manifests; their evaluated `final.pt` files represent the chosen comparison snapshot. For timing or resumption, inspect the source run log and actual checkpoint selection, not the copied planned-step field alone.

### Objectives and gradient ownership

`ntp()` computes weighted next-token CE. Pinpad actions and END have weight 1, observed positions weight 0.1 and the layout header is unscored. Text scores next tokens; math scores answer tokens and EOS.

`PQCodebook` is four segments × 32 words. It normalizes targets with running statistics, assigns nearest codewords and updates buffers with EMA-style k-means/dead-code resets. It has no optimizer-trained parameters. `NCPHead(u_g)` predicts the next segment's four code indices, and the loss sums CE across segments. Its target is stopped, so it trains E/C/head while D receives no NCP gradient.

`NextLat` predicts the next decoder residual from the current decoder residual and detached next-token embedding. It uses RMS-normalized SmoothL1 and a KL through a detached view of the final head. Hidden-state loss uses real positions, while the KL is restricted to scored targets. It is intended to be dropped at inference. The optimizer-ownership defect for its unrun arms is in §11.

### Data

The joint vocabulary has 1,090 IDs: 66 pinpad symbols and an offset 1,024-token BPE inherited from `v10_3_tiny/data`.

Family selection probabilities are 60% pinpad, 25% text and 15% math **per batch**, not per scored token. Batch sizes differ: pinpad/math 128 and text 32. Newmath is held out for later adaptation. Sudoku is excluded because the prior experiments exposed fixed-layout/chunk alignment confounding.

Pinpad is a 7×7 grid with eight colored cells, four walls, four actions and a maximum 100-step episode. Headers tag object identity and position. Ground-truth color and color-pair labels are retained for probes/controllers but are not base-model inputs. Expert trajectories use shortest paths while avoiding out-of-order colors. The dataset offers variable subgoal durations rather than deliberately matching fixed four-token groups.

### Frozen-base metacontroller

`meta.py` freezes the base and learns a Gaussian code proposal, causal switch β and rank-16 hypernetwork controller `e ← e + A(z)B(z)e`. A causal history GRU feeds the switch; a bidirectional sequence embedder feeds the posterior code proposal. The code is integrated as `z_j = β_j z̃_j + (1−β_j)z_{j−1}`.

At token rate, the controller's clock is one environment step: the position row predicting an action and its following action row share U. At concept rate, the clock is a concept group. Switch F1 is converted to the common environment-step clock, with tolerance measured in steps rather than raw tokens.

The posterior uses future information. Therefore its reported dev NLL is a **hindsight reconstruction/control diagnostic**, not a deployable causal language-model NLL improvement. `internal_rl.py` implements the causal policy that would replace this posterior, plus a raw-action PPO baseline. Implementation existence does not mean the RL experiments ran.

The declared RL gate requires a ground-truth-control advantage of at least 0.03 on the short novel task and switch F1 at least 0.8. Completed measurements are far below the switch threshold. The document reports internal RL as gated off; the Phase-2 sweep itself was still running during this review.

## 8. What the saved results support

These are local experiment results, not independent reproductions performed by this review. Cross-version absolute NLLs are not comparable when tokenizers, data mixtures, widths or scoring change.

| Study | Main recorded finding |
|---|---|
| Original v1–v3 | Concept routes can carry distant information and support retrieval; deeper recurrence is much less clearly established as general reasoning. v2 also exposes optimization/routing issues documented by independent audits. |
| v4.1 | Retrieval improves tiny pinpad capability over v4, but the transformer still leads. Constant/zero-plan controls show that a header-null switch-off can falsely suggest plan-content dependence. |
| v5 | Posterior latent control has authority; XM helps prior calibration. Persistent temporal abstraction and an internal-RL advantage were not established. |
| v6 | Removing the forecast objective at the matched 9k prefix produces a large GraphNav improvement; locked repair-versus-retry intervals do not establish a repair advantage. |
| v7 | Recorded strict task success is at zero across the main arms despite text learning; this motivates competence/exposure diagnostics in v8. The top-level report retains a historical “running” label. |
| v8 initial / full exposure | Initial engineering gate fails. The separate full-exposure pilot reaches approximately 77–81% standard dev success. Later LSL phases catastrophically lose the original standard task; completion is not retention success. |
| v9 | Independent teachers and seed-sensitive promotion are explicitly flagged. Both S3 replacement routes fail. The reduced EI/GRPO comparison is exploratory and inconclusive. |
| v10.3 | Fixes parts of the v10.1 objective damage, but concepts cost more than they return against C0 on text/math; second visits and code alternatives add little. Sudoku gains are entangled with layout. RBF and gated-adapter adaptation provide more promising local evidence. |
| v11 | Registered K1 promotion gates fail; bridge gradients harm useful learning. Separate clipping clarifies the comparison but does not rescue the bridge. |
| v12 completed 8k comparisons | NCP makes C useful, with a substantial predictive and competence cost. Frozen control does not recover a compositional subgoal representation at this scale. |

Concrete v12 values from `RESULTS.md` and the corresponding `runs/*/eval_final.json` / usage files:

| Matched snapshot | Pinpad action NLL | Text NLL | Math NLL | Closed-loop competence |
|---|---:|---:|---:|---:|
| TF7 | 0.538 | 3.555 | 0.817 | 0.281 |
| ECD, NTP only | 0.551 | 3.615 | 0.846 | 0.230 |
| ECDe, NCP | 0.579 | 3.893 | 1.120 | 0.164 |

At ECDe step 8k, zeroing C costs approximately **+0.415 nats/action on pinpad** and **+0.185 nats/token on math**. This is robust to the available mean/shuffle intervention direction on these two families: pinpad mean/shuffle costs +0.460/+0.648, math +0.272/+0.141. The effect on text is weaker: zero +0.029, mean +0.013. On newmath, zero +0.168 contrasts with mean +0.003 and shuffle −0.036, illustrating why zero ablation alone is insufficient.

The v12 concept state gains next-code predictive information on pinpad, but probes leave the abstract color-pair label around chance. The sampled TF metacontroller results have pair-switch F1 about 0.012 versus a matched random-placement benchmark around 0.265. NLL reductions under that posterior do not imply useful subgoal discovery.

## 9. The three different kinds of memory

| Mechanism | What changes | Purpose and limitation |
|---|---|---|
| Engram | Learned hashed n-gram table rows during model training | Local-pattern features; not a live database of the current document's facts |
| RBF correction bank | Appended concept-query keys, code partitions and corrective vectors; base weights frozen | Fast local functional updates; query meaning must remain fixed within a fingerprinted representation era |
| LSL/gated LoRA | Newly trained low-rank adapter banks, applied under fitted density gates | Retain older functions by restricting where new adapters act; related input distributions can still overlap and forget |

The v10.2 RBF writer snapshots the bank, computes token-loss derivatives with respect to the correction, and commits bounded negative-gradient values. It rejects stale versions, wrong eras, duplicate event IDs and capacity overflow. Zero-gradient feedback still consumes an event/version. These details are central to reproducible memory semantics.

In the v10.3 sleep experiment, GMM gates compare new-task and old-task feature densities. Gates are recorded on an adapter-off base pass so upstream adapters cannot change later gate decisions. Separate visit gates and token-rate temporal smoothing are implemented. The calibrated RBF bank obtains about −1.55 nats on unseen algebra, and distilling it into fresh gated adapters retains substantial gain after bank reset. Closely related add/sub skills remain unprotected. This is limited adaptation evidence; it is not a demonstrated lifelong-learning system.

V12 inherits these ideas in its specification. There is no new completed v12 sleep run in this snapshot, and the latest v12 class does not itself expose the v10 RBF-bank interface. Wiring or validating that integration is future implementation work, not an already available v12 feature.

## 10. Data, evaluation and operations

The early run family uses document-aware packed streams, explicit document masks, hashed n-gram indices and a shared tokenizer corpus. The v5–v9 symbolic family uses generated GraphNav-like tasks, public verifiers, exact-content split checks and variants for longer paths, shifted labels, lookup, composition and binding. Graph-isomorphism or semantic independence should not be inferred from exact token deduplication.

The v10 family uses FineWeb-Edu text, arithmetic/sorting/algebra Parquet sources and Sudoku. The initial math corpus contains serialized-byte wrapper issues; the v10.1 clean builder preserves original train/dev identities, safely normalizes math fields, quarantines normalized collisions and trains a new tokenizer. Old weights cannot be paired with this new tokenizer. Clean v10.1 data is reused by v10.2; v10.3 fits its own smaller tokenizer, and v12 offsets that vocabulary.

Strict evaluation matters: math uses answer parsing and EOS checks; Sudoku requires a complete valid board with preserved clues, every row/column/box and EOS. A well-formatted Sudoku output is not a solved board. Some routine generation panels are deliberately small and are distinguished from balanced offline checkpoint audits or sealed tests.

The trainers emphasize M2/MPS FP32 execution, CPU sparse Engram updates, source/data fingerprints, checkpoint manifests, independent dense/sparse clipping, process locks and safe stop/resume. Historical bash/zsh guards include battery/swap/thermal checks and sleep prevention. These are operational scripts to inspect before executing, not commands this review launched.

Current status files supersede launch prose: v10's saved run is stopped at step 84,778 and v10.1's at 61,384, despite old READMEs describing active planned runs. `v10_2/` has extensive validation artifacts, but no production `runs/` folder in this snapshot.

Dashboard families:

- `run_m2_25M/webapp/`: broad architecture/explanation/live-history UI, native JS/CSS/SVG with vendored Three.js; older run webapps contain copied versions.
- `v9_webapp/`: built atlas plus a separate beginner-oriented S1–S6 guide; `build.py` composes source assets, and `serve.py` exposes read-only aggregated telemetry on loopback.
- `v10_10M/telemetry/`, `v10_1/telemetry.py`, `v10_2/telemetry.py`: versioned read-only collectors and APIs backed by SQLite events/metrics; no model import or weight mutation is needed for collection.
- `v12_tiny/viz/`: a version-comparison architecture explorer with shared component vocabulary and explicit measured/paper/untested statuses.

Telemetry databases separate runs, sources, events, numeric metrics, artifacts and issues. Raw numerator/denominator and source identity matter: missing accuracy is not 0%, pre-clip gradient norms are not post-clip norms, overlapping unified-memory counters must not be summed, and batch mixture percentages are not actual token exposure.

## 11. Concrete engineering observations

1. **NextLat optimizer ownership in v12 needs correction before those arms run.** `losses.NextLat.__init__` registers the shared embedding as `self.emb`. `train.main` constructs one optimizer over `model` and another over `aux`. A fresh CPU identity check found the same parameter as `emb.weight` and `nl.emb.weight` in both TFNL and ECDeNL. Those arms would step the trunk embedding through both optimizers; auxiliary clipping would also include it. TF7/ECD/ECDe have no overlap, so this finding does not invalidate the completed main comparisons. Sources were not modified.
2. **Specifications retain historical statuses.** The root v10.2/v10.3 documents say “not implemented”, while executable packages and later tiny results exist. `REPO_WALKTHROUGH.md` dates from 2 October and stops at v4.1. Use it for the early architecture, not the current state.
3. **One documentation link is broken.** `v10_2/README.md` links to `v10_2/VALIDATION_REPORT.md`, which is absent. The actual validation JSON/logs are present.
4. **Live posterior results and proposed modules need separate labels.** v12's α sweep was active. NextLat, decoder-target HiLP, loop-of-two, learned segmentation, internal RL and v12 sleep do not acquire measured support merely because their classes or architectural sections exist.
5. **Scientific provenance is spread across files.** Resolve a run from source snapshot/hash, tokenizer/data manifest, chosen checkpoint and raw evaluation JSON together. Planned-step fields, narrative status and dashboard diagrams can lag behind actual execution.
6. **Cross-version imports create coupling.** An edit to v10 primitives can affect v10.1, v10.2, v10.3 and v11. Historical hashes and copied snapshots preserve old runs, but a new run must fingerprint all imported dependencies. v12 additionally uses local bare imports, so its scripts expect the package directory/import path used by its existing launchers.

## 12. Where to work next, depending on the task

| Intended change | Read first |
|---|---|
| Current base architecture/release mask | `v12_tiny/model.py`, `tests.py` |
| Current categorical NCP or codebook drift | `v12_tiny/losses.py`, `grad_share.py`, `drift.py`, `info_test.py`, `usage.py` |
| Current controller/switching | `v12_tiny/meta.py`, `gtctrl.py`, `phase2.py`, `switch_chance.py`, `z_check.py` |
| Current environment and competence scoring | `v12_tiny/pinpad.py`, `data.py`, `rollout.py`, `evaluate.py` |
| Causal latent RL | `v12_tiny/internal_rl.py` and the failed prerequisite gates |
| Recurrent forecast/history-reader design | `v10_10M/model.py`, `v10_1/model.py`, `v10_2/model.py` |
| Mixture exploration | `v10_3_tiny/losses.py` and candidate-gradient tests |
| Fast correction and consolidation | `v10_2/memory.py`, `v10_3_tiny/sleep.py` |
| Concept MCTS/grounding | `v9_tiny/code/hncp.py`, `hgen.py`, `csearch.py`, `train_s4.py` |
| Historical local/global and private-plan contracts | Original `hrm_encoder.py`, `hrm_decoder.py`, and the design specs |
| Live metrics/UI | Version-specific telemetry collector/store/server, then its web source |

## 13. Review artifacts

- [Full source map](SOURCE_MAP.md): every Python path, duplicate groups, imports, classes/functions and verified line locations.
- [Folder map](FOLDER_MAP.md): every top-level folder, its role, snapshot counts and sizes; every nested directory is also listed in `directories.tsv`.
- [Project inventory](inventory.json): all initial 6,179 project file records, text digests and checkpoint/archive metadata.
- [Full filesystem inventory](full_filesystem_inventory.json): later pass including environment and bytecode files.
- [Document index](document_index.json): Markdown headings and duplicate identities.
- [Array index](array_index.json): all 411 NumPy artifact headers.
- [Paper catalog](PAPERS.md): distinct PDF titles, page counts, local paths and duplicates; screening depth stated explicitly.
- [Storage index](storage_index.json): Parquet and SQLite schema metadata.
- [Inspection manifest](review_manifest.json): methods, timing, counts and fresh diagnostic scope.

The strongest current evidence is that these interfaces can carry information, that categorical NCP can make the latest concept path necessary for a trained model's predictions, and that carefully gated local adaptation can retain some new knowledge. A better matched predictive model, useful learned subgoal switching, an efficient search advantage and general continual retention remain separate, unresolved objectives.
