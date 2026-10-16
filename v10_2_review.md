# Review of the V10.2 RCDL architecture

**Review date:** 12 October 2026
**Reviewed document:** [V10_2_RCDL_Architecture.md](V10_2_RCDL_Architecture.md)
**Reviewed SHA-256:** `3e1307b0d0791396c76a7c4123df0852b7dfc72c89469bb94ecf5eef156df842`
**Scope:** evidence, architecture, objectives, causality, RCDL memory, experimental design, statistics, compute plan, and the state of the implementation that followed the document.
**New evidence for this review:** [v10_2_review_evidence/2026-10-12/](v10_2_review_evidence/2026-10-12/)

## 1. Verdict

**V10.2 is a precisely specified, honestly bounded experimental candidate whose central motivation is not yet supported by evidence.** Its contracts are correct: the release rule, the startup sentinel, a forecast head that cannot see the future, the both-route memory derivative and the parameter ledger all check out, and the later implementation follows them faithfully. Every number the document reports is exact.

Four problems matter more than the contracts:

1. **The access diagnosis is unestablished.** The document attributes V10.1's weak concept effect to limited access. But a released concept adds no evidence the decoder lacks; it can help only through computation, inductive bias or control. On the stopped V10.1 checkpoint, a fixed mean input recovers about 58% of the text concept effect; example-specific content accounts for the rest (+0.0026 of the +0.0062 cost of removing the route). On math, the mean beats the real content. On the same two routine documents at step 61,384, nearly all of the "continuous" effect is router renormalization (+0.00246 renormalized versus +0.00013 value-zero); on 64 documents, about two-thirds is. Router weights cannot separate access from content. (F01, F02, F08)
2. **The default arm removes components without a measurement.** Latent and codebook fitting are dropped with no forecast-off or baseline measurement in V10.1, and the multi-token-prediction paper the document cites reports that the method is worse than next-token prediction at small model sizes. (F06)
3. **The promotion thresholds sit near the noise floor.** A +3-point math gain is 1.0–1.5 standard errors on 256 problems. Only pooled Sudoku has power, and every V10.1 checkpoint has solved zero boards. "Competitive with C0" has no margin, and no seed rule is set, although V9 saw 17% versus 83% across two seeds of one recipe. (F03)
4. **The plan's budget is one arm.** A 30-minute pilot reaches about 2,500–3,500 updates, before any concept effect has differentiated. A defensible A/RF/C0 comparison needs about 50 sequential hours. The RBF grid also places two of three σ values in a regime where the bank is a near-global per-code average with an unscaled step. (F04, F05)

Events have also overtaken the document. Within about two hours of it, V10.1 was stopped at 11.17 of 16 active hours, so no 16-hour V10.1 result exists, and the stopped run can no longer resume because a hashed source file changed (one the trainer does not import). A V10.2 package was then written that implements only the combined RF arm. Its contracts pass, but it cannot run the comparison the document prescribes. (F09)

**Recommendation.** Before any production arm, run the two frozen-checkpoint discriminators of §9 on the stopped V10.1 checkpoint (about one to two hours). Their result decides whether Step 3 runs A/RF/C0, A/F/C0, or A and C0 first. A lone RF run is a feasibility run, not the document's §8 comparison. C0 belongs in every branch: in V7, V8 and V9 the ordinary Transformer had lower text NLL than the concept hierarchy every time, and V10 has never measured it.

**Score: 6.3/10** (§11).

## 2. Audit method and evidence boundaries

### 2.1 What was read

- The complete reviewed document, including its self-review (document §10) and its [review script](v10_2_design_evidence/2026-10-12/review_checks.py).
- The fixed capture it cites: [telemetry summary](v10_2_design_evidence/2026-10-12/telemetry_summary.json), [API overview](v10_2_design_evidence/2026-10-12/overview.json), and all twenty evaluation files from step 0 to 54,168.
- The V10.1 implementation the evidence comes from: [model](v10_1/model.py), [evaluator](v10_1/evaluate.py), [trainer](v10_1/train.py), and the two post-capture evaluations ([56,474](v10_1/runs/pretrain_16h/evaluation_0056474.json), [59,158](v10_1/runs/pretrain_16h/evaluation_0059158.json)) and [final status](v10_1/runs/pretrain_16h/status.json).
- The V10.2 implementation written after the document: [model](v10_2/model.py), [README](v10_2/README.md), and every JSON/log file under [v10_2/validation/](v10_2/validation/).
- Design history: [V10 proposal](v10/proposed_v10.md) and [reviewed RCDL contract](v10/V10_RCDL_Architecture_Reviewed.md), [V10_10M architecture](v10_10M/ARCHITECTURE.md) and [deep audit](v10_10M/audit/REPORT.md), and the [V10.1 design](V10_1_RCDL_Architecture.md).
- Earlier experiments, for the lessons the document should inherit: the [repository walkthrough](REPO_WALKTHROUGH.md) (v1–v4), [v4.1 report](v41_tiny/REPORT.md), [v5 report](v5_tiny/REPORT.md) and [review](review_v5.md), [v6](v6_tiny/REPORT.md), [v7](v7_tiny/REPORT.md), [v8](v8_tiny/REPORT.md), [v8 12-hour pilot](v8_tiny_12h/REPORT.md), [v9 sealed S1 results](v9_tiny/runs/S1_RESULTS.json), [v9 Q5 assessment](v9_tiny/runs_q5_12h/ASSESSMENT.md) and [review_v9.md](review_v9.md).
- One primary paper the document cites, to check a transfer claim: [Gloeckle et al., multi-token prediction](https://arxiv.org/html/2404.19737v1), §3.1 and Figure 3.

This is targeted source verification, not a line-by-line re-audit of every historical result or paper in the workspace.

### 2.2 What was executed

| Execution | Purpose | Result |
|---|---|---|
| Re-run of the document's review script from a copy that writes only to a temporary directory ([wrapper](v10_2_review_evidence/2026-10-12/rerun_spec_checks.sh)) | Check that document §10.2 reproduces | Every check value reproduced, including finite-difference error 1.6499736e-10 and kernel minimum eigenvalue 0.6178624; only the script's self-hash differs (§10.1) |
| SHA-256 of the reviewed document | Check that the reviewed text is the text the script certified | Matches `3e1307b0…f842` in [review_checks.json](v10_2_design_evidence/2026-10-12/review_checks.json) |
| Three new read-only CPU probes of the stopped V10.1 checkpoint (step 61,384) | Test the document's diagnosis and memory grid instead of restating them | [interfaces](v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.json), [kernel spectrum](v10_2_review_evidence/2026-10-12/probe_kernel_spectrum.json), [episode kernel](v10_2_review_evidence/2026-10-12/probe_episode_kernel.json); about 65 s in total on two CPU threads |
| One-line tie-breaking check on CPU and Metal | Confirm deterministic winner selection | [log](v10_2_review_evidence/2026-10-12/tie_breaking_check.log) |

The probes used dev data only: a fresh 64-document text panel, 64 math problems and 32 boards per Sudoku family, with a disjoint calibration draw for mean substitution (§10.2). They did not train, write memory, read optimizer state, or modify any existing file. They measure one frozen checkpoint; they are not retrained comparisons.

### 2.3 What was not done

No model was trained. No V10.2 checkpoint exists to evaluate. No sealed test data were read. The reviewed document, the V10.x code and the evidence directories were not edited. Calculations about power and kernels below are arithmetic on stated or measured quantities, not experiments.

### 2.4 Timing, references and severity

The document was finalized at **11:31 IST**. The V10.2 package (`v10_2/`) was written between **12:55 and 13:13 IST**, and the V10.1 run was stopped at step **61,384** at **12:41 IST**. This review scores the document. It reports the later implementation and run facts because they change which of the document's statements are current.

**References.** "Document §x" means a section of V10_2_RCDL_Architecture.md. A bare §x means a section of this review, except in **Location** lines and in the bold lead-ins of §7, which follow the document's numbering.

- **P1:** resolve before relying on the affected experiment or conclusion.
- **P2:** resolve so the interpretation or comparison is defensible.
- **P3:** clarity or completeness; does not change a decision by itself.

## 3. The first-principles chain V10.2 needs

### 3.1 Start with what a concept can know

Group g contains tokens `x[8g : 8g+8]` and is released at decoder row `e_g = 8g+7`. Decoder row i may read it only when `e_g ≤ i`. Everything in `U_g`, `Uplus_g` and `F_g` is therefore a function of `x[0 : e_g+1]`, which is a **subset** of the prefix `x[0 : i+1]` that E1, E3 and decoder self-attention already see at row i.

The pooled sources are computed through all 16 encoder applications, so a concept is not a summary of eight raw tokens. It is still a deterministic function of an earlier prefix. Conditioning on the full prefix therefore leaves the concepts nothing further to say:

```text
I(x[i+1] ; concepts released by row i | x[0 : i+1]) = 0
```

The concept route cannot add **evidence**. Whether the decoder can *use* that evidence is a separate question of computation. The route can help in exactly three ways:

1. **Extra serial computation.** The concept path applies 8 more block applications (4 blocks × 2 visits) at one quarter of the token rate, with attention across all earlier groups. The direct path has 16 encoder plus 4 decoder applications.
2. **A different inductive bias.** The forecast objective asks these states to summarize the next eight tokens.
3. **A control interface.** K codes, `xi`, and the RBF bank act on these states. This is the reason the project keeps them, but it is a post-training benefit, not a pretraining-loss benefit.

The repository walkthrough made the same argument for v3: the concept route was decisive only when the token path could **not** reach the evidence (the 70-token confined decoder). V10, V10.1 and V10.2 all give the decoder full causal self-attention and global E3. In that regime the concept route has to earn its keep by computation or control, and the evidence below shows it currently earns very little.

### 3.2 The necessary links

| Link | Property required | What the document establishes | What the evidence says |
|---|---|---|---|
| 1. Information | Concepts carry something useful the decoder cannot compute cheaply itself | Not argued explicitly; document §3.4 notes there is no hard bottleneck | Concepts see a strict sub-prefix; only computation, inductive bias or control can help (§3.1) |
| 2. Content | `U`, `F` carry example-specific, decision-relevant features | Assumed weak because of moving latent targets | Replacing V10.1's concept inputs by a constant mean costs **+0.0026**, versus **+0.0062** for value-zero removal: example-specific content accounts for about 42%, and a fixed mean recovers the other 58%. On math the mean is *better* (**−0.0011**) |
| 3. Access | The decoder can read the content where it needs it | Diagnosed as the main limit (document §1.2 items 1–2) | The router could have raised the concept weight above its 0.063 start; it lowered it in 3 of 4 text layers and all 4 Sudoku layers, and the routed concept contribution is 4–20× smaller than E1's. But the softmax is zero-sum and mass moved mainly to E1: E3 also fell (0.468 → 0.18–0.25), and the concept-to-E3 ratio *rose* in every text layer. Router weights therefore do not settle access versus content (§4.3) |
| 4. Learning signal | Token loss sends useful gradient into the core | Not quantified | V10.1 total-loss component norms at one logged update: core 0.028, encoder→concept CRC 0.005, versus chunk readout 1.44 and decoder 0.68. V10's token-loss-only probe: core 0.0064–0.0089 versus decoder 0.95–1.24 ([V10 audit](v10_10M/audit/REPORT.md)) |
| 5. Target | The auxiliary target is fixed and predictive | Future token IDs; correct move away from a moving latent | The cited paper finds multi-token prediction *worse* than next-token at small model sizes (F06) |
| 6. Recurrence | Visit two improves an outcome | Deep supervision of both visits | One-visit ΔNLL **+0.00028** on 64 text documents |
| 7. Selection | The prefix selector approximates the hindsight winner | Logs regret; notes noisy labels | v5 (variational prior) and v9 (hindsight-selected hypotheses) both found future-informed training hard to deploy from the prefix alone (F07) |
| 8. Answers | Token gains become complete answers | Strict scoring retained | The **same** 16 math problems and 4 boards at all 21 evaluations: math 0–3/16, every success a sorting problem; Sudoku 0/4 every time |
| 9. Memory | Kernel neighbourhoods are local enough to avoid interference | Grid `{0.25, 0.5, 1.0}`, calibration required | At σ=1 and σ=0.5 the bank behaves almost like a per-code global average (§4.4) |
| 10. Cost | Benefit per active hour beats a simpler model | Trained C0 control planned | No C0 or 16-hour A exists yet |

The chain shows where V10.2 intervenes: links 3 (access) and 4–5 (signal and target). The measured weak points are links 1–2: the concept content is weakly example-specific, and the decoder already has a path to the same evidence. **A better reader cannot create content the core does not produce, and the closest published form of the new supervision was reported to hurt at small scale.** That makes the trained C0 comparison, and a cheap access-versus-content discriminator on the frozen V10.1 checkpoint, the decisive experiments (§9).

## 4. Re-verification of the document's evidence

### 4.1 The stated numbers are correct

Every quantitative claim in the document's §1 traces to the fixed capture:

| Document claim | Source field | Captured value | Status |
|---|---|---:|---|
| Text dev NLL 9.04589 → 4.88238; perplexity 131.94 | `evaluation_initial`, `evaluation_0054168` `.families.text` | 9.0459 → 4.882379; 131.944 | Reproduced |
| Routine text NLL ≈ 4.85–4.91 since step 23,906 | evaluation history | 4.8475–4.9102 | Reproduced |
| Strict math 3/16; three sorting successes | `.generation.math.by_module` | sort 3/4, others 0/4 | Reproduced |
| Sudoku easy/hard 0/2, 0/2; TF NLL 0.21407 / 0.14079 | `.generation`, `.families` | Same | Reproduced |
| Text concept-off ΔNLL +0.006166 | `.families.text.concept_ablation_delta_nll` | +0.0061665 | Reproduced (see 4.2 for what it measures) |
| Math / easy / hard concept-off −0.004091 / −0.001496 / +0.000136 | same field, other families | Same | Reproduced |
| One-visit ΔNLL +0.000482; latent MSE 0.701525 → 0.700054 | `.one_visit_delta_nll`, `.latent_visit_mse` | Same | Reproduced |
| 7.47 effective codes over 5,000 updates | `telemetry_summary.last5000.text.effective_codes` | 7.4666 | Reproduced |
| Dense pre-clip median 3.804, p95 6.467, all clipped | `last5000_dense_grad` | 3.8038, 6.4667, clipped fraction 1.0 | Reproduced |
| 2,910 tokens/s sustained | `sustained_tps_last5000` | 2,909.7 | Reproduced |
| Exposure 101.21M / 0.722M / 4.298M / 1.004M; math 0.674% | `exposure`, `exposure_fraction` | Same | Reproduced |
| 54,480 updates, 115,059,825 tokens, 9.739 h, no gaps, no nonfinite records | top-level fields | Same | Reproduced |
| Parameter ledger 5,944,228 + 20,000,320 | review script and later [production check](v10_2/validation/production_resume_check.json) | Same in both | Reproduced; the implementation also instantiates it |

The arithmetic in document §5 is also right: `5,731,496 − 215,168 + 149,376 − 516 + 16,384 + 262,656 = 5,944,228`.

### 4.2 What the routine panel actually is

The trainer calls the evaluator with `batches=2` ([train.py:131](v10_1/train.py)), and each batch is `corpus.batch(1, family=f)` ([evaluate.py:54–56](v10_1/evaluate.py)) from a fixed dev seed. Every routine NLL and ablation number therefore comes from **two documents per family**, the same two at every evaluation:

| Family | Documents | Supervised targets |
|---|---:|---:|
| Text | 2 crops of 384 tokens | 768 |
| Math | 2 problems | 14 |
| Sudoku easy / hard | 2 boards each | 164 each |

The generation panel is likewise fixed: the identical 16 math problems and 4 boards at all 21 evaluations (one identity-set hash). The document correctly says repeated observations are not new examples, but "768 text targets" hides that n=2 for any between-document uncertainty.

Both V10.1 "concept-off" and "continuous-off" also use `ablate_continuous=True` ([evaluate.py:59–60](v10_1/evaluate.py), [model.py:79–83](v10_1/model.py)). That masks the concept router logit and **renormalizes** the two encoder weights. It is not the value-zero intervention. The document's own §7.2 rule ("any changed routing normalization is separately labeled") is not applied to its §1.1 evidence.

### 4.3 New probe: what the V10.1 concept route carries

On the stopped step-61,384 checkpoint, with 64 fresh dev text documents (22,976 targets), 64 math problems (657 targets) and 32 boards per Sudoku family. Intervals are document-level paired bootstrap 95% intervals, conditional on this one trained model:

| Intervention | Text ΔNLL | Math ΔNLL | Easy Sudoku ΔNLL | Hard Sudoku ΔNLL |
|---|---:|---:|---:|---:|
| Both routes off, router renormalized (the logged "concept-off") | **+0.00782** [0.00589, 0.00975] | +0.00226 [0.00016, 0.00425] | −0.00026 [−0.00128, 0.00072] | −0.00021 [−0.00106, 0.00063] |
| Both routes off, value-zero (router unchanged) | +0.00617 [0.00443, 0.00797] | +0.00188 [−0.00041, 0.00406] | −0.00030 | −0.00024 |
| PQ forecast off | +0.00462 [0.00318, 0.00607] | +0.00120 [−0.00060, 0.00279] | −0.00023 | −0.00015 |
| Continuous off, renormalized | +0.00290 [0.00192, 0.00389] | +0.00107 [0.00012, 0.00205] | −0.00004 | −0.00006 |
| Continuous off, value-zero | **+0.00103** [0.00017, 0.00184] | +0.00062 [−0.00004, 0.00136] | −0.00009 | −0.00010 |
| **Mean substitution** (both concept inputs replaced by calibration means) | **+0.00258** [0.00165, 0.00352] | **−0.00112** [−0.00214, −0.00016] | −0.00044 [−0.00093, 0.00004] | −0.00009 |
| One visit instead of two | +0.00028 [0.00021, 0.00035] | −0.00003 | −0.00005 | −0.00004 |

Four conclusions follow:

1. **The point estimate survives a 32× larger text panel.** The text concept effect is real, positive and small: about 0.6% perplexity (value-zero) to 0.8% (renormalized).
2. **About two-thirds of the "continuous" effect is renormalization.** Removing the continuous value while keeping the router's weights costs +0.0010, not +0.0029. On the two routine documents at the same checkpoint the split is starker: +0.00246 renormalized versus +0.00013 value-zero ([probe JSON](v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.json), `panels.routine.text`). The route the document replaces carried less than its logged number suggests.
3. **Most of the concept effect survives a content-free input.** Mean substitution leaves the router unchanged, so its comparator is value-zero both-off (+0.00617), not the renormalized figure. Against that comparator, example-specific content accounts for about **42%** of the text effect (+0.00258 of +0.00617); a fixed mean input recovers the other 58%. That 58% includes any dependence the decoder learned on the route's typical value, not only a literal offset. On math, example-specific concept content is *harmful*: the mean is better, with an interval excluding zero. This is the v4.1 lesson again: in [v4.1](v41_tiny/REPORT.md) §5, a header-null switch-off cost 0.158 of pair accuracy while constant content cost only 0.016.
4. **The routine panel misled about math in both directions.** On its two problems, concept-off was negative at every evaluation from 6,256 to 59,158. On 64 problems, the same renormalized intervention is positive. On 64 problems the calibrated control is negative, with an interval excluding zero: the opposite of what the concept hypothesis needs.

Router weights on the 64 text documents, averaged over legal rows (i ≥ 7):

| Decoder layer | E1 weight | E3 weight | Concept weight | Routed RMS, E1 / E3 / concept |
|---:|---:|---:|---:|---|
| Initialization | 0.468 | 0.468 | 0.063 | — |
| 1 | 0.696 | 0.249 | 0.055 | 0.0278 / 0.0104 / 0.0032 |
| 2 | 0.679 | 0.211 | 0.110 | 0.0291 / 0.0090 / 0.0073 |
| 3 | 0.782 | 0.191 | 0.027 | 0.0390 / 0.0097 / 0.0020 |
| 4 | 0.789 | 0.178 | 0.033 | 0.0432 / 0.0094 / 0.0027 |

On Sudoku the concept weight falls to 0.003–0.025 in every layer. The routed weights overwhelmingly favor the **early** encoder stage. This is consistent with both diagnoses: concepts may be uninformative, or E1 may simply be the best competitor in a zero-sum softmax. The concept-to-E3 ratio rose in every text layer, so the router was not simply shutting concepts out. The router does not decide the question; §9 gives an experiment that does.

### 4.4 New probe: RBF query geometry

Queries are `flatten(ExitNorm(state))/16`. With learned exit gains of RMS 1.114, every query lies near a sphere of radius about 1.1 (measured range 1.06–1.15). Pairwise distances are therefore bounded by about 2.2, and the Gaussian kernel cannot be very small at large σ. For cross-document pairs sharing a code:

| Family | Median distance | σ | Median kernel weight | Effective neighbours / same-code entries | λ_max of largest code partition (entries) |
|---|---:|---:|---:|---:|---:|
| Text | 1.14 | 0.25 | 3.2e−5 | 28 / 400 | 5.0 (466) |
| | | 0.5 | 0.075 | 253 / 400 | 80.8 (781) |
| | | 1.0 | **0.52** | **389 / 400** | **410.9** (781) |
| Math | 1.28 | 0.25 | 2.0e−6 | 4.4 / 57 | 9.3 (24) |
| | | 1.0 | 0.44 | 53 / 57 | 56.6 (103) |
| Hard Sudoku | 1.18 | 0.25 | 1.5e−5 | 13.7 / 164 | 17.9 (222) |
| | | 1.0 | 0.50 | 150 / 164 | 151.0 (222) |

Effective neighbours are `(Σw)²/Σw²` from the [interface probe](v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.json); λ_max, which also includes same-document pairs, is from the [kernel-spectrum probe](v10_2_review_evidence/2026-10-12/probe_kernel_spectrum.json). These use V10.1 queries as a proxy for a bank written from the same distribution. V10.2 keeps the same exit norm and query rule, so the sphere constraint carries over; its distances will differ.

A separate [episode probe](v10_2_review_evidence/2026-10-12/probe_episode_kernel.json) samples 200 three-document episodes per family, matching document §6.3's support size:

| Family | Median entries per episode | λ_max of largest same-code block, σ = 0.25 / 0.5 / 1.0 (median) |
|---|---:|---|
| Text | 138 | 1.4 / 6.5 / 22.0 |
| Math | 13 | 1.1 / 1.8 / 3.4 |
| Easy Sudoku | 63 | 2.2 / 6.6 / 12.7 |
| Hard Sudoku | 63 | 2.4 / 7.7 / 14.1 |

At σ=1 a read is close to a per-code average of almost every stored value. At σ=0.5 it is still broad. Only σ=0.25 is local. F05 derives the consequences for the write step and the read.

### 4.5 Evidence the capture contains but the document does not use

- **Component gradient norms** at the last logged update (`latest_training.component_grad_norms`): embedding 1.92, E1 1.98, **chunk readout 1.44**, decoder 0.68, E3 0.67, E2 0.36, Engram 0.31, exit reader 0.22, PQ head 0.074, pooler 0.058, **core 0.028**, **encoder→concept CRC 0.005**. This is one update, and under Adam a small gradient norm does not mean a small parameter update. It does show where the shared, clipped gradient budget goes: the auxiliary chunk head's own parameters carry about fifty times the gradient norm of the concept core.
- **Chunk-head loss** over the last 5,000 text updates is 5.78 nats versus 3.88 for the main decoder. The teacher-forced head is not close to the main path, which weakens (without refuting) the document's hypothesis that it solves its task by local continuation.
- **Latent MSE rose** from 0.576 (first 1,000 updates) to 0.697 (last 5,000). The document rightly says this is not worsening reasoning; it is also evidence that the target drifted.

### 4.6 What happened after the capture

| Fact | Source | Consequence |
|---|---|---|
| Two later evaluations: math 1/16 and 2/16, still sorting only | [56,474](v10_1/runs/pretrain_16h/evaluation_0056474.json), [59,158](v10_1/runs/pretrain_16h/evaluation_0059158.json) | 3/16 was the run's high-water mark, not a trend |
| V10.1 stopped at step 61,384 after 11.17 active hours, 4.83 h remaining, LR 8.6e−5, `stop_reason: null` | [status](v10_1/runs/pretrain_16h/status.json), [stop record](v10_2/validation/v10_1_stop.json) | No 16-hour V10.1 result exists. Arm A must be run in full before any matched 16-hour RF-versus-A claim |
| `v10_2/` implements only the RF arm and refuses any other architecture | [model.py:98](v10_2/model.py) (`if self.architecture!=Architecture(): raise`) | The document's "test these changes separately before combining them" has not been implemented |
| V10.2 CPU/Metal contracts pass: future perturbation at lengths 1–33, padding isolation, winner-recompute and accumulation gradients, both-route memory credit, real-model directional derivative | [contracts_cpu.json](v10_2/validation/contracts_cpu.json), [cpu_mps_parity.json](v10_2/validation/cpu_mps_parity.json) | Several document §10.3 "outstanding" gates are now met; the document is stale on this |
| 200-update full-size Metal run: 3,393 tokens/s, peak driver 4.31 GiB, minimum system-available memory 2.30 GiB | [sustained_200_final.log](v10_2/validation/sustained_200_final.log) | A two-minute measurement, not a thermal sustained benchmark; memory headroom is modest |
| The README cites `VALIDATION_REPORT.md`, which does not exist | [README](v10_2/README.md) | Implementation documentation is incomplete |
| V10.1 was resumed at steps 80 and 3,171; a checkpoint is named `before_throughput_step_0003171.pt` | [manifest](v10_1/runs/pretrain_16h/manifest.json), [v10_1/provenance.py](v10_1/provenance.py) | Whether executed code changed at that resume cannot be confirmed: V10.1's identity hashes only top-level `v10_1/*.py`, not `v10_1/optimization/` (V10.2's provenance fixes this). The run is a reference, not a clean seed-matched arm A |

## 5. What the document gets right

### 5.1 Evidence discipline

- **A fixed capture.** The evidence is frozen at one watermark with file hashes, rather than read from a moving dashboard. Every number re-traced exactly (§4.1).
- **Honest scope statements.** It says ablations on one checkpoint are not trained comparisons, that rising latent MSE is not worsening reasoning, that batch-normalized losses do not make math's gradient share 0.674%, and that hard Sudoku's lower loss does not mean it is easier. Those are the right cautions.
- **It refuses the easy repeat.** It recognizes that "add both CRC paths" was already V10.1, and that restricting decoder self-attention would not create a bottleneck because E3 is global.

### 5.2 Architecture and causality

- **The forecast head removes the teacher-forcing shortcut by construction.** It sees only a PQ slot and an offset embedding: no earlier target-chunk token, no encoder state, no raw `U`. Whether the old head used that shortcut is unmeasured, but the new head cannot. (It still shapes shared representations through the tied embedding; see §7.)
- **The release rule is exact.** A key is legal iff `same_document ∧ query_i < length ∧ e_g < length ∧ e_g ≤ i`. The review script checks all 2,376 rows for lengths 1–65 and reproduces.
- **The startup case is handled cleanly.** A parameter-free zero key/value sentinel, enabled only when no real key is legal, gives an exactly zero contribution with bias-free projections, and never an all-masked softmax.
- **Removing the zero-sum router is a real change.** In V10.1, raising concept weight required lowering encoder weight. A separate residual cross-attention sublayer removes that constraint. Combined with F01, this is a reason to test the reader, not proof that it helps.
- **The gradient-ownership statements are careful.** The RBF write derivative includes the aligned PQ path, both cross-reader sources and both K and V; the document says the attention softmax is not normalization of the memory sum.
- **The RBF claim is correctly bounded.** It restricts the kernel-descent statement to first order on a fixed replayed trajectory and disclaims finite-step, transfer and retention guarantees.

### 5.3 Objectives and experiments

- **Fixed targets instead of moving ones.** Supervising forecasts with future token IDs removes the stop-gradient-to-a-moving-target problem.
- **Visit weights sum to one** (1/3, 2/3), so two visits do not double the auxiliary budget, and the losses keep their own denominators under accumulation.
- **C0 is the first trained simpler-model control in the V10 lineage.** In V7, V8 and V9 the matched ordinary Transformer had **lower text NLL** than the concept hierarchy every time (V9 sealed text: by 0.035–0.077 nats on both seeds), while the hierarchy won on graph tasks ([v7](v7_tiny/REPORT.md), [v8 12-hour](v8_tiny_12h/REPORT.md), [V9 S1](v9_tiny/runs/S1_RESULTS.json)). V10 has never had that control. V10's math and Sudoku are, like GraphNav, exact-answer symbolic tasks, so the history suggests that C0 may win text and that the concept design's best chance is symbolic: exactly the metric that has scored 0/4 boards and at most 3/16 problems at every V10.1 checkpoint. Planning C0 is the most important experimental decision in the document; F01 and F03 ask only that it become mandatory and be judged numerically.
- **Complete-answer scoring stays strict.** Exact math with EOS and fully valid Sudoku boards remain primary; formatting, clue copying and teacher-forced loss are diagnostics.
- **Post-training is kept separate.** The draft-repair pilot is labeled a different generative protocol, not recurrence, and is not executed here.
- **Thresholds are predeclared as decision rules,** not statistical laws, and are to be frozen before results are read. F03 argues they need power, not that predeclaring them is wrong.

### 5.4 Implementation fidelity (this review's observation, not the document's claim)

The later [v10_2/model.py](v10_2/model.py) matches the specification at every point checked: decoder sublayer order (encoder CRC → self-attention → concept cross-attention → FFN); `M = W_U Uplus + W_F F` with no added norm; two-way encoder routers; query RoPE at `i` and key RoPE at `8g+7`; Xavier Q/K/V and gain-0.1 O; zero sentinel; forecast readout initialization; `1/sqrt(j+1)` offset weights normalized per document; `L_future = J_1/3 + 2J_2/3`; candidate utility `J_2(k)` with winner recomputation, where `utility.min(-1)` returned the lowest index on exact ties on both CPU and Metal in a direct check ([tie_breaking_check.log](v10_2_review_evidence/2026-10-12/tie_breaking_check.log); observed for PyTorch 2.10, not a documented guarantee); and total `L_token + 0.1 L_future + 0.1 L_explore + 0.05 L_selector`. The parameter ledger is instantiated exactly ([production check](v10_2/validation/production_resume_check.json)).

## 6. Findings requiring correction

| ID | Severity | Finding |
|---|---|---|
| F01 | P1 | The diagnosis "access-limited" is not established, and the document never states what a concept can add |
| F02 | P1 | The intervention plan lacks a calibrated constant-content control, so promotion criterion 4 can pass on non-specific concepts |
| F03 | P1 | The promotion thresholds sit near the noise floor, and "competitive with C0" is undefined |
| F04 | P1 | The screening pilot cannot discriminate the arms, the budget cannot hold the comparison, and the implementation skipped it |
| F05 | P1 | The RBF σ/η grid ignores the query geometry: at σ ≥ 0.5 the bank is nearly global, and η is not scaled to the kernel |
| F06 | P2 | Removing latent/VQ fitting rests on a hypothesis; the cited paper reports the opposite effect at small scale |
| F07 | P2 | Hindsight-winner training is never tested against a candidate-off control |
| F08 | P2 | The document's §1 evidence is presented more strongly than its panel and intervention support |
| F09 | P2 | The document is stale relative to the implementation and the stopped V10.1 run |
| F10 | P2 | The continual-learning protocol drops the parametric control and has little power per stratum |
| F11 | P3 | Smaller specification gaps: key scaling, decoder-blind utility, clipping interaction, deep supervision |

### F01 — P1: "Better access" is not yet a mechanism

**Location:** §1.2 items 1–2, §2 row 2, §3.4, recommendation paragraph.

The document attributes V10.1's weak concept effect to "fixed alignment and competition with strong encoder sources" and replaces the pointwise route with history cross-attention. Two facts weaken that diagnosis:

1. **A concept adds no evidence.** Every released concept is a function of a prefix the decoder already sees through E1, E3 and full causal self-attention (§3.1). The pointwise route was also not as isolated as "fixed alignment" implies: each `U_g` already attends over all earlier groups in the core, and the decoder's self-attention can read earlier rows that received earlier concepts. Cross-attention gives *direct, content-addressed* access, which may still help, but only if the concepts contain computed features the decoder cannot produce itself.
2. **Most of the effect survives a constant input.** Against the like-for-like value-zero comparator, example-specific content is about 42% of the text effect (+0.0026 of +0.0062), and on math the mean is better (§4.3). A better reader can only exploit the part that is example-specific.

The router data do not settle the matter either way (§4.3). V10.2 is therefore an expensive way to learn whether access was the limit.

**Required correction:**

- State the information argument and name the only routes to benefit: extra computation, inductive bias, and control.
- Before training any arm, run two **frozen-checkpoint discriminators** on the stopped V10.1 checkpoint. They cost about one to two hours, not a 16-hour run (full protocol in §9, Step 1):
  - *Access test:* freeze everything, add the proposed four cross-attention readers, and train only those readers on token loss for a fixed budget. Compare with identically trained readers over the pooled E3 sources `P3` (the core's own input) and over mean-substituted concepts. If concept readers do not beat both, better access will not rescue V10.1's content.
  - *Content test:* train the proposed forecast-only FFN head on frozen `F`, `U`, `P3` and the E3 state at row `e_g`. If the concept features do not beat `P3` and the E3 row, the concept core computes nothing predictive that the encoder lacks.
- Make the trained C0 control (document §8 arm table) mandatory in the first production comparison, not a later option.

### F02 — P1: Ablations need a calibrated constant-content control

**Location:** §7.2 intervention list; §8 criterion 4.

The document plans value-zero and route-off interventions, plus zero/shuffled forecasts for the auxiliary head. It has no constant-content control for the **main decoder** routes. §4.3 shows why this matters: on V10.1 text, value-zero both-off costs +0.0062 but a fixed mean costs only +0.0026; on math, both-off is +0.0019 to +0.0023 while the mean is −0.0011. Zeroing removes both what a constant input recovers and the example-specific content. Criterion 4 ("concept removal has a reproducible harmful effect") can therefore pass on a model whose concepts carry no example-specific information. The trained v4.1 plan-off model demonstrated exactly this false positive.

The cross-reader adds a further subtlety. Zeroing only `W_U Uplus` or only `W_F F` changes the keys as well as the values, because `M` feeds both. That is not a value-only intervention.

**Required correction:**

- Add, for every decoder route: (a) **training-derived mean substitution**, per family and slot; (b) **cross-document swap** within family, with matched group counts. Keep value-zero as a secondary diagnostic.
- Rewrite criterion 4 on the calibrated control: *example-specific* concept content must have a reproducible harmful removal effect, measured against the mean, on a declared target metric.
- Label every intervention by what it changes: values only, keys and values, or routing normalization.

### F03 — P1: The promotion thresholds lack power

**Location:** §7.2 panel sizes; §8 criteria 2–4.

The proposed large panel has 64 problems per math module (256 total) and 64 boards per Sudoku family (128 total). Criterion 3 accepts "math +3 points **or** Sudoku +5 points".

- **Math.** For two arms on the same 256 problems, the paired standard error is `sqrt(b + c)/n`, with b and c the discordant counts. At the observed accuracy range (10–20%) and a plausible 10–25% discordance, SE ≈ 2.0–3.1 points. A +3-point gain is about **1.0–1.5 SE**, well inside noise for a single seed.
- **Sudoku.** From a baseline of zero, +5 points is 7 of 128 pooled boards, which an exact sign test can detect (p ≈ 0.008). Per family (64 boards) it is 4 boards and p ≈ 0.06, and the document does not name the denominator. Pooled Sudoku is the only criterion with adequate power, and every V10.1 checkpoint so far scores 0 on every board.
- **Seeds.** Item-level intervals are conditional on one trained model. In V9's sealed S1 results, the same delayed-NCP recipe produced 16.8% and 83.2% standard accuracy across two seeds ([S1_RESULTS.json](v9_tiny/runs/S1_RESULTS.json)). A one-seed architecture ranking can be dominated by training-run variance.
- **Criterion 4** requires RF to be "competitive with the trained C0" without a margin. **Criterion 2** tolerates a 0.02-nat text regression, which is about three times the entire measured V10.1 concept contribution.

**Required correction:** predeclare a paired test and α for each criterion (McNemar or paired bootstrap for answers, document bootstrap for NLL). Size the math panel for the effect you want: detecting +3 points at about 2.5 SE needs roughly 1,000–1,750 paired problems, so either enlarge the panel (math generation is short and cheap) or raise the threshold. Replace "competitive" with a numerical non-inferiority margin against C0, at equal active wall time. Require at least two seeds before promotion, and report the seed spread beside item intervals.

### F04 — P1: The screening plan, the budget and the implementation do not match

**Location:** §7.1, §8 paragraph "First run correctness checks, then an equal 30-minute active-wall pilot…".

- **A 30-minute pilot cannot see the effect being screened.** At 2,910–4,063 tokens/s, 30 minutes is about 5–7M input tokens, or roughly 2,500–3,500 updates at V10.1's 2,112 input tokens per update (§8.2). On the two-document routine panel at V10.1's step 3,078, the text concept effect was −0.00019 and the PQ and one-visit effects were within ±0.0004 (evaluation history). A pilot can catch crashes, NaNs, memory growth and throughput; it cannot rank R, F and RF. The document says this, but then asks to continue "only a candidate with … larger-panel evidence", evidence a 30-minute pilot cannot produce.
- **The pilot's LR schedule is unspecified.** V10.1's design flagged that a shortened cosine differs from the production schedule. Running the pilot as the first 30 minutes of the 16-hour schedule would at least make it a continuable prefix.
- **The arithmetic does not fit the stated budget.** A defensible comparison needs at least A, RF and C0, each for 16 active hours on one Mac: 48 active hours (about 50 wall hours), before seeds. Adding R and F makes 80 active hours (§8.3). The document's "16 active hours, optionally 24" describes one arm.
- **The implementation already skipped the plan.** `v10_2` runs RF only and refuses other configurations (§4.6). A 16-hour RF run can say whether RF learns; it cannot attribute any change to the reader, the forecast head, or their combination, and it has no 16-hour A to compare against.

**Required correction:** state the total budget honestly. A workable minimum: run the frozen-checkpoint discriminators (F01) first; then A, RF and C0 for the same 16-hour schedule, one seed each, with R and F only if RF beats A on the calibrated criteria. Add a second seed for any arm that is promoted. Give the pilot the production schedule prefix and use it only as an engineering gate.

### F05 — P1: The RBF grid ignores the geometry it acts on

**Location:** §6.1–6.2, §6.3 "initial grids `{0.25,0.5,1.0}` and `{0.01,0.03,0.1}`".

Queries are exit-normalized states divided by 16, so they lie near a sphere of radius about 1.1 (§4.4). This fixes the kernel's dynamic range before any training:

- **Locality.** At σ=1 the median same-code weight is 0.52 and a text query has about 389 effective neighbours among 400 same-code entries. At σ=0.5 it is 253 of 400. Only σ=0.25 is local (28 of 400). Two of the three grid values turn the bank into a per-code running average of nearly every stored value. That is a global bias with a code index, not local correction, and it makes cross-stratum interference nearly unavoidable.
- **Write step.** Committing an episode changes its own replayed corrections by `−η K_episode G`. The local guarantee is first-order; a finite step behaves well only while `η · λ_max(K_episode)` times the loss curvature in `r` stays small. For sampled three-document text episodes (median 138 entries), λ_max of the largest same-code block has median **22** at σ=1, 6.5 at σ=0.5 and 1.4 at σ=0.25 ([episode probe](v10_2_review_evidence/2026-10-12/probe_episode_kernel.json)). One shared η grid therefore means about a 16× different effective step across the σ grid, before any interaction with stored entries.
- **Read mass.** Once the bank fills, a query sums `Σ w_i value_i` over every same-code entry. On the 64-document text panel the largest partition (781 entries) has λ_max ≈ **411** at σ=1 versus 5 at σ=0.25, about 80×. The per-entry bound on new value RMS does not bound this sum. V10.2 also lets each correction reach every later decoder row through cross-attention keys and values, so a correction's reach is the remaining document rather than eight rows.

**Required correction:**

- Calibrate σ on measured query-distance distributions: report effective neighbours and λ_max per stratum, and restrict the initial grid to the local regime (σ ≲ 0.25, plus smaller values).
- Set η relative to the episode kernel, for example `η = c / λ_max(K_episode)` with c on a grid, and record the achieved first-order and actual loss change.
- Bound and log the **read** norm, not only stored value norms.
- Treat the sphere constraint as a design choice: a cosine-distance kernel with an explicit temperature makes the bandwidth interpretable.

### F06 — P2: Removing latent and VQ fitting is a large change made on a hypothesis

**Location:** §1.2 item 3–4, §3.3, §4.1, §8 "F-fit, optional", §11 paragraph on Gloeckle et al.

The document concedes that the chunk head's shortcut use is "a hypothesis, not a proven defect" (document §1.2 item 3), yet the default arm removes three things at once: the teacher-forced chunk head, latent MSE and nearest-codeword fitting. Two pieces of evidence point the other way or are missing:

- The cited multi-token-prediction paper states, in its Figure 3 caption: "Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale." At 0.3B parameters its 4-token model scored 1.0 versus 1.8 on MBPP pass@1 ([Gloeckle et al., §3.1](https://arxiv.org/html/2404.19737v1)). V10.2 has 5.9M dense parameters. The document notes that gains "do not transfer automatically" but omits this directly relevant negative finding. V10.2 differs (an auxiliary head at weight 0.1, not a replaced trunk objective), so the finding is a prior, not a prediction.
- In V10, latent forecasts beat copy-current and causal-mean baselines (0.244 versus 0.322 and 0.273, V10.1 design §1.2). V10.1 never measured those baselines, so there is no evidence the latent objective had become useless.

Meanwhile, the forecast loss is sparse: one boundary per document gives at most 8 × 8 = 64 future targets per visit per update, against about 2,100 main-token targets. The old chunk Transformer made dense boundaries expensive. The new FFN head does not; the cost is mainly the vocabulary projection.

**Required correction:** make F-fit (latent MSE restored) a required arm, or at least a required frozen-checkpoint comparison, before calling F the default. State the Gloeckle small-scale result as a prior. Consider four to eight boundaries per document for the forecast loss while keeping one boundary for candidate scoring, and report the extra time.

### F07 — P2: Hindsight-winner training is never tested on its own

**Location:** §4.2; §8 arm table.

`utility_k = J_2(k)` uses future labels. `L_explore` trains the core on the winner, and the selector imitates the winner. Deployment uses the prefix argmax. This is winner-take-all multiple-choice learning. If future content decides the winner, codes may specialize to futures the selector cannot predict, making the **deployed** argmax forecast worse than a single hedged forecast. History is cautionary: V5's variational prior failed its gate by 0.34 nats while its posterior excelled ([v5 report](v5_tiny/REPORT.md) §3), and V9 recorded `prior_not_deployable: true` for its K=8 hypotheses ([S2_PROMOTION.json](v9_tiny/runs/S2_PROMOTION.json)). The main decoder never uses the winner, which limits the damage, but the shared core is trained on it every fourth update.

The machinery also has a measured cost: candidate updates run at 3,160 versus 4,063 text tokens/s in the 200-update benchmark, about 7% more time overall at a one-in-four cadence.

**Required correction:** add a **candidate-off** control that keeps K=8 codes and prefix selection but drops `L_explore` and `L_selector`. Report deployed `J_2` (argmax code), oracle `J_2`, and their gap over training. Predeclare that utility-based exploration stays only if deployed `J_2` and the main metrics are no worse than candidate-off.

### F08 — P2: The document's §1 evidence is stated more strongly than it can bear

**Location:** §1.1 table and its paragraph.

- "768 text targets" are two documents, and "14 math targets" are two problems (§4.2). Paired document intervals cannot be formed from n=2.
- Both "concept-off" numbers use router renormalization, which the document's own §7.2 requires to be labeled. On 64 documents, two-thirds of the continuous effect is renormalization.
- "Mixed" understates the math pattern on the routine panel (negative at every evaluation from 6,256 on). On a 64-problem panel the both-off sign reverses, and the calibrated effect is negative. Either way, the routine panel is too small to characterize math.
- 3/16 math was the high-water mark. The run then read 1/16 and 2/16.

**Required correction:** relabel the table with document counts and intervention type, add the larger-panel numbers, and record the post-capture readings. These change no number the document reports; they change what the numbers support.

### F09 — P2: The status is stale

**Location:** status line; §1 opening; §10.2 "no full V10.2 forward exists"; §10.3; final review decision.

The document says "Not an implemented or trained V10.2 model" and "Do not stop V10.1 or launch V10.2 on the strength of this document alone." Within about two hours, V10.1 was stopped at 11.17 active hours, a V10.2 RF implementation was written, and its contracts passed (§4.6). Several document §10.3 gates (future perturbation at lengths 1–33, winner-recompute and accumulation gradients, CPU/Metal agreement, both-route memory credit) are now met; others (sustained thermal benchmark, large-panel tasks, C0, retention) are not. The README's `VALIDATION_REPORT.md` does not exist.

**Required correction:** add a dated status section listing which gates passed where, that V10.1 ended without a 16-hour result, and that the production package implements one arm. If an RF production run is launched, label it a single-arm feasibility run, not the document's §8 comparison.

### F10 — P2: The continual-learning protocol is weaker than V10.1's

**Location:** §6.3.

The protocol is careful about leakage and ordering. Two weaknesses remain:

- **The parametric control was dropped.** V10.1's design §5.4 required a "declared-size gradient-trained linear correction control". V10.2 compares only the empty bank, exact RBF and shuffled keys. Without a parametric baseline (a small adapter or a fine-tuned head with matched storage), an RBF gain cannot be attributed to nonparametric memory. Repository history shows why retention needs a strong control: after V8's second and third local-support phases, sealed standard accuracy fell from 1,659/2,048 to **0/2,048** ([v8 12-hour report](v8_tiny_12h/REPORT.md)).
- **Little power per stratum.** Each stratum gets 8 held-out queries per round, 32 over four rounds, and those are re-scored repeatedly. Transfer and backward-transfer estimates per stratum will be very uncertain.

**Required correction:** restore the parametric control with matched storage and update budget. Increase held-out queries per stratum, or pool strata for the primary claim and report per-stratum results as descriptive.

### F11 — P3: Smaller specification gaps

- **Key scaling.** `M = W_U Uplus + W_F F` enters keys without normalization, to let correction magnitude reach K and V. A key-only RMSNorm (values untouched) would stabilize attention logits and remains differentiable for the RBF write. State why it is not used.
- **Utility is blind to the decoder.** Candidate utility scores only the auxiliary head. Code choice is never trained on what the main decoder gains from a code. This is a deliberate cost decision; it should be stated as a limit on what selector metrics mean.
- **Clipping interaction.** Every logged update was clipped. Under global-norm clipping, auxiliary gradients consume part of a fixed budget, and candidate updates (larger norm) clip the token gradient harder. Under Adam, loss coefficients barely change aux-only modules (forecast head, readers). Log clip factors separately for candidate and ordinary updates, and do not expect the "lower auxiliary weight" control in document §4.3 to change aux-only parameters.
- **Deep supervision and recurrence.** Asking visit one to forecast well may shrink visit two's increment. Report visit-two forecast and token gains with that interpretation.

## 7. Section-by-section notes

These notes follow the document's numbering. Where a finding already covers a point, the note only points to it.

**Status and recommendation.** The recommendation to keep the encoder, 8→2 pooling, two visits, soft PQ and the final RBF read is consistent with the user's fixed constraints. "Test these changes separately before combining them" is the right instruction and was not followed by the implementation (F04, F09).

**§1 Evidence.** Numbers are exact (§4.1). Panel size, intervention labeling, the math characterization and the post-capture readings need correction (F08). Document §1.2 item 3 correctly calls the chunk-head shortcut a hypothesis; the default arm then acts as if it were established (F06). Document §1.3's distinction between token exposure and objective weight is correct and useful.

**§2 Necessary conditions.** The table is a good format, but its second row ("a query must be able to retrieve relevant compressed history") presumes the binding constraint is access. The missing first row is the information argument (§3.1, F01). The falsifiable check for row 1 ("forecast beats a position-only control") is too weak: a forecast can beat position-only marginals using information the E3 row already has. Add "beats a head fed the E3 state at `e_g`".

**§3.1–3.2 Architecture and recurrence.** The pseudocode matches V10.1: the exit norm is applied inside the visit loop, so visit two starts from the normalized visit-one state ([v10_1/model.py:140–144](v10_1/model.py)). The prohibition on silently adding a third visit or halting is appropriate. Note the deep-supervision interaction in F11.

**§3.3 Forecast-only readout.** The construction and target arithmetic are correct, and the review script's 11,016 label cases reproduce. One effect is not mentioned: logits use the **tied** embedding `E`, so the forecast loss sends gradient into the token embedding that the encoder input and main output share. At weight 0.1 this is probably small, but it is a channel by which the auxiliary objective changes the main path. Measure it in the forecast-off comparison (embedding-gradient share and main-token NLL). Boundary density is in F06.

**§3.4 Decoder access.** Correctly specified and correctly implemented (§5.4). Key scaling is in F11. "This is not a hard information bottleneck" is honest; F01 asks the document to draw the consequence.

**§3.5 Release, startup and tails.** Exact and verified. The implementation rejects zero-length rows (`lengths<1`). Long text documents are cropped at a random offset ([v10_10M/data.py:33–38](v10_10M/data.py)), so group 0 usually begins at an arbitrary token rather than the document start; the group clock resets per crop, as specified.

**§4.1 Default objective.** Coefficients and reductions match the implementation. Removing latent and VQ fitting is F06. The sentence "Without explicit fitting loss, dictionary diversity and utilization may deteriorate; measure this" is right; make the measurement a promotion condition if PQ is to stay a claimed benefit.

**§4.2 K=8 candidates.** Hindsight training without a candidate-off control is F07. The list of selector diagnostics is good.

**§4.3 Clipping.** The advice not to change clipping, LR and reader together is sound. The probe-cosine plan is the right instrument. See F11 for why lowering the auxiliary coefficient will not change aux-only modules under Adam.

**§5 Parameters and compute.** Arithmetic is exact (§4.1). The cross-attention score-tensor estimate (`[8,4,384,97]`, 4.55 MiB) is correct and correctly labeled as not the total footprint. The later 200-update run peaked at 4.31 GiB of Metal driver memory with 2.30 GiB of system memory still available.

**§6 RCDL memory.** The read, write and first-order claims are precise. The σ/η grid is F05. Capacity arithmetic (84 support examples × at most 48 groups = 4,032 ≤ 4,096) holds as an upper bound; Sudoku examples release about 21 groups, so typical use is well below capacity. The parametric control and per-stratum power are F10.

**§7.1 Controlled pretraining.** Inheriting the V10.1 corpus, tokenizer, mixture and optimizer is the right choice for a controlled comparison. "Initialize new modules separately so their RNG consumption does not change the common backbone initialization" is important and should be a tested contract (same backbone tensors at step 0 across A, R, F, RF and C0).

**§7.2 Measurement.** The panel design (stratified, hashed, frozen) is good. Its sizes are F03. The text panel shrank from the 256 windows V10.1's design proposed to 128, without a stated reason; §8.1 suggests 256 is the safer size for a 0.02-nat margin. The document asks to measure the large panel's cost first; a first estimate is available now. The routine evaluation at step 54,168 took 32.3 s for 16 short math generations, 4 Sudoku boards of 82 generated tokens and the NLL panels, with prefix recomputation. Scaling Sudoku generation 32× and math 16× gives roughly 15 minutes per large-panel pass. That is an extrapolation from one routine run, not a measurement. The intervention list needs the calibrated control (F02). The "frozen-feature linear readout" control should name its features; the E3-row readout is the informative one (F01).

**§7.3 Post-training.** The draft-repair design is careful: frozen draft data, correct drafts included, no target-dependent hints, first-pass score primary. It is correctly kept outside the pretraining claim.

**§8 Ablations and promotion.** C0 is defined as "same encoder/Engram and decoder self blocks, with concept branch/auxiliary losses removed". State explicitly that C0 **keeps** the encoder-depth CRC into the decoder and the direct E3 input, so the only difference from RF is the concept branch. Thresholds and the pilot are F03 and F04.

**§9 Metal execution and telemetry.** Sound. One correction: V10.1's "component norms every 20" were a single sampled update per 20-update interval ([v10_1/train.py:155](v10_1/train.py)), not interval statistics. V10.1's design §8 asked for count/mean/p50/p95/max over each interval; carry that requirement into V10.2.

**§10 Review status.** Document §10.1's objection table is useful and mostly right. "Additional reader can still be ignored" is the residual risk the evidence now favors (F01). Document §10.2 reproduces exactly; document §10.3 is partly superseded by the implementation's passing contracts (F09).

**§11 Research context.** The TRM and GRAM distinctions are accurate and well-bounded. The multi-token-prediction paragraph omits the paper's own small-model result (F06).

## 8. Experimental design and statistical power

### 8.1 What the proposed panels can resolve

**Math, two arms paired on the same problems.** With discordance fraction d (problems one arm solves and the other does not), the paired standard error of the accuracy difference is `sqrt(d/n)`. At d = 0.20:

| Problems n | SE of difference | A +3-point gain in SE units |
|---:|---:|---:|
| 256 (proposed) | 2.8 points | 1.1 |
| 512 | 2.0 | 1.5 |
| 1,024 | 1.4 | 2.1 |
| 1,536 | 1.1 | 2.6 |
| 2,048 | 1.0 | 3.0 |

Arms trained on the same data stream from the same backbone initialization will be positively correlated, so d could be as low as 0.10. Across d = 0.10–0.25 the 256-problem SE is 2.0–3.1 points and a +3-point gain is 1.0–1.5 SE. The proposed panel cannot distinguish +3 points from noise. Math answers are short, so a larger panel is affordable.

**Sudoku from a zero baseline.** If A solves 0 of 128 boards and RF solves k, every discordant pair favors RF and the exact one-sided sign test gives `p = 0.5^k`: 0.031 for k = 5, 0.016 for k = 6, 0.008 for k = 7. A +5-point gain on the pooled 128 boards (7 boards) is resolvable at the item level. Per family it is not: +5 points of 64 is 4 boards, and `0.5^4 = 0.0625`. The document does not say which denominator criterion 3 uses; it must, before results are read. Pooled, this is the only criterion with adequate power, and every V10.1 checkpoint has solved 0 of its 4 routine boards. Item-level significance on one seed is still not seed-level evidence (§8.4).

**Text NLL, calibrated from history.** V9's sealed text comparison of two trained models on 512 documents had a 95% half-width of about 0.0057 nats (seed 901: −0.0769 [−0.0826, −0.0713], [S1_RESULTS.json](v9_tiny/runs/S1_RESULTS.json)). Scaling by `sqrt(512/n)` suggests about ±0.011 at 128 documents and ±0.008 at 256. A 0.02-nat non-inferiority margin is resolvable at 128, but 256 is safer. This is a calibration from another model family, not a prediction. The intervals in §4.3 are much narrower because they compare two interventions on the *same* trained model.

### 8.2 What a 30-minute pilot can see

| Throughput source | Tokens/s | Input tokens in 1,800 s | Updates at 2,112 tokens/update |
|---|---:|---:|---:|
| V10.1 sustained, last 5,000 updates, including pauses | 2,910 | 5.2M | 2,480 |
| V10.2, 200-update full-size run | 3,393 | 6.1M | 2,890 |
| V10.2, text non-candidate updates only | 4,063 | 7.3M | 3,460 |

At V10.1's step 3,078, on the two-document routine panel, the text concept, PQ and one-visit effects were −0.00019, −0.00036 and +0.00002. The concept route has not yet differentiated at pilot length. The pilot is an engineering gate (crashes, NaNs, memory growth, throughput), and it should use the first 30 minutes of the production schedule so it can be continued.

### 8.3 What the comparison costs

Per arm: 16 active hours, including routine evaluation and checkpointing as the document specifies, plus about 45 minutes if three large-panel passes are run outside the training budget. On one fanless Mac, arms run sequentially.

| Comparison | Arms × 16 h | Approximate wall time |
|---|---|---:|
| Minimum defensible: A, RF, C0, one seed each | 3 | ≈ 50 h |
| Attribution: add R and F | 5 | ≈ 84 h |
| Second seed for A and one promoted arm | +2 | ≈ +34 h |
| Frozen-era RBF study | separate | not estimated |

The document's "16 active hours, optionally 24" is the budget of one arm. The plan should state the total, and say which comparison is run if only one arm's budget is available.

### 8.4 Seeds

Item intervals describe one trained model. Training-run variance can be much larger. In V9's sealed S1, the same delayed-NCP recipe reached 16.8% and 83.2% standard accuracy on two seeds, and the flat model reached 13.7% and 0.0% ([S1_RESULTS.json](v9_tiny/runs/S1_RESULTS.json)). In v4.1, even a stable effect had a seed SE (0.0037) as large as its paired item SE (0.0033) ([v4.1 report](v41_tiny/REPORT.md) §3). The document asks for "uncertainty across examples/seeds" but sets no seed count and no rule for combining them. Declare both: at least two seeds for any promoted arm, the seed-level spread reported beside the item interval, and promotion only if the sign agrees across seeds.

## 9. Recommended revision and minimal next experiment

Each step lists its approximate cost, the rule that decides what happens next, and what it cannot conclude.

**Step 0 — Revise the document (hours).** Update the status (F09); relabel document §1 with document counts, intervention types and post-capture readings (F08); add the information argument (F01); state the multi-token-prediction prior (F06); define C0 as keeping encoder-depth CRC and the direct E3 input (§7); name the Sudoku denominator, the math panel size, the paired tests, the C0 margin and the seed rule (F03); add mean-substitution and cross-document-swap controls and rewrite criterion 4 on them (F02); restrict the RBF grid and scale η to the episode kernel (F05). *Cannot conclude anything about the model; every later claim depends on it.*

**Step 1 — Frozen-checkpoint discriminators on `v10_1/runs/pretrain_16h/stopped_step_0061384.pt` (about 1–2 hours on Metal).** Two small trainings with all V10.1 weights frozen and identical update budgets:

- *Access test.* Insert the proposed four cross-attention readers and train only them (with `W_U`, `W_F`) on token loss, leaving V10.1's frozen pointwise route in place so a gain cannot come from restoring a removed route. Repeat with the readers' memory replaced by (a) the **pooled E3 sources `P3`** at the same release times, which are the concept core's own input, and (b) mean-substituted concept states. Score on a 256-document text panel and the large math panel.
- *Content test.* Train the proposed forecast-only FFN head on frozen `F`, frozen `U`, frozen `P3` and the E3 row at `e_g`. Report held-out eight-token future CE for each.
- Train on the training stream; score only on dev panels.

*Decision rule, declared before running:* concept readers must beat both the `P3` and the mean-substitution readers by a predeclared margin, with no math regression. Because the readers share one frozen backbone, this is a within-model comparison with intervals near §4.3's (about ±0.002 nats at 64 documents, narrower at 256); a margin of 0.003–0.005 nats is resolvable and proportionate to V10.1's entire +0.0062 effect. A 0.01 margin would demand that access more than double the known effect; either is defensible if declared first. *Cannot conclude:* how a jointly retrained RF would behave, because a frozen decoder cannot co-adapt. Both tests also run on content trained under V10.1's latent and chunk losses, so they test R's premise (better access to existing content), not F's (that forecast supervision creates better content).

**Step 2 — Engineering gate (30 minutes per arm).** Run each arm for the first 30 minutes of its production schedule: crashes, nonfinite values, allocator growth, sustained throughput. *Cannot conclude:* any architecture ranking (§8.2).

**Step 3 — Production comparison: A, RF and C0 (about 50 hours, sequential).** Seed-matched shared tensors, the same recorded data stream, the 16-hour cosine, and the large panel at step 0, midpoint and end, sized per §8.1 (math at least 1,024 problems if generation time allows, text 256 documents, Sudoku 64 per family with the pooled test named). Judge with the calibrated controls (F02) and numerical margins (F03). *Cannot conclude:* whether the reader or the forecast head caused a difference, or whether the result survives another seed.

**Step 4 — Second seed (about 34 hours)** for A and any arm that cleared Step 3. Promote only if the sign agrees across seeds.

**Step 5 — Attribution (R, F, candidate-off, F-fit).** Only if RF cleared Steps 3–4. If it did not, attribution is moot.

**Step 6 — Frozen-era memory study (separate budget).** σ restricted to the local regime, `η = c/λ_max(K_episode)`, the parametric control restored, read norms logged, and the primary retention claim pooled across strata (F05, F10).

**Step 7 — Post-training pilot,** as document §7.3 specifies, from the checkpoint chosen in Step 3 or 4.

Two practical constraints come first:

- **The package cannot run Step 3 today.** [v10_2/model.py:98](v10_2/model.py) refuses every architecture except RF. Arm A needs the V10.1 code with fresh weights and the same implementation optimizations; C0 is a new variant without the concept branch; Step 5 needs R, F and candidate-off flags. This implementation work precedes Step 3.
- **The stopped V10.1 run cannot currently be resumed.** Its 4.83 remaining hours would give the cheapest complete 16-hour V10.1 endpoint. The resume is refused because `v10_1/telemetry.py` changed at 12:26 IST after launch, and the trainer hashes every `.py` file in `v10_1/`: the run's identity is `44012c37…`, the current tree's is `cd32d083…`. The trainer does not import that file. V10.2's provenance hashes `v10_1/*.py` too ([v10_2/provenance.py](v10_2/provenance.py)), so restoring the snapshot version for a V10.1 resume and keeping the new version for V10.2 must be sequenced, not done concurrently. A completed V10.1 run is a useful reference, not a clean seed-matched arm A (§4.6). Whether to finish it is the user's decision.

**The decision tree.** Step 1 decides which three arms Step 3 runs:

| Step 1 outcome | What it means | Arms for Step 3 |
|---|---|---|
| Concept readers beat the `P3` and mean readers (whatever the content test shows) | The core's output is more useful to this frozen decoder than its own input or a constant: the document's hypothesis survives | **A, RF, C0**, then attribution; if the content test failed, include F-fit in attribution |
| Readers do not beat the controls, but `F` or `U` beat `P3`/E3 | The core computes useful content that better access does not exploit; R's premise fails and F's remains open | **A, F, C0**; skip R |
| Neither: readers fail and `F`/`U` do not beat `P3`/E3 | Neither the core's computation nor its current supervision yields anything E3 lacks | **A and C0 first**; treat any concept arm as speculative, and ask what the core must compute that the decoder cannot |

In every branch C0 is required: three earlier versions found the ordinary Transformer better on text, and V10 has never measured it.

## 10. Validation performed for this review

### 10.1 Executions and artifacts

All files are in [v10_2_review_evidence/2026-10-12/](v10_2_review_evidence/2026-10-12/), run with `/opt/anaconda3/bin/python` and PyTorch 2.10.0. The probes ran on CPU with two threads; the document's own script sets one thread; the tie check ran on both CPU and Metal.

| File | What it does | Runtime | SHA-256 |
|---|---|---:|---|
| `rerun_spec_checks.sh` | Recreates the document's review script in a temporary directory with only the root, evidence and output paths changed, runs it, and compares with the recorded JSON | 2.4 s | `c7a018d2…2335` |
| `check_review_links.py` | Checks that every local link in this review resolves and that code fences pair | < 1 s | `b5d913e5…e8dc` |
| `probe_v10_1_interfaces.py` | Router weights and contributions, renormalized versus value-zero ablations, mean substitution, one-visit, query geometry and kernel neighbourhoods | 52.2 s | `a7ac676f…9effd` |
| `probe_v10_1_interfaces.json` (`.log` is the same text) | Its results | — | `09e89e68…a4ca` |
| `probe_kernel_spectrum.py` / `.json` | λ_max of each code partition over the large panel | 6.8 s | `04e9ffd9…8b33` / `a2f28d05…81b4ad` |
| `probe_episode_kernel.py` / `.json` | λ_max of the largest same-code block in 200 three-document episodes per family | 5.9 s | `654d42c5…88d1` / `3c8b9521…58a6` |
| `tie_breaking_check.log` | `min(-1)` tie behaviour on CPU and Metal | < 1 s | `a2fcd06e…0bc0d` |

The re-run of the document's script differs from its recorded output in exactly one field, `review_script_sha256`, which hashes the script file itself and so must differ for a path-patched copy. Every check value and the document hash are identical.

### 10.2 Panels

| Panel | Definition | Use |
|---|---|---|
| Routine | `Corpus('dev', seed=1811)`, two `batch(1)` draws per family, in the evaluator's order | Reproduces the document's panel |
| Large | `Corpus('dev', seed=20261012)`: 64 text, 64 math, 32 easy and 32 hard Sudoku | Interventions, router, geometry, spectrum |
| Calibration | `Corpus('dev', seed=20261013)`: 32 / 32 / 16 / 16 | Means for substitution only; disjoint draw |
| Episodes | 200 draws of three documents per family from the large panel, `numpy.random.default_rng(11)` | Episode kernel spectrum |

All are dev split. No sealed test data were read. The checkpoint is `v10_1/runs/pretrain_16h/stopped_step_0061384.pt`, SHA-256 `29742be3…0102`, matching the [stop record](v10_2/validation/v10_1_stop.json).

### 10.3 Primary source

[Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*](https://arxiv.org/html/2404.19737v1), §3.1, Figure 3 caption, read on 12 October 2026: "Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale." Appendix Table S7, read in the same HTML version, gives MBPP pass@1 of 1.8 for next-token and 1.0 for 4-token prediction at 0.3B parameters, versus 26.0 and 30.5 at 13B.

### 10.4 Non-modification check

The final check ran at **14:18 IST on 12 October 2026**, after all probes and edits other than this section and the last appendix row:

- The wrapper re-ran the document's own script, which re-hashes `V10_2_RCDL_Architecture.md` and every training source and manifest in `telemetry_summary.files`. All still match; the only differing field is the script's self-hash.
- Every SHA-256 in §10.1 was recomputed against disk and matches. Each probe's `.log` matches its `.json`.
- `check_review_links.py` found 80 local links (46 distinct), none missing, and paired code fences.
- `find . -newermt "2026-10-12 13:35:00"` (excluding only `__pycache__` and `.DS_Store`) returned 22 files: this review, the 12 files of the new evidence directory, and 9 monitoring or telemetry files written by three processes that were already running before this review began: the V10.1 collector (PID 75023, started 12:27 IST, `v10_1/telemetry_data/`), the V10 telemetry server (PID 56001, started 11 October, `v10_10M/telemetry/data/`), and the V9 monitor (PID 33229, started 10 October, `v9_tiny/monitor/`).
- **No file under any `runs/` directory changed.** No model, code, evidence or design file was modified, no training was run, and no memory was written.

### 10.5 Reproduce

From the project root:

```sh
sh v10_2_review_evidence/2026-10-12/rerun_spec_checks.sh
/opt/anaconda3/bin/python -B v10_2_review_evidence/2026-10-12/check_review_links.py
# The probes refuse to overwrite their outputs; move the existing JSON files aside first.
/opt/anaconda3/bin/python -B v10_2_review_evidence/2026-10-12/probe_v10_1_interfaces.py
/opt/anaconda3/bin/python -B v10_2_review_evidence/2026-10-12/probe_kernel_spectrum.py
/opt/anaconda3/bin/python -B v10_2_review_evidence/2026-10-12/probe_episode_kernel.py
```

## 11. Final score

This scores the **document's coherence and readiness**, not a trained V10.2 model. The dimensions and weights are those of [review_v9.md](review_v9.md) §12, so the two scores are comparable; the third dimension is renamed to fit V10.2.

| Dimension | Score / 10 | Weight | Reason |
|---|---:|---:|---|
| Problem framing and use of evidence | 7 | 20% | Exact, hashed, honestly caveated evidence (§4.1, §5.1); but the central access diagnosis is not supported (F01), and the panel and intervention types are under-described (F08) |
| Architectural contracts | 8 | 25% | Release rule, sentinel, forecast isolation, gradient ownership and ledger are correct, verified and faithfully implemented (§5.2, §5.4); minor gaps only (F11) |
| Objectives and training signal | 6 | 25% | Fixed future-token targets are a real improvement; removing latent/VQ fitting rests on a hypothesis against a cited small-scale negative (F06); hindsight training is never isolated (F07) |
| Experimental and statistical design | 4.5 | 15% | C0 and predeclared thresholds are strong choices; but no calibrated control (F02), thresholds near the noise floor with an undefined C0 margin and no seed rule (F03), and a pilot that cannot rank arms (F04) |
| Continual-learning design | 5 | 10% | Precise semantics and a correctly bounded first-order claim; but the σ/η grid ignores the measured geometry (F05) and the parametric control was dropped (F10) |
| Scalability and execution planning | 5 | 5% | Parameter and memory accounting are careful; the budget covers one arm, and the implementation runs only RF (F04, F09) |

The weighted result is **6.3/10**.

Against draft_v9's 5.9/10, this document is better specified and more honest about its limits. Its causal and gradient contracts need no repair; the items that do are specification choices (the σ/η grid, the C0 definition, the Sudoku denominator, the calibrated controls). Its weakness is different in kind: the motivation for the central change is under-evidenced, and its experimental plan cannot yet test that motivation at the stated budget. The score would rise substantially with the Step 0 revisions and a Step 1 result, whichever way Step 1 comes out.

**Overall score for V10_2_RCDL_Architecture.md as written: 6.3/10.**

## Appendix. Review process and consultation log

The review was built in stages, with a consultation with an independent reviewer model at each stage. The table records the advice received and what changed in this file or its evidence as a result.

| # | Stage | Advice received | What changed |
|---:|---|---|---|
| 1 | After orientation | Follow the review_v5/review_v9 format; verify numbers rather than restate them; check how V10.1 implements "concept-off"; flagged the stale status, hindsight selection, the information argument, removal of latent fitting, power, pilot length and the V10.1 stop | Adopted the structure; planned the verification steps |
| 2 | After re-running the document's script and tracing §1 | Numbers are sound; the issues are omissions and framing; run a scoped read-only probe on the stopped V10.1 checkpoint and keep evidence in a dated sibling directory | Wrote and ran `probe_v10_1_interfaces.py`; created the evidence directory and the file skeleton |
| 3 | After §2–§3 | Replace a pseudo-formula with the exact conditional-independence statement; caveat the zero-sum router | Rewrote §3.1. A check of the ratios showed the concept-to-E3 weight ratio *rose*, so the router claim was made neutral rather than adopted as suggested |
| 4 | After §4 and §6 | Compare mean substitution with value-zero, not renormalized removal; separate episode-scale from bank-scale kernel effects | Corrected "one-third" to 42%/58% everywhere; wrote and ran `probe_episode_kernel.py`; split F05 into write step and read mass |
| 5 | After §5 | Narrow "removes a shortcut" to the teacher-forcing shortcut; tie the C0 history to symbolic tasks; verify tie-breaking on Metal | Edited §5; ran the CPU/Metal tie check |
| 6 | After §7 | Add the text-panel reduction and a first large-panel cost estimate | Added both to the document §7.2 note |
| 7 | After §8 | Require the Sudoku denominator; widen the discordance range to 0.10–0.25; consider finishing the stopped V10.1 run | Edited §8 and F03; checking resume showed the run's source identity no longer matches, and that V10.2 hashes the same file |
| 8 | After §9 | Step 1 tests R's premise, not F's; branch three ways; keep V10.1's pointwise route during the access test; calibrate the margin | Replaced the two-branch tree with a three-branch table; added the margin calibration |
| 9 | After §10 | Do not assert what a filtered search hid; verify paper numbers personally; make the non-modification run the last action | Re-ran the search unfiltered and attributed every path; read Table S7 directly |
| 10 | After §1 and §11 | "Constant offset" overstates what mean substitution measures; scope the resume clause; avoid "measured-working" | Reworded §1, §3.2, §4.3 and F02; named the specification items that do need repair in §11 |
| 11 | After pass 1 (numbers) | Fix the "ninety minutes" timing, the link-1 wording, and a 64-document fraction attached to the logged panel; listed overreaching sentences | Fixed all; added the V10 token-only gradient figures; harmonized F01/F04 with §8–§9 |
| 12 | After pass 2 (overreach) | Bare § references collide between this review and the document; label the perplexity figure by intervention | Added the reference convention in §2.4 and prefixed every document reference; labeled perplexity and thread counts |
| 13 | After pass 3 (consistency) | Check links and code fences as the document's script does; trim the longest paragraph | Wrote `check_review_links.py` (all links resolve); moved resume provenance into §4.6 |
| 14 | After pass 4 (readability) | Do not embed a self-hash; time the wrapper; anchor every §1 number in the body; fix the closing order | Removed the self-hash; added the two-document figures to §4.3; wrote this appendix |
| 15 | Closing checks | Unavailable: the reviewer model was temporarily overloaded | The §1 re-read against the body, the evidence-directory pass and the final run in §10.4 were completed without a consultation |
