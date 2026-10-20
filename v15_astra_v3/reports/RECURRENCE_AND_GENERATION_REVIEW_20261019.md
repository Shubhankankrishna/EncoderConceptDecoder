# Recurrence, capacity, and the repetitive-generation result — 19 October 2026

Recurrence increases computational depth while reusing a fixed parameter set. It can expand what a compact network can compute. There is no justified conversion from six recurrent block applications to a model with six times as many independently learned weights. Parameter count alone is insufficient to explain this experiment's current failure to produce coherent text.

This review uses the registered configuration, source, and saved evaluations. It launches no model, codec, profiling, or training job. The 26 registered top-level Python source hashes still match the live plan.

**Verified configuration and parameter accounting**

| Quantity | Current run |
|---|---:|
| Unique trainable parameters | 10,000,333 |
| Tied input/output code-table parameters | 6,291,456 (62.9%) |
| Distinct fast recurrent block parameters | 545,280 |
| Distinct slow recurrent block parameters | 545,280 |
| Fast block applications per predicted block | 4 |
| Slow block applications per predicted block | 2 |
| Recurrent slots / width | 16 / 192 |
| Context code positions | 2 × 641 = 1,282 |
| Context memory vectors after pooling | 2 × 12 = 24 |
| Coarsest root's share of code positions in the loss | 1/641 = 0.156% |
| Last four levels' share of code positions in the loss | 404/641 = 63.0% |

The recurrent-block counts follow directly from `Block`: two attention modules (8d² weights), three feed-forward matrices (3df weights), and four LayerNorms (8d parameters), with d=192 and f=432. These counts are not FLOP estimates. Context encoding, Gaussian heads, hierarchical output decoding, softmax, and the frozen codec add computational work. The frozen codec's 111.4M parameters are additional to the trainable model ledger; their existence does not give GRAM a pretrained language generator.

This run performs one short phase with two slow transitions and two fast refinements per transition. The hidden state is freshly initialized for each next-block prediction. There is no state carried across independent training examples, outer deep-supervision cascade, search, or persistent strategy. This is an Astra adaptation of GRAM, not a complete paper reproduction.

**Verified implementation differences that deserve investigation**

1. Learned attention pooling reduces each observed code level to one vector before the context transformer. The same summary mechanism is used for earlier output levels. Positional embeddings are present, so order is not entirely absent, but local detail may be difficult to preserve. Additional reasoning can only work with the information supplied to it. The original NCM generator conditions on position-resolved cumulative codec reconstructions; our code-summary interface changes that design substantially.

2. Both training likelihood and saved prior-only code likelihood use the correct earlier target levels while scoring a target block. Free generation uses the model's own earlier predictions. This is legitimate likelihood evaluation, not target leakage. It nevertheless creates an important free-generation robustness test. Current GRAM training explicitly rejects conditioning corruption to retain the declared clean terminal ELBO. The NCM paper uses conditioning-code corruption as a separately modified objective to improve robustness against accumulating code errors. Adding it would require a new registered objective/control, not a silent modification to the live run.

3. Saved continuations sample the recurrent Gaussian trajectory but choose every categorical code using argmax (temperature=0). NCM's generation algorithm samples categorical codes and then takes argmax only at the final token decoder. Greedy code selection may amplify modal, repetitive code combinations. A fixed-checkpoint sampling comparison is needed before attributing these outputs to all possible samples from the trained distribution.

4. The loss gives equal weight to individual code positions. Its root code contributes one of 641 terms, while the last four levels contribute 404. This is a valid likelihood weighting and does not imply root learning is necessarily weak; coarse levels can also receive gradients through later-level conditioning. It does mean aggregate loss is dominated by fine-level code terms. Per-level losses and generated-prefix diagnostics are required to tell whether the coarse choices are the bottleneck. Level balancing would be a deliberately different objective.

5. Logged total KL is around 1e-4 nats per block at the time of review, despite nonzero prior sigma. This says the target-informed posterior is close to the prior on those trajectories. It does not prove that noise or recurrent state is ignored, and it does not establish an advantage from stochastic reasoning. Context shuffling and fixed-noise/mean-state interventions are required to measure use of those inputs.

6. Training reuses a small curated public corpus (3,275 complete documents across all splits). Repeated epochs add optimization work, not new document diversity. The official codec is frozen, but the GRAM predictor and its own code embeddings started from fresh weights. The GRAM paper primarily evaluates controlled puzzles and generation probes; it does not establish this exact raw-text/NCM hybrid as a validated language model.

**What current observations establish**

At saved step 9,998, held-out prior-MC code NLL was 6.7120, improved from 9.0117 initially. The 64 saved greedy-code continuations were all dominated by a repeated word. Correct reference codes reconstruct largely normal reference text, consistent with the separate 96.70% token-reconstruction audit. These observations support investigating generated code combinations, conditioning and decoding; they do not isolate a unique cause, prove GRAM is inherently unable to model text, or prove that more recurrent steps will fix it.

**Highest-value diagnostic sequence for a separately scheduled checkpoint audit**

- Hold checkpoint, examples, and latent noise fixed; compare greedy code selection with bounded categorical sampling at declared temperatures. Score repetition, context relevance and code likelihood separately.
- Measure per-level NLL and coarse-code accuracy. Compare rollouts with correct coarse prefixes against entirely generated prefixes, labeling the former as an oracle diagnostic.
- Hold noise fixed and shuffle context blocks across examples. Measure changes to logits, likelihood and generated outputs to test context dependence.
- Compare fixed stochastic trajectories with prior-mean trajectories and multiple fixed seeds. Near-zero KL alone cannot answer whether stochasticity matters.
- Sweep recurrent depth at a frozen checkpoint as a sensitivity experiment. Extra inference steps are outside the trained unroll; failure would not rule out training a deeper model. A real depth comparison requires separately trained controls and measured compute budgets.
- Compare pooled and position-preserving/hybrid conditioning using retrained models and the same data. Checkpoint weights cannot simply be transferred to a changed interface and treated as a matched result.

These diagnostics should use an isolated source snapshot and the project's heavy-job scheduling safeguards. They are proposed here; none was launched alongside the live run.

Primary sources:

- [Generative Recursive Reasoning, Sections 1–2 and experiments](https://arxiv.org/html/2605.19376v1).
- [Next-Concept Modeling, Sections 3.2–3.4 and Appendix A.2](https://arxiv.org/html/2610.14162v1).

Implementation evidence: `model.py` (pooling, conditional decoder, greedy code generation), `gram.py` (short stochastic trajectory and clean terminal ELBO), `evaluate_raw.py` (prior likelihood and default generation), and the registered `gram/manifest.json`.
