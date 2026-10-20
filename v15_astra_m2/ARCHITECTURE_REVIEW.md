# V15 Astra implementation review and M2 experiment contract

Reviewed 18 October 2026, Asia/Kolkata. Implementation version: **V15-Astra-M2-2-GRAM-first**.

The six files in `V15_Astra_Full_Training_Blueprint_Reviewed` were inspected, including
the complete 1,767-line training specification. The nine supplied mathematical
checks were independently rerun. The supplied blueprint and earlier V15 documents
are preserved; this directory records an explicit implementation variant.

## Verdict

The blueprint is coherent as a staged research program. It is **not ready to run
unchanged as a complete symbolic reasoning system**. Actual pretrained-codec tests
exposed representation failures that the earlier static review could only flag as
risks. A deterministic Stage-1 HRM experiment is implementable after selecting an
appropriate codec, fixing device compatibility, and testing causal interfaces.

The user explicitly requested **GRAM first** after the initial deterministic HRM
startup tests. The long run therefore trains GRAM first and deterministic HRM
second as a comparison. Both start from random initialization; no long HRM run has
been performed or used to initialize GRAM. This is an explicit departure from the
blueprint's requirement to establish held-out Stage-1 superiority before Stage 2.
The deterministic 28-example overfit passed, but it is not held-out evidence.
Stages 3–7 remain gated research work, not features of this run.

## Findings that affect implementation

1. **TinyStories is unsuitable for these symbolic probes.** On 2,000 development
   blocks, numeric-field lists were preserved in only 14.5% of blocks, and payload
   token fidelity was 78.72%. None of the seven chosen action sentences passed the
   exact action parser. For example, a preliminary probe reconstructed 42 as 52.
   It must not be used as a lossless state representation. Raw results and failure
   examples are in `reports/codec_audit.json`.
2. **LM1B has a different schema.** The released LM1B tokenizer uses 8,192 codes at
   each of 12 levels, not the TinyStories 4,096. It uses BERT WordPiece, not the
   TinyStories 8,192-entry BPE. The 128 text-token / 641 latent-code layout remains.
   The actual config and checkpoint are loaded with strict state-dict matching.
   On the same 2,000 probes its payload-token fidelity is 99.36%, and all seven
   action sentences parse correctly. The strict rendered-number-list match is
   57.35%; that metric also counts BERT's spaced minus signs and extra decoded
   suffix digits, so it must not be interpreted as a semantic numeric-error rate.
3. **Short-block padding is out of distribution.** LM1B preserves the beginning of
   the tested action sentences but emits text in padded positions. Action parsing
   therefore uses an explicitly registered first-complete-sentence protocol, with
   no target-length input at inference. The complete decoded output is retained in
   evaluation examples. This is a restricted action protocol, not faithful
   reconstruction of an entire padded block or a general language-model result.
4. **Use the specification's state-conditioned track.** Exact current simulator
   state is an explicit model input. Future state, labels, reward, optimal actions,
   distances, and verifier output never enter the model. Grid walls are represented
   in this current-state input. NCM codes also encode the observed textual prompt.
   This experiment does not claim that NCM codes alone retain exact state.
5. **NCM does not shorten this training sequence.** There are 641 code positions for
   128 input tokens. Reusing the frozen codec removes codec training and offline
   batching avoids re-encoding every epoch. The paper's speedups against its
   baselines do not establish a speedup for this 10M HRM on Apple Silicon.
6. **Apple MPS needs an area-pooling adaptation.** Stock PyTorch MPS rejects the
   non-divisible adaptive-average sizes used by NCM. The adapter implements the
   floor/ceil bin definition with fixed matrices. Its numerical error and hard-code
   differences are measured against the official CPU path. Device, dtype, batch,
   source revision and bundle identity are recorded rather than claiming universal
   bitwise equivalence after nearest-neighbor quantization.
   A wider 2,048-block test changed 283 of 1,312,768 codes (0.0216%), affecting nine
   blocks, despite passing all action round trips. MPS encoding measured 50.2
   blocks/second but failed the stricter CPU-equivalence gate. The experiment
   retains CPU encoding and decoding; only the trainable GRAM/HRM runs on MPS.
7. **Avoid unsafe upstream convenience paths.** The pinned `VQVAE.reconstruct`
   expects an obsolete quantizer return tuple, and `idx_to_zhats` appends aliases
   of an in-place cumulative tensor. Our full decoding calls the working official
   `decode`; partial decoding explicitly accumulates the official residual
   upsampling/projection path. The original source is unmodified.
8. **Root and future-target leakage need executed tests.** Unlike the released NCM
   generator's optional unconditional root prior, this model predicts the target
   root. Current/future target IDs cannot enter their own query. Parallel training
   agrees with level-by-level inference under identical previous levels.
9. **Full-code CE can hide failed actions.** Many target positions are common across
   the seven templates. Low loss or high code agreement is not enough. Report free
   decoded parsing, legal actions, optimal actions, and exact closed-loop success.
10. **10M must count the code tables and posterior.** GRAM has **10,000,333 trainable
    parameters**, including **237,505 training-only posterior/target-encoder
    parameters**. Its deployment package contains **9,762,828 parameters** and
    physically excludes both modules. HRM has **10,001,292 parameters**. Both
    counts include tables and all output parameters. The frozen LM1B
    codec has approximately **111.4M parameters**, plus frozen buffers; these are
    additional and are not represented as part of the 10M trainable budget.

## Review of later stages

| Stage | Assessment | Required evidence before adoption |
|---|---|---|
| 0: codec/data | Implemented and audited; domain and padding caveats above | Frozen hashes, runtime source parity, action preservation, split integrity |
| 1: HRM | Implementable normalized code model; recurrent benefit remains an experiment | Overfit, causal/gradient tests, real free-action and closed-loop results |
| 2: GRAM | Implemented short-unroll full trajectory ELBO; diagonal-Gaussian Astra adaptation | Prior-only overfit, separate full/truncated gradient tests, physical posterior removal, held-out comparison |
| 3: strategy modes | Formal mixture is normalized; noisy ELBO inside logsumexp is only a surrogate | Valid strategy diversity beyond independent GRAM samples at equal cost |
| 4: latent optimization | Score-function gradients are suitable for discrete sampled codes | Fixed-noise finite differences, independent baselines, verified reward, equal-cost sampling controls |
| 5: MCTS/value | Valid only with authoritative external transitions and complete actions | Root visits, terminal reward counted once, legal actions, calibrated value and matched best-of-N |
| 6: search distillation | Plausible after a demonstrably stronger search teacher | No test replay, verified canonical actions, improved student without search |
| 7: continual learning | Reversible design, no general retention guarantee | Both acquisition and retention, task stream matrix, replay controls, rollback |

Additional mathematical precision: if the frozen decoder is sampled from
`p(text|codes)` and has no direct dependence on steering `z`, the joint score
gradient in `z` needs only the code-policy score. A decoder score is necessary only
if its conditional distribution itself depends on the optimized variable. The
blueprint's conditional warning should not be read as requiring a nonzero extra
decoder gradient in the frozen, codes-only case. This run uses greedy decoding.

The supplied reference script writes to a hard-coded `/mnt/data` path. Its nine
test functions were executed through a wrapper, with results saved locally and
without changing the reviewed file. Passing these checks is separate from the
new integration and training tests.

## Registered implementation changes

* LM1B frozen codec replaces TinyStories because of measured domain fidelity.
* One complete observed block (`W=1`) plus typed current state is sufficient for
  these fully observed Markov tasks. Hidden states reset for every decision.
* Trainable code tables have width 64, projected to width 192. This factorization
  fits 12 × 8,192 input/output tables in the 10M budget. Tables are tied to output
  weights; no code IDs or refinement levels are dropped.
* Two context layers; two output layers; 192-wide hidden states; six attention
  heads; 16 slow and fast slots; two slow cycles with two fast updates each.
  HRM's SwiGLU width is 544; GRAM's is 432 to fit the prior and posterior in the
  same approximate parameter budget (a difference of 959 total parameters).
  The two arms have the same seed, but their different widths preclude identical
  initial weight tensors or an architecture-isolated causal comparison.
* Training output queries are vectorized over all 641 positions with a same-level
  self-attention mask and strictly earlier-level label-memory mask. This saves
  Python and Metal dispatch overhead without changing the conditional model.
* GRAM adds reparameterized diagonal Gaussian noise only after the slow proposal.
  Both conditional Gaussian distributions are evaluated on the same posterior
  trajectory. The training-only encoder uses the complete true target; generation
  never accesses it. The prior sees the proposal and observed context only.
* GRAM uses `(summed code NLL + summed trajectory KL) / 641`, beta=1, one posterior
  sample, two slow transitions, and full BPTT. No code corruption is used in GRAM.
  HRM uses clean per-code NLL as well for the comparison. This is a short-unroll
  full-ELBO adaptation, not a reproduction of GRAM paper Eq. (14)'s truncated
  surrogate. The truncated routine exists separately and its gradient cutoff is
  tested. Slot-isotropic variance is also unit-tested but is not a trained arm.
* GRAM's validation NLL is a four-sample prior `logmeanexp` Monte Carlo estimate,
  not exact marginal likelihood. Report prior sampling, prior-mean decoding,
  posterior success (training diagnostic only), valid diversity, KL and sigma
  statistics separately. Final tests use physically posterior-free weights.
* No target length or posterior is present in generation. No separate seven-class
  action head, nearest-template selection, simulator-based action correction or
  oracle fallback is used to make generated actions look successful.

## Dataset and scope

Executed startup checks: 10 base architecture tests, 10 GRAM tests, and the nine
supplied mathematical checks pass. GRAM's fixed-noise CPU/MPS loss discrepancy is
zero; tested Gaussian gradient discrepancies are below 7.5e-9. A stochastic MPS
optimizer/RNG save-resume test reproduced the next update exactly. At 1,200 overfit
updates, prior-only decoding gets 27/28 training examples correct for each of
three sampling seeds, including after physically removing the posterior. This is
an implementation diagnostic, not a validation or generalization result.

The aggressive 0.002 overfit learning rate produced transient instability. The
main run uses 0.0003, 500 warmup steps, gradient clipping, and batch 8 with two
accumulation steps. A 13-step production-shape GRAM benchmark measured about
18 blocks/second and 2.72 GiB MPS driver allocation; this is not a sustained
thermal or end-to-end throughput claim.

140,045 independently simulator-verified transitions: 109,943 train, 14,031
validation, 13,090 ID test and 2,981 OOD. There are 28,733 unique text blocks including
the seven canonical action templates. Each block is encoded once and stored as
641 uint16 values; training examples contain indices into that array.

Modular tasks vary modulus, add/subtract amount, multiplier, initial number and
goal. Grid tasks vary wall topology, start and goal. Rule/topology group assignment
precedes trajectory slicing; all starts, goals and steps from one group remain in
one split. Larger moduli and 9×9 mazes form an additional OOD set. Validation is
used for checkpoint selection; final ID/OOD tests are deferred until the run ends.

Actions are canonical sentences for add, multiply, subtract, left, right, up and
down. The simulator supplies the next actual state only after the generated action
is parsed and applied. This is a small action-policy experiment, not broad language
pretraining. Closed-loop results at horizons 5/10/20 do not imply that every puzzle
requires 10/20 actions; minimum solution lengths are reported explicitly.

## Resource and run policy

Float32 MPS training, batch 8 and two-step gradient accumulation, no concurrent
model training arms. AdamW, clipping, bounded device allocation, available-memory
and disk checks, atomic resumable checkpoints, and a persistent absolute deadline.
Codec preparation finishes before long model training. Two arms receive equal
allocated wall time within a 48-hour envelope, reserving 60 minutes for final
evaluation. This single-seed pilot cannot establish statistical superiority across
training seeds. Group bootstrap intervals measure held-out task variation only;
they describe a group macro-average, while the primary action rate is weighted
by sampled examples.

The supervisor prevents idle sleep while active, stops on the deadline, and limits
restarts. Closing the lid, shutting down, or removing power can interrupt execution;
resumption retains the original deadline. Stop files and commands are in README.

## Primary sources

* [NCM paper](https://arxiv.org/abs/2610.14162)
* [Official NCM code](https://github.com/HazyResearch/ncm), pinned commit
  `c70274273f489abc73832b2b98584a8483b6b203`
* [Official NCM checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints),
  pinned revision `d342ce8aaea58830eef3b8ad473664673e41f692`
* [BERT tokenizer](https://huggingface.co/google-bert/bert-base-uncased), pinned
  revision `86b5e0934494bd15c9632b12f734a8a67f723594`
* [HRM paper](https://arxiv.org/abs/2506.21734)
* [GRAM paper](https://arxiv.org/abs/2605.19376), especially Eqs. (4)–(14).
* The supplied reviewed blueprint is the source for the later-stage audit; those
  components have not been implemented or empirically reproduced in this run.
