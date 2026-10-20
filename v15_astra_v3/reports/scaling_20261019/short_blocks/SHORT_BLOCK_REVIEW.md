# A 30M GRAM generator with shorter prediction blocks

Assessment, 2026-10-19: a 30M capacity pilot is defensible, and shortening the prediction target while retaining broader context is a useful experiment. Neither change has yet been shown to cure the present repetitive generation. This report and its JSON proposal do not launch or replace training.

## 1. Define the units before comparing them

There are three different lengths: whitespace-separated words, BERT subword IDs, and NCM latent code IDs. The frozen local codec maps 128 subword positions to 641 code positions across 12 levels. A 128-position input is not 128 words, and the 641 codes are correlated representations of that input, not 641 additional text tokens.

In the saved evaluation example matching the quoted neuroscience passage, the passage belongs to the observed history. That history contains two 128-ID blocks (256 IDs, 214 whitespace-separated items in this example). Its reference continuation is a separate 128-ID target, containing 113 whitespace-separated items. Shortening the next prediction need not discard the broader observed history.

A CPU-only check using the pinned pre-tokenizer on the first 200 training documents gave:

| Chunk length in whitespace words | Median subword IDs | 90th percentile | 95th percentile |
|---|---:|---:|---:|
| 10 | 12 | 16 | 17 |
| 20 | 24 | 30 | 33 |

The sample follows corpus order and is not a random corpus-wide estimate. A 32-subword target is a reasonable initial approximation to the proposed short-word chunk; it is not a guaranteed 20-word segment. Sentence boundaries and longer words remain relevant.

## 2. What recurrence means for 30M parameters

The existing-codec candidate in `../candidate_configs/30m.json` contains 30,042,637 unique trainable scalar parameters. It retains two fast transitions per slow transition, two slow transitions, and one phase.

| Quantity | Count |
|---|---:|
| Unique trainable generator parameters | 30,042,637 |
| Unique fast-block parameters | 3,747,840 |
| Unique slow-block parameters | 3,747,840 |
| Fast applications per GRAM trajectory | 4 |
| Slow applications per GRAM trajectory | 2 |
| Fast/slow weight occurrences in the unrolled trajectory | 22,487,040 |

The last row is 4 × 3,747,840 + 2 × 3,747,840. These occurrences reuse 7,495,680 unique fast/slow weights. This is an accounting illustration for those two blocks only: it is not the whole graph's parameter-use count, a FLOP estimate, or an equivalent independent model size. There is no valid conversion from this recurrence schedule to a claim of 180M unique parameters or 180M-model capability. Shared transitions increase processing depth without adding independent weights; see the [GRAM paper](https://arxiv.org/html/2605.19376v1).

The separately frozen codec contains 111,404,672 model parameters. Thus the existing codec plus this trainable generator would contain 141,447,309 physical parameters, of which only 30,042,637 are updated. Cached codes avoid running the codec in every optimizer update. Training-only generator modules are omitted from its smaller inference export.

## 3. Why a shorter target might help

The task is to learn the conditional distribution of the next chunk given its context. A smaller chunk asks each completed hierarchy to coordinate fewer text positions and supplies generated context to the next chunk more frequently. That could reduce the burden on this small hybrid, particularly at chunk boundaries. This is a hypothesis requiring generated-text evaluation.

There is no universal rule that a 10M–30M model cannot generate 128-token passages. [TinyStories](https://arxiv.org/abs/2305.07759) demonstrates coherent multi-paragraph generation by small models on an appropriate restricted corpus; that result does not validate our GRAM–NCM architecture on general web text.

Shorter chunks also require more sequential chunk-generation calls for a fixed text length. They can introduce boundary discontinuities. More parameters do not automatically fix pooled context, earlier-level conditioning, free-generation robustness, decoding choices, or limited training diversity. The existing recommendation to try 20M first remains the cheaper capacity control; 30M is a reasonable larger pilot if that is the selected budget.

## 4. Why padding does not deliver the hoped-for compute saving

The local pretrained codec is tied to 128 positions through its configured hierarchy, learned resampling, decoder positions, and wrapper shape checks. It is not enough to change the raw window size to 32. Padding shorter content to 128 is compatible with the codec shape, but still yields all 641 target codes.

| Content IDs per chunk | PAD fraction for a full chunk padded to 128 | Training windows before shared-block filters | Relative window count |
|---|---:|---:|---:|
| 128 | 0% | 14,012 | 1.00× |
| 64 | 50% | 26,657 | 1.90× |
| 32 | 75% | 52,025 | 3.71× |
| 16 | 87.5% | 102,754 | 7.33× |

These are hypothetical resegmentation counts from training-document lengths, retaining a 256-ID observed context. No shorter cache was created. The live 128-ID dataset has 13,974 examples after its shared-block filters, so the table's unfiltered baseline should not be confused with the live count. Context-block handling would also need changes to retain the same span.

With the existing codec, shortening content does not reduce per-target generator sequence length and may increase total work for the same text. Padding-dominated reconstruction could also hide poor content fidelity; measure non-PAD accuracy separately. A native 32-position hierarchy requires a compatible trained codec checkpoint or codec adaptation/retraining, followed by reconstruction and boundary audits. No such compatible short-block checkpoint is selected here. A different number of hierarchy levels would also change the generator parameter count, requiring a new budget calculation.

The [NCM paper](https://arxiv.org/html/2610.14162v1) generates code levels from coarse to fine, predicting positions within a level in parallel. Its efficiency results do not establish an automatic benefit for padding short content into this different, fixed local hierarchy.

## 5. M2 Air feasibility and next experiment

The existing 128-position 30M candidate's analytical dense-forward estimate is about 14.35 GFLOPs per example. The earlier scaling assessment projects approximately 1.5–3.8 seconds per optimizer update at the current effective batch of eight. These are unprofiled planning estimates, not MPS benchmarks, and they do not apply unchanged to a redesigned short-block codec.

FP32 weights, gradients and Adam's two moments alone occupy about 0.448 GiB. Peak RAM additionally includes activations, attention workspaces, Metal caches, codec assets, checkpoint copies and other applications. Allocation feasibility is plausible; sustained throughput and memory headroom remain unverified for 30M. The running experiment retains exclusive access to heavy compute.

The proposed progression is:

1. Finish the protected run and evaluate saved-checkpoint generation, including sampled versus greedy codes, context sensitivity and coarse-level errors.
2. Choose the codec route. Short text padded into the existing codec is only a curriculum/fidelity diagnostic. A native shorter codec needs its own reconstruction audit before generator training.
3. Keep 256 observed subword IDs available and compare 32-, 64- and 128-ID targets. Include 16 only as an additional short-target control. Measure continuity, context relevance and repetition over complete continuations.
4. Compare 10M and 30M using the same codec and chosen target setting, with both equal wall-time and equal-data views. Do not attribute a combined codec-and-size change solely to model capacity.
5. Before any long 30M run, perform a short exclusive MPS preflight for finite gradients, checkpoint/resume correctness, sustained throughput and peak working set. Compare code likelihood only under the same codec; compare changed codecs using text-level quality, fidelity and end-to-end throughput.

## Scores

| Proposal | Research-priority score |
|---|---:|
| Shorter prediction target while preserving broader context | 8/10 |
| A controlled 30M GRAM capacity pilot | 7/10 |
| Padding 10–20 words to 128 positions as a compute optimization | 2/10 |

These scores are subjective judgments, not probabilities of success. The refined design goal is approximately 30M trainable generator parameters, a native short target near 32 subword IDs, and 256 observed subword IDs. Its benefit is unmeasured and codec compatibility must be resolved first.

Supporting machine-readable files: `analysis.json` and `proposal.json`. No live source, data, configuration, deadline or process was changed for this assessment.
