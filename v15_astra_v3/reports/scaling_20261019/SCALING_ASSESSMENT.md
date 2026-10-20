# Should Astra GRAM scale to 20M, 30M or 50M?

Assessment: about 20M is a sensible next controlled capacity experiment. A 30M run is conditional on useful results from 20M. A 50M run is a poor next use of a single M2 Air and a 24-hour budget. These are research-priority judgments, not predictions of model accuracy.

1. Define the constraint.

This estimate uses the current FP32 MPS experiment, effective batch 4 × 2, and a recent median update time of 0.517 seconds over 60 logged updates. The anchor ends at step 14,750. The laptop has 16 GB of unified memory and a fanless M2 design. Sustained performance, background applications, allocator behavior and kernel shapes matter. No larger model was allocated, profiled or trained for this estimate.

2. Allocate the extra parameters where they can help.

The present 10M model uses 6.29M parameters in tied code lookup/output tables. The candidate recipes keep those tables at code_dim=64 and retain two context layers, two decoder layers, 16 latent slots, four fast block applications and two slow block applications. Their extra weights go mostly into wider processing blocks, rather than expanding the code tables. The full hybrid still includes the unchanged, separately frozen 111.4M-parameter NCM codec.

| Candidate | Width / heads / FF / Gaussian hidden | Total trainable | Fast+slow block weights | Model+grad+Adam state |
|---|---|---:|---:|---:|
| 10M current | 192 / 6 / 432 / 128 | 10,000,333 | 1,090,560 | 0.149 GiB |
| 20M proposal | 384 / 6 / 832 / 192 | 19,963,789 | 4,282,368 | 0.297 GiB |
| 30M proposal | 512 / 8 / 1072 / 256 | 30,042,637 | 7,495,680 | 0.448 GiB |
| 50M proposal | 640 / 10 / 1936 / 320 | 50,087,053 | 13,998,080 | 0.746 GiB |

These are illustrative budget-matched configurations, not hardware-optimized prescriptions. They preserve code_dim=64, so increasing width does not raise the rank of the final code-logit projection. The registered parameter ledger is reproduced exactly by the arithmetic for the 10M anchor, including the unused typed-state tables retained in the shared architecture.

FP32 Adam state is 4 bytes for each weight, 4 for its gradient, and 8 for its two moment tensors: approximately 16P bytes. This is NOT total or peak RAM. Activations, attention workspaces, Metal caches, masks, the CPU codec, OS/apps and temporary checkpoint copies are additional. The present checkpoint verifier can temporarily read roughly another 12P bytes of weights and optimizer state on CPU. A 50M model is not automatically too large for 16GB merely because of its weights; actual peak memory remains unmeasured. The runner also caps MPS allocation at half the recommended device working set, which is not the same as half of installed RAM.

3. Account for work instead of multiplying parameter counts.

The calculator counts dense matrix products and attention products for a training forward pass, treating each multiply-add as two FLOPs. It omits most elementwise work and does not profile fused/sparse kernels. Larger update times assume forward-work ratios roughly carry over to backward/optimizer costs. The same observed hardware efficiency is used only as a central scenario; the timing envelope is a subjective 0.65–1.60× of that estimate, not a confidence interval or guarantee.

| Candidate | Forward arithmetic ratio | Central seconds/update | Planning seconds/update | Central updates in a fresh 24h run |
|---|---:|---:|---:|---:|
| 10M current | 1.00× | 0.52 | measured anchor | ~157,805 |
| 20M proposal | 2.82× | 1.46 | 0.9–2.3 | ~55,872 |
| 30M proposal | 4.58× | 2.37 | 1.5–3.8 | ~34,426 |
| 50M proposal | 7.88× | 4.07 | 2.6–6.5 | ~20,027 |

Update budgets reserve ten minutes for final evaluation and 5% of the remainder for periodic validation/checkpoints. Those overheads are assumptions, not measured for the proposed models. Batch size and accumulation are held fixed. Reducing microbatch size to fit memory can change throughput and requires a new estimate.

Width scaling costs more than the headline parameter ratio here because the 6.29M code tables stay fixed while many processing matrices grow quadratically. The baseline proxy assigns about 61% of forward arithmetic to the hierarchical decoder blocks, about 21% to output-logit products, and only about 4% to the repeated fast/slow blocks. These are analytical shares, not measured runtime shares. The implementation therefore needs a profiler before claiming that recurrence is the main speed limit.

The table widens the whole current backbone. Selectively expanding the small-slot recurrent core could have a different cost than expanding the 641-position output decoder. Conversely, adding many small layers can increase MPS kernel-launch overhead even when the arithmetic looks cheap. Model size does not uniquely determine latency.

4. Separate representational bottlenecks from capacity.

The present context interface pools 1,282 code positions into 24 vectors. A wider model gives those vectors more dimensions and could improve retention, but leaves the number of summary vectors and the pooling mechanism unchanged. It also leaves clean earlier-level teacher conditioning, greedy categorical-code generation, equal per-code loss weighting, and the small training corpus unchanged. More weights might help the conditional distribution; they are not a demonstrated remedy for the repetitive output.

Hybrid context is an existing alternative that retains the finest 128 code positions plus 12 summaries from each observed block, for 280 context vectors. It changes the conditioning distribution and must be trained/evaluated as a separate control. Its estimated cost is also higher:

| Candidate | Hybrid forward work vs current 10M pooled | Central hybrid seconds/update |
|---|---:|---:|
| 10M current | 1.35× | 0.70 |
| 20M proposal | 3.97× | 2.05 |
| 30M proposal | 6.49× | 3.35 |
| 50M proposal | 11.13× | 5.75 |

These estimates change only observed-context pooling. The earlier-target-level summary interface remains the same. Hybrid context alone is not a complete faithful implementation of the NCM cumulative-reconstruction generator.

5. Account for data and evidence.

The training split contains 2,618 documents and 2,294,881 pre-tokenizer IDs, with 13,974 next-block windows. Their distinct supervised target blocks contain 1,624,621 non-PAD text-token positions. The 641 hierarchical code positions per block are correlated representations of 128 pre-tokenizer positions; they must not be counted as 641 new independent text tokens.

| Candidate | Central repeated passes over the existing target windows in 24h |
|---|---:|
| 10M current | 90.3× |
| 20M proposal | 32.0× |
| 30M proposal | 19.7× |
| 50M proposal | 11.5× |

More epochs do not add document diversity. Larger models can be more data-efficient in some regimes, so these counts alone do not prove that data is the current bottleneck. They do show that the same small corpus is a weak basis for committing immediately to 30M–50M general text pretraining. Expanding toward tens of millions of unique raw tokens would be a useful next pilot budget, not a proven sufficiency threshold.

A new CPU-only baseline check fits add-one per-level code frequencies on the training split and scores the exact same 64 validation examples as saved step 13,305. Its NLL is 7.192, compared with GRAM's 6.700. The model beats this particular context-free frequency baseline by 0.492 nats/code. This establishes more than improvement against uniform predictions. It does not establish use of document context: output position and correct earlier output levels can also help. It does not predict benefit from increasing parameter count. No sealed-test examples were used.

Compute-optimal scaling literature supports balancing parameters, training data and computation. Its numerical token/parameter rules were measured on other architectures and size ranges; they are not calibrated laws for this small recurrent, hierarchical-code experiment. No 20-tokens-per-parameter requirement is imposed here.

6. Make the decision with a small controlled progression.

- First audit a saved checkpoint for greedy versus sampled codes, coarse-prefix errors, shuffled context, prior-mean versus stochastic trajectories, and per-level likelihood. These tests should run under the existing heavy-job scheduling safeguards after the live job is idle.
- Establish a 10M baseline with the chosen improved conditioning/decoding recipe. Keep a clean version of the present baseline for comparison.
- Test approximately 20M with that same recipe and a larger, provenance-preserved corpus. A short real MPS preflight should verify finite gradients, checkpoint recovery, measured peak memory and sustained throughput before a long run. Prefer retaining at least 2GiB available host RAM in this proposed pilot; this is an engineering margin, not the existing 1.2GiB trainer stop threshold.
- Compare both equal wall time and equal examples processed. Require better held-out likelihood together with more coherent, context-sensitive free generation. A reduction in the repeated-word metric alone is insufficient.
- Consider 30M if 20M demonstrates a useful quality gain per unit of compute. Defer 50M until that evidence, adequate data, and a longer or faster compute budget exist.

A 20M capacity experiment is reasonable because its recurrent-block weights grow nearly fourfold, while its FP32 optimizer-state storage remains only about 0.30GiB. It is not justified as a guaranteed cure for the current repetition. A 30M or 50M model may be feasible to allocate, but feasibility and research value are separate questions.

Only CPU arithmetic, small cached-data statistics and metadata reads were performed. The live training source hashes remain unchanged. The candidate configurations are proposals under this report directory; no new production plan, model or job was launched.

Sources:

- [Apple: M2 MacBook Air fanless design and unified-memory bandwidth](https://www.apple.com/ca/newsroom/2022/06/apple-unveils-all-new-macbook-air-supercharged-by-the-new-m2-chip/).
- [PyTorch: MPS process memory fraction](https://docs.pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html).
- [Hoffmann et al.: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- [NCM: conditioning, generation and objective choices](https://arxiv.org/html/2610.14162v1).

Reproduce the arithmetic from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 v15_astra_m2/.venv/bin/python v15_astra_v3/reports/scaling_20261019/calculate.py
```

`estimates.json` records the exact module counts, calculation assumptions and timing scenarios. `candidate_configs/` contains the three illustrative configurations. `unigram_baseline.json` records the separate frequency-only diagnostic.
