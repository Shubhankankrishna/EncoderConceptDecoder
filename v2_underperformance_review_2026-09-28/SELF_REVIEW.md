# Self-review of the v2 underperformance report

This is a self-review by the same analyst, not an independent second reviewer. It checks the report against frozen source, raw results, counterexamples, and recomputed arithmetic. The evidence cutoff remains 28 September 2026 at 22:34:33 IST; subsequent training is outside the scored snapshot.

## Corrections made during review

1. **Depth-benefit row count corrected.** The draft said R6 improved every sampled row at both checkpoints. Directly recounting the stored per-row losses gives 16/16 at checkpoint 138 and 15/16 at checkpoint 267. WikiText row 123 worsens by about 0.00325 nats at checkpoint 267. Both language averages improve. The final report preserves the exception and makes no universal per-example claim.
2. **NCP summary counts corrected.** All 11 logged final-pass NCP values exceed pass 1, not 10/11. Pass 1 is best in the first two records; pass 2 is best in the remaining nine. The report and validation script use those exact counts.
3. **Padding confound removed from routing claim.** A severe last-quarter drop statistic included mostly padding in one batch. It was not used as evidence of real-token starvation. A separate four-batch routing diagnostic identifies real versus fully padded chunks and reports both assignment loss and loss of both experts.
4. **Loss denominators made explicit.** Training logs average per-microbatch token means; validation pools valid token sums/counts within each language. NCP/state summaries have different denominators. Capacity and carry intervention results were recomputed from sums/counts rather than averaging differently sized row losses.
5. **Training-window comparisons corrected before drafting.** v1 uses 50-step logs, v2 25-step logs. Pairs of v2 records are averaged for the matched comparison. A single v2 endpoint record is not compared with a v1 50-step mean as though the windows were identical.
6. **The older codebook-collapse interpretation was rejected.** The second checkpoint shows substantial recovery of code usage and explained variance. The final report distinguishes codebook health from concept-head prediction and acknowledges moving target entropy.
7. **Gradient credit and numerical correctness were separated.** A missing final-objective path to `W_s` was verified with autograd. The actual per-window gradient bridge was separately checked against a retained graph with identical detach boundaries. Passing that test does not validate full-backpropagation credit assignment.
8. **The prior carry-normalization concern was checked against implementation.** Current carry normalization runs inside `one_pass` and receives a gradient. The report does not repeat an obsolete design-pseudocode criticism as a current bug.
9. **The figure legend was moved.** Visual inspection found the gradient-panel legend overlapped carry-gain points. The final figure places it in unused space.

## Claims challenged before acceptance

| Claim | Challenge | Final treatment |
|---|---|---|
| v2 is worse early | Could this merely compare a scratch model with mature v1? | Matched-update and matched-time official evaluations confirm the gap. |
| Loops are destructive | Does token loss actually worsen as depth increases? | No on average in the diagnostic samples. Later NCP worsens, but token loss slightly improves. |
| NextLat is responsible | Do scalar losses or one batch suffice? | Used four batches and multiple shared matrices; still framed as a local optimization suspect requiring a training control. |
| Routing drops explain the whole deficit | Does disabling them recover the missing loss instantly? | No. Fixed-weight NTP change is tiny; long-run learning impact remains unknown. |
| A healthy LB/entropy score proves routing health | Can an overloaded router have those metrics? | A synthetic counterexample and actual per-row drops show it can. Counterexample is labeled synthetic. |
| State detachment prevents all language training of the core | Are parameters shared across later passes? | Yes. Shared later applications get NTP gradients; the unique initial-state projection does not. |
| More Engram parameters replace lost width | Is that supported by a controlled comparison? | No. Parameter reallocation is measured; its causal contribution is unquantified. |
| Current probes prove failure | Was v1 already solving them at this stage? | No. Early probe results cannot settle eventual reasoning ability. |
| Current configuration should be changed immediately | Has a winning candidate been trained? | No. The report proposes controlled continuations and leaves the active run intact. |

## Mechanical verification

The [validation script](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/validate_report.py) recomputes central comparisons, gradient-sign claims, routing ranges, pass counts, parameter arithmetic, and score weights. It verifies source-line bounds and local artifact links, checks the data continuity ledger and inventory counts, and compares active v2 source hashes with the frozen code. Detailed outcomes are written to [review_checks.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/review_checks.json).

The independent numerical backward check uses K12, three detach windows, nonzero capacity dropping, and accumulation 16. Maximum relative gradient error is approximately `5.93e-15`. It validates the actual truncated graph and encoder bridge, not the merits of the objective or the equivalence of truncated and full backpropagation.

The figure was inspected after rendering. Axes identify separate loss units, fixed-checkpoint versus training comparisons, raw-gradient cosines, and real-chunk assignment drops.

## Limits retained in the final report

- One run/seed, about 2.2 hours of a 48-hour plan, two official evaluations.
- Only eight sampled validation rows per language for fresh depth/branch diagnostics; no claim that these equal the 128-row official evaluation.
- Four training microbatches for replicated routing/gradient checks, selected deterministically from subsequent data positions rather than sampled as an independent population study.
- Gradients cover selected weights and encoder-output leaves, not every encoder parameter and not Adam-preconditioned updates.
- Existing v1 logs supply the early baseline; equal-step v1 binary checkpoints are no longer available.
- Fixed-weight interventions do not establish how a separately trained variant would perform.
- Data continuity was checked by inode/size/mtime against the earlier comprehensive audit; the complete corpus was not rehashed during active training.
- Old design papers and vendored dependencies were not individually re-audited. The coverage appendix explicitly distinguishes inherited review, current source inspection, and mechanical inventory.
- Training checkpoint files may rotate out later; this evidence bundle preserves measurements and code, not full immutable model weights.
- The 5/10 score is a transparent subjective rubric for the demonstrated recipe, not a standardized capability metric.

No unresolved observation justifies calling the full performance deficit causally explained. The report identifies confirmed defects and measured tensions, then specifies experiments that could confirm or reject their contribution.
