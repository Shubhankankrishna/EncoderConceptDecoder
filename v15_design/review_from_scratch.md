# V15 architecture: fresh first-principles review

Reviewed 16 October 2026, IST. Scope: `v15_Architecture.md`, completed v14 results, earlier experiment reports/configurations, and the two supplied papers. This is an author self-review, not an independent reviewer or model-training result. The main document contains the full review in §16; this record explains what was corrected and what remains uncertain.

## 1. Restart with the objective

The objective is better generated answers and text at a controlled total cost. Neither high code utilization, high auxiliary accuracy, a nonzero gate, nor sensitivity to an artificial intervention is the objective. The same prefix already determines every predicted latent; a latent can improve computation or training, but does not add new information to that prefix.

The design therefore separates target content, target predictability, and downstream use. An auxiliary-only arm, a trained plan-off arm, a generic prefix-feature bridge, and matched ordinary models prevent one metric from standing in for all three.

## 2. Recheck the evidence before selecting a remedy

The prior proposal was written before final evaluation. It has been archived, and the current document explicitly uses the completed checkpoint at step 12,977, not the last periodic log at 12,950. V14 improved Reasoning from 0.2829 to 0.3725 and text NLL from 2.8004 to 2.6861, with 12.09% fewer model parameters and more token exposure. This remains a single-seed historical recipe comparison.

V14 passes the copy/marginal and local-view diagnostics, so a blanket statement that it learned only copying would be false. It fails the incremental-C probe criterion and global teacher/online agreement. Its 86.22% reassignment is technically lower than v13's 92.36%, but is still very high. These findings motivate stable targets and a direct test of useful C computation, not deletion of the useful historical U route.

The continual result was also incorporated: the specialized method's lower forgetting accompanies lower acquisition and worse final average than replay. The old claim that comparative continual evidence was unavailable was removed.

## 3. Distinguish the papers' actual contributions

NCM trains a codec first, freezes it, then learns a hierarchy generator. Its next level is a detail level, whereas this project's next concept is a future time group. V15 adopts the training separation and conditional hierarchy, not the parallel token-output algorithm. Both training stages and label extraction count in the cost. The default allocation is an explicitly proposed 2.2-hour codec stage plus 8.8-hour student stage.

The codec's 25-bit summary of 16 tokens is much more restrictive than NCM's potentially redundant full hierarchy. Reconstruction gates and task fidelity therefore remain essential. The document includes the paper's likelihood-bound qualifications, conditional-dependence interpretation, exposure mismatch, whole-system parameter costs, and PDF/release configuration discrepancies.

Depth Memory retrieves contributions across layers within a forward pass. It is not persistent knowledge storage. V14 already has depth attention, so a published gain over PreLN does not establish a gain here. A plain residual C and trained U=p bypass were added to isolate whether retrieval, residual organization, or the entire C stack is useful. The paper's 7.8→14.7 and 19.4→21.2 figures were explicitly labeled as GiB of peak memory, not speed measurements.

## 4. Audit information flow and optimization

The future codec is label-only. Student plans derive exclusively from its own prefix state, with no true future lengths, masks or target codes in the forward path. Completed-group release is explicit. The copy baseline uses a fully observed past window; copying the previous overlapping future target would leak.

The root/fine bank selects joint pairs and retains a null tail. It does not claim exact text marginalization or consistent mode commitment. All metadata, including entropy and retained mass, is detached before the bridge. Under a uniform prior the four entries retain only 0.390625% of probability, so a weak early bridge can reflect diffuse predictions rather than a software fault. Alternative-root residual labels are recomputed under that root, and the robustness term remains a separately falsifiable hypothesis.

Freezing includes the entire codec, embedding, normalization, preprocessing and codebooks. EMA update order uses pre-update residual coordinates. A student's optimizer cannot change the target identity.

V11 exposed a clipping confound. The revised design clips auxiliary-only heads separately from the shared student parameters, while explicitly retaining the possibility of conflicting auxiliary gradients inside the encoder/C. No stability theorem is inferred from the bounded homogeneous delta write.

## 5. Audit measurement and cost

V14's probe split is by individual groups; correlated windows may cross split boundaries. V15 requires document/semantic-instance splits and grouped uncertainty estimates.

The earlier blanket dismissal of teacher-forced exact evaluation was corrected. For unique-reference deterministic causal decoding with matching termination, all-token exact correctness is equivalent to greedy exact correctness. Average token accuracy is not. Multi-valid-output tasks still require generation and verifiers.

An Engram-matched ordinary transformer now has an explicit formula count of 6,873,424, beside the 6,849,984 no-Engram control. The generic bridge head was recomputed as 26,085 parameters after including its norm; with delta C the model is 6,873,748. These are analytical counts, not instantiated-model measurements.

Cost includes codec training, readiness checks, overlapping-window encoding, student training, caches and blocking diagnostics. Research/calibration cost is disclosed separately. Component tensors are not presented as whole-system peak memory. The hardware gate includes long-running events rather than a short successful smoke alone. Primary comparisons are locked before sealed evaluation; claims against several test-scored controls require simultaneous or multiplicity-adjusted intervals.

## 6. Remaining reasons to reject the design

- A frozen codec can be stable but irrelevant, or encode future details the prefix cannot predict.
- Twenty-five bits may be insufficient, particularly for entities, arithmetic and exact outputs.
- Four selected joint alternatives may retain too little probability mass or blend inconsistent modes.
- The continuous history route may make the forecast bridge unnecessary.
- A shallow depth memory may lose to AttnRes, a plain residual stack or a trained bypass.
- Shared auxiliary gradients can hurt token learning despite separate head clipping.
- Extra codec and bridge cost can erase a matched-token gain at equal total time.
- A frozen code vocabulary and frozen prior may restrict continual acquisition.

Each has a corresponding comparison or failure gate in the main document. If only one component wins, keep that component under a distinct variant name. If no component beats competent matched controls, retain the baseline.

## Verification status

`check_design.py` performs small CPU-only formula, probability, release, ridge-regression, source-arithmetic and reference checks. The generated `review_checks.json` records actual outcomes and the reviewed document hash; `evidence_manifest.json` hashes the source artifacts. These checks do not execute a v15 model and do not validate its future implementation or quality. No training, evaluation job, telemetry service or existing experiment result was changed by this design revision.

**Disposition:** ready as a falsifiable implementation specification, with empirical benefit unresolved. The purpose of the review is to make that uncertainty testable rather than hide it behind an architectural name.
