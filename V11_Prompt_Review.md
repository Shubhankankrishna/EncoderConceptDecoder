# Review of the V11 architecture design prompt

Reviewed 12 October 2026 against the supplied papers, the V10.3 experiment artifacts, and selected implementation files. This is a review of a design brief; it is not validation of an implemented V11 model.

**Verdict: ready to give independently to GPT-6 Astra Ultra, Opus 5.5, and Fable 5.1.** The prompt preserves the requested ECD direction while requiring an implementable causal specification and an experiment capable of rejecting its explanation.

## Coverage of the request

| Requested direction | Where the prompt makes it concrete |
|---|---|
| Successor to V10.3, informed by its performance | Section 3 includes both seeds, smaller C0 control, route interventions, probe results, and sleep outcomes; separates older V10.1 diagnostics from V10.3 measurements |
| Encoder–Concept–Decoder; E1/E2/E3 | Sections 1–2 retain the structure and E2's two gated streams |
| Recurrence of two | Section 2 fixes two shared-stack visits in E2 and the concept core; Section 11 separates parameter count from executed compute |
| Existing pooling, concept head, shift/repetition | Sections 2, 5, and 6 retain the 8→2 reference, require a boundary table, and preserve segment next-concept prediction |
| Additional NextLat/HiLP head | Sections 4–6 distinguish the papers and require a token-to-group interface plus explicit target and gradient choices |
| Explorative Modeling | Section 7 compares hard selection, soft mixtures, and continuous candidates, and requires a causal deployment policy |
| All three CRC directions | Sections 2, 5, and 8 specify alignment, release times, source selection, and intervention tests |
| IRC, Kimi attention residuals, Qwen gating, mHC | Sections 4 and 8 distinguish the mechanisms and require a primary choice rather than stacking names |
| RBF and gated LoRA sleep | Section 9 defines addressing, correction placement, representation changes, calibration exclusions, and multi-phase retention |
| Meta-controller ambition | Section 10 requires persistence, termination, controllable behavior, and a staged-training comparison |
| Small decoder and capacity allocation | Sections 2 and 11 retain a reference allocation and compare alternatives at matched budgets |
| Mac experiment and long-term scale | Sections 11–12 separate a tiny test bed, width-128 <10M reference with separate Engram accounting, and conditional 1–3T scaling estimates |
| Reusable prompt and self-review | Section 13 fixes the expected output and requires correction of the resulting design's own inconsistencies |

## Material checks and corrections

**1. The key causal distinction is explicit.** NextLat trains a transition after receiving the next token. Its predicted state cannot be fed back to help predict that same token. HiLP's upper transition instead predicts a later abstract state without future token inputs. The prompt makes the designer show information availability and incremental generation, not merely draw two heads. Sources: supplied NextLat PDF §3.2, pp. 5–6, Algorithm 1 p. 25; supplied HiLP PDF §2, pp. 1–2.

**2. Token and group clocks cannot be silently identified.** The existing two pool queries each attend to all eight states. Both slots are therefore released together after the eighth token. The review added an explicit distinction between completed-group causality, which can permit mutual attention between the two released slots, and strict slot-by-slot triangular attention. It also requires a separate audit of the raw concept-state CRC.

**3. GradCuit terminology was corrected after inspecting the implementation.** An earlier draft referred ambiguously to a GradCut/δ scalar. The inherited implementation actually uses a temporary code-space offset `xi` in `(e_k + xi) W_z`, constrained to zero in pretraining. The final brief preserves that contract and does not mistake it for a gradient multiplier. Source: `v10_10M/model.py`, `ConceptCore.forward`; reviewed V10 RCDL specification §1.2.

**4. NCP is represented accurately.** Its segmented soft-codebook forecast is supervised by MSE to a stopped continuous target; the brief does not invent mandatory discrete-label cross-entropy. All three requested CRC directions are present in the paper. Its IRC coefficients can be signed and unnormalized, while CRC source coefficients are normalized. Calling the skip paths mHC without expanded-state equations would be unjustified. Source: supplied NCP PDF §§2.5–3.5, pp. 7–9.

**5. Gradient detachment remains an explicit design decision.** Freezing a KL output head's parameters must preserve derivatives through the predicted input. HiLP's stopped SWA input does not by itself stop the direct token-state input to the combined head. The prompt identifies this ambiguity and requires an unambiguous implementation contract rather than copying prose mechanically.

**6. Exploration is neither dismissed nor assumed effective.** The supplied XM paper describes best-of-K and soft mixture interpretations; V10.3 used an all-branch mixture objective. The brief asks for a conditional-uncertainty argument, objective-consistent metrics, and a deployable prior. It avoids treating entropy or hindsight-winner classification as universal success metrics. Source: supplied XM PDF §3 and Appendix C; V10.3 `losses.py` and comparison artifacts.

**7. The meta-controller analogy has a staging constraint.** The temporal-abstraction paper's useful controller is learned over a frozen pretrained model, and its jointly trained comparison fails to develop the same structure in those experiments. V11 is therefore asked to compare staged and joint training. This is not promoted into a theorem that joint ECD training cannot work. Source: supplied temporal-abstraction PDF pp. 5–7.

**8. The V10.3 evidence is qualified correctly.** The four main NLL columns were checked against `comparison.json`. Removing a route can hurt its own model while a separately trained simpler model still wins. The brief reports the small positive Sudoku probe increment instead of repeating an overbroad “no information” claim. It identifies fixed-layout Sudoku, incomplete baseline matching, and the generation cap. It does not infer results from a missing equal-wall-time run.

**9. RCDL's promise is preserved without a retention guarantee.** The brief includes the full per-family sleep comparison. In particular, gated LoRA still harms related arithmetic and consolidation worsens add/sub NLL relative to RBF-only adaptation. A global token-weighted retention score can conceal this. Explicit write/calibration ID exclusion, per-family acceptance rules, memory-era checks, and multiple phases are required.

**10. The reference remains faithful while allowing research.** The review restored an explicit 3:1 sliding/full encoder attention reference and preserved the historical width/layer/FFN baseline. Smaller decoders, different targets, and residual choices remain labeled comparisons. Two recurrent visits, auxiliary rollout depth, exploration K, and controller duration are not conflated.

**11. Scaling claims are constrained by accounting.** A 120/60/20 allocation is analyzed as a hypothesis. Unique parameters, shared recurrent execution, compressed sequence length, expanded residual storage, Engram, teacher copies, RBF values, and active MoE parameters have separate accounting. Neither a 16-hour run nor a trillion-parameter performance claim is treated as established.

## Evidence and packaging checks

- Extracted text from all five supplied PDFs and inspected rendered method/equation pages from each; the brief cites the attached editions.
- Read primary web material for the additional references. The precise “Qwen 3.8 Next gated residual attention” mechanism remains unresolved and is explicitly labeled for verification.
- Copied the supplied papers and selected evidence into `V11_Paper/`; originals were preserved. `manifest.json` records source paths, sizes, and SHA-256 hashes for 30 copied files, including selected inherited implementation and design files.
- The source subset is for inspection, not a runnable package. The compact packet excludes datasets, model weights, and full training logs. The brief contains the essential measured findings and points to the original experiment when more data are needed.
- Re-read the written prompt for scope, causal clocks, objective ambiguity, source distinctions, and budget feasibility. The final task asks for one chosen architecture and a prioritized experiment plan, with optional mechanisms separated from the first test.

## Remaining limitations

The prompt cannot establish that the proposed extra head will add predictive information, that two visits will become useful, or that a small decoder will preserve quality. Those are its central experiments. The V10.3 control is smaller, the seed count is limited, and the strongest adaptation evidence covers one phase. No new training, model-code edits, or external model submissions were performed for this deliverable.

The brief is intentionally detailed because the dangerous ambiguities are at the interfaces between otherwise reasonable ideas. The recipient should produce a minimal primary design, not interpret the breadth of the comparison list as a requirement to implement every option.
