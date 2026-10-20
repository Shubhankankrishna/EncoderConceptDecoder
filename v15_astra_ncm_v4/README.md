# Astra v4: pretrained NCM with grounded recurrent planning

This is the reviewed implementation of the supplied LM1B / LoRA / recurrent GRAM
inspired / strategy / action / MCTS proposal. Read [REVIEWED_SPEC.md](REVIEWED_SPEC.md)
for the detailed first-principles review and corrections. The original proposal is
archived at [reports/ORIGINAL_ENGINEERING_SPEC_V1.md](reports/ORIGINAL_ENGINEERING_SPEC_V1.md).

The experiment tests exact graph planning and a separately trained language renderer.
It does **not** establish general reasoning, reproduce the complete GRAM paper, or
continue the prior 32-token language-model experiment. The original official
128-token / 641-code NCM hierarchy is retained for pretrained compatibility.

## Implemented architecture

1. Complete graph adjacency, current node, goal and risk features enter a
   permutation-equivariant state encoder. Oracle distances are labels only.
2. A shared spatial residual cell makes two refinements. Candidate action features
   score legal moves and discounted returns. Exact simulator transitions are canonical.
3. An explicit categorical strategy preference selects shortest or risk-averse paths.
   Two independent prior draws may repeat. No future-conditioned posterior is used.
4. Optional PUCT searches exact legal actions with discounted backups and local
   cycle/depth guards. No NCM decoding runs inside search. Search is evaluation-only.
5. A successful verified selected path conditions the official frozen NCM generator
   through zero-initialized FiLM and rank-4 attention LoRA. A canonical structured
   answer remains available if generated text is inaccurate. Failed plans are not
   repaired with an oracle and counted as neural successes.

| Component | Physical parameters | Training policy |
|---|---:|---|
| Official LM1B neural codec | 111,404,672 | Frozen, CPU canonical |
| Official LM1B Small generator | 97,065,600 | Original weights frozen |
| Attention rank-4 LoRA | 221,184 | Trainable in renderer stage |
| Full renderer additions, including LoRA/FiLM/plan encoder | 528,704 | Trainable in renderer stage |
| R2 graph planner | 346,564 | Trainable in structured stage |
| R2 planner with two preference embeddings | 346,820 | Trainable in strategy stage |

The fully instantiated strategy-plus-renderer configuration contains 209,345,796
physical parameters, of which 875,524 belong to new trainable modules. The two stages
train separately. Codec buffers, activations and optimizer states are additional
memory. Recurrence reuses 98,881 cell parameters twice; it does not create extra
independent parameters. R0, R1 and two untied controls isolate different costs.

## Data and staged run

There are 1,600 training, 200 validation, 200 test and 200 larger OOD graphs. Global
directed topology isomorphism checks reject renamed duplicates. Starts are balanced
across 2 / 3 / 4+ required moves, and 2–3 / 4–5 / 6+ in OOD. Training includes all
current states and all legal counterfactual actions, including unreachable states.

The offline NCM cache has 1,024 train, 96 validation, 96 test and 96 OOD verified plan
texts, plus 192 train and 64 validation ordinary-text retention examples. Retention
uses existing FineWeb train families, **not an official LM1B benchmark**. The cache
holds canonical float32 cumulative position-resolved features and integer codes.

The supervisor runs twelve structured controls: R0, R1, R2, parameter-matched untied,
arithmetic-matched untied and two-preference R2, each with seeds 42 and 43. Each gets
4,000 optimizer updates, batch 32, and the same data order where applicable. Then,
if both R2 seeds meet the registered grounded-interface gate, it runs up to 12,000
renderer updates, batch 1 accumulated four times, with frozen-base retention KL.
Renderer training stops early after six non-improving validation checks.

All work shares a **24-hour maximum from replacement setup**, including preparation
and final evaluation; this is a ceiling, not a promise to consume 24 hours. The
absolute deadline is recorded in `runs/20261019_lm1b_grounded/plan.json`. No source
or data mutation is allowed after registration. The supervisor serializes heavy
jobs and stops on nonfinite values, low memory, low disk or low unplugged battery.

## Run, stop and resume

Use the existing virtual-environment executable without resolving its symlink to
the base Python binary; the environment contains the required codec dependencies.

```bash
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder
TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
  v15_astra_m2/.venv/bin/python -u -m v15_astra_ncm_v4.run
```

This command resumes the same registered run and never extends its deadline.
The detached running process is recorded in the run's `process.json`. Send that
supervisor SIGTERM for a clean save, or create the run's `STOP` file. Remove a STOP
file only when explicitly resuming. Do not kill a trainer to start a competing job.
Production sources are frozen; changes require a separately versioned experiment.

The new [local dashboard](http://127.0.0.1:8767/) reports checkpoint steps, validation
success, value error, research gates and generated text. The older dashboard remains
available on port 8766 as history. Telemetry never loads tensor checkpoints.

## Validation evidence and limitations

- `reports/upstream_audit.json`: strict official loading, 100 ordinary and 100
  symbolic codec examples, all-level FiLM parity, greedy code parity, CPU/MPS
  agreement and frozen-weight invariance after optimization. Published decoder
  helper and explicit boundary-normalized output are reported separately.
- `reports/preflight.json`: real MPS optimization and exact next-update recovery for
  each structured variant; eight-graph memorization control, not held-out evidence.
- `reports/adapter_preflight.json`: exact float32 cache parity, no target/future-code
  leakage, conditioning gradients, LoRA optimization and exact optimizer recovery.
- `reports/unit_tests.log`: oracle, mask, search, graph permutation/padding, data
  stratification, prior and label-leakage regression tests.
- `reports/cli_smoke_venv/`: actual CLI clean stop/resume and complete renderer
  loss/evaluation/checkpoint smoke tests. Their weights are never production initializers.
- `reports/replacement_provenance.json`: preserved predecessor checkpoint hashes.

Graph success can be high on a small fully observed task without useful language
generation. Successor prediction can copy the action destination. Hard masking
guarantees legality without learning. A codec roundtrip score does not measure free
generation. A low code loss is not text perplexity. The dashboard keeps these claims
separate; final test/OOD scores and end-to-end planner-to-renderer cases are written
by `finalize.py` only after checkpoint selection on validation.

Equal-step and measured-cost controls are exploratory two-seed evidence. The MCTS
implementation does not by itself justify replacing BFS or another simpler planner.
No search superiority or pretrained transfer to planning is claimed without those
additional comparisons.
# Current status — 19 October 2026

The original staged pilot completed. Its graph planners learned the synthetic
task, but free NCM text did not faithfully render their routes. A separate,
tested continuation compares the original training procedure with model-generated
prefixes and correctly recomputed residual labels. See
[the diagnosis and comparison protocol](followup/README.md). Original registered
Python sources and checkpoints are preserved; the original deadline still applies.

