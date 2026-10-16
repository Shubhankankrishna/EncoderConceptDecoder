# V8 tiny: competence before complexity

This is a self-contained implementation and a **single-training-seed, 12-hour pilot** of `draft_v8.md`. The complete research matrix in the draft requires more than this pilot. Original V6/V7 code, checkpoints and experiment outputs remain untouched.

The queue first checks fixed-instance memorization and fresh graph lookup/composition for both the hierarchy and a parameter-matched ordinary stacked Transformer. It then trains fresh alpha-zero, delayed-NCP and ordinary-Transformer foundations for 24,000 updates each. Every completed foundation receives exactly 576,000 GraphNav examples, plus 96,000 lookup, 48,000 two-hop composition and 48,000 pointer examples. No global token stop truncates this exposure.

Replacement-core refitting, exploration, search and continual learning require competent accepted anchors. A failed gate produces a useful negative report and prevents expensive dependent stages. The queue never labels preservation of zero success as successful refitting.

The hierarchy has 2,448,640 learned parameters before exploration; the ordinary Transformer has 2,450,048. Smooth two-site Z injection adds 8,448. Continual learning adds 20,480 rank-4 Q/V parameters per phase on the dense hierarchy, plus separately reported frozen projection/density buffers. Recurrent replacement uses six separate execution-visit caches and has different active/resident counts.

The new continual phases are a withheld encyclopedia/science text domain, a visibly marked four-step graph binding query, and a changed edge serialization with ordered node labels. Foundation training excludes all three. Text comes from local TinyStories and previously deduplicated FineWeb-Edu corpora. V8 partitions are fresh for V8's new models, but historical V6/V7 models may have seen those documents; historical checkpoint replay is a diagnostic, not independent confirmation.

Artifacts:

- `spec/draft_v8.md`: design snapshot.
- `spec/PRE_REGISTRATION.md`, `spec/LOCK.json`: exact implementation scope, budgets and frozen source hashes.
- `data/manifest.json`: partitions, hashes, task definitions and exposure provenance.
- `tests/`: production-shaped contracts, trained V6 replay and throughput/gradient checks. Deliberately unsuccessful preflight iterations are historical debugging evidence, not running experiment failures.
- `runs/status.json`, stage `live.json` and `runs/queue.jsonl`: authoritative progress.
- `REPORT.md` and `PRIMARY_RESULT.json`: evidence and partial/skipped work at completion.

All training is FP32 on Metal. Compilation is selected only where eager-gradient checks pass; recurrent exploration uses eager execution if its parity check fails. Fast decoding is validated against full-prefix decoding on trained graph prompts before use. Reported slot throughput includes padding and is distinct from scored-target throughput.

To inspect without changing a job, read the status, current-stage log/live JSON and matching process. Do not launch a second driver. Checkpoints contain model/version/gate state, optimizer, sampler exposure state and random-number states; stage trainers support `--resume` only with identical source/configuration.
