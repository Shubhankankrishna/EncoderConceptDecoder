# V6 draft implementation and test coverage

This is a tiny-scale development implementation of the revision-2 draft, copied into `spec/v6_draft.md` at launch. Implementation correctness, measured quality and scaling claims have different status.

| Draft requirement | Implementation / evidence |
|---|---|
| 1–2: token/control causality | `checks.py`: future mutations, pooled-boundary timing, document isolation |
| 3–4: zero and transplant anchors | Initial and arbitrary-adapter checks; `audit.py` repeats zero/frozen checks on trained checkpoints; likelihood at all five blend fractions |
| 5–6: gradient reach and initialization | Nonzero Z derivative at zero, finite differences, adapter gradients, frozen weights, safe/capped writes; independent nonzero factors and saved initialization rescaling |
| 7: realizable selection | One winner for a complete schedule; selection counterexample and retained-graph/score-then-replay gradient equality |
| 8: loss accounting | First text targets included, cross-document/padding masks, complete VQ/NCP chunks, per-domain token/row denominators |
| 9: exact generation/replay contract | Cached/full greedy output equality; fixed windows and fixed temperature-one retries; generation/replay share model/control timing; primary repair is greedy |
| 10: rejection gradient | Small-step likelihood decrease fixture; per-token-type gradient attribution after training; task success is checked separately |
| 11–13: verification and accepted states | Final update generated/verified, U=0 checks original, accepted/unused Adam state frozen, strict full DONE chunk, BFS reference |
| 14,16,18: caching and optimization | Fresh dependent caches after each attempt, six distinct core visits, CPU/MPS cache equality, ordinary/compiled losses and gradients |
| 15: finite behavior | Safe epsilon norm, finite zero derivatives, invalid gradient/update rejection, numerical failure records, context exhaustion rejected |
| 17: per-item cost records | `evaluate.py`: attempts, cumulative wall time, outputs, verifier reasons, generation/backward/replay counts; `items.jsonl` and summary curves |
| 19: codebook restarts | Complete-chunk candidates, usage accounting, replacement-row optimizer moment reset and preservation of other rows |
| 20: packing and accumulation | Document masks, exact global denominators, CPU/MPS microbatch fixtures, per-problem row-permutation fixture; distributed execution not exercised |
| 21: recovery | Atomic writes, no incomplete latest checkpoint, RNG/sampler/config state, exact CPU and tolerance-based MPS next-update fixtures |
| 22: precision | FP32 executed; BF16/FP16 comparison deferred and labeled untested |
| 23: memory-efficient candidate replay | Gradients checked against retained execution; candidates vectorized into one retained graph, with measured throughput/memory |
| 24: parameter ledgers | Instantiated active/resident/trainable counts; original teacher concept/A removed from active deployed count |

The software checks are grouped, so the number of executable groups is not the number of draft requirements. `tests/optimized_checks.json` is the authoritative actual outcome; this table is not a blanket claim that every empirical gate passed.

Scaling formulas, wider/context-longer runs, elastic recurrence, independent foundation seeds, low precision and distributed execution remain design requirements for later experiments. Hardware-fit evidence here applies only to d=128,T=192, FP32 and the measured local batch sizes. Shared-batch cost comparisons do not establish individual-request tail latency.

The original generator was adapted into independent `code/graphnav.py`; V6's tokenizer/text reads use the existing immutable V5 data. New masks, uniform-document packing, graph split exclusions and restart moment resets are explicit recipe changes. A standard transformer baseline and the longer bridge-control ladder are deferred under the requested time budget.
