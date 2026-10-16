# v4: go / no-go after the tiny experiment (2026-10-01, evening)

Sources:
- `v4_tiny/REPORT.md` and every result file under `v4_tiny/runs/`;
- `draft_v4.md` revision 2;
- v3's hourly evaluations, `run_v3_19M/runs/monitor.log`.

No new experiments were run for this note.

## Verdict in three parts (different confidence, kept separate)

| question | verdict | confidence |
|---|---|---|
| **1. Is v4 implemented correctly?** | **Go.** The full-size model instantiates to the draft's ledger exactly (19,898,011 at k = 6; 19,899,043 at k = 8). 15 acceptance tests pass on both attention paths. 13 of 13 deliberate defects are caught. v4's encoder and decoder reduce bit-exactly to v3's. No leak, no dead branch. | High: deterministic tests on the same file that builds the full model. |
| **2. Is the architecture sound?** | **No disqualifying defect found; one load-bearing claim untested.** The concept route is live and used (causality test; core off +0.12–0.25 nats). Every new branch trains. But v4 changes the whole concept path at once (k = 6 with a 2-token query margin, M = 2 instead of 3, terminal-only supervision, a new pooler, a two-level codebook, the F loop), and **no run has shown retrieval through that path.** v3 proves retrieval for *its* path; v4's is unproven until gates G3 / G4 at full scale. | Medium: the tiny model never reached retrieval onset (a budget limit, not a failure). |
| **3. Can it launch right after v3?** | **Conditional.** Go to implementation now, and to a *gated* launch once the checklist below is done. "Good to go" means the gates will tell within ≈ 12 h, not that success is guaranteed. | — |

## What the tiny experiment settled, and what it did not

**Settled (transfers):**
- Correctness and parameter count.
- The concept route carries row-specific information that the decoder cannot get from its 70-token window.
- The pooler weights tokens by content, unlike v3's per-slot pool, which in addition collapses the coarse code.
- The two-level codebook is healthy (fine-level VE on the residual 0.72, against the 0.75 bound).
- Both shared FFNs, the second pass and ĉ_pred are used.

**Measured with a negative or null sign (at d 64, single seed):**
- Decoder Engram reads: a memorisation channel. +0.47 to +0.70 nats on memorised text, −0.14 to −0.17 on unseen text,
  ≈ 0 to −0.02 on ordinary text. The encoder's reads on the same table generalise (+0.14 to +0.17).
- Pooler FFN: it writes, but contributes Δ ≈ 0.
- F loop: +0.022 at k = 4, only +0.002 at k = 6.
- k = 6 vs k = 4: the gap widens to ≈ +0.03 nats by 4–5M tokens, right at the draft's gate, for 1.07× speed.

**Not settled:**
- Retrieval under v4's path (recall, carry, multi-fact).
- Whether the pooler drops stop words once retrieval pays (at tiny scale it selected structure).
- Any full-scale speed number.
- Behaviour on MPS.

## Recommended changes to v4's defaults (owner decisions; the draft is not edited)

The principle: the 48 h main run should carry **as few simultaneous untested changes as possible**. Changes with a null
or negative sign move to arms.

| part | draft default | recommendation | why | effect on the ledger |
|---|---|---|---|---|
| decoder Engram reads | on | **off in the main run; "reads on" becomes a mandatory arm**, with switch-offs on repeated vs novel text | the only addition with a negative sign, and a mechanism that can fire at full scale (v3 sees ≈ 2 epochs of its corpus). **This reverses an item of the owner's brief** ("decoder as v3 plus Engram"); the evidence is the memorisation table in REPORT §2.1; owner's call | −427,522 |
| pooler FFN | on | **off in the main run; arm "FFN on"** with the 0.01-nat kill line at step 4,050 | writes, but Δ ≈ 0 | −294,913 |
| gated conv (owner) | — | **priced arm** beside attention | tied with attention on language; 130k cheaper | arm only |
| k = 4 control | mandatory | **keep mandatory** | the only clean within-v4 comparison; the k = 6 cost reached the gate at tiny scale | — |
| no-F-loop arm | recommended | **keep** | the loop's value depends on k | — |
| pooler telemetry | slot-averaged ratios | **within-slot class ratios + position share of variance** | slot averages confuse recency with content | — |

With both removals, ≈ 722k parameters return to the budget (19.18M), and the draft's §4.1 option B (5 core layers × 2)
fits again if wanted. Recommendation: **keep option A** (4 × 2). Spend nothing new until the gates speak.

## Launch checklist (draft §8.4–§8.5, against what exists)

| # | item | state | where it runs | rough time |
|---|---|---|---|---|
| 1 | model and acceptance harness | **done** (`v4_tiny/code/model_v4.py`, `test_v4.py`, `mutants_v4.py`) | — | — |
| 2 | data rebuild at k = 6 (T 1,020, block 60; documents and probe items padded to multiples of 6) | not started | CPU, can start now | 1–2 h plus checks |
| 3 | v4 trainer: port v3's `train.py` to `model_v4` (time-based cosine, divergence / MPS / memory guards and re-exec, two-level codebook init and restarts, separate loss terms, step memo, dashboard seeded from the logs), plus the eval suite with v4's switch-offs | not started (`run_v4tiny.py` is a CPU harness without guards) | CPU (code) | about half a day |
| 4 | extend the harness: the two "not covered" tests (§3.8 (5) shared gather; §4.9 (3) residual-VQ reproduction) and a full-config smoke | not started | CPU | 1 h |
| 5 | `compile_check` / `mps_check` with the new modules (pooler, F loop, decoder reads if kept); keep the k_norm eager pin from V3-006 | not started | **GPU, after v3's final eval** | 1 h |
| 6 | `profile_step` + `ab_interleave` on the real model; the CPU 1.07× does not transfer | not started | GPU | 1–2 h |
| 7 | memory growth (§8.4.1): v3's footprint still grows ≈ 0.9 GiB/h and re-execs at 24 GiB (tonight, ≈ 22:30–23:00) | open; the re-exec guard carries over as the fallback | GPU, 2 h trace | 2 h |
| 8 | owner decisions §10.6: (1) RL door A + C vs B; (2) core 4 × 2 vs 5 × 2; (3) k = 6 default with k = 8 arm, gates binding; (4) learned boundaries as v5 | open | owner | — |

Order: 2, 3 and 4 now (CPU, while v3 finishes). Then 5, 6 and 7 once the GPU is free after v3's final evaluation. Then
launch with gates G1–G5 and the mandatory k = 4 control.

## Optional pre-launch de-risk (owner decision)

Without it, the first reading of retrieval through v4's path is gate G3, ≈ 5 h after launch.

The one cheap partial test available before the GPU is free is the corrected tiny run:
- fillers drawn fresh from all 457k documents;
- long enough to seek onset: ≈ 20–30M tokens, ≈ 5–6 h on the P-cores.

The cost: v3 runs ≈ 15–30% slower during its last hours, at the lowest learning rate. Those tokens are not recovered
(V3-018 mechanism).

## What in v3's 48 h evaluation would change this call (pre-registered, read the final with this in hand)

| v3 48 h reading | if it holds | if it does not |
|---|---|---|
| second-pass gain (R2 − R1 multi4) ≥ 2 SE on test | M = 2 stands (§1.4) | revisit M before launch (M = 3, or 2 with the pass-1 arm) |
| shared-FFN switch-off still costs multi4 on test | keep the core's full shared FFN | consider option B or a smaller shared FFN |
| ĉ_pred off still ≈ 0.01 nats | RL-door framing stands (A + C) | if larger, the code matters more than assumed: revisit §5.4 |
| V3-016's rules hold on test + OOD | §4 choices stand | revise §4 before launch, not after |
| token count | compare v4 at matched tokens, noting V3-018's ≈ 2.5% shortfall | — |

**v3 48 h readings for the table above (read off `run_v3_19M/runs/v3_48h/eval_final.json`, ckpt_last step 15,588,
test split; facts only, the call is the owner's):**
- Second pass: test multi4 R1 0.764, R2a 0.822, R2b 0.818, R3 0.830 → R2 − R1 ≈ +0.055 to +0.059, R3 − R1 +0.066; the file
  has no per-item hits, so only an unpaired binomial SE is available (≈ 0.025): ≈ 2.2–2.4 SE. OOD multi4 is 0.031–0.041
  at every R (difference +0.010 ± 0.012). Language: R1 costs +0.008 (TinyStories) / +0.006 (WikiText) nats.
- Shared FFN off: test multi4 0.830 → 0.666 (−0.164), carry3 0.967 → 0.926; language +0.009 / +0.004 nats; OOD multi4
  0.041 → 0.031.
- ĉ off (`c_off`): +0.021 (TinyStories) / +0.025 (WikiText) nats — above the 0.01 assumed; test multi4 0.830 → 0.779.
- V3-016's rows: c_off > 0.02 on both sets; R1 − R3 < 0.02 nats on language but multi4 −0.066 and hop 0.170 → 0.162 at
  R1; shared_ffn_off < 0.02 nats on language but −0.164 multi4; engram_read2_off +0.016 / +0.031 (one set each side of
  0.02); table_off +0.009 / +0.008. Language and items point in different directions on M and the shared FFN.
- Token count: V3-018 ≈ 2.5% plus V3-019 ≈ 2.4% of the run lost to the two owner-approved CPU experiments.

**New input (2026-10-02 05:30), not folded into the call below — the owner decides:** the v4-vs-LLM-stack tiny
experiment (`VS_LLM_REPORT.md`) gave the pre-registered verdict **caution**: text at parity (−0.007 ± 0.004 nats),
gridworld-pinpad clearly in the LLM stack's favour on 3 seeds (held-out optimal-action rate −0.133 ± 0.004; the LLM
learns the subgoal structure by ≈ 1.5–3M tokens, v4 not by 6M); v4's second core pass and F loop contributed nothing at
that scale. Also: "token count" above — add V3-019's ≈ 2.4% to V3-018's ≈ 2.5%.

## Bottom line

**Go to implementation now, and launch v4 after v3's final evaluation, gated.** Two changes are recommended first:
decoder Engram reads off and pooler FFN off in the main run, both kept as arms. The gates at steps 1,792 (recall onset)
and 4,050 (carry3, multi4, the k = 6 gate) answer the one thing the tiny experiment could not: whether v4's concept path
retrieves. They do so within ≈ 12 h of launch.
