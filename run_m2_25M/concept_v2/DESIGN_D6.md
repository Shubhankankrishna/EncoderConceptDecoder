# D6 — combining the best of D2, D3 and D4 (user request, 2026-09-28)

## Step 1 — first principles: what each design contributes, and what it lacks
Today's failure: every pass does the same thing (R1..R8 identical, step angle 6.7° → 0.1°). Three independent causes are possible,
and each of D2/D3/D4 attacks a different one:
| Cause of "passes don't matter" | Who fixes it | How |
|---|---|---|
| the LOSS does not reward later passes (any R gives the same answer and is equally rewarded) | D2 (LoopFormer) | main loss on the long route; shorter routes distilled toward sg(long endpoint) → passes must move toward a better plan ("drift") |
| nothing FORCES a pass to improve on its input (a fixed point reachable from anywhere satisfies the loss) | D3 (deep-sup groups) | detached windows: each window starts from the previous window's state and must improve it; every pass graded on the true next-chunk code (grounded) |
| the COMPUTATION is identical every pass (one dense map applied again) | D4 (Looped-MoE) | state-conditioned expert routing: pass 2 picks different experts than pass 1 (Sparse-Layers: 25–53 % fully disjoint), no pass index needed |
The three are complementary (loss, optimisation signal, capacity), so a combination is not redundant — provided the pieces do not
conflict. Two potential conflicts to resolve:
(a) D2's (t, Δ) pass conditioning vs D3's "no pass index for extrapolation". Resolution proposed: drop (t, Δ); MoE routing is the
    per-pass variation source (state-conditioned, defined for any R). Keep LoopFormer's LOSS asymmetry, which is the part the audit
    identified as the fix (audit item 3).
(b) D2's second (shortcut) trajectory vs D3's single long trajectory. Resolution proposed: the long trajectory's own intermediate
    passes ARE the short routes — each pass's readout is distilled toward sg(final plan) (LoopFormer's consistency, ELT's
    distillation) at no extra trajectory cost.

## Step 2 — D6 draft v0 "LOOPFORMER-MoE-DS" (before advisor)
- Chassis: S-space (X̃ re-read every pass through the GR gated read; carry S), S_0 = W_s c_hat, 4 distinct d-256 layers, full causal
  attention 4 × 64 GQA kv 2, sigmoid output gate, no attention shortconvs, zero-init block outputs, LN_h carry + separately normed
  readout (from D3). No pass-index conditioning.
- FFN: looped MoE — PER-LAYER expert sets (what Sparse-Layers actually tested), E = 4 SwiGLU experts × 512 per layer, top-2 (active
  width 1024 = spec), LB + z-loss. (Options: E = 8 per layer = 14.4M params; shared pool E = 8 = 4.9M params but untested.)
- Schedule (D3): K ~ U{8..12} passes, windows of 4, remainder merged into the last window, carry detached between windows, all windows
  graded, per-window backward.
- Loss: L_NTP(final pass) + λ_NCP · mean_r CE_NCP(u_r) [grounded per-pass] + 0.1 · mean_{r<K} ‖sg(u_K) − u_r‖² [LoopFormer/ELT
  consistency toward the final plan] + L_VQ + LB/z.
- Eval R = 6 (default); R-sweep to 16.
- Budget (core_budget.py): per-layer E 4 × 512: params 8.06M (+6M vs 2.06M today), inference R 6 = 7.19M MACs/token = 21.2 %, training ×1.40.

## Step 3 — D6 v1 (after advisor D6 #1 = row #35)
WHERE LOOPFORMER LIVES IN D6 (stated for the user): D6 keeps LoopFormer's TRAINING RECIPE — the asymmetric loss and "distil the short
route toward the long endpoint" — and REPLACES LoopFormer's (t, Δ) conditioning by D4's state-conditioned experts, because the
user's Sparse-Layers paper is the stronger evidence for per-pass variation at our scale and routing is defined for any R (no
out-of-range pass inputs). The (t, Δ) AdaLN is NOT deleted: it is ablation A1 (with vs without) on the eval sheet.
CONSISTENCY ONLY IN THE LAST WINDOW: ‖sg(u_K) − u_r‖² is applied only to passes r inside the final graded window (the window that
contains K). Earlier windows are graded ONLY on the true next-chunk code (D3) — distilling them toward u_K would give them two targets,
one of which is wrong wherever the final plan is wrong. Inside the last window it means what LoopFormer meant: the late passes are
short routes to the endpoint.
FFN: ONE SHARED EXPERT POOL, E = 8 SwiGLU experts × 512, top-2 (active 1024), per-layer routers, LB + z-loss — the economical layout
(4.92M params). Trade-off stated: per-layer E = 8 is the layout Sparse-Layers tested (14.4M params); per-layer E = 4 (8.1M) has only 6 expert
pairs to diverge into and is rejected. Check: E15 (expert overlap across the 4 layers within a pass) + E11 (routing divergence across
passes, per layer). If E15 shows the pool collapsing, fall back to per-layer E = 8.
EVAL R = 6: it lies inside window 2 for every K ∈ {8..12} (as D3), so the default inference depth is a supervised pass.
BUDGET: params 4.92M; inference R 6: 7.20M MACs/token = 21.2 %; training ×1.40 (single trajectory; consistency + LB statistics negligible).
PRE-COMMITTED SCORES (before review): C1 inherits D3 (puzzle-regime discount, 8 at most); C2 may exceed D3 only if the last-window
consistency adds something E4/E5 will show; C3 gets D4's conditional credit (+1 only with E11 + E15); C7 21.2 %.

## Step 4 — D6 v2 (after advisor D6 #2 = row #36)
HONEST DESCRIPTION: D6 = D3's training (detached windows, every pass graded on the true next-chunk code) + LoopFormer's consistency
applied to the FINAL window (the last 4–7 passes are short routes distilled toward the endpoint; earlier windows get no consistency
signal by design) + D4's state-conditioned experts throughout.
WHY IT SHOULD BEAT D2, D3 AND D4 (first principles):
- vs D3: in D3 a window can recompute from the clean input X̃ and ignore the carry (risk E13). In D6 the experts are routed on the
  STATE, so different S fire different experts — the carry gets a job that X̃ alone cannot do.
- vs D4: D4 keeps D1's loss placement (sampled-R loss can be satisfied by invariance). D6 replaces it with D3's detached windows.
- vs D2: D2's gains are elasticity only (never tested beyond L) and depend on (t, Δ) inputs; D6 keeps LoopFormer's consistency where it
  is well-defined and gets per-pass variation from routing that is defined at any R, so its claim covers more passes than trained.
MoE × PER-WINDOW BACKWARD: load-balance and z-losses are accumulated PER WINDOW and backpropagated with that window's NCP loss, so the
gradient reaches the routers that made those decisions (window 1's routing is out of the graph when window 2 backprops).

## Step 5 — D6 budget + eval sheet
BUDGET (core_budget.py, moe = shared 8 × 512 top-2, K ~ U{8..12}, per-pass head): params 4.92M (compute params ≈ 27.9M, +11 %);
inference R 6: 7.20M MACs/token = 21.2 % of the model (cap 25 %); training ×1.40 vs today (≤ 1.5×). Activation memory ≤ one window
(≤ 7 passes × 4 layers) thanks to per-window backward.
EVAL SHEET: E1–E10 (common) + from D3: E11-D3 per-window NCP improvement (≤ 0 on average), E12 settle-vs-wander at R 16, E13 GR-read gate
mass S-branch vs X̃-branch (carry used?) + from D4: E11-D4 routing divergence across PASSES per layer, E15 expert overlap across the 4
layers within a pass, expert-load entropy + NEW: E16 routing divergence across WINDOWS (does window 2 fire different experts than
window 1 for the same chunk? — the direct test of "the carry gets a job") + ABLATIONS: A1 with vs without LoopFormer (t, Δ) AdaLN;
A2 consistency off (does the last-window term add anything over D3?); A3 per-layer E = 8 instead of the shared pool.
EVAL SHEET ADDITIONS (after advisor D6 #3 = row #37):
- A1 wiring: the "with (t, Δ)" arm uses D2's wiring — AdaLN α REPLACES the GR write scalar, α init 0.05, W_o/W_2 default init — otherwise
  it would be a dead variant (zero-init output × near-zero gate) and "conditioning does not help" would be guaranteed.
- E17 LoopFormer signature: cross-pass CKA inside the final (consistency) window vs across earlier windows — low CKA = drift, the
  check that the LoopFormer piece did something.
- A4 dense FFN 1024 (D6 without experts): isolates the looped-MoE contribution the user considers most important; if A4 ≈ D6 on E1/E2,
  experts did not matter at this scale.
- Note for E12: eval R 6 lies in the consistency window only for K ∈ {8, 9}; for K ≥ 10 the last window starts after pass 6.
