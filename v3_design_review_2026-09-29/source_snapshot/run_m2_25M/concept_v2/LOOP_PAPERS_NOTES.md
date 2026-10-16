# Loop Papers — per-paper notes (concept module v2 design, 2026-09-28)

Source: `~/Desktop/EncoderConceptDecoder/Loop Papers/` (42 PDFs), text extracted with PyMuPDF to the session scratchpad.
Read in full except bibliographies. One fixed block per paper:
(a) core mechanism · (b) how extra passes count (injection, carried state, per-pass loss / deep supervision, truncated BPTT, R sampling)
· (c) length / step extrapolation numbers · (d) failure modes the authors admit · (e) implication for a 4-layer d=256 looped concept core
· (f) key equation.

Target the notes serve: concept core v2 = 4 distinct layers, d=256, FFN 1024 (SwiGLU or GELU), full causal attention over
chunk-rate concepts (T/4 = 256 positions), recurrence kept. Current core (baseline, row 0): one shared LoopLayer, S-space
gated residual read/write over [X=c_hat, S], S0 = 0, R ~ U{1..4} train, R = 4 eval, out zcRMSNorm, PQ 48x128x8, NCP head.
Measured failure of the baseline: R1..R8 equal to 0.001 nats, step angle 6.7° -> 0.1°, hop bridge probe flat across passes,
OOD more-steps tasks all 0, hop flat at ~0.10 from hour 13.5, u carries ~1 answer token (probe 0.59 -> 0.21 -> 0.06).

## 1. LoopFormer (Jeddi et al., ICLR 2026, arXiv 2602.11451) — read in full (18 pp)
(a) Cycle loop of k distinct blocks Φk (k ∈ {1,2,3}) applied M ≤ L times; every pass is conditioned on (t_{i-1}, Δ_i):
normalized time t ∈ [0,1] and step size Δ (ΣΔ = 1). t and Δ → sinusoidal features (256) → 2-layer SiLU MLP → e_t + e_Δ = c.
Per block an AdaLN-Zero modulator (SiLU → Linear d→4d, zero-init weights+bias) gives (α_msa, α_mlp, γ_msa, γ_mlp):
x ← x + α ⊙ Attn(RMSNorm(x) ⊙ (1+γ)); same for FFN. RMSNorm without affine. h0 = embeddings (no separate injection).
(b) Why passes count: training runs the full L-step trajectory AND one shortcut trajectory, S ~ U{1..L-1}, with a random
schedule ΔS summing to 1. L = L_L + 0.1·L_S + 0.1·L_cons, L_cons = ||sg(h_L) − h_S||² (Alg. 1; text says "logits" — the
paper is inconsistent). Full BPTT, no truncation. Short routes are distilled toward the long endpoint (self-distillation inside
the loop). The "drift": CKA across loop steps stays low → representations keep evolving (progressive drift), whereas early-exit
baselines (Base-Loop-EE, TMLT-EE) have high cross-step CKA = stagnation, and flat curvature/anisotropy/entropy.
(c) Not length/step extrapolation. Budget elasticity only (M ≤ L). 1B, d=2048, 25B Pile tokens, 3⊗8: Pile ppl 10.28 at 24×
vs 10.91 plain loop, 10.38 TMLT; at 6× (M=2) 14.3 vs 15.07/15.79 for the other elastic baselines. Loop 3⊗8 > 3-layer vanilla
(10.28 vs 12.93). More loops beyond L not tested. Schedule matters at fixed budget (spread ~1.4 ppl; best = big steps early,
fine steps late).
(d) Admitted: sequence-level budget only; ~1.5× training FLOPs (1.3× wall-clock) from the second trajectory; representation
analysis is correlational; instabilities at depth need 4k warm-up.
(e) For us: the stagnation they fix is exactly our measured failure (step angle → 0.1°, R1..R8 equal). Two portable pieces:
(1) pass conditioning — a (t, Δ) or pass-index embedding driving AdaLN-Zero gates/scales on each of the 4 layers, so the same
weights can do different things at different passes (cheap: one d→4d linear per layer); (2) a shortcut-consistency term that
ties the R-short plan to sg(R_max plan) while the NCP/NTP loss sits on R_max — this trains "more passes = refine toward a
target" instead of our current "any R gives the same answer". Caveat: it targets elasticity, not OOD step extrapolation;
consistency pulls short → long, it does not by itself make long better than long−1.
KEY DIFFERENCE vs our baseline (loss placement): we put the full NCP/NTP loss on whichever R was sampled, equally weighted —
that rewards R-invariance. LoopFormer puts the main loss on R_max only; the short route is 0.1-weighted and distilled toward
sg(h_L). This asymmetry is the candidate MINIMAL FIX of the current S-space loop (Design 1 seed).
C6 note: AdaLN-Zero init is identity-at-init (α = 0 ⇒ branch weights get no gradient until α moves) — fine at 1B/25B tok,
a warm-up risk at 25M / 24 h on MPS; init α small non-zero or zero-init only the γ part.
(f) x ← x + α(c) ⊙ F(RMSNorm(x)⊙(1+γ(c))), c = φ(t)+φ(Δ); L = L_L + 0.1 L_S + 0.1 ||sg(h_L) − h_S||².


## 2. Loop, Think & Generalize (Kohli, Parthasarathy, Sun, Yao; OSU; arXiv 2604.07822) — read in full (19 pp)
(a) Plain looped GPT-2: a block of L = 4 distinct layers (d 768, 12 heads) repeated R times, h(r+1) = fθ(h(r)); full state
replaced each pass; NO input injection, no gating, no halting in training, no middle loop. Loss only on the final position
(tail entity) after the last pass. Zero-init of both c_proj (attention + FFN output) so every block is exactly identity at
init — they call this essential (App. E: default Gaussian init → mixed, seed-dependent, no consistent scaling, overthinking).
NoPE for the depth runs. AdamW 1e-4, wd 0.01, 2k warm-up.
(b) Why passes count: (1) training recurrence must be LARGE — "this scaling effect only emerges for R > 4"; the max train R
sets the extrapolation range (fixed R=6 → 14-hop, R=8 → 19-hop, trained on ≤ 12-hop). (2) Dynamic R per batch,
R ~ clip(Poisson(λ), Rmin, Rmax) (λ = 4/6/8, Rmin = 2, Rmax = 8/12/16) gives the largest learnable depth (22-hop vs 16 for
fixed R=8) and far slower overthinking decay. (3) Easy-to-hard curriculum over hop depth, 95 % gate per stage, joint replay.
(4) Very long training: 1.3 M steps to grok 4-hop, then 19-hop comes fast (phase transition; "the difficulty is discovering
the rule"). Systematic generalization only after ~10^4 epochs (three-stage grokking: memorise → ID → OOD).
(c) Numbers: fixed R=1..8 learn 2/6/9/10/12/13/16/16-hop ID; dynamic 22-hop ID; extrapolation only when inference R > train R;
dynamic generalises ~1.5–1.8× beyond its ID depth. 2-hop systematic generalisation: R=1 (vanilla) 0 %, R≥2 non-trivial;
iso-FLOP 8-layer vanilla never generalises OOD. Adaptive halting: stop when KL(p_t||p_{t−1}) < 0.01 AND H(p_t) < 3.0
(KL alone stops too early).
(d) Admitted: overthinking — logit margin rises then falls with extra passes, peak margin lower for harder tasks; input
injection (Bansal/Geiping) does NOT fix it here; results seed-sensitive without zero-init; no clear trend from larger L or
larger max-R on the generalisation ratio; toy KG with dedicated tokens; shortcut illusion (App. H): without a permutation KG
the model "solved" 80-hop by mapping a relation suffix to the answer — activation patching exposed it.
(e) For us — the most diagnostic paper so far. Differences from our baseline that plausibly explain our flat R-curve:
  • our max train R is 4 (U{1..4}, mean 2.5); they find inference scaling appears only for train R > 4 and max-R sets range;
  • our state is S-space with gated writes (S0 = 0, X re-read each pass); theirs replaces the full state; they found input
    injection irrelevant for overthinking — so injection is not the cure, the R distribution and init are;
  • INIT AUDIT (verified in code): our LoopLayer is NOT identity-at-init — GRWrite s = 2σ(0) = 1, and GatedAttention.o /
    SwiGLU.w2 use PyTorch default init, so pass 1 is a full-strength random residual update (S0 = 0). Loop-Think zero-inits
    exactly these (c_proj of attention + FFN) and App. E shows default init → seed-dependent, no inference-time scaling,
    more overthinking. A concrete baseline defect, inherited fix for every design;
  • our loss is on NCP + NTP through the decoder at every token position, they supervise one final answer token only;
  • we train 24 h on mixed LM data, they train 10^5–10^6 steps on one task: grokking-style depth rules are unlikely to
    appear in our budget unless the synthetic mix is large and curriculum-ordered.
  TRANSFER CAVEAT: their hops are over PARAMETRIC facts (entity→entity stored in weights, dedicated tokens, no context);
  our hop/bind/pointer tasks are IN-CONTEXT pointer chasing — composition must go through attention across chunk positions
  each pass, not FFN lookup. The R-distribution and zero-init findings transfer; 'learnable depth grows with train R'
  numbers do not transfer 1:1 (discount in C1 for any design leaning on this paper).
  Design implication: Poisson R with Rmin 2, mean ~4, Rmax 8 on the 4-layer block (cheap at ¼ rate), zero-init outputs,
  eval R-sweep beyond train max, KL+entropy halting for the decoder-facing plan. Also: audit our OOD tasks for suffix
  shortcuts (their App. H warning).
(f) h(r+1) = fθ(h(r)); R ~ clip(Poisson(λ), Rmin, Rmax); halt when KL(p_t||p_{t−1}) < ε and H(p_t) < H_th.

## 3. Gated Recurrent Transformers (GRT; Hegazy, Alanwar, Elhoushi; arXiv 2608.15062 v4) — read in full (27 pp)
(a) Prelude (fixed blocks, once) → shared core of n_rec distinct blocks iterated R times → coda → LM head. Each recurrence:
h̃ = W_proj[h(r−1) + ε_x, h_pre] (concat 2d → d; ε_x ~ N(0, 0.1²)), o = B_shared(h̃) (all n_rec blocks),
g = σ(f_g([LN(h(r−1)), LN(h_pre)])/τ + ε_g) elementwise (f_g = 2-layer SiLU MLP, hidden d, last bias +4 ⇒ g ≈ 0.98 at init,
ε_g ~ N(0, 0.1²), τ = 1), h(r) = g ⊙ h(r−1) + (1−g) ⊙ o. A GRU over depth, per element. h(0) = h_pre.
(b) Why passes count: the gate + anchor re-injection let one weight set act as several "layers": gate write-heavy at steps 1–2
(mean ≈ 0.82), copy-heavy later (0.87; applied/proposal ratio 0.18 → 0.07). W_proj learns a CONTRASTIVE projection (state and
anchor halves have row cosine −0.19) — it feeds the block "what changed since the prelude". Depth r ~ U{1..R} each step, loss
only at the sampled depth, no per-iteration loss, full BPTT. Extra per-step cost ≈ 10d² FLOPs (W_proj + f_g).
(c) LM only (GPT-2 family, 9.8B tokens). isoFLOP: 1+1×10+1 (35M) 3.14 vs dense 12L 124M 3.15; medium 2+5×4+2 2.89 vs dense
2.84 (Ouro 2.93, RRT 2.95, Poisson 2.97, MoR 3.02); large 2.77 vs 2.71. isoParams: 2.76 vs 2.84 (medium). Ablation (small,
20k steps, Δ nats): recurrence alone +0.107 vs dense; +prelude/coda −0.035; +state noise −0.018; +prelude re-injection
−0.022; +elementwise gate −0.048 (at 2k steps re-injection −0.198, gate −0.115). Gate bias +4/+2/0/−2 → 3.141/3.152/3.160/
3.151 (shallow). σ_x 0 / 0.1 / 0.2 → +0.018 / 0 / +0.019. Anchor ablation: fixed h_pre 2.68, drifting h(r−1) 3.38, input
embedding 3.73, zeros 8.08. NO step extrapolation: trained R = 6, inference R = 8/10/12 → 2.69/2.72/2.74 (slightly worse).
77 % of loss reduction in steps 1–2; KL to final 0.014 by step 5. Harder tokens gain 10× more over the steps (R² 0.998).
(d) Admitted: R fixed at inference (no per-token halting); gate bias / noise may need retuning beyond GPT-2; representations
across steps are MORE similar than across dense layers (CKA ≥ 0.907 for steps 2–6); head roles are fixed by weight — the same
program runs each step, "the gate, not the heads, decides how much of each step's output to incorporate"; steps do not
replicate deep-layer hierarchy (all steps ≥ 2 align with dense layer ≈ 11 of 36).
(e) For us: our LoopLayer is already a cousin (re-reads anchor X = c_hat and state S, gated write) — GRT's anchor ablation
(fixed anchor worth 0.70 nats over a drifting one) supports keeping X = c_hat re-injected every pass. Portable upgrades:
(1) a per-element GRU-style gate over the WHOLE 4-layer block output, conditioned on [LN(S), LN(X)], bias +4 (≈ identity at
init — also fixes baseline defect #1); (2) a contrastive input projection W_proj[S + ε, X]; (3) small state noise σ 0.1 and
gate noise 0.1 (anti-collapse, cheap). Cost at d = 256: W_proj 2d² + f_g (2d·d + d·d) ≈ 5d² ≈ 0.33 M params, ~0.33 M MACs per
concept per pass (÷4 per token). Caveat: GRT's own evidence is that uniform-r + loss-at-sampled-depth yields early-exit
robustness and convergence (fixed point by step 4), NOT extrapolation — the same R-invariance we already measured. The gate
alone won't make R = 8 beat R = 4; it needs a training signal that rewards later passes (LoopFormer / Loop-Think).
(f) h̃ = W_proj[h + ε_x, h_pre]; o = B(h̃); g = σ(f_g([LN h, LN h_pre])/τ + ε_g); h ← g⊙h + (1−g)⊙o.

## 4. Recurrent Looped Transformer (RLT; Yifan Zhang, 2026-09-12) — read in full (16 pp)
(a) Recurrence over the SEQUENCE (temporal), not over depth. A causal encoder builds KV memory M≤t; a decoder of L_D blocks
runs once per token and carries its final output s_{t−1} (plus layerwise SWA caches) to the next token:
u_t = e_t + α·g_t ⊙ W_s·RMSNorm(s_{t−1}), g_t = σ(W_g[e_t; r_{t−1}] + b_g); decoder block = causal SWA over decoder activations
→ cross-attention to encoder memory → FFN. Learned initial state s⋆. Optional tying of encoder/decoder weights (two logical
passes through one backbone). The state path to token t traverses t·L_D blocks ("unbounded temporal depth").
(b) Passes don't exist in the depth sense; depth grows with position. Full BPTT across the whole sequence (checkpointed);
truncation must name every detached tensor. Most of the paper is RL replay semantics (prompt/response use one transition).
(c) NONE — the report states it "does not report measured efficiency or scaling results". Purely a specification.
(d) Admitted: sequential decoder recurrence over T tokens — no parallel prefill, lower hardware utilisation; gates and
contraction may suppress long paths ("structural depth alone is not a reasoning guarantee").
(e) For us: orthogonal to depth recurrence. The only transferable idea is a TEMPORAL carry inside the concept core (chunk j's
first pass reads chunk j−1's final state), which would give composition depth that grows with chunk index — but it makes the
256 chunk positions strictly sequential in training (256 × 4 layers serial per sequence): far too slow on MPS for our
throughput-bound 24 h budget, and it breaks the parallel causal attention the concept core relies on. No evidence either.
Low weight in scoring; record as a rejected axis. Note the gated-merge form u = e + α g ⊙ W_s RMSNorm(s) (a GR-style read).
(f) u_t = e_t + α σ(W_g[e_t; RMSNorm(s_{t−1})] + b) ⊙ W_s RMSNorm(s_{t−1}); H_t = D(u_t; M≤t, C_{t−1}).

## 5. T²MLR — Transformer with Temporal Middle-Layer Recurrence (Cai, Zhu, Dong, He, Arora; Princeton; arXiv 2607.15178 v2) — read in full (34 pp)
(a) Temporal recurrence of a MIDDLE block: the representation after layer ℓ_end at token t−1 is fused into the input of layer
ℓ_start at token t. Cache R_t = RMSNorm(h_t^(ℓend) + R_{t−1}) (temporal residual). Fusion
Φ(h, R) = h + tanh(γ_cur)·σ(f_cur[h;R]) ⊙ h + tanh(γ_rec)·σ(f_rec[h;R]) ⊙ W_rec R, γ's zero-init ⇒ exactly a standard
Transformer at init. Learned: γ_rec → positive, γ_cur → negative (recurrence acts as an additive refinement).
(b) PARALLEL training via Jacobi fixed-point iteration (not a strict serial carry): seed R⟨1⟩ = ShiftRight(h^(ℓend)) from a
normal pass, then d_forward times: run the middle block on Φ(h^(ℓstart−1), R⟨k−1⟩), R⟨k⟩ = Norm(ShiftRight(h⟨k−1⟩^(ℓend)) +
R⟨k−1⟩); d_forward = 16, d_backward = 4 (truncated BPTT — only the last 4 iterations carry gradient). Column t is exact after t
iterations; gradients match the (32,32) anchor with cosine ≥ 0.96 once d_forward ≥ 8 and d_backward ≥ 4; Jacobi-trained ≈
exact-trained (within 0.001–0.0045 loss, 135M–1B). Inference: exact sequential carry, ≤ 8 % per-token overhead.
(c) S5-Retrieval (state tracking of S5 permutations + in-context dictionary retrieval; train N ≤ 32, test ≤ 48): 4-layer
T²MLR near-perfect ID exact match and non-trivial OOD token accuracy; parameter-matched 4-layer Transformer and LSTM fail.
Pretraining 135M, 10B tokens: val CE ≈ 3.07–3.09 vs 3.11 baseline; best zero-shot avg with a 6-layer middle block (13–18)
44.14 vs 42.83; full recurrence (1–30) and 2-layer recurrence are weakest; middle > early/late at fixed width. Beats 2× full
loop (42.99), 3× middle loop (42.68) and 2× pause tokens (43.31) at matched params/data. Fine-tune: Variable assignment (depth
5), ProsQA-Hard, HotpotQA-Easy, GSM-Aug all improve; 1.7B retrofit GSM8K 35.8 → 39.9, MATH500 12.8 → 18.0.
(d) Admitted: 2–4× training wall-clock; a training-compute-matched Transformer (2.24 epochs) beats it on zero-shot NLP;
no multi-seed variance; the gain is inference-side and on reasoning tasks.
(e) For us: the temporal axis rejected under RLT comes back in PARALLEL form — and it maps exactly onto a depth loop over a
causal core: Jacobi iteration k = one pass of the block over all chunk positions, reading the previous pass's state shifted
by one position. Our LoopLayer already has this structure implicitly — at pass r, causal attention over [X, S_{r−1}] lets
chunk j read the previous pass's state at every earlier chunk. So T²MLR is paper evidence that "iterate a block while
reading earlier positions' previous-iteration state" solves STATE TRACKING that a same-size Transformer cannot (S5, our
state_machine / stack OOD families), and that exactness grows with the number of iterations (column t exact after t passes).
WHAT T²MLR HAS THAT OUR BASELINE LACKS (so 'we already have it' does not mean 'it fails'): (i) an ACCUMULATING prefix cache
R_t = Norm(h_t + R_{t−1}) — our S_j is per-position, any prefix-sum must be learned by attention; (ii) an UNBOUNDED horizon at
inference via the exact sequential carry — a depth loop's serial horizon is ≤ R passes unless attention learns a parallel scan
(~log2 N layers, Merrill/Li); (iii) 150k steps on S5 alone. Alg. 1 line 6 ('if k > D_fwd − D_bwd then StopGrad') looks
inverted relative to the text (the LAST d_backward iterations carry gradient) — cite the text's intent; paper inconsistency.
Portable pieces: (1) an explicit shifted carry (chunk j−1's last-pass state) fused before layer 1 of each pass with
zero-init tanh-scalar gates — a cheap, exact-at-inference path that makes each extra pass extend the serial horizon by one
chunk; (2) truncated BPTT through only the last few passes (d_backward = 4) to bound the cost of large R; (3) put the recurrence
on a middle block (in our pipeline the concept core IS the middle block — consistent). Caveat: 2–4× training cost; their S5
result used long training (150k steps).
(f) Φ(h,R) = h + tanh(γ_c)σ(f_c[h;R])⊙h + tanh(γ_r)σ(f_r[h;R])⊙W_r R; R_t = RMSNorm(h_t^(ℓend) + R_{t−1});
Jacobi: R⟨k⟩ = Norm(Shift(F_mid(Φ(h, R⟨k−1⟩))) + R⟨k−1⟩).

## 6. TRM — Less is More: Recursive Reasoning with Tiny Networks (Jolicoeur-Martineau, Samsung SAIL; arXiv 2510.04871) — read in full (12 pp)
(a) One tiny 2-layer network, two carried states: y = current embedded ANSWER (decodable), z = latent reasoning (not
decodable). One "full recursion": n times z ← net(x + y + z); then y ← net(y + z) (no x — the presence of x tells the single
net which job it is doing). deep_recursion = T−1 full recursions under no_grad, then 1 full recursion WITH gradient (full
backprop through its n+1 evaluations — no IFT 1-step approximation). n = 6, T = 3 → 42 effective layers per supervision step.
(b) Why passes count — DEEP SUPERVISION: up to N_sup = 16 supervision steps per example; (y, z) carried across steps DETACHED;
each step has its own CE loss on y and its own optimizer step; ACT halting head trained by BCE on "answer correct?" (no
Q-learning continue loss → no second forward pass). The model learns "take any (y, z) and improve it" — an improvement operator,
not a fixed-depth function. At test time all 16 steps run.
(c) Sudoku-Extreme (1k train): 87.4 % (HRM 55.0). Ablations: 1-step gradient 56.5; separate fH/fL 82.4; no EMA 79.9; 4 layers
n=3 79.5; self-attention instead of MLP-mixer 74.7 (on 9×9); T=2,n=2 73.7; single z 71.9; multi-scale z (n+1 features) 77.6.
Maze-Hard 85.3 % (att), ARC-AGI-1 44.6 %, ARC-AGI-2 7.8 % at 7M params. Cited ARC-Prize analysis of HRM: deep supervision
doubled accuracy (19 → 39 %), recursion inside a step only 35.7 → 39.0 %. Depth sweep: more recursions help up to 42–72
effective layers then flatten/OOM.
(d) Admitted: supervised, deterministic (not generative); why recursion beats a deeper net is unexplained (suspect overfitting
avoidance); MLP-mixer only works for small fixed contexts; more layers / MoE hurt on 1k-example data; backprop through fewer
than all n+1 recursions (k = 4 of 7) did not help; fixed-point (DEQ) iteration was slower and worse ("converging to a fixed point
is not essential"); weight-tying embedding/head hurt.
(e) For us: (1) TWO STATES beat one: our core carries a single S and reads u = norm(S) as the plan — TRM's single-z ablation
(71.9 vs 87.4) says split into a latent scratch z (reads X, y, z) and a decodable plan y (reads y, z only; the NCP head reads y).
(2) Deep supervision as the 3rd C2 mechanism: per-step loss with a detached carry trains an improvement operator; test-time
extra steps keep helping up to the trained N_sup. In LM pretraining we cannot take 16 optimizer steps per batch — the portable
form is a per-pass NCP loss (cheap: the NCP distance head runs at chunk rate, ~0.1 M MACs/token) with the state detached between
passes groups, NTP only on the final pass (the decoder is too expensive to run per pass). (3) Full BPTT through the last
recursion group beats 1-step gradients by 31 points — do not truncate to 1 step; T−1 no-grad groups + 1 grad group is a cheap
way to run large R (cost of grad ≈ one group). (4) EMA of weights helped a lot on small data (0.999). Caveat: 1k-example
puzzle regime, 60k–100k epochs, heavy augmentation — the "fewer layers is better" result is an overfitting effect that does not
transfer to our data-rich LM regime.
(f) z ← f(x + y + z) ×n; y ← f(y + z); deep_recursion = [no_grad × (T−1)] + [grad × 1]; per-step CE(y) + BCE(q, correct), carry detached.

## 7. PTRM — Probabilistic Tiny Recursive Model (Sghaier, Parviz, Jolicoeur-Martineau; Mila; arXiv 2605.19943) — read in full (14 pp)
(a) Inference-only change to a trained TRM: K parallel rollouts, Gaussian noise σ added to the latent z at EVERY deep-recursion
(supervision) step, pick the rollout with the highest Q-head logit (the ACT halting head trained as a correctness classifier).
No retraining, no task-specific augmentation.
(b) Diagnosis: deterministic recursion gets trapped in "bad basins" (oscillates in a bounded latent region with negative Q);
"delayed successes" escape late; noise gives each rollout a probability of escape (K=100: 8 of 100 escape on a failed puzzle).
Q tracks cell accuracy at every step (≈ +6 correct vs −6 incorrect at convergence).
(c) DEPTH beyond the trained N_sup helps a little under deep supervision: TRM K=1, D 16 → 48: PPBench val 76.4 → 78.3/79.5 %,
golden 62.6 → 66.0 %. Width: best-Q@100 PPBench golden 62.6 → 91.2 %; Sudoku-Extreme 87.3 → 98.75 % (K=100, D=64, σ=0.3);
Maze-Hard 83.8 → 86.7 % (pass@K 95.6 %; Q head a weak verifier there); ARC-AGI-2 pass@1 7.36 → 8.47 %. mode@K voting gains only
+1.3 pp — selection by the learned verifier is what matters. Langevin steering with ∇Q adds nothing over the noise alone.
Noise only at the initial z (other work) gave no improvement — noise must be injected at every step.
(d) Admitted: puzzles only, small grids; gains depend on verifier quality (weak on Maze/ARC-2).
(e) For us: (1) C1 datapoint — under DEEP SUPERVISION, running more recursion steps than trained IMPROVES accuracy (+3 pp at 3×
depth), unlike GRT (uniform-r, loss at sampled depth) where extra steps slightly degrade. That is evidence that the training
signal, not the architecture, decides extrapolation. (2) Deterministic loops fall into attractors; per-pass noise during
TRAINING (GRT σ 0.1) and at test time is a cheap anti-attractor measure. (3) A learned halting/correctness head over the plan
is a verifier: for our chunk-rate core, a tiny Q head on u (predicting "NCP top-1 correct?") could drive KL+entropy-style halting
and best-of-K plan selection at eval — an optional test-time extra, not a training requirement.
(f) z_t ← z_{t−1} + ε, ε ~ N(0, σ²I); (z_t, y_t) = rec(x, z_{t−1}, y_{t−1}); answer = argmax_k Q(y_D^(k)).

## 8. HRM-Text — Efficient Pretraining Beyond Scaling (Wang, Liu, … Abbasi Yadkori; Sapient + MIT; 2026) — read in full (26 pp)
(a) Hierarchical Recurrent Model for language: two DISTINCT modules (H slow, L fast; 16 layers each, d 1536 at 1B). z_H0 from
input embeddings, z_L0 fixed. Schedule H2L3 = 2 cycles × (3 L updates + 1 H update) = 8 module steps; logits from the final H.
MagicNorm: each module = L PreNorm blocks + a final Norm at the module EXIT on the recurrent path, z_n = Norm(z_{n−1} + Σ blocks)
(PostNorm-like bounded forward variance over N steps, PreNorm-like gradients inside the truncated horizon). Parameterless
RMSNorm, SwiGLU, RoPE, sigmoid-gated attention (like ours). Adam-atan2, constant LR 2.2e-4, EMA 0.9999 instead of decay.
(b) Why passes count: two roles by distinct weights + WARM-UP DEEP CREDIT ASSIGNMENT — truncated BPTT horizon K = 2 → 5 (of 8
steps) linearly during training. NO deep supervision, no ACT: a single loss on the final output (task-completion objective
−log P(x_a|x_q) with a PrefixLM mask). Full BPTT produces rare, very large gradient events (paired max ratio 35.7×, 95th pct 1.89×;
Jacobian growth rises with backward depth, mostly with H-cycle depth) — K = 5 is the deepest stable setting.
(c) FLOP-matched (1e21, 60B tokens): HRM 1B (H2L3) MMLU 60.7 / ARC-C 81.9 / GSM8K 84.5 / MATH 56.2 vs Looped Transformer 1B
56.5/74.1/75.1/48.3, RINS 56.1/76.7/77.7/48.9, Transformer 3B deep 56.7/80.5/75.7/50.5. TRM-1B (shared H/L, H2L1) UNSTABLE
(46.4 MMLU); TRM-0.6B at H2L3 matches HRM-0.6B only with 2× the compute. Objective ablation: P(x) causal → P(x_a|x_q) → PrefixLM
adds far more than the architecture swap (Transformer 40.6 → 53.2 MMLU; HRM 43.7 → 60.7). Effective depth: HRM keeps large
‖Δh‖ and large logit-lens KL to the final layer deep into the unroll; standard and looped Transformers converge early.
Auto-guidance logits_w = (1+w)·logits(h) − w·logits(h′) (h′ = earlier H loop): tiny gains, best w varies by task.
(d) Admitted: gains are architecture + objective + curated instruction data jointly; knowledge (MMLU) still scale-bound;
contamination test flags DROP at n = 13; up to 1B only; ACT not used.
(e) For us: (1) STATE-NORM DEFECT in our baseline: our S grows linearly (state_norm 9.77 → 18.49 → 27.07, ≈ +8.5 per pass) because
the recurrent path has no norm; a constant-size update into a growing state shrinks the relative change every pass — part of the
measured angle decay (6.7° → 0.1°) is just ‖S‖ growth. MagicNorm (norm at the exit of each pass on the recurrent path, PreNorm
inside) is the cheap fix → baseline audit item #4. (2) DISTINCT WEIGHTS FOR DISTINCT ROLES beat sharing at scale (TRM-1B
unstable; HRM > looped Transformer at matched FLOPs) — supports 4 distinct layers, and suggests a 2+2 split (L-layers iterate
fast reading the input, H-layers update the plan slowly) as one design axis. (3) Truncated BPTT with a warm-up horizon (K 2 → 5)
bounds the gradient-tail risk of large R. (4) Logit-lens KL to the final output staying large at depth is the "depth is being
used" diagnostic — the opposite of our R1..R8-flat measurement; add it to the eval for any v2 run. (5) The biggest lever in
HRM-Text is the objective/data (PrefixLM, response-only loss), not the loop — consistent with C5 (loops buy reasoning, not LM loss).
(f) z ← Norm(z + Σ_l Block_l(Norm(·))) per module step; H2L3: [L,L,L,H]×2; TBPTT K: 2 → 5; logits_w = (1+w)ℓ(h) − wℓ(h′).

## 9. ELT — Elastic Looped Transformers for Visual Generation (Goyal, Agrawal, Anil, Jain, Paul, Kusupati; Google; arXiv 2604.09168) — read in full (25 pp)
(a) N distinct layers g_Θ looped L times (N×L), MaskGIT (BERT backbone) and DiT backbones, no conditioning on the loop index,
no gates (but NOTE: the student prefix does receive GT loss, λ starting at 1 — intermediate supervision exists; what differs from
GRT is that the max-depth pass keeps its full GT loss every batch and the intermediate loss decays into distillation). Intra-Loop Self-Distillation (ILSD): train at a FIXED L_max; sample a student exit L_int ~ U(L_min, L_max) each step;
the student is a strict PREFIX of the teacher's trajectory, so it costs only one extra head evaluation (no second forward pass —
unlike LoopFormer's separate shortcut trajectory, 1.5× FLOPs).
(b) Loss: L = L_GT(F_Lmax) + λ·L_GT(F_Lint) + (1−λ)·L_dist(F_Lint, sg(F_Lmax)), λ linearly 1 → 0 over training (anchor the
student to labels early, to the teacher late); L_dist = CE between teacher and student token distributions (MaskGIT) or MSE
(diffusion). Main loss always on L_max — the asymmetric placement again.
(c) ImageNet-256: ELT-XL 7N×4L FID 2.0 at 111M vs MaskGIT-XL 2.0 at 446M (4× fewer params, iso-inference-compute); DiT
backbone 8N×4L 3.16, 16N×2L 2.83 vs 32-layer DiT 3.43; 1N×32L = 10.30 (a single unique layer fails regardless of loops — minimum
block size matters). Without ILSD quality collapses away from L_max (FID 41.7 at L=1 → 2.4 only at L=L_max); with ILSD smooth
(22.6 at L=1, 2.2 at L=7–8). MODEST EXTRAPOLATION: UCF-101 trained L_max = 4, best FVD 69.2 at L = 6 (72.9 at L = 4), degrades at
L ≥ 9. UCF-101 76M ELT FVD 72.8 vs MAGVIT-L 306M 76. Up to 3.5× throughput on TPU (weights stay on-chip).
(d) Admitted: N too small fails; L ≫ L_max deteriorates ("over-iterates beyond its trained convergence regime"); extrapolation
behaviour needs more study; visual domain only.
(e) For us: (1) the cheapest form of the LoopFormer idea — prefix students: our R_max forward already computes every
intermediate pass, so a per-pass NCP loss/distillation costs only extra NCP-head evaluations at chunk rate (~0.03–0.1 M MACs per
token per pass), no extra core passes. (2) Loss recipe with evidence of modest positive extrapolation beyond train depth: main
GT loss at a fixed R_max, prefix students get λ·GT + (1−λ)·KL-to-sg(teacher), λ 1 → 0. (3) 1N × 32L failing supports
"≥ 4 distinct layers per pass" over one shared layer (C3). (4) No loop-index conditioning needed for elasticity — ILSD alone did it
(cf. LoopFormer, which uses both). Caveat: vision; the task structure (iterative denoising/unmasking already outside the loop)
is different from in-context composition.
(f) L = L_GT(F_Lmax) + λ L_GT(F_Lint) + (1−λ) L_dist(F_Lint, sg(F_Lmax)), L_int ~ U(L_min, L_max), λ: 1 → 0.

## 10. Ouro — Scaling Latent Reasoning via Looped Language Models (Zhu, Wang, Hua, … Bengio, Eshraghian; ByteDance Seed et al.; arXiv 2510.25741) — read in full incl. appendix (55 pp)
(a) LoopLM: the whole stack of N shared layers (24 layers for 1.4B, 48 for 2.6B; d 2048, MHA, SwiGLU, RoPE, SANDWICH norm) looped
T_max = 4 times; an exit gate λ_t = σ(Gate(h_t)) at every step gives an exit distribution q(t|x) = λ_t Π_{j<t}(1−λ_j). 7.7T tokens.
(b) Objective (Stage I): L = Σ_t q(t|x)·L(t) − β·H(q) (entropy reg. = KL to a UNIFORM prior; β 0.1 → 0.05). ALL T_max steps run
every batch and every step's LM loss enters, weighted by the learned q — ADDITIVE, not replacing; no detach (full BPTT over 4
steps). Learned q on MMLU ≈ {0.0004, 0.086, 0.379, 0.535} → effectively deep-heavy. Stage II trains only the gate on detached
per-step improvements I(t) = max(0, L(t−1) − L(t)), target w = σ(50·(I − 0.005)). Uniform prior beats geometric priors (which
starve deep steps of credit). T_max = 8 was UNSTABLE (loss spikes) → cut to 4; batch 4M → 8M tokens; LR must be lower than dense.
(c) Ouro-1.4B ≈ Qwen3-4B (MMLU 67.4, BBH 71.0, GSM8K 78.9, MATH500 82.4); 2.6B ≥ 8B dense on reasoning. R > R_TRAIN DEGRADES
moderately: 1.4B T=4 → 8: MMLU 67.45 → 64.49, ARC-C 60.9 → 58.2 (Thinking models peak at T=4–5, fall after; only safety keeps
improving to T=8). Step agreement: A[2,3] = 55 %, A[3,4] = 36 % of labels change between steps ≤ 4; beyond 4 the answer is fixed.
Knowledge capacity unchanged (~2 bits/param looped or not); MANIPULATION improved: Mano modular-arithmetic trees at L=24 ops:
Loop 3⊗4 92.2 % vs dense 12⊗1 (iso-FLOP) 34.8 % vs 3⊗1 11.0 %; 3-hop QA learned with fewer unique samples than iso-param and
iso-FLOP dense. Theory: one looped layer solves graph reachability in O(log D) loops (repeated squaring). MMLU gains largest in
elementary maths (+156 %), formal logic (+143 %), smallest in global facts (+8 %). Scaling-law appendix (53M–1.36B, 20B tokens):
at equal (params × steps) the UNSHARED model is always better; the gap grows with steps and shrinks with size. Small models show
"exploitation": SHALLOW-step loss rises with more data while the gate shifts weight to deep steps. Hidden-state-difference
early exit is within 1–2 % of the trained gate.
(d) Admitted: extrapolation beyond trained depth degrades on benchmarks; T=8 training unstable; RL gave no gain; vocabulary
limits; "future research should focus on enhancing performance extrapolation at greater depths".
(e) For us: (1) C2 taxonomy — Ouro is ADDITIVE per-step loss weighted by a learned exit distribution (deep-heavy after training),
no detach, no distillation. It gives monotone "deeper is better" up to T_max and faithful step-to-step revisions, but NOT
extrapolation beyond T_max — so additive weighting alone does not buy step extrapolation; it removes the GRT-style collapse.
GRT's 3.19 vs 3.14 used Ouro's objective at small scale: fair as an LM-loss comparison, not about extrapolation. (2) The
small-model "exploitation" effect is directly relevant at 25M: with learned per-step weights the model sacrifices shallow passes to
make deep passes better — acceptable for us (it is the opposite of R-invariance), but it means R1 plans get worse: keep a floor
weight on the shallow NCP loss so the decoder always has a usable plan (C5). (3) Mano/multi-hop: loops help MANIPULATION
(composition) not storage — the concept core is exactly the composition module; parametric storage lives in the encoder/Engram.
(4) T_max 8 unstable at 1.4B with full BPTT — another vote for grouped BPTT / truncation if we train R up to 8. (5) Sandwich norm
+ lower LR for looped parts.
(f) q(t|x) = λ_t Π_{j<t}(1−λ_j); L = Σ_t q(t|x) L(t) − β H(q); Stage II: w_t = σ(k(I_t − τ)), I_t = max(0, L_{t−1} − L_t).

## 11. MoR — Mixture-of-Recursions (Bae, Kim, Bayat, … Courville, Yun; KAIST/Mila/Google; arXiv 2507.10524 v3) — read in full incl. appendix (38 pp)
(a) Recursive Transformer with MIDDLE-CYCLE sharing (unique first and last layer, a shared block of middle layers cycled N_r
times) + lightweight routers that give each TOKEN its own recursion depth. Expert-choice routing: at step r a router
g = σ(θ_rᵀ H) picks the top-k tokens (capacity N_r/N_r, …, 1/N_r), H ← g·f(H) + H for selected tokens only, hierarchical
filtering (only tokens kept at r can continue). Token-choice: one routing decision up front assigns depth i. KV: recursion-wise
caching (only active tokens store KV at that depth) or recursive KV sharing (reuse first-recursion KV).
(b) Router training is end-to-end through the output scaling g·f(H); expert-choice top-k over the sequence LEAKS future
information (non-causal), fixed with an AUXILIARY BCE loss on the main router (coef 0.001, sigmoid, α = 0.1, linear router —
best) or a separately trained auxiliary router (worse, positional "dead token" bias up to 66.7 %). Token-choice needs a balancing
loss (MaxVio 0.2–0.7) and still struggles to balance "heterogeneous experts" for half of training. No per-step loss, no detach.
(c) 360M base, 16.5e18 FLOPs: MoR expert N_r=2 (167M params) NLL 2.7511 vs vanilla 315M 2.7824 vs recursive 2.8079 (MoR sees 27B
vs 20B tokens at equal FLOPs); N_r=3 2.7925, N_r=4 2.8204; token-choice N_r=3 2.9163 (worse). At 135M MoR is WORSE than vanilla
(capacity bottleneck); ≥ 360M it matches/exceeds. Fixed 118M params, 10B tokens: recursion 1 → 2 → 3 valid NLL 2.968 → 2.921
→ 2.911. Sharing: Middle-Cycle best (360M N_r=3: 2.876 vs Cycle 2.936, Sequence 3.025, M-Seq 2.975); Sequence worst. Per-recursion
LoRA/DoRA/prefix relaxation from scratch: no real gain. KV sharing: small loss. Test-time: more recursion helps only up to the
trained N_r. Compute-optimal MoR prefers bigger models trained shorter. Up to 2.06× decode throughput with depth-wise batching.
Router depth tracks next-token predictability (second sub-word pieces exit early).
(d) Admitted: 1.7B vanilla slightly better (scaling concern); expert-choice capacity can't be changed after training (router
outputs become binary); token-choice load imbalance; no reasoning-task evaluation.
(e) For us: (1) Adaptive per-CHUNK depth is possible in the concept core (route hard concepts through extra passes), but it is an
EFFICIENCY lever, not an extrapolation lever — depth stays capped at N_r and the core is already cheap at ¼ rate. Expert-choice
top-k across chunk positions is non-causal (would leak future chunks into the plan — violates our causal plan rule R7) → only
token-choice/threshold routing with an aux loss would be allowed; complexity high for little gain → LOW priority. (2) Middle-Cycle
is exactly our macro layout (unique encoder → looped concept core → unique decoder) — consistent. (3) At small scale (135M) sharing
costs capacity ("recursive capacity bottleneck") → supports distinct layers inside our tiny core rather than one shared layer.
(4) Per-recursion LoRA/prefix does not rescue sharing from scratch — prefer distinct layers or state-conditioned modulation (GRT,
LoopFormer) over low-rank per-step adapters.
(f) H_t^{r+1} = g_t^r f(H_t^r; Φ′) + H_t^r if g_t^r > P_β(G^r) else H_t^r; L_bal = α Σ_i f_i P_i.

## 12. DiscoLoop — Looping Discrete Embeddings and Continuous Hidden States (Fu, Guo, Wang, Zhu, Lee, Jiao, Russell, Mei; Berkeley/Princeton; arXiv 2607.00341) — read in full (16 pp)
TAXONOMY: ORTHOGONAL to the C2 rule — it changes what is handed between passes (representation), not where the loss sits
(final pass only, fixed K = number of hops, no R > R_train test; length generalization named as future work).
(a) H(k+1) = f(H̃(k)); H̃(k+1) = H(k+1) + α(k) ⊙ RMSNorm(Φ(H(k+1))), Φ(h) = Σ_v softmax(Wh/τ)_v · W[v] (soft decode through the
tied LM head then re-encode through the embedding); α fixed (1) or a token-wise gate σ(w·Φ + b) (d+1 params); α = 0 on the last
loop; differentiable, no stop-gradient; top-128 truncation of Φ in LM pretraining.
(b) Diagnosis on in-weight two-hop: after loop 1 the bridge entity is DECODABLE (P = 1.000) but the hidden state is MISALIGNED with
its embedding (cos 0.33 ID / 0.27 OOD) — loop 2 is fed a noisy mixture instead of a clean token embedding. A training-free swap
H ← (1−α)H + α·Norm(W[argmax]) at the bridge position: OOD 8.3 → 25.9 % (α 0.1) → ≈ 100 % (α 0.5). Stage-1 (one-loop) accuracy
collapsing to ~0 marks the model offloading composition to loop 2 (the phase transition), earlier/sharper with DiscoLoop.
(c) Symbolic 2-hop (500 entities × 2 graphs, 3000 epochs): vanilla loop ID 71.1 % / OOD 8.3 %, no-loop (2× params) < 20 % / 0 %,
DiscoLoop ≈ 100 % / ≈ 100 %. 3-hop: DiscoLoop ID ≈ 100 %, OOD ≈ 65 %; vanilla and no-loop ≈ 0 % OOD. Synthetic English (direct and
reverse): DiscoLoop ~100 % ID / ~95 % OOD, vanilla ~90 % ID and fails OOD. LM 440M (Ouro backbone, 24 layers, K = 4, 20B tokens
FineWeb-Edu + FineMath): avg 50.5 vs vanilla loop 49.3 vs PonderLM 49.8; its loss overtakes vanilla after ~13B tokens.
(d) Admitted: moderate scale; no depth/length extrapolation test; in-weight (parametric) composition only.
(e) For us: LOCAL NEGATIVE EVIDENCE EXISTS — HRM_THINKER_PAPERS / Concept Module P3: DiscoLoop was TESTED in our concept testbench
and rejected (+750 steps slower in 2 seeds; bridge decodability at pass 1 fell 35–62 % → 2–5 %: "a channel read from an
unsupervised pass-1 state was co-adapted away"; the in-weight two-hop arms all stayed 15–27 % ID / ≤ 1.5 % OOD at ~700 epochs,
far below the paper's 3000-epoch regime). SHARPENED VERDICT: the local in-weight replication arm was UNDERPOWERED — every
variant including the vanilla loop stayed at 15–27 % ID (paper's vanilla reaches 71 % at 3000 epochs), so it adjudicates nothing
about the paper's task. What does stand: (i) +750 steps on our testbench task and (ii) bridge decodability DESTROYED (35–62 % →
2–5 %) — a genuine contradiction of the paper's mechanism, where P(bridge) = 1.000 is preserved. So: 'local test underpowered on
the paper's task; contradicts the paper's mechanism on ours; conditional option only with per-pass supervision; not retested this
cycle.' Do not cite P3 as a replication. It could change only if pass-k states become SUPERVISED to decode something (e.g. an
ELT-style per-pass NCP loss), in which case a soft decode through the PQ codebook (Σ softmax(−dist/τ)·codeword per segment) would be
a principled channel — but our NCP head decodes the NEXT concept, not an intermediate bridge, so the meaning only partly matches
DiscoLoop's bridge. Our P4 fix (re-read X = c_hat every pass) already addresses the same input-distribution mismatch. Keep as a
conditional option for a design that has per-pass supervision; not a core ingredient.
(f) H̃ = H + α ⊙ RMSNorm(Σ_v softmax(Wh/τ)_v W[v]).

## 13. DeepLoop — Depth Scaling for Looped Transformers (Li, Zhang, Guo, Gu, Wang; Princeton/UCLA; arXiv 2607.13491) — read in full (25 pp)
TAXONOMY: ORTHOGONAL to the C2 rule — a stability/parameterization result (C6); every row trains at a fixed R with the loss on
the final pass only; no R > R_train test.
(a) Keep Post-LN DeepNorm x ← Norm(α·x + f(x)) but with a LOOP-AWARE exponent: α = (2N)^{1/2}, β = (8N)^{-1/2} (β = init gain on the
residual-branch matrices: value/output projections and FFN), N = K·R unrolled blocks. Reason: a shared parameter's update is the SUM
of R visit-wise gradients and is then READ by all R visits; with aligned visits (κ_R = Θ(R)) the first-order output change grows
by an extra factor R, so the untied DeepNorm exponent p = 1/4 must become p = 1/2 at fixed physical depth. For HRM-style
truncated training, count only gradient-visible visits (M_g).
(b) Passes count because training at larger R is STABLE: the benefit appears only when blocks are revisited (neutral at R = 1).
(c) GPT-2 small (50B tokens) val loss R = 1/3/5/7: base 2.8627 / 2.8077 / 2.7910 / 2.7700 vs DeepLoop 2.8631 / 2.7917 / 2.7679 /
2.7514; GPT-2 medium: base 2.6253 → 2.5558, DeepLoop 2.6264 → 2.5280 (gap widens with R: −0.028 nats at R = 7). Downstream 8-task
avg (medium, 1-shot) R = 7: 55.20 vs 54.62. p-sweep at R = 3: p ≤ 0.40 ALL seeds diverge (stuck at the unigram floor), p = 0.45 mixed,
p ≥ 0.50 reliable; among trained seeds lower p learns slightly faster. HRM on ARC-AGI-1 with p = 1/2 (only change): voted K=2
36.50 → 39.75 % (≈ 6σ over a 4-seed control).
(d) Admitted: single-seed LM runs; one scale for the p-sweep; conservative worst-case bound (κ_R not measured); Post-LN only.
(e) For us (C6): (1) a looped block's parameter update is amplified by R when visits align — exactly our regime once R_max goes to
8 with 4 distinct layers (N = 32). Consequence for any design: shrink the residual-branch init of the looped layers (zero-init as in
Loop-Think is the β → 0 limit; DeepLoop's β = (8N)^{-1/2} = 1/16 at N = 32 is the non-zero alternative) and/or lower the core's LR
(Ouro reports looped models need a smaller LR). (2) Monotone val-loss gains with trained R up to 7 at 124M–350M — consistent with
"more trained R is fine if stable". (3) Our GR-read/gated-write core is pre-norm-like, so DeepLoop does not transfer literally;
the transferable rule is "scale the looped branch down by ~√(R·layers)". (4) The per-module view: only gradient-visible visits count
→ grouped/truncated BPTT also reduces the effective alignment factor.
(f) x_{i+1} = Norm(α x_i + f_j(x_i; φ_j)), α = (2N)^{1/2}, W^(0) = β·W̃^(0), β = (8N)^{-1/2}; condition M κ_R (β/α)² = O(1).

## 14. LOTUS — Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers (Fan, Svete, Lee; MSR/ETH/KRAFTON; arXiv 2606.31779 v2) — read in full (30 pp)
NOTE: the file name suggests a skeptic's paper; it is not — it is a POSITIVE latent-CoT method. Its skeptic content is the failure
analysis of prior latent methods (below).
TAXONOMY: CONFIRMS the rule's grounding clause — supervised (grounded) latent states + fixed trained R; beyond R_train: flat/dip.
(a) Looped PADDED Transformer: K = 6 latent blocks × c tokens (c = 25; 150 learnable latent positions) placed between question and
answer; the base LM is iterated R = 6 times over the latent region with input injection h(t) = f(E + h(t−1) | C_pre) (question KV
computed once), full BPTT through R. Step loss L_step = CE of each post-loop latent position (read through the base LM head) vs.
the gold CoT token of its step (block i ↔ CoT step i, all in parallel) + answer loss from a final forward that conditions on h(R);
λ_step = 0.05. Curriculum (Coconut-style, one more step latent per epoch; R grows with K).
(b) Why passes count: (i) grounding — each latent block has a direct, position-aligned target; (ii) parallel width — all steps
refined together over R passes (R ≪ N CoT tokens); (iii) post-loop readout (latents may use all R passes before being read).
(c) GSM8K (Llama-3.2-3B): LOTUS 70.0 vs explicit CoT 71.5 vs CODI+SIM-CoT 62.3; OOD avg 63.9 vs CoT 62.1; thought latency 2.5×
faster (6.9× on natural-language CoT, 68.1 vs 68.4). Ablations (3B): no latent supervision 63.3; CODI-only 64.4; per-iteration
head supervision 68.2 vs post-loop 70.0. Trained R = 2/3/4/5/6: 14.6 / 23.2 / 52.6 / 68.1 / 70.0 (depth essential). Inference R
(trained 6) = 1…7: 22.7, 40.0, 55.0, 63.5, 68.7, 70.0, 69.3 → peaks at the trained R, slight dip beyond. Block width c = 1 → 49.7,
5 → 67.5, 25 → 70.0 (one slot per step is too narrow). Latents read through the LM head recover gold steps (top-1 70.9 %) and put
mass on UNSEEN valid alternative chains (15.3 % top-1, 64 % top-5). L_step-only and L_ans-only both far worse (complementary roles).
(d) Admitted / failure modes of latent reasoning it documents: prior latent methods (Coconut, CODI, SIM-CoT) fall further behind CoT
as scale grows; Coconut at scale suffers "latent instability — latent representations become homogeneous and lose semantic
diversity, causing training to collapse"; PCCoT (Jacobi) peaks at ~3 iterations and degrades beyond; ungrounded latent traces
"drift from meaningful computation and destabilize training at scale". LOTUS itself: fixed K/c/R; math only.
(e) For us: (1) C1 DISCOUNT RULE (skeptic content): latent passes whose intermediate states have NO grounded target drift/collapse
(Coconut homogenisation ≈ our step-angle → 0.1° stagnation); consistent with our own P3 rule. A design earns full C1 only if its
passes are grounded (per-pass NCP targets, TRM decodable y, or task intermediates). (2) WIDTH matters as much as depth: LOTUS c = 1
slot per step loses 20 points vs c = 25 — mirrors our measured capacity limit (u carries ≈ 1 answer token: probes 0.59 → 0.21 →
0.06). A multi-slot concept (m latent slots per chunk, read by the decoder) is a C4 option. (3) Post-loop readout beat per-iteration
readout on the main head (70.0 vs 68.2) — per-pass losses should be light (λ small, ELT-style decaying), with the main loss post-loop.
(4) No extrapolation beyond trained R (69.3 at R = 7) — again: fixed-R training does not buy extra depth at test time.
(f) h(0) = f(E|C_pre); h(t) = f(E + h(t−1) | C_pre); L = L_ans + λ_step Σ CE(head(h(R)_{i,j}), T_{i,j}).

## 15. PLT — Parallel Loop Transformer (ByteDance Seed; arXiv 2510.24824) — read in full (16 pp)
TAXONOMY: ORTHOGONAL (fixed L = 2–3, final-pass loss; an inference-efficiency method). Parallel across TOKENS, not passes — the
T²MLR/Jacobi family, but exact (not approximate).
(a) Cross-Loop Parallelism (CLP): before loop l, right-shift the previous loop's states by one position and add the embeddings,
B = E + shift(H^(l−1)); H^(l) = f(B). Loop l at token i therefore depends on loop l−1 at token i−1 (a diagonal dependency), so at
decode time loop 1 of token i, loop 2 of token i−1, loop 3 of token i−2 run in ONE batched forward pass. Training is parallel over
tokens and serial over loops (exact, no fixed-point approximation). Efficient representation enhancement: all non-first loops
attend GLOBALLY to the first loop's shared KV (one KV cache) plus a private gated sliding-window attention (w = 64),
ỹ = g ⊙ y_local + (1−g) ⊙ y_global, head-wise scalar sigmoid gate.
(b) Passes count via plain depth; nothing forces pass-specific behaviour; fixed L.
(c) Seed-MoE 680M-act/13B, 150B tokens: vanilla avg 34.7; loop-2 39.7 (latency 1.96×, KV 2×); +CLP 39.6 (latency 1.23×); +KV share
36.2 (−3.4, latency 1.00×); +G-SWA = PLT-2 39.7 (latency 1.02×, KV +1.4 %); PLT-3 40.8. PLT-2 1.7B-act/40B beats vanilla 2.5B/60B
(62.6 vs 62.1) at ~30 % lower latency. Open-source OLMo 1.2B dense (400B tokens): loss 2.577 → loop-2 2.532 → PLT-2 2.537.
(d) Admitted: gains shown at fixed loop count; confidential details for the in-house runs.
(e) For us: answers the "parallel across what?" question — across tokens via a one-position shift between passes, i.e. the same
structure as T²MLR's Jacobi carry but exact. It changes the serial-horizon arithmetic only in the sense already noted under T²MLR:
a shifted inter-pass connection makes pass r at chunk j consume pass r−1 at chunk j−1, so R passes extend the serial horizon by R
chunks while training stays parallel. Our core is tiny and not latency-bound, so the KV-sharing/G-SWA efficiency pieces are not
needed; the one transferable idea is the SHIFTED carry as an explicit extra input (already an ingredient). Also a C5 datum: a 2-loop
block gains +5 avg points / −0.045 loss at ~1B scale when compute is not matched.
(f) H^(l) = f(E + shift_right(H^(l−1)); K_share, V_share); ỹ = g⊙SWA_w(Q,K,V) + (1−g)⊙Attn(Q,K_share,V_share).

## 16. Loopie — Loop the Loopies! (Gao, Chen, Xiao, Yang, Tao, Zhou, Dai; IQuest Research; arXiv 2607.16051) — main text + appendices A–D read in full; appendix E (the model's own IMO-2025 proof transcripts and judge rubrics, ~1,400 lines) checked by search only — no architecture content (67 pp)
(a) Is this the skeptic's paper? PARTLY: its premise is the compute-matched critique — "given N× pre-training compute, N× parameters
usually beats looping N times"; Ouro-2.6B×4 should be compared with a ~10B dense model, Huginn-3.5B×32 with ~112B. Its answer:
looping wins under matched compute only with (i) LAYER-LOOP scheduling (L1 L1 L2 L2 … — each stored layer applied twice before the
next) instead of MODEL-LOOP (L1 L2 L3 L1 L2 L3), (ii) R = 2 only, (iii) the "Loopie recipe": halve stored layers, loop twice, spend
the freed activation memory on 2× micro-batch, reinvest the measured speed-up in width/depth (Qwen3-30B-A3B → D 2304, L 27, R 2),
matched by measured optimizer-step wall-clock, not FLOPs. SmallInit std (2.5d)^{-1/2} for block matrices; WSD-without-decay LR.
(b) Taxonomy: ORTHOGONAL — fixed R = 2, final loss only, no extrapolation test.
(c) Loopie-20B-A2B (3.5T tokens) vs compute-matched Qwen3-like 30B-A3B (800B tokens): lags early, OVERTAKES after ~600B tokens;
scaling ladder 0.15B/0.25B/0.5B/1B baselines: Loopie +1.1/+0.6/+1.7/+2.2 avg points (1000× active-param tokens). Layer-loop vs
model-loop (6B-A0.6B): model-loop AHEAD EARLY, layer-loop overtakes after ~1.2T tokens. Removing layer-loop at equal compute:
clearly worse ("2.14× speedup" to reach the same score). N× loop vs N× stored layers (250B tokens): the marginal benefit of recurrence
is largest at R = 2 and decays for R = 3, 4. Post-trained: AIME24 92.1, IMO-2025 35/42 with heavy test-time scaling.
(d) Admitted: pretraining compute is the binding constraint → R = 2; larger R "may be useful … where the goal is to study recurrent
reasoning rather than pre-training efficiency"; related-work summary: looped models "can drift, overthink, or collapse to shallow
computation unless their state injection, normalization, supervision, and stopping rules are designed for stable iteration";
learned loop allocation "can correlate with token difficulty without necessarily extrapolating to longer inputs"; stochastic stopping
improves extrapolation across unseen depths (Kuo et al.); CART anchors each iteration to context.
(e) For us: (1) C7 SKEPTIC DISCOUNT — every MAC spent on core passes could instead buy encoder/decoder width; our core runs at ¼
rate so the exchange rate is 4× better than in Loopie, but any design must show its core cost (MACs/token) stays a small fraction of
the ~18 M MAC total, else C7 drops. (2) SCHEDULE: at OUR horizon (~0.2B tokens in 24 h) the evidence favours MODEL-LOOP (the 4-layer
block cycled R times: MoR Cycle > Sequence at 10B tokens; Loopie model-loop leads until ~1.2T tokens); layer-loop is a
large-token-budget effect → designs use the block cycle. (3) Loopie's R = 2 optimum is a pretraining-compute statement about the
WHOLE model; our reasoning core is a separate budget where R up to 8 is cheap — not a contradiction, but a reminder to cap R_max by
measured MACs. (4) SmallInit (2.5d)^{-1/2} for block matrices is another "scale the branch down" datum (C6).
(f) C(D, L, R) ∝ L·D²·R; Loopie recipe: (D0, L0, 1) → (D0, L0/2, 2) → widen until t_step(2b, g/2) ≈ t_step,ref(b, g).

## 17. LoopCoder-v2 — Only Loop Once for Efficient Test-Time Computation Scaling (Yang, Guo, … Dai; Beihang / IQuest; arXiv 2606.18023) — read in full (20 pp)
TAXONOMY: fixed R (train = eval), final loss; a loop-count study of PLT. Not about extrapolation — about the cost side of CLP.
(a) 7B PLT coders (14 layers, d 5120, looped block = the whole 14-layer stack) trained from scratch on 18T tokens at R = 1, 2, 3, 4
with CLP (B^(r) = Embed(x) + shift(h^(r−1))), first-loop shared KV + G-SWA (w = 64, per-head gate zero-init to an even blend).
(b) Gain–cost view: each extra loop gives refinement (hidden-state update, attention re-routing, output shift) but the CLP shift
feeds token i the loop-(r−1) state of token i−1 instead of its own → an intrinsic offset cost Ω(r) = mean ‖h_i^(r−1) − h_{i−1}^(r−1)‖.
(c) SWE-bench Verified R = 1/2/3/4: 43.0 / 64.4 / 27.6 / 22.4; suite avg 38.0 / 46.5 / 36.9 / 34.3 — strongly NON-MONOTONIC. Loop 2
carries the largest attention re-routing, output shift, and the effective-rank PEAK (174.6); later loops: cos θ between successive
updates NEGATIVE (−0.72 at r = 2, −0.46 at r = 3 → oscillatory), effective rank falls (158.2 at r = 4), heads grow redundant
(head-head sim 0.57 → 0.67 → 0.71), middle extra loop ≈ dead pass-through. ANSWER TO THE COMPOUNDING QUESTION: Ω(r) is ROUGHLY
CONSTANT per loop (a fixed tax, not compounding), but it exceeds the shrinking per-loop gain by 30–45× beyond loop 2. Explicit CoT +
loop is super-additive (LiveCodeBench +26.9 at R = 2). Secondary pointers (not in folder, unverified): Schwethelm et al. — looping a
block r times is worth only r^0.46 unique parameters in val loss; Yang et al. 2026 — performance can peak at an intermediate loop
depth then collapse, fixed-point regularisation stabilises.
(d) Admitted: no non-shifted (sequential) looped variant at 7B is compared, so the regression's attribution to CLP rests on the Ω
diagnostic, not an ablation; train = eval loop count only.
(e) For us: (1) CAVEAT FOR THE SHIFTED-CARRY INGREDIENT: the damage comes from REPLACING the same-position recurrence with the
neighbour's state (PLT/LoopCoder: token i no longer sees its own previous-loop state). T²MLR instead ADDS the shifted cache to the
unchanged same-position stream (zero-init tanh gates) and improves. ⇒ any design using a shifted carry must keep S_j (same chunk,
previous pass) as the main path and treat chunk j−1's state as a gated EXTRA input, zero-init. (2) Negative-cosine successive updates
+ effective-rank decline are the failure signature of useless passes — add "per-pass effective rank" and "update alignment cos θ"
to the eval diagnostics (our baseline's +0.99 alignment / 0.1° is the other failure: stagnation). (3) CONFOUND — R = 3 < R = 2 here compares SEPARATELY TRAINED fixed-R
models, each WITH CLP; attribution to the offset rests on the Ω diagnostic, not an ablation. It is NOT a 'training at larger R
hurts' result, does not contradict Loop-Think's 'train R_max > 4' rule, and is not entered in the C2 table.
(f) B^(r) = Embed(x) + shift(h^(r−1)); Ω(r) = (1/S) Σ_i ‖h_i^(r−1) − h_{i−1}^(r−1)‖₂; cos θ(r) = ⟨Δh^(r), Δh^(r−1)⟩ / (‖·‖‖·‖).

## 18. SMELT — Scaling Laws for Compute-Matched MoE Looped Transformers (Wang, Zhang, … Li; Tsinghua / ByteDance Seed / M-A-P; arXiv 2609.01343) — read in full incl. appendices (35 pp)
TAXONOMY: ORTHOGONAL (fixed R = 2, final loss); the quantitative compute-matched study.
(a) MoE Transformer; loop the MIDDLE HALF of layers twice; match per-token FLOPs (narrow H), total params (add experts), KV cache
(head size / GQA) to the unlooped baseline within < 4 %; inside the looped span every sublayer's residual update is SCALED BY 1/r
(from Wang et al. "On the residual scaling of looped transformers") to stop correlated tied updates inflating the stream.
(b) Ablations at 200M: span 50 % best (val loss 1.9257 vs full-stack 1.9322 vs none 1.9445 at S≈85 %); looped model prefers a larger
effective depth-to-width ratio (physical 12 / executed 18 best); loop count 2 best (1.9257) vs 3× 1.9385, 4× 1.9360 — more loops force a
thinner model under matched FLOPs.
(c) Scaling ladder to 54B non-embedding params: separate Chinchilla-style fits, frontier exponent γ 0.250 vs 0.237 → compute saving
(CE Gain) 6.8–10.0 % at 1e20 FLOPs, 14.7–18.0 % at 1e21. Wins DCLM Completion 96/96 pairs, Core 83/96, MMLU 29/30; downstream gain
EXCEEDS what val loss predicts, growing with scale. Domain CE Gain: Code 20.4 % > Finance 16.8 % > Math 16.6 % > Knowledge 14.9 % > Web
14.8 %. Gain tilted to LONG samples (512–4096 tokens: 1.52× the short-sample gain; adding params/experts shows no tilt) and to IN-
CONTEXT LEARNING (gap 0.9 pp at k = 0 → 1.9 pp at k ≥ 1; Dyck k = 32: 29.8 vs 26.4 %). Mechanism: visit 2 reuses retrieval coordinates
(Q/K cosine 0.89–0.93 across visits, top-8 attended tokens 56–66 % shared vs 28–34 % control), V diverges (0.65–0.74), writes 1.2–3.5×
LARGER, ALIGNED updates (same-layer cross-visit cosine 0.42–0.65); ATTENTION SINK collapses on visit 2 (Dyck head: BOS mass 0.60 →
0.02, demonstration-answer mass 0.24 → 0.85), against the baseline's depth trend. EXCHANGE-RATE NUMBER cited (Schwethelm et al.,
dense, iso-depth, not in folder): r recurrences ≈ r^0.46 unique-block equivalents — looping LOSES at matched compute when unique params
must shrink; SMELT's MoE parameter recovery is what turns it into a win.
(d) Admitted: ablations only at 200M; arithmetic FLOPs, not wall-clock, matched; mechanism descriptive, not causal.
(e) For us: (1) C7 QUANTITATIVE — by the r^0.46 rule a 4-layer core cycled R times is worth ≈ 4·R^0.46 unique layers IN LM-LOSS TERMS
(R = 4 → 7.6, R = 8 → 10.4) for 4R layer-passes of cost: loops are poor LM-loss value per FLOP. The core must be justified by
COMPOSITION, which is exactly where SMELT's gains concentrate (long samples, ICL, Dyck, code) and Ouro's Mano result — consistent with
C5 (don't claim val-loss gains). (2) C1 SUPPORT FOR IN-CONTEXT RETRIEVAL: the second visit keeps "where to look" (Q/K) but reads
different content (V), kills the attention sink and moves mass onto demonstration answers — the closest mechanistic analogue in the
folder to our in-context hop/bind/pointer tasks. (3) Fifth "scale the branch down" datum: 1/r residual scaling inside the looped span.
(4) Loop count 2 optimal under matched FLOPs at the whole-model level — for our ¼-rate core with its own budget, the constraint is
the C7 share, not the loop count per se.
(f) Δx = (1/r)·Attn(LN x); L(F,S,D) = E + A(1−S)^b/F^a + K/D^c; CE Gain = 1 − C_tgt/C_ref; Schwethelm: N_eff = N_once + r^0.46 N_rec.

## 19. ETD — Encode, Think, Decode: recursive latent thoughts (Koishekenov, Lipani, Cancedda; FAIR/UCL; arXiv 2510.07358) — read in full incl. appendices A–E (20 pp)
TAXONOMY: fixed k per model, final loss (standard mid-training objective), no detach; inference-k vs train-k not separated.
(a) Split a pretrained OLMo-2 1B (16 layers) into latent encoder E / recursive "thinking" block T / latent decoder D, chosen by the
angular-distance knee (Kneedle): 7-4*k-5. Re-run only the mid-training stage (50B tokens, 1.25 % of pretraining) with T looped k
times; no new params, data or hyperparameters; T is SHARED weights (4 distinct layers looped = our block-cycle layout).
(b) Passes count via extra effective depth on reasoning-heavy tasks; adaptive variant: ACT-style router (linear + sigmoid,
accumulate halting ≥ 1−ε, N_max 10), trained end-to-end with the task loss.
(c) k = 1/2/3/4/5: GSM8K 44.05 / 51.10 / 54.36 / 55.50 / 56.56 (+28.4 %); MATH 4.57 / 5.45 / 6.22 / 3.73 / 4.33 (peaks at k = 3);
factual knowledge flat (37.6 → 38.2); BBH needs k ≥ 3. At equal FLOPs, selective 7-4*4-5 beats 2-12*2-2 (middle looping with thin
ends) and full looping 0-16*2-0. Adaptive depth beats fixed k = 5 on DROP/OpenBookQA with fewer iterations.
(d) Admitted: single seed/data order; layer-role analysis is correlational; instruct models untested.
(e) For us: (1) confirms the settled constraint (loop a MIDDLE block; our encoder → core → decoder), and adds that the loop block
should sit where representations change slowly (angular-distance plateau) — our concept core is exactly that stage. (2) Another datum
that loops help reasoning, not knowledge (factual flat) — C5. (3) Non-monotone per task (MATH peaks at k = 3) — per-task optimal
depth varies → supports ACT/KL+entropy halting at eval rather than one fixed R. (4) 4-layer shared block, k up to 5 trained from a
warm start in 1.25 % of the tokens: evidence that a looped block can be introduced late (warm-start option for v2).
Appendix: angular distance d = arccos(cos(x_T^l, x_T^{l+n}))/π on the last token, 10k C4 examples; Kneedle knee detection with
a degree-2 polynomial fit. Per-task (k = 5 vs 1): DROP +19.9 %, ARC-C +20.8 %, OpenBookQA +15.5 %, Social IQa +13.9 %, MMLU-Pro
+22.7 %, but NaturalQuestions −2.1 % (−4.8 % at k = 3) — loops slightly HURT pure factual recall. Block-position sweep: 7-4*2-5
best overall; 3-4*2-9 best for reading comprehension; putting the block at the very start (1-4*2-11) or end (11-4*2-1) is worst.
(f) x_L = x_0 + Σ_E f + Σ_{j=1..k} Σ_T f + Σ_D f; halt when Σ_t w_t ≥ 1 − ε.

## 20. LoopWM — Looped World Models (FaceMind Research Asia; arXiv 2606.18208) — read in full incl. appendix tables (34 pp)
TAXONOMY: depth loop (inner) + temporal carry across environment steps (outer); inner loop trained with Poisson T and a final
loss — weakly confirms the rule; EVIDENCE QUALITY LOW (see d).
(a) Observation encoder + action embedder → looped dynamics core (prelude P → shared recurrent block R × T → coda C) → prediction
heads. Recurrent update borrowed from Parcae: h(t+1) = Ā·h(t) + B̄·e + R̄(h(t), e) with Ā = exp(Δ·diag(−exp(a))) — a DIAGONAL
state-retention matrix whose entries lie in (0, 1) by construction (contractive, bounded for any T); B̄ = Δ·B; e = LN(P([h_{k−1};
e_k; u_k])) (normalised injection prevents late loss spikes). h(0) ~ N(0, σ²I) or the previous step's final state (dual loop).
(b) T ~ Poisson(μ_rec) with LEARNABLE mean, sampled PER SEQUENCE (not per micro-batch — "eliminates most loss spikes"); truncated
BPTT at ⌈μ/2⌉ steps; optional exit gate g = σ(w·h + b) with a binary-entropy regulariser; inference T_max may exceed μ_rec.
"Deferred decoding": K action steps rolled out in latent space, decoder called once at the end; intermediate latents kept grounded
by a latent-consistency loss ‖g_ω(h_k) − sg(E(o_k))‖² plus a cumulative-drift penalty; curriculum over K.
(c) ScienceWorld text world-modelling (5 chained actions): EM 68.4 % vs claude-opus-4-6-max 47.2 %, gemini-3-flash 30.8 %, qwen-3.5-
flash 10.0 %; AlfWorld EM 51.6 vs 53.0 (Claude). Per-step table for its own model: EM 67.2 → 68.4 % from step 1 to 5, F1 78.0 → 87.5 →
85.3 % (best at step 3).
(d) Admitted / our reading: NO ablation of looped vs non-looped or of the spectral constraint; baselines are general API LLMs, not
matched models; the paper states it is "intentionally selective in disclosure scope". Treat as a source of IDEAS, not evidence.
(e) For us: (1) Parcae-style contractive retention S ← Ā⊙S + B̄⊙inject + R(S, X) with Ā ∈ (0,1) diagonal (learned) is a principled
bounded-state alternative to the exit RMSNorm (audit #4): it keeps ‖S‖ bounded for ANY R — useful if we extrapolate R at eval.
(2) Per-sequence (rather than per-batch) R sampling reduces loss spikes — a free change for our trainer. (3) Learnable mean of the R
distribution is an idea, but it can drift to small R (Ouro's geometric-prior warning) — keep R fixed-distribution. (4) Grounding
intermediate states with a stop-grad target (latent consistency) is the same idea as LOTUS's grounding and the C1 grounding condition:
for us the natural stop-grad target at every pass is the NCP target code / sg(final-pass plan). (5) Temporal outer loop = rejected
axis (RLT) for training throughput.
(f) h(t+1) = Ā h(t) + B̄ e + R̄(h(t), e), Ā = exp(Δ·diag(−e^a)) ∈ (0,1)^d; T ~ Poisson(μ) per sequence; TBPTT ⌈μ/2⌉.

## 21. Nanbeige4.2-3B tech report (Nanbeige LLM Lab, Boss Zhipin) — read in full (17 pp)
TAXONOMY: fixed R = 2 (whole stack reused once more), final loss; no R > R_train result. Mostly a post-training/agentic report.
(a) Looped Transformer: after the stack runs bottom→top, the hidden states pass through the SAME stack once more (2 passes);
3B non-embedding params, 28T tokens.
(b)/(c) Loop design notes (no tables): training the looped model FROM SCRATCH beat upcycling a trained dense model ("significantly
better" — representations adapt to reuse throughout pretraining); 2 passes best trade-off — keeps ~75 % of the dense model's token
efficiency with a significant capacity gain; MORE passes give marginal gains, "substantially slows training and makes optimization
LESS STABLE"; sharing the KV cache across passes halves KV but gives consistently lower gains. Base model: GSM8K 92.7, BBH 81.6,
MMLU-Pro 63.8 vs Nanbeige4-3B 85.9/70.7/47.6 (loop + better data jointly — not separable). Post-training: SFT curriculum
64K→128K→256K, two-stage RLHF (think/non-think), length-controlled reasoning RL, agentic RL with action-centric process rewards.
(d) Admitted: loop effects are reported qualitatively only; data recipe changed at the same time.
(e) For us: (1) Ouro (T = 8 loss spikes) and Nanbeige ('less stable') are the two STABILITY sightings, both whole-stack at ≥ 1.4B
with full BPTT; Loopie/SMELT's R = 2 is a COMPUTE-MATCHING result (SMELT 3×/4× still train fine, they lose by forcing a thinner
model) — different reasons, not to be merged. Only the stability half is a caveat for us; our tiny chunk-rate core with grouped/truncated BPTT and
branch down-scaling is the regime where larger R is still viable (Loop-Think trained R up to 16 at small scale). (2) Train the loop
from scratch rather than upcycle — relevant to the warm-start option: a warm start from the current checkpoint is only an
initialisation, the loop must still be trained with the new recipe for real steps. (3) Separate KV per pass > shared KV (also
MoR/PLT) — our core attends over its own pass-r states, fine.
(f) h = f(f(x)) (two passes of the full stack).

## 22. URM — Universal Reasoning Model (Gao, Chen, Xiao, Xing, Tao, Luo, Zhou, Dai; Ubiquant; arXiv 2512.14693) — read in full (12 pp)
TAXONOMY: fixed inner loop M = 8 with TRUNCATED backprop (first N = 2 loops forward-only) and the loss accumulated over the
graded loops t = N+1..M (per-loop supervision on the later loops) + ACT outer loop (max 16, TRM-style deep supervision) —
CONFIRMS the rule (detach/no-grad warm loops + graded later loops).
(a) Decoder-only Universal Transformer: 4 distinct layers (d 512, 8 heads) looped 8 times per inner loop; ConvSwiGLU — a depthwise
short conv (kernel 2) applied to the gated SwiGLU features after the MLP expansion, Y = σ(W_dw ∗ (SiLU(G)⊙U))·W_down; TBPTL; EMA;
AdamAtan2 (Muon: same final accuracy, ~2× faster convergence).
(b) Why passes count: recurrent inductive bias + strong nonlinearity; later loops graded, early loops forward-only.
(c) ARC-AGI-1 pass@1 53.8 (TRM 40.0, HRM 34.4); ARC-AGI-2 16.0; Sudoku 77.6. Ablations: w/o short conv 45.3; w/o truncated backprop
40.0. DECOMPOSITION (Table 2, ARC-1 pass@1): vanilla Transformers up to 32× params/FLOPs reach at best 23.75 (32 layers × d512) — 16×
d1024 collapses to 0.0; UT 2 layers × 8 loops 36.25, 4 layers × 8 loops 40.0 at 4× params / 32× FLOPs. So at EQUAL FLOPs (32×) a
4-layer block looped 8× beats 32 unique layers (40.0 vs 23.75); at EQUAL PARAMS (4 layers d512) looping takes 5.13 → 40.0; and 4 distinct
layers per loop beat 2 (40.0 vs 36.25). TBPTL sweep (8 loops, 2-layer URM): grad/no-grad 8/0 36.25, 7/1 37.75, 6/2 39.13 (best), 5/3
39.50 pass@1 but lower pass@10, 4/4 38.75, 2/6 34.25, 1/7 22.50. Short-conv placement: inside the attention path (after Q, K, V or SDPA)
no gain or WORSE; after the MLP expansion the dominant gain; kernel 2 best. Nonlinearity ablation: SwiGLU → SiLU 29.75, → ReLU 28.63,
no attention softmax 2.0.
(d) Admitted: remaining nonlinearities (RMSNorm, QK dot product) not ablated; ARC/Sudoku regime (tiny data, heavy augmentation).
(e) For us: (1) C3 — in the data-scarce reasoning regime a looped 4-layer block beats 8× more unique layers at equal FLOPs and 4
distinct layers per loop beat 2: direct support for "4 distinct layers AS A LOOPED BLOCK", with the caveat that the LM regime (MoR,
ELT, SMELT) values unique params more. (2) TBPTL recipe with numbers: run the first ~25 % of passes without gradient, grade the rest
(per-loop loss) — a concrete C2/C6 ingredient for R_max 8 (e.g. passes 1–2 no-grad, 3–8 graded). (3) SHORT CONV PLACEMENT: our
concept core's GatedAttention applies ShortConv to q/k/v over chunk positions — URM finds convs in the attention path useless or
harmful, and the gain lives after the FFN expansion. Candidate change for any design: move the core's short conv from q/k/v into the
FFN (ConvSwiGLU, kernel 2 over chunk positions, causal). CAVEAT: AXIS TRANSFER, not a direct port — URM's conv runs over the
token/cell sequence; ours would run over concept-chunk positions (one position = 4 tokens; cdoc = doc[:, ::4], T 1024 → 256 chunk positions), where locality means something else. Credit it as
a conditional ingredient only. (4) Keep SwiGLU (the nonlinearity matters most; our FFN is SwiGLU).
(f) [verified, p.5 eq.: "gradients only on the loss accumulated in the latter (M−N) loops" — a per-loop CE on EVERY graded loop,
equal weight, not final-pass-only; the loss sits on top of TRM-style outer deep supervision] L_TBPTL = Σ_{t=N+1}^{M} CE(h_t^{(D)}, y); Y = σ(W_dw ∗ (SiLU(X W_g) ⊙ X W_u)) W_down.

## 23. GRAM — Generative Recursive Reasoning (Baek, Jo, Kim, Ren, Bengio, Ahn; KAIST/Mila/NYU; arXiv 2605.19376v2) — read in full (27 pp; bibliography skipped)
(The folder has ONE GRAM PDF; the scratchpad held a duplicate extract "Generative_Recursive.c.txt" — same paper.)
TAXONOMY: TRM-style deep supervision (N_sup = 16 supervision steps, detached carry z_T^(n) → z_0^(n+1)), gradient ONLY through the final
transition of each supervision step (1-step truncation), + ELBO KL on the stochastic transition. Depth fixed at train time (T = 3
high-level × K = 4 low-level (K = 6 Sudoku) per supervision step); ACT halting optional at inference. EXTRAPOLATES in depth:
MNIST trained with 16 steps improves monotonically BEYOND training depth, 16 → 256 steps (FID 77.79 → 73.34, IS 1.89 → 2.04; the
8-step row is below training depth); Sudoku Fig. 4 runs 8 → 320
iterations and both TRM and GRAM keep improving → CONFIRMS rule (ii): detached-carry deep supervision = improvement operator.
(a) Recursion on a per-position latent state z = (h, l) (like HRM/TRM; D 512, 8 heads, FFN 512; f_L, f_H = [Attn + SwiGLU] × 2
layers; fixed z_0 sampled once and stored). Low level: l ← f_L(l, h + e_x) K times (deterministic). High level: u_t = f_H(h, l)
(deterministic proposal) then STOCHASTIC GUIDANCE h_t = u_t + ε_t, prior ε_t ~ N(μ_θ(u_t), σ_θ²(u_t) I); training samples from a
target-conditioned POSTERIOR q_φ(ε_t | u_t, y) (same transition module, separate SwiGLU heads for μ, σ). Decoder reads only h.
Noise only at the high level ("noise in the low-level state did not improve performance").
(b) Why passes count: each pass is a refinement step of a latent trajectory; stochasticity turns a single deterministic attractor
into a distribution over trajectories (multi-hypothesis), trained by amortised VI; posterior sees y so the prior learns to steer.
(c) Sudoku-Extreme 97.0 (TRM 87.4, HRM 55.0, Looped TF 61.3); ARC-1 52.0 (TRM 44.6); ARC-2 11.1 (TRM 7.8), 10M params. N-Queens 8×8
acc 99.7 / coverage 90.3 (TRM 66.8 / 36.1; AR 96.3 / 84.8; MDLM 96.1 / 87.2). Graph colouring conflicts 2.7 / 3.3 (AR 19.0 / 61.3).
Parallel width: N = 20 samples at 16 iterations beats TRM at 320 iterations (97.0 vs 90.5). Ablation (Table 3b): w/o stochastic
guidance 82.87 / 72.91; stochasticity only N(0, σ²) 94.88 Sudoku but 50.27 N-Queens; GUIDE ONLY (σ = 0, deterministic
target-conditioned) 0.0 / 0.0 ("severe overfitting" — the posterior leaks y); TRM + stochastic decoding 82.87 / 71.66; TRM + random
z_0 78.53 / 71.82 → naive randomness does NOT help; the variational posterior/prior pair does. Cumulative (3a): Looped TF 61.25 →
+SG 65.64 → +DS+SG 73.90 → +DS+HR+SG 93.96 (Sudoku). Hyper: AdamW 1e-4, wd 1.0, EMA 0.9999, KL balance 0.8, β 0.04–0.5 (task-tuned),
N_sup 16. ELBO check (A.3): the truncated surrogate and the full ELBO both fall monotonically.
(d) Admitted: sequential deep supervision limits training efficiency — "a significant barrier to scaling toward larger foundation
models"; β is task-tuned; with heavy augmentation (Aug = 50) the sampling gain vanishes (diversity already internalised);
LPRM/ACT heads add machinery. Puzzle regime only; no LM result.
(e) For us: (1) OUR TARGET IS MULTIMODAL. The core predicts the next chunk's concept/PQ code — many valid continuations exist,
GRAM's multi-solution setting where deterministic recursion covers only one solution (their Fig. 4 right / 15: TRM coverage drops as
#solutions grows). Stochastic guidance may therefore help COVERAGE / plan quality (C4). It is NOT a diagnosis of our R-invariance:
GRAM's deterministic RRMs converge to one valid answer among many (a coverage failure), whereas our loop gives the SAME output for
every pass count — the loop contributes nothing (audit item 3, loss placement). Noise does not fix that; see (5). (2) GRAM gives a
PRINCIPLED version of TRAJECTORY randomness: GRAM's ablation shows naive trajectory noise — random z_0 init and stochastic
decoding — does NOT help TRM. GRAM never tested random R: random R is DEPTH randomisation (C2 mechanism (i), per-sequence R sampling,
a settled ingredient) and is untouched by this ablation — a design keeping random R loses no credit here. What helps is a per-pass Gaussian guidance with a TARGET-CONDITIONED POSTERIOR during
training and a learned prior at inference. VERIFIED: `cdoc` in ConceptCoreV2.forward is the per-chunk DOCUMENT ID (doc[:, ::4],
used only for masking) — NOT a target, so it cannot feed the posterior. The posterior must condition on the NCP target, i.e. the next
chunk's PQ code n[:, 1:] (or sg(c_hat[:, 1:])). LEAK WARNING: that code is computed from future tokens the decoder must predict, so
the posterior path is a Free-Transformer-style latent — only the KL bottleneck (with free bits) limits leakage, and train/infer
mismatch (posterior vs prior) is a real C5 risk. (3) Guardrails:
σ must stay > 0 (σ = 0 posterior = target leak → 0 %); KL-balance 0.8 and β ~0.05–0.1; noise only on the slow/high state (our S),
never on the read path the decoder uses directly; at eval I PROPOSE the prior MEAN (deterministic) or few samples — UNTESTED:
every GRAM result samples, and the σ = 0 ablation is a TRAINING-time leak, not evidence about mean-decoding a σ > 0 model; C5 safety
of stochastic guidance is a hypothesis until the per-pass eval shows it. (4) C1
grounding: the posterior conditioned on the true next-chunk code is a grounded intermediate target — BUT GRAM applies the KL only
at the final transition of each supervision step, so per-pass grounding requires each pass to BE a supervision step with a detached
carry (mechanism (ii)). Credit C1 and C2 from that one structural choice once, not twice. (5) Depth extrapolation came from detached deep
supervision, not from the noise — keep rule (ii) as the C2 mechanism and treat stochastic guidance as a C3/C4 add-on.
(6) Cost: μ/σ heads are two small SwiGLUs (~2·3·d·d_ff/4 params); the KL term is cheap; training stays one pass per step.
(f) h_t = u_t + ε_t, ε_t ~ N(μ_θ(u_t), σ_θ²(u_t)) (prior) / q_φ(ε_t | u_t, y) (posterior);
L^(n) = E_q[log p(y | z_T^(n))] − KL(q_φ(ε_T^(n) | u_T^(n), y) ‖ p_θ(ε_T^(n) | u_T^(n))), gradient through the last transition only.

## 24. On Learning to Think (Schmidhuber, IDSIA technical report, arXiv 1511.09249, 30 Nov 2015) — read in full (15 pp body; bibliography skipped)
TAXONOMY: not a looped-transformer paper; NO EXPERIMENTS ("Experimental results will be described in separate papers"). Conceptual
framing only → zero empirical weight for C1/C2/C5; cited for design vocabulary.
(a) RNNAI = a small RL controller C plus a large predictive RNN world model M. M learns to compress the entire lifelong history
("all data is holy") by predictive coding under an MDL objective (bits_M + bits_H: weight-description bits + bits for prediction
errors; grow/prune units whenever that lowers the total). Sec. 5.3 (the main contribution): add learnable C→M and M→C connections
(owned by C); M's weights frozen; C "thinks" by feeding activation sequences into M and reading activations out, before and while
acting. A special gate unit û lets C feed û(t)·all(t) into M — setting û = 0 makes M ignore the environment and be used purely for
"thinking". Training alternates (Algorithm 1): freeze M, train C on new trials; unfreeze M and retrain it in a "sleep phase".
Sec. 4.3: M may predict the COMPRESSED CODES of an FNN/autoencoder preprocessor rather than raw inputs. Sec. 6: intrinsic reward for
experiments that improve M's compression (curiosity / PowerPlay).
(b) Why "thinking steps" count (AIT argument, Sec. 2.1): if the mutual algorithmic information K(q | p) between C's task solution and
M's program is high, C's search for a short program w̄ that exploits M is far smaller than learning from scratch; C can short-cut to
M's abstract units ("hierarchical planning over a few relevant abstractions, ignoring irrelevant spatio-temporal details") instead of
simulating futures step by step.
(c) Numbers: none new. Recalled prior result: self-invented recursive code sped up program search ~1000× (OOPS).
(d) Admitted: C may find M hard to exploit and learn to ignore it (zero connections); step-by-step simulation only works in
(near-)deterministic worlds; M only approximates conditional expectations in noisy worlds (→ multimodality problem, cf. GRAM).
(e) For us (vocabulary, not evidence): (1) Our architecture already IS this pattern: a small concept core (C) reading and writing
the large encoder representation X = c_hat (M) through learnable interfaces (GR read/write over [X, S]); "M predicts compressed
codes of a preprocessor" = our NCP on PQ codes. (2) The û GATE maps to an input-injection gate on the per-pass read of X: letting some
passes run with the read of X gated off ("pure thinking" passes) is a cheap, zero-init-able design knob — no empirical backing here;
it would need the per-pass eval. (3) Plan over abstractions (chunk concepts) rather than token-level rollouts — consistent with the
middle-span, chunk-level core. (4) Alternating/frozen training suggests one C5-safe experiment path: train a new core with the
encoder/decoder frozen first, then unfreeze (cf. the Nanbeige/ELT warm-start caveats). (5) Depth vs width: NOT separated; no LM
result → the C5 evidence gap the advisor flagged stays open after this paper.
(f) C(t+1) = f_C(w_C, C(t), M(t), sense(t), w_M) (w_C learnable), M(t+1) = f_M(C(t), M(t), w_M) (w_M frozen); M input = û(t)·all(t);
M objective min bits_M + bits_H.

## 25. Sparse Layers are Critical to Scaling Looped Language Models (Lee, Biloki, Hu, May; USC-ISI/Netflix; arXiv 2605.09165v2, 30 Jun 2026) — USER-ADDED paper #42 (root folder); read in full (15 pp; bibliography skipped)
TAXONOMY: fixed R (8×2 in the scaling study; 4×4 and 2×8 at one budget), final-pass LM loss only, no depth extrapolation, no
per-pass loss; early exit is TRAINING-FREE (entropy threshold at loop boundaries). Not a C2 paper; it is the cleanest
LM-REGIME, ISO-FLOP C3/C5 evidence in the set.
(a) Controlled 2×2: {dense, MoE FFN} × {16 unique layers, 8 layers × 2 loops}, all with equal N_active (C = 6·N_active·D;
looped layers counted per invocation, only k active experts counted). MoE = top-2 of E = 8 SwiGLU experts (each d_ff/k wide so
active FFN = dense FFN), token-choice routing, load-balance loss + router z-loss. µP extended to router/expert/unembedding
(init variance and LR ∝ 1/w_ratio; looped layers need no change). FineWeb-Edu 10B, GPT-2 tokenizer, WSD with sqrt cooldown; isoFLOP
budgets 5e16–1e18; widths 128–1024 (≤ 305M active / 711M stored). Pre-RMSNorm without gain, RoPE, SwiGLU.
(b) Why passes count: a dense FFN applies the SAME operation to every token on every pass — "an invariance that compounds when
looped". MoE routing is input- and state-dependent, so the same physical layer picks DIFFERENT experts on pass 2 → depth-unique
computation from shared weights.
(c) Scaling: Looped-MoE beats Base; dense Looped is STRICTLY worse than Base; MoE (non-looped) best on loss; all exponents ≈ 0.076–0.077
→ the architectures differ by OFFSET, not slope. Downstream (Core 9 @1e18): Looped-MoE 39.6 (216M stored) > Base 38.7 (246M) > Looped
37.4 (168M) > MoE 36.4 (366M) — the spread is dominated by BoolQ (63.9 vs 51.4 vs 49.6 vs 39.0); other tasks move ≤ 2–3 pts → weak.
Routing divergence (Fig. 5, 8×2, k 2/8): layers 1–6, 8: 25–53 % of tokens get fully DISJOINT expert sets on pass 2, only 4–14 %
identical, most share exactly one expert; layer 7 (last before the loop boundary) is the outlier, 37 % identical / 3 % disjoint
(a "stabilising" loop-invariant layer). Looped-MoE tokens reach 3–4 unique experts per physical layer across loops.
LOOP SHAPE at fixed width (d 704, 16 effective layers), full-depth test ppl: Looped-MoE 8×2 35.9, 4×4 35.4 (BEST), 2×8 37.3; Base 34.8,
dense Looped 8×2 36.2. Early exit (training-free entropy, Table 7), ppl at 10 % FLOPs saved: Base 55.4, MoE 75.7, Looped 50.2,
Looped-MoE 51.0, 4×4 44.3, 2×8 42.0; JSD to final: looped models jump toward the final distribution AT LOOP BOUNDARIES (Fig. 6),
"architectural, not learned": each loop ends with the output-facing layers.
(d) Admitted: µP not transferred across depth (width-only scaling); ≤ 305M active; single budget for the loop-shape study; no
measured throughput. Not admitted but material: at equal WIDTH Looped-MoE stores MORE non-embedding params than Base (8·(A + 4F)
vs 16·(A + F); Table 5 at d 256: 45M stored vs 39M active incl. embeddings) — "without additional parameters" is true only vs
dense Looped at its own width; the Base-vs-Looped-MoE stored comparison (216M vs 246M) is between DIFFERENT compute-optimal widths.
Part of the MoE gain is stored capacity (consistent with SMELT/MoR: the LM regime values unique params).
(e) For us: (1) DIAGNOSIS SUPPORT: our core's dense FFN (and attention) apply one fixed map every pass; the paper names this
invariance as the looped bottleneck. Our R-invariance (step angle 6.7° → 0.1°) is consistent with it — but careful: the paper shows
a LOSS-OFFSET penalty, not R-invariance; loss placement (audit item 3) remains the primary cause. (2) STATE-CONDITIONED
PER-PASS COMPUTATION: an MoE FFN in the core gives each pass different experts WITHOUT a pass-index embedding. Because routing
depends on the state, not on r, it is defined for any R — a principled reason it could extrapolate to R beyond training where a
learned per-pass embedding would not (hypothesis; the paper never tests R > trained). (3) LOOP SHAPE: 4 layers × 4 loops was the
best Looped-MoE shape at fixed compute (35.4 vs 35.9 for 8×2, 37.3 for 2×8) — first LM-regime evidence for exactly our
"4 distinct layers looped ~4×" config (single run, one budget). (4) PER-PASS READOUT FOR FREE: since every pass ends with the same
layers + the same readout, per-pass NCP / logit-lens readouts are well-formed without extra training → cheap per-pass eval and a
natural grounded per-pass target (C1) and exit point. (5) SIZING at d 256 / FFN 1024: E = 8 experts × d_ff 256 (k = 2 → active 512,
HALF the dense FLOPs) or E = 8 × 512 with k = 2 (active 1024 = dense FLOPs; stored FFN 4×: 3·256·512·8 = 3.15M per layer vs 0.79M
dense; ×4 layers = 12.6M vs 3.1M). A cheaper variant: E = 4 × 512, k = 2 (stored 2×, 6 expert pairs). Data per expert: the core sees
T/4 = 256 chunk positions per sequence, so each expert gets ~k/E of those; load-balance + z-loss are mandatory. (6) C5: in the LM
regime at equal compute, dense looping costs LM loss (8×2 36.2 vs Base 34.8 ppl) while Looped-MoE nearly closes it (35.4–35.9) —
the most direct C5 evidence so far: a looped core should not use a plain dense FFN if it DISPLACES unique layers. Our core ADDS
compute on top of the encoder/decoder rather than displacing it, so the comparison applies only partially.
(f) T = top-k(W_router x); p_i = softmax over T; y = Σ_{i∈T} p_i FFN_i(x); L_LB = E·Σ_i f_i p_i; L_RZ = mean_j (log Σ_i exp h_i(x_j))².

## 26. SMT — Pretraining Recurrent Networks without Recurrence (Kumar, Isola; MIT; arXiv 2606.06479v2) — read in full (30 pp; bibliography skipped)
(Scratchpad also held a duplicate extract "Pretraining_Recurrent_Networks.c.txt" — same paper, one PDF in the folder.)
TAXONOMY: TEMPORAL recurrence (an RNN over time steps), not depth recurrence — the axis we rejected for the core. "Without
recurrence" is a TRAINING-TIME claim: the RNN is never unrolled during SMT pretraining; an on-policy fine-tune (DMT) unrolls it
afterwards. It is NOT the "warm-start a recurrent core from a non-recurrent stack, watch LM loss at the transition" study → the
C5 warm-start evidence gap stays open.
(a) Teacher: bidirectional Transformer encoder E_φ compresses the past x_{≤t} into M = 16 memory register tokens m_t (a predictive
state), a causal decoder D_ψ predicts the future y_{t..t+T_f} from m_t + teacher-forced future inputs. Student RNN f_θ (8-layer
bidirectional Transformer over [memory tokens, x_{t+1}], output RMSNorm; also MLP and GRU variants) learns the ONE-STEP transition
(m_t, x_{t+1}) → m_{t+1} by MSE. Joint loss L = λ_dec L_dec + λ_dyn L_dyn + λ_unif L_unif (uniformity loss against memory collapse),
one random t per sequence. DMT: freeze E/D, unroll the RNN on its own memories, regress m̂_t → m_t (on-policy imitation, DAgger);
readout initialised from the decoder and finetuned on RNN memories. Defaults: d 256, T_c 256, T_f 64, λ_dec 1, λ_dyn 0.1, λ_unif 0.001.
(b) Why it works: decouples WHAT to remember (learned time-parallel by the teacher, O(1) credit path) from HOW to update (one-step
supervised); L_dyn also shapes the teacher's memory to be Markovian.
(c) Synthetic (retrieval, copy, stack, KV, modular ICL): SMT→DMT beats BPTT in every setting (associative recall still unsolved).
TinyStories (char-level): about the SAME data efficiency as BPTT (language needs little long-range memory); MNIST pixel sequences:
much better. GRU backbone: SMT FAILS (memory-space collapse). Joint vs detached (Fig. 10): with T_f = 1 the DETACHED variant fails the
needle task but JOINT training solves it — "credit is amortised into gradient optimisation steps", one step back per update (like
value bootstrapping). λ sweep: best λ_dyn 0.1, λ_unif 0.001; λ_unif = 0 keeps RNN loss but COLLAPSES the memory space. DMT: lower
equilibrium drift and better loss across runs; one-step error only partially predicts rollout drift (corr 0.45–0.50). Length
generalisation (stack tracking): the RNN student beats its Transformer teacher beyond the training length.
(d) Admitted: the teacher is time-parallel (TC0-limited) → the student inherits limits and needs BPTT/post-training to exceed it;
"not necessarily for learning reasoning, since intermediate steps are not supervised"; DMT is sequential; training all m_t gave no
gain at their scale.
(e) For us (transfer from time → depth, by analogy only): (1) CREDIT AMORTISATION across optimisation steps is a mechanistic reason
why DETACHED-CARRY deep supervision (rule (ii): TRM/GRAM/URM) can learn long pass chains without full BPTT — the same bootstrapping
argument, now with an ablation behind it (joint > detached at T_f = 1). (2) UNIFORMITY LOSS (λ ≈ 1e-3) on the looped state: cheap
guard against the S-space collapsing to a few directions (our "u carries ≈ one answer token", per-pass effective rank) — a C4/C6
ingredient; λ = 0 left the loss unchanged but the space collapsed, so it must be monitored with a rank metric, not loss. (3) DRIFT
LESSON: supervising one pass from teacher states (off-policy) drifts at rollout; train on the model's OWN pass-r state (on-policy),
which detached-carry deep supervision already does. (4) Our code already carries a related hook: HRMThinker(nextlat=…) — SMT names
NextLat (Teoh et al.) as the closest method. CHECKED (code/hrm_decoder.py:225): our NextLat is
training-only TOKEN-level latent dynamics on the DECODER state h (ĥ_{t+1} = h_t + f(LN[h_t; rms(E x_{t+1})]), SmoothL1 to sg h_{t+1} +
KL through the frozen head), horizon 1 — it does not touch the concept core, so an SMT-style one-step target inside the core would be new. (5) Memory as M register
tokens updated by a full-attention block over [memory, input] ≈ our S-space read/write — supports keeping S as a small set of slots.
(f) L_dyn = MSE(f_θ(m_t, x_{t+1}), m_{t+1}); L_unif = log E_{a,b} exp(−2‖m_a − m_b‖²); L_dmt = E_t MSE(m̂_t, m_t) with m̂ from the RNN's own rollout.

## 27. MoDA — Mixture-of-Depths Attention (Zhu, Fang, Liao, … Wang; HUST/ByteDance Seed; arXiv 2603.15619) — read in full (18 pp; bibliography skipped; no appendix)
TAXONOMY: not looped — a DEPTH-STREAM / residual-family primitive (same family as the Qwen gated residual, Delta Attention Residual
and Attention Residuals the user named). No R, no C2 content. LM-regime evidence (OLMo2 recipe, 400B tokens).
(a) "Read, operate, write" lens on stacking: depth residual (read identity, write add → the stream is repeatedly superposed →
"information dilution"), depth dense (DenseNet: lossless but O(L²D²)), depth attention (read earlier layers by attention over the
same token's per-layer KV), and MoDA: each head attends JOINTLY — one softmax — over the sequence KV of the current layer AND the
depth KV {K_i, V_i}_{i<l} of the SAME token position from all preceding layers. Write: the attention layer appends its K/V to the
depth stream; the FFN appends K/V via a light extra projection of its input. Query projection reused → params O(L·D²/G). Fused
Triton kernel (flash-compatible depth layout, chunk-aware, group-aware indexing) = 97.3 % of FA2 at 64K; ~1458× over naive torch.
(b) Why depth retrieval helps: features formed early are diluted by repeated residual updates; data-dependent retrieval of earlier
layers' K/V recovers them. Attention maps put persistent mass on depth KV in middle/late layers and LESS mass on attention sinks.
(c) 700M (Table 3, C4 val ppl / downstream avg): OLMo2 36L 18.59 / 56.93; +2 layers 18.31 / 57.11; depth KV reusing the sequence KV
(ZERO params, +0.12 % FLOPs) 18.48 / 58.10; + FFN KV proj (default) 18.21 / 58.87; + extra attn KV proj 18.17 / 58.97 (saturated, not
worth +37M params). 1.5B: downstream 62.28 → 64.39, avg per-domain ppl 13.67 → 13.47 (all 10 domains). SMALL-MODEL ablation
(Table 6, width 384, 6 q-heads, 2 kv-heads = OUR width config; FineWeb-Edu val loss): 48L pre-norm 3.3800 → depth KV 3.3759 → +FFN KV
3.3656; 48L POST-norm 3.4062 → 3.3653 → 3.3484 (best); 24L post-norm 3.4740 → 3.4537 → 3.4338. Pre-norm gain from depth KV alone is
tiny (−0.004); post-norm gain is 10× larger (−0.041) — MoDA makes post-norm the better choice in deep stacks.
Kernel overhead grows with L (+8.6 % at L 64 → +30.5 % at L 256, T 16K).
(d) Admitted: full depth-KV cache grows linearly with depth → proposes a bounded depth-KV SLOT buffer (top-S by utility or sliding
window); sink-pattern change not understood. Not admitted: pre-norm gains are small at 24–48 layers; downstream gains lean on
COPA/WinoGrande (small, noisy sets).
(e) For us: (1) A LOOPED core has an unusually long depth stream per chunk position — 4 layers × R passes (32 entries at R = 8) — and
our core's S is repeatedly overwritten by GR writes (dilution by construction). "Depth KV across PASSES" (attend to your own
earlier-pass K/V at the same chunk position, jointly with sequence attention) is a natural, near-free way to let pass r retrieve
pass-1 features: zero extra params when reusing the sequence K/V, trivial cost at T/4 = 256 chunk positions (implementable in plain
PyTorch: per-position depth logits [B,H,T,4R] concatenated to the sequence logits before one softmax). (2) R-EXTRAPOLATION RISK: the
depth set grows with R, so R > R_train shows the softmax more keys than trained → use the paper's own bounded SLOT buffer (last S
passes or top-S) to keep the key count fixed. (3) C6: post-norm + depth KV beat pre-norm at depth — a post-norm (or exit-norm)
looped core with depth KV bounds the state norm (audit item 4) while keeping a gradient highway. Evidence is non-looped; transfer
is an extrapolation. (4) Candidate for the "delta attention residual / attention residual" slot of the user's brief — compare with
those two papers at the design phase. Small-model evidence at OUR width shows the effect is real but small in pre-norm (−0.004 loss).
(f) o_t = softmax([q_t K_seq^⊤ ; q_t K_depth,t^⊤] / √d + M) · [V_seq ; V_depth,t], K_depth,t = {K_i,t}_{i<l} (same position t).

## 28. Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning (Drozdova, Sirbu, … Niklasson, Richards; Google PoI / Mila; arXiv 2609.01449) — read in full (15 pp; bibliography skipped)
(Scratchpad duplicate "Diffusion_Timestep.c.txt" = same paper; one PDF in the folder.)
TAXONOMY: PER-STEP supervision (masked MSE on EVERY outer step, equal weight) + TRUNCATED BPTT over L = 4 outer steps + DETACHED
carry between segments + RANDOM rollout length per sample (K ~ U[20, 160]) + staggered persistent rollouts (a rollout's hidden state
survives across optimisation steps in a pool, reset only when its episode ends) + NO timestep / pass-index conditioning.
EXTRAPOLATES FURTHEST in the whole set: trained on K ≤ 160 with 4-step gradient windows, solve rate rises MONOTONICALLY to
K = 10,000 (60× the longest training rollout) → strongest confirmation of rule (ii) (+ random-length (i)).
(a) Denoiser = looped pre-LN Transformer block (J = 8 inner loops, H 128, 4 heads, row/col/box-masked attention for Sudoku; a local
3×3 conv NCA for mazes). Each outer step k: u = E_x(Pin(x_k)) + pos + clue flag + h_k → J shared blocks → z_k; TWO separately
normalised views: h_{k+1} = LN_h(z_k) (carried memory, never supervised directly) and ŷ_k = E_o(LN_o(z_k)) (prediction). Training:
x_k = fresh VP corruption of the TARGET at annealed level t_k (teacher forcing); inference: x_{k+1} = Pin(a(t)ŷ_k + b(t)ε).
(b) Why passes count: the input channel is re-corrupted every step, so the persistent h is the ONLY clean path from earlier steps →
computation must accumulate in h; the annealed corruption provides a graded per-step curriculum.
(c) Sudoku-Extreme 99.90 % at K = 10,000 (constant max noise at inference); annealed inference 99.56; NO inference noise (t = 1)
49.39 → stochastic forcing at inference doubles the solve rate. Maze-Unique 98.93 % (K 100); Maze-Hard exact 77.87 / optimal 86.43
(TRM exact 85.3). Time conditioning: no significant effect. Memoryless (no h): slower and strictly worse. Training regime (Table 2,
K 10k): annealed + teacher forcing 99.54; fixed max noise 82.73; FREE-RUNNING (own prediction re-noised) 80.50; i.i.d. noise level
per step 20.59; clean-only 0.00. BPTT window (Fig. 3): L = 1 fails (no cross-step propagation), L = 16 vanishing gradients, L = 4 best.
Inner loops J: too few or too many worse, J = 8 best. Memory width 64 → 256 helps at every K (diminishing). Interventions at step 200
of 400 (annealed): reset memory 82.50 → 58.03; shuffle memory across cells 29.38; shuffle all values 0.27 → h carries instance- and
position-specific progress. SETTLING (Table 5): trained+tested on unique-solution boards 99.24 % settle-and-stay; trained on
UNDER-DETERMINED boards: 57.93 % settle-then-WANDER (88.94 % when tested on fully-determined boards).
(d) Admitted: unique-target benchmarks; the under-determined result does not isolate solution multiplicity (exploratory); Maze-Hard
exact below TRM. Not admitted: tiny models, puzzle regime, no language.
(e) For us: (1) C2 RECIPE, strongest evidence yet: shared update with NO pass index + normalised carried state (LN_h) + per-pass
loss + BPTT over short 4-pass groups with DETACHED carry + per-sample random pass count → depth extrapolation far past training.
Directly supports grouped/truncated BPTT with group size ~4 (L = 1 and L = 16 both worse) — matches URM (first ~25 % no-grad) and
HRM-Text (K 2 → 5). (2) DUAL NORMALISED VIEWS: carry h = LN_h(z), output ŷ = E_o(LN_o(z)) with separate norms — maps to our S (carry)
vs u (decoder read): give the carry its OWN norm (bounds its norm → audit item 4 / C6) and the readout its own norm. (3) The user's
"random recurrence" in its best-evidenced form: random rollout LENGTH per sample (K ~ U[20, 160]) + no timestep → the update must
be useful at any depth. (4) NOISE: the escape mechanism is noise in the OBSERVABLE/input channel with a CLEAN memory (contrast GRAM:
noise in the latent via a learned prior). Inference with max noise beat no-noise 99.90 vs 49.39. (5) CAUTION FOR LM (multimodal
next-concept target): trained on under-determined targets, trajectories settle then WANDER (57.93 %) — more passes on a multimodal
target need not converge; any R-extrapolation claim for our NCP target must be MEASURED, and the decoder read should come from a
stable readout (e.g. averaged or confidence-gated), not assume a fixed point. (6) A "concept-diffusion core" (feed a corrupted
next-chunk concept as an input channel, h carries reasoning) inherits a TRAIN/INFER LEAK: with teacher forcing the core sees the
noised target, and the decoder reading u would exploit it. Leak-free variant = FREE-RUNNING (feed the core's own re-noised previous
prediction): 80.50 vs 99.54 on Sudoku — much worse but no leak. Treat as a conditional/high-risk design idea (C5), not core.
(f) h_{k+1} = LN_h(z_k), ŷ_k = E_o(LN_o(z_k)), z_k = B_θ^J(E_x x_k + p + m⊙g + h_k); L = (1/L) Σ_{k<L} masked‖ŷ_k − y‖²; detach h every L = 4.

## 29. CALM — Continuous Autoregressive Language Models (Shao, Li, Meng, Zhou; WeChat AI; arXiv 2510.27688) — read in full (33 pp; bibliography skipped; appendix = sampling proofs: Thm 1 read, Thms 2–3 skimmed)
TAXONOMY: NOT looped. Relevant as the closest LM-regime analogue of OUR concept pipeline: compress a chunk of K = 4 tokens into
one vector and predict the NEXT VECTOR (our chunk = 4 → c_hat → PQ code → NCP). Informs the concept TARGET space (C4/C5) and the
multimodality question raised by GRAM / Diffusion-Timestep.
(a) Context-free chunk autoencoder (K tokens → FFN → flatten → linear → FFN → l-dim z; decoder mirrors; ~75M params, d 512,
reconstruction > 99.9 %). A reconstruction-only latent is BRITTLE ("practically impossible" to model downstream) → robust latent via
VAE (β = 0.001) + per-dimension KL CLIPPING (floor 0.5; without it 71/128 dims collapsed) + dropout on z (0.15) + token dropout
(0.15); l = 128, posterior σ ≈ 0.3 yet still > 99.9 % reconstruction. Backbone = LLaMA-style Transformer over T/K steps; output
= a single-step ENERGY-SCORE generative head (L residual MLP blocks refining uniform noise conditioned on h; ~10 % of params) trained
with the energy score (strictly proper, likelihood-free; N = 8 model samples, M = 100 target samples from the AE posterior, α = 1).
Input: the previous chunk's K DISCRETE tokens embedded + a 2-layer compression MLP. Evaluation: BrierLM (unbiased from 2 samples;
Pearson −0.966 with CE); temperature by rejection / batch approximation.
(b) Why it works: raising semantic bandwidth per step cuts AR steps K-fold; the robust smooth latent makes next-vector prediction
learnable; the energy score handles a multimodal continuous target without a likelihood or an iterative sampler.
(c) Pile, Llama-3 tokenizer, ~230B tokens. CALM-M (K 4) 371M: BrierLM 5.72 vs Transformer-S 281M 6.05 at 44 % fewer train FLOPs;
CALM-XL 1.82B 8.53 vs Transformer-L 849M 8.98 at ~13 % fewer train FLOPs. K sweep (CALM-L): K = 1 worse than a discrete LM at more
FLOPs; K = 2 halves cost for a small drop; K = 4 beats the baseline frontier; K = 8 drops (capacity). AE ablation (BrierLM): recon-only
3.99; + KL 3.48 (collapse); + KL clip 4.13; + both dropouts 4.70; β 0.1 breaks reconstruction (~99 %); l = 128 best of 32–256; scaling
the AE does not help. Heads: energy > flow matching > diffusion (diffusion needs many steps; flow midpoint ~4). N 2→12: 4.37 → 4.72;
α < 1 fails, α = 2 collapses to 0. INPUT (Table 5): discrete-token input 4.70 vs CONTINUOUS latent input 3.25 vs both 4.40 — "compact
and brittle" vectors are hard to unpack as INPUT.
(d) Admitted: K = 1 gap vs discrete LM; AE focuses on reconstruction not semantics; no RL/KD toolkit for likelihood-free models;
temperature sampling needs many samples at low T.
(e) For us: (1) VALIDATES our chunk size: K = 4 is CALM's sweet spot (K = 8 capacity-limited at ≤ 1.8B). (2) TARGET SPACE: CALM shows a
deterministic reconstruction latent is too brittle to be a prediction target; our NCP predicts DISCRETE PQ codes with CE — CE over a
codebook is already strictly proper and multimodal, so we sidestep CALM's brittleness problem. Keep a discrete (PQ) target for the
core; a continuous-target core would need CALM's VAE + KL-clip + dropout recipe. (3) INPUT CAUTION for looped designs: feeding the
core's OWN compact predicted vector back as input (e.g. a "concept-diffusion" observable channel, or looping over predicted codes)
is the weak option (3.25 vs 4.70); keep the loop grounded in rich encoder states (X = c_hat) and carry S, not the predicted code.
(4) MULTIMODALITY, leak-free: an ENERGY-SCORE head trains a sampler to match the next-concept distribution from model samples ONLY —
no target-conditioned posterior (contrast GRAM's q(ε | u, y), which leaks future tokens into the decoder path). If a design wants a
stochastic plan, the energy score is the leak-free way; cost N× head evaluations. (5) Note the multimodality location: our NCP
softmax is proper per codebook, but the plan vector u handed to the decoder is ONE deterministic vector (a "mean" of modes), and the
PQ codebooks are predicted independently (factorised — cannot represent correlated modes across codebooks). That — not the loop —
is where a stochastic/energy design would add C4 value.
(f) L_energy = Σ_i [ (2/NM) Σ_{n,m} ‖z_{i,m} − z̃_{i,n}‖ − (1/(N(N−1))) Σ_{n≠k} ‖z̃_{i,n} − z̃_{i,k}‖ ]; Brier ≈ 1{x₁=y} + 1{x₂=y} − 1{x₁=x₂}.

## 30. Intern-S2-Mobius — Foundation Model with Decoupled Knowledge and Reasoning (Intern-S2-Mobius Team, Shanghai AI Lab; arXiv 2608.14290) — read in full (bibliography skipped)
(Scratchpad duplicate "Intern_S2_Mobius.c.txt" = same paper; one PDF.)
TAXONOMY: claims "dynamic latent reasoning" (iterating latents over a few layers, decoding several tokens at once) but gives NO
loop count, NO R sweep, NO equations for the iteration, NO ablation → C2 content nil; low C1 weight (technical-report style,
confounded comparisons).
(a) Mobius-v0: the FFNs of all layers are pulled out of their layers and concatenated horizontally into ONE globally shared
knowledge-vector memory (block-partitioned, MoE-style sparse activation at scale); each layer keeps its own self-attention
("Reasoner"); hidden states are the cache/carrier. Every layer can read all knowledge → "backward residual connection" (deep states
reach knowledge that used to live in shallow FFNs and vice versa). Multi-token (MTP-style) decoding from refined latents.
(b) Why (claimed): FFN = key-value knowledge store, attention = composition; cross-layer redundancy of stored knowledge is removed;
reasoning compresses into dense latents instead of long CoT.
(c) Train-from-scratch 7B-A1B MoE, 1T tokens: same MMLU as a Transformer with 0.626× the data ("1.6× data efficiency"; one curve,
no seeds). CPT from Qwen3.5-35B-A3B (1T tokens) + SFT + RL: MMLU-Pro 89.05 vs 85.31, AMO 58 vs 50, SimpleQA 28.9 vs 21.4, but UGD-hard
73.0 vs 78.0 and HLE 19.1 vs 22.4 (worse); ~4× request throughput, mainly from 1.2–5× SHORTER outputs. Appendix: expert routing in
the CPT model keeps a block-diagonal (inherited layer-wise) pattern, the from-scratch model routes more uniformly; a two-hop
compositional task (Physics of LLMs: train 500 one-hop + 400 two-hop entities, test 100 held-out two-hop) favours "a yet-to-be-released
newer Mobius"; layer-lens shows more target-aligned intermediate predictions (one example).
(d) Admitted: mechanism of data efficiency and of shorter CoT "remains to be fully explored"; whether it "hacks" or truly reasons
"remains to be determined" (argued from case studies). Not admitted: the CPT comparison mixes architecture change with 1T extra
tokens + SFT + RL; the compositional result is from an unreleased model.
(e) For us: (1) A concrete, cheap variant of the Looped-MoE ingredient (#25): ONE expert pool SHARED by all 4 core layers and all
passes, with per-layer attention kept distinct — every (layer, pass) routes into the same pool. At d 256: a 16 × 512 pool (k = 2) stores
6.3M FFN params once (vs 12.6M for 4 layers × 8 experts), keeps active FFN = 1024, and gives each layer access to 16 experts; the
router (state-conditioned) still diverges across passes. Evidence for the shared-pool part is WEAK (no ablation). (2) "Distinct
attention (reasoners) + shared FFN (knowledge)" is the mirror image of SMELT's finding (visit 2 keeps Q/K, changes V) — both say
the parts of a looped block need not be shared the same way; a design may share FFN knowledge while keeping per-layer attention.
(3) Nothing here on R-extrapolation, loss placement or stability.

## 31. The Free Transformer (Fleuret; FAIR at Meta; arXiv 2510.17558) — peripheral; read in full (16 pp incl. appendices A–C; bibliography skipped)
TAXONOMY: not looped. A conditional-VAE latent injected MID-STACK in an LM — the LM-regime reference for any design that feeds a
target-conditioned posterior into our core (GRAM #23 idea). Quantifies the LEAK BUDGET.
(a) Decoder of L blocks; after block L/2 a per-token latent Z_t (one-hot of 2^H = 65,536 values, H = 16 bits) enters via a linear
map R added to the KEYS/VALUES of block L/2+1 (queries stay x). Training/prefill: Z from an encoder = the decoder's first half + ONE
extra NON-causal block whose queries are a learned constant ζ (prevents a token-wise copy) + linear readout of H bit-logits → Binary
Mapper (independent bits, gradient pass-through Y + G − sg(G)). Generation: Z uniform (prior 1/2^H). Loss = CE + (1/T) Σ_t
max(0, KL(Q(Z_t|S) ‖ P) − κ) (token-wise FREE BITS). Overhead: 1 block ≈ 3.1–3.6 %.
(b) Why: some structure ("positive vs negative review", "where the target is") is a latent decision; an AR-only model must infer it
post hoc from its own tokens (costly, error-prone, fragile OOD). A latent lets the model decide once.
(c) Synthetic (target letter ×8 at a random position + noise): κ = 1/64 bit ≈ vanilla; 1/8 bit → Z encodes the target POSITION; 1 bit →
position + noise; 8 bits → encodes the whole sequence → wrong generations (leak). LM: κ = 4 bits/token COLLAPSES (the encoder channels
the tokens to predict; baseline CE ≈ 2.59 bits) → downstream collapse; ≤ 2 bits trains fine, CE differences ~0.01. 1.5B/47B tokens: HumanEval+
0.055 → 0.079–0.085, MBPP 0.112 → 0.144–0.152, GSM8K small; many multi-choice tasks ±. 8B/200B: mixed by κ (½ bit best: MMLU +11 %, CSQA
+26 %, others ±). 8B/1T tokens, κ = ½ bit: HumanEval+ +11 %, MBPP +2.8 %, GSM8K +2.8 %, MMLU +5.2 %, CSQA +5.8 %; last-third averages +4–6 %;
others flat. Curves often unstable (encoder/decoder coupling). Hyper-parameters NOT retuned.
(d) Admitted: arbitrary latent form; unstable curves; scale untested. Not admitted: many per-benchmark swings are within noise; no
seeds.
(e) For us: (1) LEAK BUDGET for any posterior/latent that sees future tokens (GRAM posterior on the next-chunk code, or a
concept-diffusion core with teacher forcing): free bits at ~½ bit per TOKEN (≈ 2 bits per 4-token chunk) is the evidenced safe point;
≥ 4 bits/token collapses. Our chunk carries ~4 × 2.2 nats ≈ 12.7 bits of next-token entropy at current NTP 2.2 → a per-chunk latent must
be capped far below that (≤ ~2 bits/chunk) with token-wise free bits, and collapse monitored by a train-CE drop that does not show in
val with the prior. (2) INJECTION SITE: Z enters mid-stack through keys/values only — the same "middle-span" place our core sits;
a stochastic plan could enter the decoder's K/V the same way. (3) Constant-query encoder (ζ) is a cheap trick to stop a token-wise
copy — reusable if a design builds a posterior. (4) Reasoning-leaning gains (code, GSM8K) from a ½-bit latent are the LM-regime
support for "a decision variable before tokens" — the role our concept plan plays deterministically today.
(f) L = CE + (1/T) Σ_t max(0, KL(Q(Z_t | S) ‖ U(2^H)) − κ); Z enters as K,V ← x_{L/2} + W_R Z (Q ← x_{L/2}).

## 32. Unified Latents (UL): How to train your latents (Heek, Hoogeboom, Mensink, Salimans; Google DeepMind; arXiv 2602.17270) — peripheral; read in full (15 pp + appendices A–B; bibliography skipped)
TAXONOMY: not looped; image/video latent diffusion. Relevant only through LATENT BITRATE vs MODELLABILITY (our concept/PQ target).
(a) Deterministic encoder z_clean, forward-noised to a FIXED minimum noise (log-SNR λ(0) = 5, σ₀ ≈ 0.08) that is tied to the diffusion
PRIOR's maximum precision → the KL term becomes a weighted MSE over noise levels and gives a tight, interpretable UPPER BOUND on latent
bits. Diffusion decoder with sigmoid-reweighted ELBO; a loss factor c_lf (1.3–1.7) up-weights the decoder (prevents posterior
collapse). Stage 2: retrain the prior as a base model with sigmoid weighting on frozen latents.
(b) Why: the prior measures and regularises the bits the latent carries; low-bitrate latents are easier to model, high-bitrate ones
reconstruct better — the loss factor sets the trade-off explicitly.
(c) ImageNet-512 gFID 1.4 at lower training FLOPs than SD-latent baselines; Kinetics-600 FVD 1.3 (SOTA). Bitrate sweep (Table 2):
LF 1.3 → 2.1 raises bits/pixel 0.035 → 0.116, PSNR 25.7 → 30.1, rFID 0.79 → 0.27, but SMALL-model gFID worsens 1.42 → 2.38 (medium model
1.37 → 1.58, less sensitive). Channels: insensitive between 16 and 64; ≤ 8 cannot reconstruct. Ablations: no prior gradient (plain KL to
N(0, I)) needs 8 channels and is worse; near-noiseless latents (λ 10) → the prior cannot bound the bitrate, decoder takes everything;
LEARNED encoder variance collapses to 0 and becomes unstable at λ 5 → fixed encoder noise is a stabilising simplification.
(d) Admitted: less-informative latents may just move modelling into the decoder; diffusion decoders are expensive to sample; AE data
confounds comparisons.
(e) For us: (1) BITRATE OF THE CONCEPT TARGET is a design lever: "small models prefer LOW-bitrate latents; larger models tolerate
higher". Our PQ target is high-bitrate: 48 codebooks × log2(128) = 336 bits per 4-token chunk (vs ≈ 12.7 bits of actual next-token
entropy per chunk at NTP 2.2). For a d = 256 core, a lower-bitrate PQ layout (16 × 16 → 112 bits, or 32 × 8 → 224 bits) should be easier
for the core to predict — at the cost of plan detail for the decoder. Pick the layout per design with this trade-off stated (C4).
(2) Fixed (not learned) noise on a latent is the stable way to cap its bits (their learned-variance ablation collapsed) — relevant if a
design injects a stochastic plan (cf. GRAM σ > 0 requirement). (3) No looping content.

## 33. GradCuit — Credit-Assigned Gradient Flow for Test-Time Latent Reasoning (Yu, Shen, Li, … Zheng; BIGAI/PKU; arXiv 2608.02585) — peripheral; read (body + appendix A; bibliography skipped; appendix C extra cases skimmed)
TAXONOMY: not looped, not a training method — TEST-TIME optimisation of inserted latent states in a FROZEN LM.
(a) Insert N latent states (initialised from a textual prefix's hidden states, plus a zero-init trainable offset Δz) at the input of
decoder block l = M/2; the continuation attends to them causally; after greedy generation, a self-verifier gives R ∈ {0, −1} and Δz
is updated by Adam (lr 1e-3) on −R Σ_t log π(x_t | x_<t, z̃); ≤ 10 updates, stop when the verifier accepts.
(b) Why: every continuation token has a differentiable path to every inserted latent through the remaining blocks → direct credit
assignment (vs LatentSeek, which reaches latents only through decoded prefix tokens).
(c) 5 instruct backbones (3B–14B) × GPQA-D / GSM8K / MATH-500 × 2 formats: avg 64.5 % vs CoT 57.9 (+6.6) and best baseline 62.1; LR
robustness std 0.82 vs 1.53 (LatentSeek). RANDOM-WALK updates (Gaussian directions) reach 60.6 % ≈ reward-guided LatentSeek 60.3 %.
Ablation (avg of 15): CoT 60.5 → fixed inserted prefix 62.0 → random latent updates → full GradCuit 66.6. Layer: 25–50 % depth best;
75 % no gain or worse. Gradients concentrate on reasoning-connector tokens (because/therefore/then).
(d) Single seed; self-verifier reward; greedy; one case study.
(e) For us: (1) MID-STACK latents are a good exploration/control space (25–50 % depth) — our concept plan sits exactly there; even
random perturbation of such latents plus a verifier helps, echoing the GRAM / Diffusion-Timestep finding that noise in a
mid-level state enables exploration. (2) Possible INFERENCE-TIME add-on for any design (optimise S or u with a self-consistency or
NCP-confidence reward), not a training-time design element; no C2/C3 evidence for the core itself. Low weight.

## 34. EBM-CoT — Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit CoT (Chen, Cui, Ye, Zhang, Bian, Zhu; Oxford et al.; arXiv 2511.07124) — peripheral; read (body + appendices A–E skimmed for numbers; bibliography skipped)
TAXONOMY: not looped in depth; an ITERATIVE LATENT REFINEMENT at inference (S = 3 Langevin / gradient-descent steps on an energy
over soft thought tokens) with frozen base + assistant LMs.
(a) SoftCoT pipeline: a small assistant LM emits latent thought tokens, a trained projection maps them into the frozen base LM's
embedding space; EBM-CoT adds an MLP energy E_φ(c, l) and refines the projected thoughts by S = 3 Langevin steps
l ← l − η∇_l E + √(2η) ε during training (at inference the appendix says the updates are DETERMINISTIC, no noise). Loss = L_LM (through
the unrolled refinement) + α (hinge ReLU(E(l_neg) − E(l_pos) + m) + λ‖l_pos − l_neg‖²). Only E_φ and the projection train.
(b) Why: AR next-token prediction is a locally normalised EBM; a learned energy over the whole latent thought enforces global
consistency → one chain behaves like a self-consistency ensemble.
(c) LLaMA-3.1-8B: single-chain avg 72.49 vs SoftCoT 70.52 (GSM8K 85.26 vs 81.03), N = 10: 76.60 ≈ SoftCoT 76.4-ish; Qwen3-8B: AQuA 86.02 vs
75.04 (single chain). Consistency rate (pass@1 / pass@N) near 100 %. Thought tokens: more than 4 hurts ("unstable optimisation"); α
best ≈ 0.5 in the sweep (0.1 in main runs); stronger assistants help.
(d) Admitted: Langevin cost; shallow MLP energy. Not admitted: gains vs SoftCoT are small except AQuA; N = 10 numbers nearly tie.
(e) For us: (1) A learned ENERGY over the plan as a refinement operator: each loop pass could be read as a descent step on a learned
energy (cf. EBT, next) — a principled way to make "more passes = lower energy = better plan", i.e. a monotone improvement operator
(the property our loop lacks). (2) Few steps (S = 3) and few latent tokens (≤ 4) were best — longer latent refinement destabilised,
another hint that refinement needs an explicit objective to keep improving with depth. Low direct weight (frozen-LM regime).
(f) l^(s+1) = l^(s) − η ∇_l E_φ(c, l^(s)) + √(2η) ε; L = L_LM + α[ReLU(E(l_ℓ) − E(l_c) + m) + λ‖l_c − l_ℓ‖²].

## 35. EBT — Energy-Based Transformers are Scalable Learners and Thinkers (Gladstone, Nanduru, … Du, Ji, Li, Iqbal; UVA/UIUC/Harvard; arXiv 2507.02092) — peripheral; read body in full + appendices B–E (F–I = cognitive-facet essay, intro and pseudocode, skimmed); bibliography skipped
TAXONOMY: iterative refinement of the PREDICTION (not the hidden state) by gradient descent on a learned scalar energy
E_θ(context, ŷ); training unrolls N steps from noise and puts the loss ONLY at the last step, backprop through the steps (Hessian-vector
products). S2 recipe: RANDOM number of steps (2–3) and RANDOM step size, Langevin noise, replay buffer (simulates longer trajectories),
truncated backprop without detaching between steps. Extrapolates in "thinking" steps at inference (more steps → lower perplexity), which
a Transformer++ cannot do per token.
(a) Decoder-only EBT: observed states z_o plus predicted states z_p (2N tokens), custom attention so each prediction sees the observed
prefix and itself (no leakage); Llama-2 backbone; categorical CE on the final ŷ for text. Bidirectional EBT for denoising.
(b) Why: learning a VERIFIER is easier than learning a generator; prediction = optimisation against the verifier → dynamic compute,
uncertainty (energy level), self-verification (best-of-N by energy).
(c) Pretraining scaling (small models, RedPajama-100B sample, ctx 256, FFN = d): up to 35 % higher scaling RATE than Transformer++ in data,
batch, depth, params, FLOPs, width (higher intercept; FineWeb check holds). Thinking: up to 29 % perplexity improvement with more
forward passes on OOD sets, Transformer++ 0 %; gains grow ~linearly with OOD shift; BoN-5 self-verification gain grows 4–8 % → 10–14 %
with more training data. Ablation (Table 2): removing RANDOM STEP SIZE "nearly eliminates thinking gains"; removing Langevin noise
helps single-path thinking but hurts self-verification; replay buffer and random step count also help. Downstream ppl (Table 3) with
slightly WORSE pretrain ppl (33.43 vs 31.36): GSM8K 43.3 vs 49.6, BB-Math 72.6 vs 79.8, Dyck 125.3 vs 131.5, SQuAD 53.1 vs 52.3. Denoising: beats
DiT with 1 % of the forward passes (PSNR 27.25 vs 26.58 in-dist, 23.29 vs 19.56 OOD). COST: one optimisation step ≈ 3.33× a
Transformer++ step; 2-step training ≈ 6.66× FLOPs.
(d) Admitted: step size / step count hyper-parameters destabilise training; expensive (second order); ≤ 800M; FAILS ON MULTIMODAL
targets ("many modes" → blurred averages, because training makes the landscape convex around each example).
(e) For us: (1) RANDOM DEPTH AS A REGULARISER, strongly evidenced: randomising the number of refinement steps AND the step size was
the single most important ingredient for gains that grow with more steps — direct support for keeping the user's "random
recurrence" (per-sample random R) and adding a random per-pass step scale (e.g. random residual gain on the loop update). (2) The paper
frames looped/recurrent-depth models as IMPLICIT EBMs that amortise the score ∇E but cannot verify: an explicit scalar ENERGY/verifier
head on the core (cheap: one linear readout of S) could supply halting and best-of-R selection even without EBT's second-order
training. (3) Full EBT training is too expensive for us (≥ 3.33× per step; C7) and fails on multimodal targets — our NCP target is
multimodal (same caution as Diffusion-Timestep's wandering and GRAM's coverage). Use as an ingredient source, not a design.
(f) ŷ_{i+1} = ŷ_i − α ∇_ŷ E_θ(x, ŷ_i) (+ η_i Langevin), loss J(ŷ_N, y) backpropagated through the unrolled steps; α and N randomised.

## 36. RecursiveMAS — Recursive Multi-Agent Systems (Yang, Zou, … He, Zou; UIUC/Stanford/NVIDIA/MIT; arXiv 2604.25917) — peripheral; read (body + appendices A, D; prompt templates E and case studies F skimmed; bibliography skipped)
TAXONOMY: system-level recursion over FROZEN LLM agents; only small links train. FULL BPTT through n ≤ 3 rounds, CE only at the final
output; inner links warm-started by a cosine regression of latent thoughts to the input embeddings of the ground-truth text.
(a) Each agent generates m latent thoughts autoregressively (last-layer h fed back through an INNER link R_in(h) = h + W2 GELU(W1 h));
an OUTER link R_out(h) = W3 h + W2 GELU(W1 h) maps one agent's latents into the next agent's input space; the last agent's latents close
the loop back to the first; only the final round decodes text. Two stages: inner-loop (1 − cos(R_in(H), Emb(y))) then outer-loop CE
through all rounds.
(b) Why: latent links avoid the vocabulary bottleneck; Thm 4.1: a text link R_text = W_in softmax(W_out h) has vanishing Jacobian when
tokens are confident (O(ε)), a residual latent link keeps ≈ 1.
(c) r = 1/2/3: accuracy rises steadily (avg +3.4 / +6.0 / +7.2 over text recursion) with 1.2× / 1.9× / 2.4× speedup and 34.6 % / 65.5 % / 75.6 %
fewer tokens; text recursion does NOT improve with r. Train × inference rounds landscape (Fig. 1): more inference rounds help models
trained with fewer rounds, deeper training shifts the whole frontier up. Link ablation: Res + 2-layer best (GPQA 65.3 for Res+1-layer vs
64.5 plain 2-layer). Latent length m saturates ≈ 80. 5 runs.
(d) Admitted: none substantive. Not admitted: n ≤ 3 only; frozen agents; heavy engineering around off-the-shelf models.
(e) For us: (1) RESIDUAL LATENT LINKS: a zero-risk interface between modules is h + small MLP(h) (identity-dominated) — consistent with
our zero-init / identity-at-init audit item 1. (2) A soft-max/text bottleneck between loop steps kills gradients (Thm 4.1); our
loop carries continuous S, which is right; do NOT route the loop through the discrete PQ code (cf. CALM input caution). (3) "Training
depth sets the frontier, inference depth adds on top" — another (weak) data point that R_train matters most. Low weight.

## 37. A Survey on Latent Reasoning (Zhu, Peng, Cheng, Qu, … Eshraghian; M-A-P et al.; arXiv 2507.06203v2) — peripheral; read in full (28 pp; bibliography skipped)
TAXONOMY: survey, no new experiments ("does not offer a direct empirical comparison"). Useful as a map of design TRENDS.
(a) Vertical (activation) recurrence: loop/UT, Coconut/CoTFormer hidden-state feedback, training-induced recurrence (pause/filler/
planning tokens, CODI, CCoT), training strategies (MIDAS progressive stacking, looping-inspired cosine regulariser (Saunshi), stepwise
internalisation, RELAY L = L_ans + λ L_iter). Horizontal (hidden-state) recurrence: linear-state (Mamba-2, GLA, RWKV, DeltaNet…) and
gradient-state (TTT, Titans, Atlas) with the DUALITY: DeltaNet's update S_t = S_{t−1}(I − β k kᵀ) + β k vᵀ equals ONE gradient step on
½‖S k − v‖². Mechanistic section: intermediate layers carry the latent CoT; deep Pre-LN layers degrade (variance growth, rank-1
attention) — "Curse of Depth". Infinite-depth: text diffusion, TTT family, implicit fixed-point RNNs (≤ 16 self-iterations per token).
(b) Claimed trends for looped architectures (Table 1): convergence on PRE/LOOP/CODA; per-iteration input re-injection (Recurrent-Depth
feeds both the prelude output x¹ and the previous state); DEPTH EMBEDDINGS DEPRECATED (UT sinusoidal → CoTFormer learnable → dropped
by Recursive Transformer/AlgoFormer, "tried, dropped" by Recurrent-Depth); dynamic stopping SIMPLIFIED (ACT → router → change-magnitude
early exit max Δh < ε / fixed iterations / fixed-point criterion).
(c) Numbers: only cited ones (e.g. Titans-S ≈ 250M matches 1.3B Transformer on 1-shot recall; implicit RNNs ≤ 16 iterations).
(d) Admitted: no apples-to-apples comparisons; field moving fast.
(e) For us: (1) Our encoder → core → decoder IS the Pre/Loop/Coda consensus (middle-span loop settled). (2) NO PASS-INDEX EMBEDDING:
three independent lines (this survey's trend, Diffusion-Timestep "no timestep needed", Sparse-Layers' state-conditioned routing)
say the pass index is unnecessary or harmful for depth generalisation — any per-pass variation should be STATE-conditioned (MoE
routing, per-pass V from the state) not r-indexed. (3) DELTA-RULE WRITE for the loop state: replace the additive gated write
S ← S + g⊙W(x) by an error-correcting write S ← S + β (v − read(S)) (DeltaNet duality: one gradient step on a regression objective) →
bounded state (audit item 4) and an explicit "improvement operator" reading of each pass. This is the natural home for the user's
"delta attention residual" option — cross-check with the Delta Attention Residual paper at design time. (4) Input re-injection each
pass (our GR read of X) matches Recurrent-Depth. (5) Change-magnitude early exit (Δh < ε) is the simplest halting rule to evaluate
R-adaptivity cheaply.

## 38. Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation (Gladstone, Ji, Du; UIUC/Harvard; arXiv 2607.27372) — peripheral; read body in full + appendices A–C (D experimental details, E related work, F theory skimmed); bibliography skipped
TAXONOMY: not looped; a TRAINING-LOOP factorisation for MULTIMODAL targets: best-of-K ("Forward XM": generate K candidates,
backprop only the one closest to the data) or "Reverse XM" (one generation, closest of K data targets). Handles multimodality
WITHOUT a posterior/encoder (contrast VAE/GRAM/Free Transformer).
(a) L = min_i J(ŷ_i, x) over K candidates from different latents/noise (for MDLM: K learned discrete latent embeddings sampled per
masked position). Theory: smooth large-K Forward XM = maximum likelihood of the K-candidate mixture (mass-covering); Reverse XM =
reverse KL + model entropy (mode-seeking, needs an entropy/coverage term). Each extra Forward candidate costs ≈ 1/3 of a training step
(forward only); memory-saving mode re-forwards the winner.
(b) Why: direct regression has "generative expressivity" 1 (its minimiser is the blurred mean of modes); K candidates let each latent
specialise to a mode; exploration and factored generation (more diffusion steps / AR factorisation) are SUBSTITUTABLE.
(c) Images (RAE recipe): 6.2× sample efficiency, 4.1× FLOP efficiency, near-SOTA unguided FID 1.43; SiT: 2.5× data, +52 % FLOP eff.;
Large + XM-5 beats XLarge (+47 % params) without. FID/FVD improve monotonically with K (e.g. XJumpy FID 63.3 → 53.2 at K 25; FVD 26.9 →
21.2 at K 8). MDLM: better perplexity–entropy frontier at K = 8 (XM-1 MDLM collapses to repeating "the"). Generalisation: min FVD 30.0 vs
37.5 (less overfitting). Gains grow with scale (7 → 36 % with data, 13 → 23 % with size). Minibatch-OT coupling HURTS (FID 46.3 → 54.5).
End-to-end XM policy/world model match Diffusion Policy/Diffuser at 16–256× fewer NFEs.
(d) Admitted: AUTOREGRESSIVE LMs are the hardest case ("injecting a latent is less natural", "less bottlenecked by generative
expressivity") — only "modest" early gains; Forward cost grows with K; Reverse needs coverage tricks; losses not comparable across K.
(e) For us: (1) LEAK-BOUNDED MULTIMODAL PLAN: our plan u is ONE deterministic vector while the next-chunk target is multimodal (→ a
blurred "mean plan"; notes #23, #29). A Forward-XM core: run the core with K different learned latent seeds (e.g. K learned S_0 slots
or K latent embeddings added at pass 1), score each by the NCP loss against the true next-chunk code, backprop only the best into the
core AND the decoder. The selection leaks at most log2 K bits per chunk: K = 4 → ≤ 2 bits/chunk = exactly the Free Transformer's
evidenced safe budget (½ bit/token × 4 tokens). At inference: sample a seed uniformly (Free-Transformer-style) or pick by an energy/
confidence head (EBT). (2) Cost: K core forwards (the core is ≤ 25 % of MACs, so K = 4 adds ≈ +0.75× core-forward cost; decoder runs
once). (3) Caveat: AR-LM gains were only modest in the paper; our decoder is AR — treat as a C4 add-on with a measurement plan, not a
core mechanism. (4) Reinforces a scoring rule: any design that injects randomness should say how it avoids the blurred mean
(GRAM posterior / Free-Transformer latent / XM selection / energy head).

## 39. DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression (DeepSeek-AI tech report) — peripheral; §1–2 (architecture + optimisation) read in full; §3 infra, §4 data/pretraining setups, §5 post-training/RL/evals skimmed (no architecture ablations reported there); bibliography skipped
(Scratchpad duplicate "DeepSeek_V41.c.txt" = same report.)
TAXONOMY: not looped. Production MoE (552B backbone + 196B Engram; 8B active in prefill, 16B in decode; 45T tokens). Relevant for
residual design (mHC), cross-layer K/V sharing, encoder→decoder interface and Engram.
(a) CAUSAL ENCODER–DECODER (CED, after YoCo): 40 layers = 20-layer causal encoder + 20-layer decoder; the decoder's GLOBAL K/V are
projected from the FINAL ENCODER hidden state H_{L/2} with layer-dependent projections (C_l = H_{L/2} W_l^{KV}), while each layer keeps
its own sliding-window K/V (window 128) → prefill ≈ halved. CSA2: global main K/V shared across layer groups (Full / Reindex / Reuse
modes; e.g. encoder groups of 6 layers = 1 Full + 5 Reuse), each layer keeps its own Q and SWA K/V; Hierarchical Sparse Indexer.
mHC (manifold-constrained hyper-connections): n = 4 residual streams X ∈ R^{n×d}, X_{l+1} = B_l X_l + C_l F_l(A_l X_l) with token-wise
coefficients predicted from X; B made doubly-stochastic by 20 Sinkhorn–Knopp iterations; "Single-Pass mHC" shifts A by one block for
kernel fusion ("negligible degradation"). Engram (N-gram orders 2–4, 8 hash heads, ~16M-row prime tables) at layers 1 and 14, short
conv dropped. Head-wise Muon for Q/K; Sinkhorn-balanced momentum updates for Engram/embedding/head; FP4 main KV via QAT.
(b) Why: KV storage/bandwidth is the deployment bottleneck; the upper half can read the lower half's final state instead of
recomputing its own global context.
(c) Base model ≈ DeepSeek-V4-Pro-Base quality with 1/3 total and 1/4 active params; global KV 890 B/token (≈ 1/4 of V4-Flash); prefill ≈ ½;
45T tokens "with no instability"; LR 2.6e-4 constant to 28T then cosine to 2.6e-5. No per-component ablation tables.
(d) Admitted: gap to giant closed models on expert agentic science tasks. Not admitted: CED / CSA2 / mHC gains are asserted, not
ablated in the report.
(e) For us: (1) C7 INGREDIENT — K/V REUSE ACROSS PASSES: in our looped core the X-side (c_hat) keys/values of the GR read do NOT change
across passes → compute K_X, V_X ONCE and reuse for all R passes (exact, not an approximation; CSA2 shows even approximate cross-layer
reuse is production-safe). Only S-side K/V and all Q are recomputed per pass → per-pass variation lives in Q/S (consistent with SMELT's
"visit 2 keeps Q/K, changes V" and Intern-Mobius's "distinct reasoners"). (2) RESIDUAL SLOT: mHC is DeepSeek's residual choice — a
doubly-stochastic stream-mixing matrix has spectral norm ≤ 1, i.e. a NON-EXPANSIVE residual; for a loop this is a principled bounded-norm
carry (audit item 4 / Parcae-style contraction) and n = 2 streams map naturally onto a TRM-style (y plan, z reasoning) split. Candidate
alongside Qwen gated residual and delta residual at design time. (3) CED supports our interface: decoder layers reading ONE mid-stack
state through PER-LAYER projections is production-validated (552B) — our decoder reading u/plan per layer is the same pattern.
(4) Engram at two depths with N-gram orders 2–4 — we already use Engram in the encoder; no change needed in the core.

## 40. "AI inside AI" — Emergent temporal abstractions in autoregressive models enable hierarchical RL (Kobayashi, Schimpf, Schlegel, … Meulemans, Sacramento; Google PoI; arXiv 2512.20605v2) — peripheral; body read in full; appendix B–D partly read (B.2, B.3, D.2), A (environments), C (probing/RL details), E (ELBO) skimmed; bibliography skipped
TAXONOMY: not looped in depth. A mid-depth latent "controller" that PERSISTS OVER TIME with a learned switching gate — the
closest paper in the folder to a chunk-level PLAN that is held across steps. RL-regime (grid world, MuJoCo ant), small models.
(a) Pretrain a causal transformer / SSM on unlabeled expert observation–action sequences (next-action + λ next-observation). Linear
probes decode the agent's latent subgoal from the residual stream (peaks by mid-depth, stays to the top). A LINEAR residual controller
e ← e + U_t e inserted NEAR MID-DEPTH turns the frozen model into a subgoal-pursuing policy; sequencing per-subgoal controllers solves
longer, recombined tasks (length + compositional generalisation). Unsupervised METACONTROLLER (recurrent hypernetwork): a
non-causal GRU encoder (sees a whole-sequence embedding) proposes a code z̃_t ~ N(μ_t, Σ_t); a CAUSAL switching unit gives
β_t ∈ [0,1]; z_t = β_t ⊙ z̃_t + (1 − β_t) ⊙ z_{t−1}; a decoder maps z_t → controller U_t. Loss = next-action NLL + α KL(N(μ,Σ) ‖ N(0,I)).
Internal RL: replace the non-causal encoder by a causal policy π(z_t | e_{1:t}) trained by RL in z-space with binarised switching.
(b) Why: next-token pretraining already builds temporally-abstract belief states; acting in the compressed, slow z-space shrinks the
horizon and search space.
(c) Mid-depth insertion gives the best controllability across base depths (Fig. 4, 5 seeds). β_t becomes quasi-binary and aligns
with ground-truth subgoal switches without any sparsity regulariser. Rate–distortion (Fig. 7): with a FROZEN base there is an α range
where subgoal-aligned switching is optimal; CO-TRAINING base + metacontroller loses this and collapses to a single switch at t = 0.
Internal RL solves sparse-reward tasks where GRPO on raw actions, CompILE, the β ≡ 1 ablation and the co-trained variant all fail
within 1M episodes. B.2: longer pretraining and moderate weight decay improve the abstract representation.
(d) Admitted: controlled toy environments; scaling to LLMs untested.
(e) For us: (1) CHUNK-LEVEL PLAN CARRY WITH A SWITCH GATE: our plan is recomputed per 4-token chunk with no persistence. A causal gate
z_j = β_j z̃_j + (1 − β_j) z_{j−1} (β from the current core state, zero-bias-init towards "switch" to stay identical to today at init)
would let a plan persist over several chunks and change sparsely — a principled, cheap version of the "gated zero-init shifted carry"
already listed as an ingredient (T²MLR), with the β ≡ 1 ablation = our current behaviour. (2) MID-DEPTH is where control works
(third independent source with GradCuit and Free Transformer) — our middle-span core is well placed. (3) FROZEN-BASE lesson: abstract
latents were discovered only on top of a FROZEN pretrained model; co-training collapsed. Supports a staged experiment path: train the
new core (and its plan interface) against a frozen, already-trained encoder/decoder first (cf. #24 (e)(4)), then unfreeze with a low
LR. (4) The variational metacontroller is another posterior-with-KL design → same leak budget as #31 applies.

## 41. From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence (Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson; CMU/NYU; arXiv 2601.03220) — peripheral; body (§1–8) read in full; appendices (proofs A, compute-optimal B, experiment details C, RASP D, E–H) NOT read — disclosed
TAXONOMY: information theory; one looped-vs-non-looped experiment (Fig. 6). No architecture proposal.
(a) Epiplexity S_T(X) = |P*| and time-bounded entropy H_T(X) = E[log 1/P*(X)], where P* minimises the two-part code |P| + E[−log P(X)]
over programs running in time T. Estimators: prequential (area under the training-loss curve above the final loss) and requential
(cumulative teacher–student KL). Three "paradoxes" resolved for bounded observers: (1) deterministic computation CAN create
information (CSPRNGs, ECA rule 54, AlphaZero); (2) information DEPENDS on factorisation order (one-way functions; chess
board-then-moves vs moves-then-board); (3) likelihood training can learn MORE structure than the generating process (induction,
emergence).
(b) Why it matters: loss measures only the random residue; epiplexity measures the reusable structure a model absorbs, which is what
transfers OOD.
(c) ECA: rule 15 low S/low H, rule 30 high H/≈0 S, rule 54 high S. Chess: the REVERSE order (final board first, then moves) has higher
S_T and H_T at high compute and gives better OOD (centipawn evaluation much better, puzzles equal). LOOPED vs NON-LOOPED on ECA rule 54
(Fig. 6): below a compute threshold the looped model is WORSE (cannot afford to unroll); above it the looped model ABRUPTLY wins (MDL and
epiplexity drop) by learning the short "iterate the rule" program, while the non-looped model keeps learning ever more emergent
shortcuts. Natural data: OpenWebText has the most epiplexity per token, CIFAR the least (> 99 % random); VQ tokenisation raises image
epiplexity; ADO data selection ≈ higher prequential epiplexity + better downstream.
(d) Admitted: epiplexity measures amount of structure, not task relevance; prequential estimator is heuristic; the looped brute-force
regime is rarely reachable on natural data.
(e) For us: (1) THRESHOLD BEHAVIOUR OF LOOPS: a loop pays off only once training compute lets it learn the iterative program; below
that it underperforms a straight stack — a plausible reading of our R-invariance at ~0.2B tokens and a reason to measure R-benefit
over training (not only at the end). (2) FACTORISATION: predicting an abstract "plan first, tokens after" (our NCP on the next chunk's
concept before the decoder emits tokens) is the harder, board-first-like direction, which the chess result links to richer, more
transferable internal structure — weak, analogical support for keeping the concept-first factorisation. (3) EVAL ADD-ON: prequential
epiplexity of the NCP loss curve (area above final loss) is a free, comparable measure of how much structure each core design
absorbs — add to the per-design eval sheet.

## 42. The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook (Yu, Chen, He, … Yan; NUS/Fudan/Tsinghua et al.; arXiv 2604.02029) — peripheral survey; §4.1 Architecture, §4.3 Computation, §4.4 Optimization, §5.1–5.3 Reasoning/Planning/Modeling and §6 Outlook read (prose; method tables NOT read); §1–3 (intro/foundation/evolution), §4.2 Representation and §5.4–5.7 (perception, memory, collaboration, embodiment) skimmed — disclosed; bibliography skipped
TAXONOMY: survey; no experiments. Architecture = backbone (parameter-shared / iterative / augmented — Huginn, Looped Trans., Ouro,
LoopFormer, PonderLM-2, DLCM "shifts computation from tokens to a compressed concept space"), component (generation, projection,
alignment, control, storage), auxiliary model. Computation = compressed / expanded (depth, width, structure) / adaptive (depth,
width, semantic-unit, control) / interleaved. Optimization = pre-training (AR supervision incl. looped models; auxiliary
supervision e.g. CoCoMix, ConceptLM; reinforcement pre-training LoopRPT) / post-training / inference-time.
(e) For us: (1) Our encoder → looped concept core → decoder sits in "augmented backbone + concept space" (the DLCM / ConceptLM /
CoCoMix neighbourhood) — the survey confirms the design family but gives no comparative numbers. (2) Width expansion (parallel latent
trajectories aggregated before decoding: SoftCoT++, PCCoT, CoT2) is the survey-level name for the K-candidate plan idea (#38) — it
lists it as an active direction without controlled evidence. (3) Its main open problem — EVALUABILITY of latent trajectories — matches
our situation (R1..R8 equal was only visible because we measured per-pass behaviour): keep the per-pass eval sheet (logit-lens KL,
per-pass NCP, step agreement, effective rank, OOD more-steps) as a first-class deliverable of every design. Low direct weight.

# EXTRA — the three residual papers named in the user's brief (read for the residual slot of the designs)

## X1. Qwen3.8-Next Gated Residual (GR) — "On the Design of Qwen3.8-Next Architecture" (Qwen Team) — §1, §2.2 (Residual) and §3.3 (stability) read in full; other sections (GDN/QSA token mixing, n-gram layer, infra, scaling-law refit) skimmed
(a) Residual stream widened to n_r = 4 branches. READ: per-branch RMSNorm R̂_i (own gain γ_i), elementwise sigmoid gate predicted
from ALL branches through a bottleneck (rank d/8): G = σ(W_u SiLU(W_d vec(R̂)/n_r)), block input x = (1/n_r) Σ_i G_i ⊙ R̂_i (this also
replaces the block's pre-norm; it is GatedNorm applied to the widened stream). WRITE: one data-dependent scalar per branch,
s = 2σ(W_w vec(R̂)/n_r) ∈ (0, 2), R_i′ = R_i + s_i y. No branch-mixing H_res ("adds little"). Standard init, "no special initialization".
(c) 25B-A3B, 560B tokens: pre-norm 1.617 → mHC static → mHC dynamic 1.594 → GR 1.590; widening alone +1.58 pts avg benchmark, dynamic
read/write +1.98 more (loss moved only 0.002 — benchmark and loss disagree); read granularity (elementwise) matters, write granularity
does not; read ALL branches. vs Attention Residuals (28 layers): Full AttnRes = GR = 1.762, Block AttnRes S = 2/4 +0.008/+0.011; at 48 layers
Block AttnRes 1.711 vs GR 1.707. GatedNorm lowers loss 0.002–0.005 everywhere. Stability: at 4× optimal LR the Muon + GR run has zero loss
spikes. Branch analysis: one branch carries LONG-RANGE paths (layer-0 outputs delivered past layer 10 to softmax-attention layers), the
other three stay local (median skip 1.2–3.5 layers).
(e) For us: GR IS our current residual (GRWrite/GR read in hrm_encoder.py and the core's read/write over [X, S]). In a LOOP its write is
EXPANSIVE: R_i accumulates s_i·y every pass with s ∈ (0, 2) and s = 2σ(0) = 1 at init → (i) not identity-at-init (audit item 1),
(ii) branch norms grow with R (audit item 4; read-time RMSNorm hides it from the next block but not from the carried state). Keep GR's
READ (elementwise gated, all branches — the evidenced part) and change the WRITE for the looped core: zero/negative-bias init of the write
logit (s ≈ 0 at init) and a bounded/normalised carry (delta-rule write or LN_h), per Diffusion-Timestep and the Survey's DeltaNet duality.

## X2. Delta Attention Residuals (Luo, Cai, Hu; NeurIPS 2026; arXiv 2605.18855) — body read in full + appendix B (AttnRes comparison); NeurIPS checklist skipped
(a) Route over DELTAS v_i = h_{i+1} − h_i (per-sublayer outputs; or block deltas Δ_b) instead of cumulative states, and ADD the routed
mixture to the residual stream: ĥ_l = h̃_l + Σ_i α_{i→l} v_i, α = softmax_i(w_lᵀ RMSNorm(v_i)), w_l ZERO-INIT (uniform α → bounded
perturbation; a zero-init null source gives an exact identity path). Delta Block = block-level deltas (B + 1 sources).
(b) Cumulative states are redundant (adjacent h_i nearly equal) → softmax routing over them collapses to near-uniform (max weight ≈ 0.2
in deep layers); deltas are diverse (attention vs MLP outputs, different depths) → sharp routing (max ≈ 0.6, 1.8× higher).
(c) FineWeb-Edu, 10K steps: 220M PPL 38.71 → Delta AttnRes 36.83 / Delta Block 37.08; 533M 32.00 → 31.05 / 31.16; 1044M 29.70 → 29.13 / 29.19
while their AttnRes reimplementation DEGRADES at depth (1044M 31.76, Full 33.36); Qwen3-0.6B 32.22 → 31.45; 7.57B 17.43 → 16.00 (−8.2 %),
AttnRes 18.58. Block count B = 4–6 best at 220M, flat at 533M. Cost: Delta Block −20 % (1B) to −35 % (8B) throughput, +3–26 % memory;
Delta AttnRes (2L sources) −69 %. Fine-tuning Qwen3-0.6B into Delta Block (dual LR 5e-5 / 5e-3): no loss spike, 55.6 vs 55.0 avg; AttnRes
spikes (2.8 → 3.96) and lags. Deep layers route > 50 % weight to specific early outputs; the token embedding gets disproportionate weight.
(d) Short training (10K steps), their own AttnRes reimplementation (official code unavailable) — the AttnRes degradation conflicts with
Kimi's own report (X3); treat the head-to-head as contested, the delta-vs-baseline gains as the solid part.
(e) For us: the natural "delta attention residual" for a LOOPED core: sources = PER-PASS DELTAS v_r = S_{r+1} − S_r (what each pass
added); at pass r the layer input is S̃ + Σ_{r′<r} α v_{r′} with a zero-init query → EXACT identity at init (audit item 1), additive
(residual preserved), sharp selective access to early passes (counters the dilution/overwrite of S), and a readable routing map per
pass (visualisable in the webapp). Cap the source set (last K passes or a fixed number of pass-blocks) so R > R_train does not change
the softmax's support (same R-extrapolation guard as MoDA #27). Cost at our scale is trivial (≤ R sources of size T/4 × d).

## X3. Attention Residuals (Kimi Team technical report) — read in full except infrastructure detail (§4 pipeline caching / two-phase inference skimmed); bibliography skipped
(a) Replace h_l = h_{l−1} + f(h_{l−1}) by softmax attention over earlier outputs: h_l = Σ_{i<l} α_{i→l} v_i, v_0 = embedding,
v_i = f_i(h_i); α = softmax(w_lᵀ RMSNorm(v_i)) with ONE learned, ZERO-INIT pseudo-query w_l per layer (uniform average at init,
"prevents training volatility"). Block AttnRes: sum layer outputs within blocks of S, attend over N ≈ 8 block summaries + embedding +
the current partial sum. Frames residuals as depth-wise LINEAR attention (Highway = stick-breaking, (m)HC = m-semiseparable, DDL ↔
DeltaNet) and AttnRes as depth-wise SOFTMAX attention.
(c) Scaling (5 sizes, 8K ctx): L = 1.891 C^−0.057 (baseline) vs 1.870 C^−0.058 (Block) vs 1.865 C^−0.057 (Full) → Block ≈ 1.25× compute.
Table 2: e.g. 528M: 1.766 baseline, 1.746 Block, 1.737 Full, 1.747 mHC-lite. Ablations (16-head model): DenseFormer 1.767 (no gain —
input-independent weights useless), mHC 1.747, Full 1.737, Block (S = 4) 1.746, sliding window over last 8 layers 1.764 (DISTANT layers
matter), input-dependent query 1.731, input-independent scalars 1.749, sigmoid instead of softmax 1.741, multi-head 1.752, no RMSNorm on
keys 1.743 / 1.750. Kimi Linear 48B-A3B, 1.4T tokens: all benchmarks ≥ baseline (GPQA-D +7.5, Math +3.6, HumanEval +3.1, MMLU +1.1);
BOUNDED per-block output magnitudes (baseline grows with depth — PreNorm dilution) and more uniform gradients; under fixed compute AttnRes
prefers DEEPER, narrower models. Learned maps: locality dominates, persistent weight on the embedding, selective long skips.
(e) For us: (1) Evidence that input-dependent SOFTMAX selection over depth beats fixed/scalar mixing and that distant sources matter —
in a loop the "distant" sources are EARLY PASSES and the encoder input X. (2) Replacement routing bounds magnitudes (audit item 4) but
X2 reports it degrading at depth in a reimplementation; the additive delta form (X2) keeps the identity path and is the safer choice
for a loop whose depth grows with R. (3) Zero-init pseudo-query + RMSNorm on keys are REQUIRED details (ablated). (4) "Prefers deeper
models" is consistent with using depth attention exactly where depth is multiplied — the looped core.
