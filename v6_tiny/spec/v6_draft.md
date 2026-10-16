# V6 draft: scalable NCP, anchored exploration, and bounded GradCuit

Date: 4 October 2026  
Revision: 2 — repository-wide evidence review, with explicit scaling and stability requirements.  
Status: proposed architecture; no V6 implementation or training results yet.  
Scope: establish the mechanism at the tested tiny scale, then scale width and context through the gates in section 8. Increasing recursion depth is a separate experiment.

“V6” here means the new architecture. The old run B_v6_s0 is a **V5 experiment variant**, not an implementation of this design.

## 1. Decision and research question

Keep the successful sequence: **NCP foundation → replacement core trained through the frozen decoder → useful Z exploration → complete transplant → verifier-guided Z adaptation at inference**.

Start the stochastic system from the strong deterministic replacement. Preserve that function exactly at Z=0, then train small Z-dependent adapters around it. Train exploration directly from a deployable prior. Use fixed control windows and bounded GradCuit. Omit learned switching and internal guidance noise from the first version.

The primary question is:

> Can a protected deterministic core with a trained, bounded Z interface make latent repair more effective than token retries at the same inference cost, while retaining language and graph competence?

V5 did not establish that advantage. GradCuit improved success and beat random control changes, but the deterministic core with token retries won the final comparison. V6 must beat that reference, not merely recover the cost of injecting noise.

The proposal is specific:

1. Retain the NCP representation and decoder interface.
2. Train the deterministic recursive replacement first.
3. Freeze it and train Z-dependent input adapters that vanish at Z=0.
4. Train exploration with coherent prior-candidate selection and a modest average-candidate quality term.
5. Start inference at Z=0. If verification fails, optimize bounded controls and regenerate.
6. Keep token retries as the mandatory competitor and the deployment fallback.

The new mechanisms below are hypotheses. Implementation correctness and empirical usefulness have separate gates.

Scalability means that quality improves with additional data and compute while memory, latency and recovery remain practical. Stability means finite, controlled training updates, preserved zero-control behavior, reproducible resume, and bounded per-problem search. Neither follows from a small parameter count or a normalized hidden state alone.

## 2. Evidence the design must explain

### 2.1 Sources and limits

The authoritative evidence is the completed JSON under [v5_tiny/runs](v5_tiny/runs). At drafting time, [REPORT.md](v5_tiny/REPORT.md) still calls D/E pending and the generated result tables omit final control runs.

The V5 validation problems have been examined repeatedly. They now count as development evidence for V6. Confirmation needs fresh locked evaluation instances and independent training seeds.

The numbers below retain their original verifier and generation protocols. A stricter V6 end-of-answer checker requires rerunning every comparison under that checker; do not treat a new V6 score and an old table entry as a matched comparison.

Most architecture results use one foundation seed. Item-level uncertainty cannot substitute for training-seed variation. The small flat-transformer baseline also had 2.7% fewer parameters and a different compute profile; its GraphNav failure does not establish a general NCP advantage over transformers.

### 2.2 Foundation and replacement findings

NLL: lower is better. Success: generated output accepted by the task verifier.

| Evidence | Measured result | Consequence for V6 |
|---|---|---|
| NCP foundation | Text NLL 1.8128; graph success 89.26% | Retain the tested foundation. |
| Same foundation without forecast loss | Text NLL 1.8143; graph success 19.53% | Keep forecast training; replicate the large task-specific effect. |
| Flat transformer | Text NLL 1.7799; graph success 0% | Keep language and symbolic results visible together. |
| Deterministic replacement B_v2 | Standard 88.87%; hard 37.89%; passes both NLL gates | Make this the competence anchor. |
| One-block replacement B_v2a | Standard 87.30%; hard 35.74% | Recursion may help, but capacity and compute differ. |
| Posterior-trained Z, B_v3 | Prior NLL gaps +0.3452 text / +0.3394 graph; fails both gates | Posterior reconstruction does not establish deployable control. |
| Z + internal noise without XM, B_v4 | Gaps +0.3163 / +0.3332; fails both gates | Internal guidance alone did not fix the prior. |
| Candidate-selection-only B_v5 | Passes both gates; standard mean-control success 90.04%; persistence +3.10 ± 0.43 percentage points | Keep prior-driven exploration. Persistence did not improve overall task success. |
| Combined B_v6 | Passes both gates; standard prior mean 91.80%, sampled hold4 85.35% | Separate deterministic quality from sampling cost. |
| C_v3 with candidate selection | Prior gaps +0.0222 / +0.0204 | XM can repair prior calibration. |
| Matched continuation without XM | Gaps +0.3701 / +0.3390 | Extra continuation alone did not explain that recovery. |
| Stronger Z KL, B_v3b1 | Standard 89.65%; Z-swap first-colour effect 2.15%; final KL about 0.00003/token | Good prediction can coexist with nearly unused Z. |
| C_v6 continuation | Standard 85.74% sampled / 91.02% mean; hard 36.33% / 41.60% | Compatibility worked; continuation added modest movement. |

Sources: [NCP](v5_tiny/runs/A_ncp_s0/eval_a_val.json), [forecast-loss ablation](v5_tiny/runs/A_ncp_a0_s0/eval_a_val.json), [flat baseline](v5_tiny/runs/A_v0_s0/eval_a_val.json), [B_v2](v5_tiny/runs/B_v2_s0/eval_b_val_lam1.0.json), [B_v2a](v5_tiny/runs/B_v2a_s0/eval_b_val_lam1.0.json), [B_v3 corrected](v5_tiny/runs/B_v3_s0/eval_b_val_nllfix.json), [B_v4](v5_tiny/runs/B_v4_s0/eval_b_val_lam1.0.json), [B_v5](v5_tiny/runs/B_v5_s0/eval_b_val_lam1.0.json), [B_v6 corrected](v5_tiny/runs/B_v6_s0/eval_b_val_nllfix.json), [C_v3](v5_tiny/runs/C_v3_s0/eval_b_val_lam1.0.json), [C_v3 no-XM](v5_tiny/runs/C_v3_noxm_s0/eval_b_val_lam1.0.json), [stronger KL](v5_tiny/runs/B_v3b1_s0/eval_b_val_lam1.0.json), [C_v6](v5_tiny/runs/C_v6_s0/eval_b_val_lam1.0.json).

### 2.3 Controllers and repair

| Method | Standard | Hard goals | Evaluation budget |
|---|---:|---:|---|
| Original trained RL controller | 81.05% | 35.94% | One trajectory |
| Gated-KL trained RL controller | 75.20% | 31.64% | One trajectory |
| Deterministic prior mean within Phase D | 90.63% | 39.45% | One trajectory |
| GradCuit, best tested learning rate | 97.27% | 54.30% | Up to 11 generations plus backward passes |
| Random Z-offset updates | 90.43% | 44.14% | Up to 11 generations |
| Z resampling | 99.80% | 59.77% | Up to 11 generations |
| Token retries from prior mean | 99.80% | 60.16% | Up to 11 generations |
| Deterministic B_v2 token retries | **100.00%** | **64.26%** | Up to 11 generations |

Sources: [D](v5_tiny/runs/D_v6_s0/phase_d.json), [D gated](v5_tiny/runs/D_v6_gated_s0/phase_d.json), [E standard](v5_tiny/runs/E_v6_s0_std/phase_e.json), [E hard](v5_tiny/runs/E_v6_s0_hard/phase_e.json), [B_v2 retries](v5_tiny/runs/B_v2_s0/bestof11_T1.json).

- GradCuit's best tested arm exceeded random updates by 10.16 hard-goal percentage points. Its direction carried useful information.
- Its advantage over the trained RL controller used a larger inference budget. It is not a matched-compute comparison of learning algorithms.
- Both methods are reward-guided: Phase D changes a shared controller during training; GradCuit changes current-problem controls during inference.
- V5 used greedy rejection gradients, not an unbiased stochastic policy-gradient estimator.
- Ungated discovery switched almost everywhere. Gated discovery reached 99.60% precision but 29.36% recall; hardening increased NLL from 0.0450 to 0.1052. The soft model's behavior did not survive deployment.
- These are failures of the tested controller, switching objective and training budget. They do not establish that a shared RL controller cannot work with a better control interface.
- Equal generation limits are not equal compute. Backward passes, verifier calls, generated lengths and stopping all count.

The implementation also differs from a stale comment in [phase_e.py](v5_tiny/code/phase_e.py): it recomputes a causal hold4 prior schedule and adds **one shared offset** at walk positions. It does not hold one fixed base Z for the entire episode.

The published [GradCuit mechanism](Loop%20Papers/GradCuit.pdf) inserts latent states into an intermediate attention sequence. This repository instead optimizes offsets through a concept-core control interface. V6 continues that adaptation, with new bounds/windows; it should be reported as GradCuit-style repair rather than an exact reproduction of the paper.

## 3. Derivation from first principles

### Step 1 — preserve the information needed for prediction

The encoder carries observed facts. Forecast training shapes compressed representations toward future prediction. The decoder still needs token-level detail. Keep both routes; do not force every identifier through a bottleneck.

The forecast-loss ablation supports that training signal. It does not prove that discrete PQ entries are semantic actions or carry all useful information; the continuous port has substantial measured influence.

### Step 2 — separate competence from exploration

A reliable deterministic model already exists. Sampling should add alternative behaviors while preserving access to that model. Make Z=0 an exact deterministic path. A learned prior mean is insufficient protection because it can move during training.

### Step 3 — train the inference distribution

The posterior sees future targets; deployed controls cannot. Weak KL left a prior mismatch, while strong KL nearly removed Z's influence. The first V6 model therefore has no target-conditioned posterior: train responses to controls sampled from a fixed bounded prior.

A fixed prior does not guarantee coverage. Context-conditioned adapters must learn useful responses to it, and coverage remains an evaluation gate.

### Step 4 — define exploration by useful alternatives

Different vectors or large gradients do not establish useful exploration. A modest number of control draws should produce distinct valid behaviors for the same task. Measure verified coverage and single-draw quality. Do not force diversity where only one answer is valid.

### Step 5 — make temporal credit explicit

A shared offset changes many decisions together. Learned continuous switching can fail when hardened. Begin with fixed windows represented in training, then compare one shared offset with separate window controls. Persistence is a behavioral hypothesis, not a consequence of storing the same vector.

### Step 6 — distinguish rejection from correction

Rejecting one trajectory can redistribute probability to other failures. Preserve a competent starting point, bound the search, regenerate after every update, and compare with fresh sampling. A small distribution-preservation penalty is a hypothesis to test, not a guarantee of improvement.

### Step 7 — charge every computation

Frozen weights still cost backward computation when differentiating with respect to Z. Promote latent repair using task success at matched end-to-end cost. Publish attempt-count comparisons separately.

### Step 8 — separate scale from depth

Wider states, longer histories, more independent training data and more recursive calls solve different constraints. Increase one axis at a time. Shared weights save storage; every additional use still costs computation, activations and potentially a distinct attention cache. A useful tiny result is a reason to run a controlled scale test.

## 4. Architecture

### 4.1 Tiny configuration

| Component | Proposed default |
|---|---|
| Vocabulary / context | Existing tokenizer, 2,048 entries; T=192 |
| Width / heads / FFN | d=128; four heads; SwiGLU hidden width 384 |
| Foundation stacks | Three encoder, two concept, three decoder blocks |
| Chunking | k=4; completed-chunk mean pooling and no-affine RMS |
| Concept output | Soft PQ: eight segments × 32 entries × 16 dimensions |
| Continuous output | Existing A port and gamma |
| Deterministic replacement | Two shared blocks; two high transitions, two low updates each |
| Stabilization | Existing post-RMS after low/high updates |
| External control | Eight dimensions per fixed response window |
| New adapters | Two context-conditioned injections, intermediate width 16 |
| Internal guidance / posterior / learned switch | Absent |
| Prior | Standard Gaussian draws projected into the declared control ball |
| Training window length | Uniform per-row choice from {1,2,4} chunks |
| Default deployment window | One chunk; four chunks is a mandatory comparison |
| First answer | Z=0, greedy |
| Repair budget | Ten updates, at most eleven generated candidates |

One-chunk control is an identifiable starting point, not an established optimum.

### 4.2 Data flow

~~~mermaid
flowchart LR
    X["Observed or generated prefix"] --> E["Frozen NCP encoder"]
    E --> H["Token features H"]
    E --> C["Completed-chunk concepts C"]
    C --> R["Frozen deterministic recursive core"]
    C --> I["Trainable Z-dependent adapters"]
    Z["Bounded window controls; zero first"] --> I
    I --> R
    R --> P["Frozen PQ readout and A port"]
    P --> S["Causal shift"]
    H --> D["Frozen decoder"]
    S --> D
    D --> Y["Candidate"]
    Y --> V["Public verifier"]
    V --> G["On rejection: bounded GradCuit"]
    G --> Z
~~~

The foundation trains in A, the deterministic core in B0, and only control adapters in B1. At inference, every shared weight is frozen; only current-problem Z changes.

### 4.3 Causal timing

H_t encodes x_0 through x_t. A complete chunk produces:

~~~text
N(x) = x / sqrt(mean_features(x²) + 1e-6)
C_j = N(mean(H_(4j), ..., H_(4j+3)))
j(t) = floor((t+1)/4) - 1
D_input,t = H_t + chat_j(t) + gamma ⊙ A_j(t)
~~~

N is the foundation's no-affine normalization, not a scalar norm. Position t predicts x_(t+1). Negative j uses the foundation's zero plan. C_j may first influence t=4j+3. The core at j reads only C_0…C_j.

Use actual pooled concepts from the observed/generated prefix at deployment. Do not feed predicted concepts back as observed facts. Reset attention, chunk alignment, controls and optimizer state at task/document boundaries. Mask padding and cross-document targets. An incomplete final chunk never becomes available early.

### 4.4 Deterministic anchor

Fit the existing V2-style replacement through the frozen foundation. Preserve all of its weights:

~~~text
e = W_in C
h = h_init
l = l_init
repeat R=2:
    repeat nL=2:
        l = post_L(f_L(l + h + e))
    h = post_H(f_H(h + l))
u = B_u(out_norm(h))
A = B_A(out_norm(h))
~~~

The blocks attend causally over concept positions. This is depth recurrence over a sequence, not a newly introduced persistent private state across chunks.

### 4.5 Z-dependent adapters

For q in {H,L}:

~~~text
r_epsilon(x) = sqrt(max(mean_features(x²), epsilon²))
g_q(e) = 1 + 0.5 tanh(G_q N(e) + b_q)
raw_I_q(e,z) = U_q [g_q(e) ⊙ (V_q z)]
cap(e) = 0.1 · r_epsilon(e)
I_q(e,z) = raw_I_q · min(1, cap(e) / r_epsilon(raw_I_q))
~~~

G is r×d, b is r, V is r×d_z, U is d×r; r=16, d=128, d_z=8 and epsilon=1e-6. Scalar r_epsilon clamps the mean square before taking its square root, so its derivative is finite at zero and it equals ordinary RMS above epsilon. N retains the foundation's separate normalization convention.

Here r is the adapter's intermediate width. Since d_z=8, its local Jacobian with respect to one window's Z has rank at most eight, even when r=16. Increasing r alone does not add independent control coordinates.

Change only:

~~~text
h at initialization = h_init + I_H(e,Z)
low-step input = l + h + e + I_L(e,Z)
~~~

Share the adapters across recursive calls. Keep high updates, output projections, PQ, port order and causal timing unchanged.

**Exact anchor:** I_q(e,0)=0 for finite adapter parameters and states. With the core frozen and deterministic evaluation, Z=0 reproduces the deterministic replacement. No second reference-core forward pass is needed to obtain the model's ordinary logits.

Do not short-circuit the adapter expression when differentiating at Z=0. Its value is zero, but its derivative with respect to Z is generally nonzero. A fast zero-control bypass is allowed only for ordinary inference after equivalence checks, never for the first repair gradient.

Initialize U,V independently with normal standard deviation 0.02 and G,b at zero. Zeroing both projection factors would kill their initial learning. Use one fixed training calibration minibatch and two prior schedules, then rescale each U downward so its maximum observed raw-write RMS / r_epsilon(e) is at most 0.01. Save the calibration seed and rescaling factors. This is an initialization check, not a guarantee over all future prefixes.

The 10% cap is per injection. Repeated writes may accumulate, so also monitor final port shifts, state norms and token-distribution changes. The cap is a proposed guard, not a measured optimum.

Freezing the core limits expressivity. If diagnostic supervised control fitting cannot recover known valid trajectories, investigate adapter capacity or optimization before spending more search attempts.

The mandatory simpler control is two bias-free direct lifts W_H z and W_L z, using the same write caps and injection sites. It needs only 2dd_z=2,048 parameters at tiny scale and also preserves Z=0. The context-conditioned adapter is the main candidate, but it must justify its added complexity against this control.

### 4.6 Prior, windows and prompts

Draw eta_w independently from N(0,I_8), then:

~~~text
z_w = eta_w · min(1, 2 / r_epsilon(eta_w))
~~~

Use the same bound RMS(z_w)≤2 during training and repair. This projected distribution has boundary mass; it is not an ordinary Gaussian density. The default computes no Gaussian KL for it.

A shared bound does not make optimized controls distributed like prior draws. Probe paths from zero through 0.25z, 0.5z and z on development examples, and report quality and gradient behavior along them. Calibration near zero and under repeated repair remains a hypothesis.

Controls are constant within each L-chunk window, indexed relative to response start:

- GraphNav: header controls are zero; response controls start at the boundary predicting the first walk token.
- Text: choose a prompt boundary among completed chunks in the first half of the row, with at least one complete prompt chunk. Apply controls only to the scored suffix.
- Sample L uniformly from {1,2,4} per training row. All lengths used at deployment are represented.
- Hidden segment labels never set a control window.

At inference controls may depend on feedback from previous completed attempts; that feedback is already available. Future control positions must not influence earlier logits.

The primary experiments use chunk-aligned prompts. For an arbitrary text prompt ending inside a chunk, keep controls zero until the next completed chunk can causally supply a response plan; report those initial uncontrolled response tokens. Do not silently insert padding tokens into the user's prompt. A separate experiment may train a finer-grained boundary interface.

## 5. Training stages

### A — NCP foundation

Retain the implemented V5 objective, with the masking correction below explicitly separated from historical replication:

~~~text
L_A = L_NTP + 1.0 L_NCP + 1.0 L_VQ
~~~

NCP is per-feature MSE between chat_j and detached C_(j+1). VQ fits nearest codebook entries to detached concepts. “Codebook-only VQ” describes that loss's gradient: it does **not** imply that other losses cannot train the codebook. V5's soft PQ path also permits NTP/NCP gradients into Q during A.

Keep no-affine normalization, soft feedback, no code-label CE, data-based codebook initialization and the V5 restart schedule. Freeze Q after A.

Initial recipe: 18,000 steps, batch 64×192, 32 text and 32 graph rows, AdamW learning rate 0.002, 300 warmup steps, cosine decay, matrix weight decay 0.1, clipping 1.0 and the documented 10× codebook learning-rate group.

NTP is a token mean within each domain, then 0.5/0.5 across domains. Graph headers and padding are unscored for NTP. NCP uses the existing target-chunk mask and within-domain reductions. V5 VQ fits all pooled chunks, including graph headers and padding-only chunks; it does not reuse the NCP mask. Preserve these distinct reductions for the historical replication.

For fresh V6 experiments, use only complete, non-padding chunks from one document for VQ, and require both endpoints of an NCP pair to be complete chunks from that same document. Retain valid header chunks for VQ. Exclude cross-document attention. These masking changes and the codebook moment-reset correction below are declared recipe corrections, applied equally to every newly trained foundation and matched baseline where relevant. Report their foundation-only effect before attributing downstream changes to the adapters.

Use document IDs in all three attention stacks and align packed document boundaries to k=4. A mask on the loss alone does not prevent cross-document information flow. Exclude incomplete/padded chunks from codebook initialization, usage statistics and restart candidates, and clear Adam's first and second moments only for codebook rows that are reinitialized. The older V3/V4 trainers already implement this moment-reset rule; V5's restart method does not.

Keep the no-forecast-loss control and parameter/compute-accounted flat baseline in confirmation. Report language, validity and task success separately.

### B0 — deterministic replacement

Freeze the foundation; train only F_det for an initial 6,000 steps with the existing V2 recipe:

~~~text
L_B0 = L_NTP(frozen_decoder(F_det(C)))
       + L_NCP
       + delta(s) L_port_distillation
~~~

Use V5's distillation definition: match the old continuous core output and A port, with delta=1 for 10% of steps and decay to zero by 30%. The frozen PQ readout maps the new continuous output to the decoder-facing soft concept. Train through the real decoder at replacement fraction lambda=1.

Require the deterministic replacement to preserve foundation NLL and generated competence before adding Z. Retain the one-block replacement control with capacity and compute differences recorded.

### B1 — prior-driven exploration

Freeze F_det too. Train only the 8,480 adapter parameters. Gradients still traverse frozen core/PQ/decoder computations into the adapters: freezing weights is not a no-grad wrapper. Encoder computations can use no-grad because controls do not enter them.

Initial recipe: 6,000 steps, the same input data mixture, K=2 independent complete prior-control schedules per row; AdamW learning rate 0.001, 200-step warmup, cosine, zero adapter weight decay, clip 1.0. These adapter optimizer values are proposed defaults.

For candidate i, score the entire valid suffix under its one coherent causal control history:

~~~text
ell_i = -sum_t log p(y_t | prefix_t, Z^(i))
L_row = [(1-beta) min_i ell_i + beta mean_i ell_i] / n
beta = 0.25
~~~

n is valid suffix length. Exclude empty suffixes; average rows within domains, then 0.5/0.5.

The winner is a **complete candidate schedule**. Do not choose separate winners in different windows while ignoring how their states affect later logits. Every scored candidate must correspond to a realizable forward trajectory.

The mean term discourages rare excellent winners alongside many unusable samples. It may also weaken specialization; beta=0 is a mandatory ablation. This is a composite selection objective, not exact marginal likelihood.

Candidate selection is teacher-forced: candidates see the same observed previous tokens but different control histories. It does not train on their own generated prefixes or guarantee multimodal free generation. Keep the selected-candidate score, average-candidate score and verified coverage as separate measurements.

Keep Z=0 outside the winner pool: its loss is independent of adapter weights, so letting it win would create selection rows without an adapter gradient. Log it as the reference.

No posterior reconstruction, latent KL, epsilon, learned termination or shared-controller RL is in B1. Their removal is tested against V5-style full-core controls.

Require both deployable single-draw quality and improved verified coverage across multiple controls. Neither NLL nor vector diversity alone is sufficient.

### C — transplant validation

Blend both decoder-facing ports for lambda in {0,0.25,0.5,0.75,1}:

~~~text
chat_lambda = (1-lambda) chat_foundation + lambda chat_replacement
A_lambda = (1-lambda) A_foundation + lambda A_replacement
~~~

Lambda=0 reproduces the foundation. Lambda=1,Z=0 reproduces the deterministic replacement. These are different anchors.

Deploy at lambda=1. An intermediate blend is a diagnostic, not an endpoint substitute. B0/B1 already train at full replacement, so extra transplant training is not assumed necessary.

If continuation is needed, declare a fixed 1,500-step adapter-only arm and give its comparison the same additional budget. Keep F_det frozen.

Evaluate free generation with actual generated-prefix concepts. Never attach the original demonstration's continuation labels after generation reaches a different state. Future on-policy supervision must recompute valid targets or use task feedback only.

### D — bounded GradCuit calibration

Replace the learned switching-controller stage with per-instance adaptation. All model and adapter weights remain frozen.

1. Initialize every response control to zero.
2. Generate greedily and verify.
3. If accepted, return immediately: this first candidate is exactly F_det's answer.
4. Otherwise update Z using the rejected trajectory, regenerate from the original prompt, and verify again.

For rejected y with n scored tokens:

~~~text
L_repair(Z;y) =
    (1/n) sum_t log p_Z(y_t | prefix_t)
    + lambda_KL (1/n) sum_t KL[p_0(.|prefix_t) || p_Z(.|prefix_t)]
lambda_KL = 0.05
~~~

p_0 is the zero-control model evaluated on the **same rejected token prefixes**. Detach p_0 probabilities, tokens, verifier decisions and acceptance masks. Exclude prompt targets and targets after the complete declared terminator. For the current GraphNav format, generate and score the full [DONE, SEP, SEP, SEP] chunk; DONE alone is not an accepted complete answer. Stop each row after that chunk and leave the remaining buffer as padding.

Minimizing the first term lowers rejected-sequence probability. The penalty limits broad distribution changes on those prefixes. It does not bound task regret on every possible future history.

Use Adam on per-window Z: proposed learning rate 0.1, betas (0.9,0.999), optimizer epsilon 1e-8 and no weight decay. Clip the gradient's Euclidean norm across active window coordinates to 1.0 **separately for each problem**. Sum per-problem normalized losses before differentiation so batch size does not rescale a problem's gradient.

After every update, project each active z_w with z_w ← z_w · min(1,2/r_epsilon(z_w)). Optimizer state and step counters are separate per problem and reset between tasks. Mask never-used future windows, including optimizer moments; they remain zero until reached. If a later attempt ends earlier, freeze its currently unused coordinates and moments. Accepted problems retain their controls, answers, moments and step counters while other problems continue.

Default U=10: at most eleven generated candidates. Generate and verify after **every** update, including the last. Return accepted answers with verification status; after exhaustion report failure and explicitly label any returned candidate unverified.

This masking contract requires more than zeroing a gradient: an Adam momentum update can still move a coordinate with a zero current gradient.

For continuity with V5, the main arm uses greedy rejection repair. It is an RL-style surrogate, not an unbiased score-function estimator. A separate sampled arm must sample from and replay the actual temperature/masking distribution.

The penalty, bounded coordinates and per-window parameterization are individually ablated. None is an established improvement.

Before tuning repair, measure three quantities on fixed development failures: target-NLL oracle reachability, random-control verified coverage, and verifier-only repair gain. Add per-window Z-gradient RMS and the fraction of writes hitting their cap, evaluated at zero and along the prior interpolations. A small finite-difference check on selected coordinates must agree with the replay gradient.

When the greedy answer repeats, record that explicitly: reducing its log probability may still leave its rank unchanged. Do not call a decreasing repair loss a successful correction. Report gradient contribution by colour, destination-node, format and stopping tokens to detect updates dominated by suppressing otherwise valid output.

### E — choose the inference policy

Compare under the same verifier:

1. Deterministic replacement, one greedy answer.
2. Deterministic replacement, T=1 token retries.
3. Independent prior-Z draws with greedy tokens.
4. V5-style shared-offset GradCuit.
5. Per-window GradCuit, KL penalty zero.
6. Per-window GradCuit, KL penalty 0.05.
7. Random per-window updates matched to the actual gradient method's step norms.

For incremental comparisons, all methods start with the identical Z=0 greedy answer, then receive at most ten more candidate slots. Preserve the original V5 protocols as separately labeled reproduction arms.

The paired direction diagnostic may obtain random-step norms from a corresponding GradCuit trace, but must be labeled as using that trace. For a standalone deployment baseline, fix a step-size schedule on development data. Charge any online gradient computation used to obtain norms to that baseline's cost.

Measure success at 1,2,4,8,11 attempts and at fixed end-to-end time/FLOP budgets. Choose the deployment policy on development data. If token retries win, they remain the default.

## 6. Execution and gradient contracts

### Replay

Replay generated tokens under the exact control schedule used for that attempt. Default inference has no internal stochastic guidance. Any optional noise arm must save and replay its base noise.

Differentiate conditional token log probabilities through the frozen model into Z. Do not differentiate through argmax, sampled tokens, verifier decisions or acceptance. Frozen parameters remain unchanged, while input gradients through their computations remain available.

### Caching

Cache by dependency, not by whether a token is called part of the prompt. The first response control acts on the final completed prompt chunk's plan, at the decoder position predicting the first response token. That core output and decoder position depend on the new control.

Encoder caches for an unchanged token prompt remain reusable. Core/decoder caches are reusable only strictly before the earliest control-affected position. A response-control update invalidates that position and every later dependent state, including the boundary position still inside the token prompt.

The first correct implementation may regenerate and recompute the complete response. Optimize incremental execution only after cached/uncached logits and Z gradients agree across chunk/window boundaries. Report optimization differences between search methods.

Give each of the six recursive block applications its own K/V cache slot: fL is used four times and fH twice, and their inputs differ across visits. Weight sharing does not permit sharing the contents of those caches. Also retain the fewer-than-four token features needed to finish the next pooled chunk.

Inference caches built under no-grad are not automatically valid for repair backward. Recompute every Z-dependent activation needed for that gradient; detach only computations proven independent of the optimized controls. Checkpointed replay must retain the same controls, masks and any random draws.

### Temporal coordinates

L=1 exposes eight coordinates per response chunk. This increases control dimension relative to V5's shared eight-dimensional offset. Compare shared and local controls with the same backbone and budget, and include random local updates.

Changing one window may affect later outputs through causal attention. A window is not automatically an isolated semantic action.

At the same coordinate-wise Adam learning rate, a larger number of control coordinates can produce a larger total update. Report active dimension, total and per-window step norms, cap-hit rate and induced replay KL for shared/L=1/L=4 comparisons. Include a diagnostic with matched perturbation magnitude before attributing gains solely to temporal separation.

## 7. Lessons inherited from older architectures

| Earlier evidence | V6 decision |
|---|---|
| Original isolated concept experiments found XM coverage gains, while the end-to-end XM test did not reproduce them | Require coverage through the actual frozen encoder/core/decoder pipeline. |
| Original GradCuit probe repaired failures, but rank enumeration and extra passes were strong competitors | Separate latent steerability, useful update direction and advantage over other compute uses. |
| V2's NextLat gradients opposed NTP on several shared weights; truncated recurrence cut final-loss credit to the initial-state projection | Keep the small core fully unrolled; audit loss-specific gradients before adding auxiliary objectives. |
| V3 extra core passes and shared computation helped some composition probes more than language | Retain the small recursive core and measure task behavior; do not assume extra untrained passes help. |
| V4 larger chunks traded quality for modest speed; some added branches wrote without helping | Keep k=4 and mean pooling. No new pooler FFN by default. |
| V4.1 direct retrieval improved fact access | Preserve V5's existing full-causal token route; do not add a redundant retrieval block. |
| V4.1 null injection exaggerated route importance | Use trained plan-off controls and calibrated mean/swap interventions; audit PQ and A separately. |
| V4.1 oracle offsets showed latent steerability | Keep supervised reachability as an oracle diagnostic, separate from verifier-only performance. |
| V5 XM repaired priors; epsilon alone did not | Prior-driven exploration with no default epsilon. |
| V5 switching saturated or broke during hardening | Fixed windows; learned durations remain a separate research question. |
| V4.1 long-episode NLL degradation often came from stopping/non-action probability rather than action choice | Report decision quality, format and stopping separately on longer horizons. |

Sources: [V3 final-control readings](v4_tiny/GO_NOGO.md), [V4.1 assessment](v41_tiny/ASSESSMENT.md), [V4.1 first principles](v41_tiny/FIRST_PRINCIPLES.md), [V5 methods](v5_tiny/METHODS.md), [V5 review](review_v5.md).

Additional lineage evidence: [original control experiments](Concept%20Design/RESULTS_H1_H2.txt), [end-to-end concept experiments](Concept%20Design/RESULTS_E2E.txt), [decoder experiments](Decoder%20Design/RESULTS_DECODER.txt), [V2 optimization audit](v2_underperformance_review_2026-09-28/REPORT.md), [V3 design review](draft_v3_review.md), [long-horizon decomposition](v41_tiny/LONG_AUDIT.md). Some original component scripts depend on experiment files no longer present; their saved reports are historical evidence, not a claim of a fresh reproduction.

Pinpad, GraphNav and text test different capabilities. Transfer mechanisms as hypotheses, not their numerical effect sizes. These results do not establish learned hierarchical reasoning or a scaling guarantee.

## 8. Budgets

### Parameters

| Quantity | Parameters |
|---|---:|
| Tested NCP foundation | 2,022,144 |
| Original two-block concept module plus its output norm | 426,624 |
| Original teacher A projection, also inactive after full transplant | 16,384 |
| Deterministic replacement | 476,288 |
| Deployed deterministic transplant | 2,055,424 |
| Two adapters: 2×[(128×16+16)+(8×16)+(16×128)] | 8,480 |
| Proposed deployed V6 | **2,063,904** |

This is an analytic ledger, not an instantiated count. Check it before implementation acceptance. Revision 1 incorrectly retained the old teacher A projection in the deployed total. Both that projection and the original concept module are excluded here. If retained for blend diagnostics, count their 443,008 parameters in loaded memory. Per-problem Z is optimization state, not shared model parameters.

Adapter injections can be computed once per concept position and reused across recursive calls. Ordinary inference needs no second core/decoder pass.

### Training

- A: 18,000×64×192 = 221,184,000 input token slots.
- B0: 6,000×64×192 = 73,728,000 input token slots.
- B1: 73,728,000 input token slots, **two candidate trajectories per row**; not B0-equivalent compute.
- Optional C: 1,500×64×192 = 18,432,000 input token slots.
- Calibration, oracle diagnostics and all control experiments have separate budgets.

Small trainable parameter count does not make B1's backward passes free. Report elapsed time, forward/backward work and peak memory.

Slots include unscored headers and padding. Record non-padding input tokens, loss-bearing targets by domain, unique document/graph exposure and repeated-data passes separately. B1's row-normalized suffix loss intentionally differs from A/B0's token-normalized domain loss; preserve each declared denominator across microbatch accumulation and distributed workers.

Carry the documented MPS reduction safeguard until new correctness checks justify changing it: compiled regions return per-token tensors; sensitive reductions run eagerly. This drafting task does not execute those checks.

### Inference

~~~text
total cost = encoding + all generations + all probability replays
             + backward passes + anchor-KL replays + verifier work
~~~

Include cache construction, output lengths, early stopping, batch size and a fixed compilation-warmup policy. Report wall time and analytic FLOPs separately. Token retries may afford more attempts than GradCuit at the same cost.

Select learning rates on development data. Do not headline the best test arm after inspecting all its results.

### Scaling contract

The first scaling family holds k=4, encoder/concept/decoder depths 3/2/3, R=2, n_L=2, FFN width 3d, PQ S=8,V_Q=32, control dimension 8 and adapter intermediate width 16 fixed. Attention head width stays 32. The vocabulary remains 2,048 for this controlled family.

For this family, including tied embeddings and excluding inactive teacher modules:

~~~text
P_foundation(d,V) = 105d² + (V + 308)d + 256
P_deterministic(d,V) = 107d² + (V + 312)d + 256
P_adapters(d,r,d_z) = 4dr + 2rd_z + 2r
P_V6 = P_deterministic + P_adapters
~~~

| Scale | d / heads / FFN | Foundation | Active V6 | Trainable B1 adapters |
|---|---|---:|---:|---:|
| Tiny reference | 128 / 4 / 384 | 2,022,144 | 2,063,904 | 8,480 |
| First width test | 256 / 8 / 768 | 7,484,672 | 7,633,440 | 16,672 |
| Second width test | 512 / 16 / 1,536 | 28,731,648 | 29,291,040 | 33,056 |

Each width needs its own trained foundation and deterministic anchor. Widening an existing checkpoint or changing its normalization is not the same frozen-anchor experiment. These are parameter targets, not demonstrated quality or hardware-fit claims.

Use this order:

1. **Mechanism pilot:** reuse the existing tiny A_ncp/B_v2 checkpoints and historical data convention to compare direct Z lifts, context-conditioned adapters and retries. Treat all outcomes as development evidence. This avoids retraining the foundation before discovering whether the restricted interface can work.
2. **Fresh tiny confirmation:** train the chosen tiny recipe and matched baselines with the corrected document/VQ handling, multiple seeds and locked evaluation.
3. **Width:** compare d=128 and d=256 at T=192, with both equal-data and equal-total-training-compute views. Include A+B0+B1 in the total, plus a separately reported incremental B1 cost.
4. **Context:** at fixed d=256, evaluate and train supported lengths 192→512→1,024→2,048. Use real long documents or longer fact-composition tasks, not extra padding around the same short graph. Keep tokenizer, width and recursion fixed.
5. **Further width:** test d=512 at a context length that passed the previous gate. A larger vocabulary or additional depth is a separate, priced revision.

Keep the initial optimizer-update budget at 12,288 input slots using packing and gradient accumulation; log actual scored targets and examples by domain. Multiple graph episodes in a long row need separate document IDs, chunk-aligned boundaries and attention isolation. Larger batches or changed domain exposure are separate variables.

The available V5 text arrays contain 133,553,935 training tokens and 5,349,196 validation tokens. Longer training can repeat this corpus without adding new evidence. Before the larger width run, publish unique-data exposure and add independently split text if the experiment needs more diversity. Reuse the fixed tokenizer only for the controlled width study; a vocabulary change requires retraining and repricing the embedding/head.

At each width, include an ordinary transformer and the deterministic replacement trained with the same data protocol. Give the deterministic baseline an additional-training arm matched to B1's extra compute. Otherwise extra adaptation data/compute could be mistaken for a better architecture. Small graph benchmarks alone cannot establish language-model scaling.

### What actually scales in time and memory

Let M=T/k and N_c=R(n_L+1)=6. Ignoring constants and the separate vocabulary head, dense prefill/training work is:

~~~text
matrix work   ∝ B d² [6T + N_c M]
attention work ∝ B d [6T² + N_c M²]
LM-head work  ∝ B T d V
~~~

Only the concept path runs at T/4. The six token-level blocks retain quadratic full attention. V6 does not yet have a subquadratic long-context design. Efficient attention kernels reduce activation storage and improve execution; they do not remove all attention arithmetic.

For standard multi-head attention, a simple incremental decoder stores approximately:

~~~text
KV elements = 2Bd [6T + N_c(T/k)] = 15BTd
~~~

At B=1,T=2,048 this is 15, 30 and 60 MiB in FP32 for d=128,256,512 respectively, before weights, encoder-output buffers, activations, temporary kernels and repair backward. Half-sized storage applies only to a validated two-byte cache dtype. Shared fL/fH weights still need all six cache slots.

Use memory-efficient causal/document-aware attention where supported, microbatch accumulation, and activation checkpointing through frozen modules that carry Z gradients. For B1, compute frozen encoder features once per minibatch. Two candidate strategies are valid:

- Keep K candidate graphs when they fit, then differentiate the declared selection/mean loss.
- Score all K candidates without gradient, save their complete schedules and deterministic winner indices, then replay candidates one at a time with weights (1−beta)·1[i=winner]+beta/K. Accumulate all gradients before clipping or updating. At beta=0, replay only winning rows.

The second strategy trades additional forward work for lower activation memory. It must match the first strategy's loss and gradients on a fixed fixture, including tie handling. Preserve the exact frozen weights and random draws between scoring and replay.

Bound positional-cache storage by the configured maximum context and supported device/dtype combinations; do not retain a new full cache for every observed generation length. Release per-problem activations, controls and optimizer state when that problem finishes. These lifetime rules are part of the sustained-memory check.

For FP32 Adam, budget roughly 16 bytes per trainable parameter plus 4 bytes per frozen parameter before activations and extra copies. B1 and repair usually become activation/decoder-compute problems, despite their small optimizer states. Chunk the LM-head/loss computation only with an implementation that preserves full-vocabulary normalization and gradients.

### Numerical and optimization stability

1. **Precision:** FP32 is the reference. Keep norm/cap reductions, codebook distances, log-softmax, KL, loss sums and optimizer moments in FP32. In particular epsilon²=1e-12 must not underflow in an FP16 clipping norm. Validate a supported BF16/mixed-precision path against that reference before using it at scale; backend availability is a measured implementation property.
2. **Width initialization:** for a new wider foundation/core, start with the reference linear-weight standard deviation 0.02·sqrt(128/d) for matrices whose fan-in scales with d. Retain the tiny reference at d=128. Embeddings, the tied output head, constant initial states and adapter factors have different fan-ins: calibrate their activation/logit RMS explicitly instead of applying this rule indiscriminately. This is a proposed calibration recipe, not a claim of implemented maximal-update parameterization.
3. **Learning rates:** do not copy a peak learning rate to every width without a pilot. Use the same declared small LR sweep and token budget for paired architectures. Record gradient RMS, actual update/weight RMS, clipping frequency, logits and loss by domain. Treat normalization gains, codebooks, embeddings and matrices as distinct diagnostic groups.
4. **Loss units:** NCP already averages features. For width comparisons, normalize VQ's segment squared-distance by 16/(d/S), so its tiny-width definition is unchanged and increasing segment width alone does not multiply the objective. Apply this to all matched architectures and log the raw loss as well. Empty masks contribute finite zero and no optimizer update from that term.
5. **Recurrence:** retain full unroll and post-normalization with R=2,n_L=2. Monitor state RMS and gradient/parameter-update effects at every shared-block visit. Normalized forward states do not prove stable backpropagation through repeated weights. Do not increase R or borrow a residual-scaling formula without an architecture-matched depth experiment.
6. **Codebooks:** initialize from valid complete training chunks, count usage before clearing counters, and reset moments for replaced rows. Record the 10× codebook LR separately and validate its update scale at each width; it is not an automatically safe scaling law.
7. **Controls:** measure write saturation, port shifts, replay KL, Z-gradient RMS and finite-difference agreement. The write cap and Z ball are bounded-input safeguards; output quality still needs verification. A nonfinite candidate update is rejected before modifying the accepted answer or trusted optimizer state.
8. **Normalization and anchor:** save all epsilon values, learned norm gains, precision and output scaling in the configuration. Compare Z=0 with the deterministic anchor under the same execution path and dtype. Cross-backend numerical tolerance and mathematical zero-control equivalence are different checks.

The local [DeepLoop paper](Loop%20Papers/DeepLoop.pdf) motivates auditing repeated parameter visits, but its Post-LN residual formula is not directly validated for this mixed pre-norm-block/post-state-norm core. [Sparse-layer scaling experiments](Sparse%20Layers%20are%20Critical%20to%20Scaling%20Looped%20Language%20Models.pdf) and [compute-matched looping](Loop%20Papers/Scaling%20Laws%20for%20Compute-Matched%20MoE%20Looped%20Transformers.pdf) motivate matching compute and checking width transfer; they do not justify restoring the repository's problematic V2 MoE by default. [LoopFormer](Imp%20Paper/LoopFormer.pdf) provides a separately trained elastic-depth direction, not permission to add untrained passes to V6.

### Recovery and promotion gates

Save atomic checkpoints containing model/adapter weights, optimizer moments, scheduler progress, global and per-stream RNG state, data cursor, codebook counters, precision state and the full immutable run configuration. Hash the foundation/anchor, tokenizer, data manifests and implementation used. A resumed run must reproduce the next batch, candidate schedules, loss and update within the declared same-backend tolerance.

The first invalid loss/gradient must prevent optimizer.step. Preserve a diagnostic record and the last finite checkpoint. A recovery with a reduced LR is a recorded recipe deviation with a fixed retry limit, not an invisible successful continuation. Resource interruption and numerical failure are separate statuses.

Before promoting a width/context:

- Pass G0 and the relevant quality gates at the current scale.
- Complete a fixed development pilot including warmup and at least 1,000 post-warmup optimizer updates without nonfinite updates. A codebook-restart and checkpoint-resume fixture must be included; pilot success is not a long-run guarantee.
- Pass eager/optimized loss and gradient agreement, batch-row permutation, microbatch-accumulation and cached/replay checks.
- Measure memory at the largest intended prompt, response, K and batch size; retain at least 20% headroom against the declared device/process memory budget. Include restart/serialization peaks and check for sustained growth.
- Show a useful quality-versus-compute trend and an affordable end-to-end repair curve. If the wider model only improves its zero-control answer, report that scaling result without claiming that latent repair scaled.

Benchmark competing inference policies in interleaved order with fixed hardware, warmup, batch sizes and background-load policy. The V3 decision log documents large thermal drift on the fanless M2; a single “method A then method B” timing is insufficient. Report median and tail per-problem latency as well as aggregate throughput. A multi-device implementation additionally needs globally correct domain denominators and independent, reproducible data/control streams.

## 9. Evaluation and acceptance

### Data and replication

1. V5's inspected validation sets become development evidence.
2. Create fresh training/development/locked-test generator seeds and split manifests. Keep the tokenizer fixed initially. Different generator seeds alone do not guarantee disjoint instances: check exact prompt/graph-instance overlap and record duplicate removal.
3. Use disjoint text documents and prevent evaluation text or problems from entering training/repair replay.
4. Evaluate ordinary graphs, hard goals, longer supported horizons, node relabelings and changed graph structures. Lengths beyond supported context are a separate extrapolation study.
5. Use development training seed 0 for engineering and limited tuning. Confirm the selected design on at least three independent foundation/replacement seeds, such as 1,2,3.
6. Report each seed, paired item differences within seeds, and seed variation. Three seeds are limited evidence; pooling their items does not create more independent training runs.
7. Add a prior repository composition task only after the primary GraphNav mechanism is understood.

Standard V5 success reached 100%, so hard goals and distribution shifts are the primary discriminators.

For each generated failure, record malformed output, illegal transition, valid walk ending at the wrong goal, early/late stopping, missing terminator and numerical/resource failure separately. Report colour-choice, destination-lookup and format/stopping likelihood. A lower whole-sequence NLL can conceal a worse decision policy, as the V4.1 long-action audit demonstrated.

Keep the primary comparison unconstrained. A separate syntax-constrained decoding comparison must apply the identical mask to every search policy and replay its renormalized distribution. A graph solver can recover a shortest path directly from the public header; use it to validate reachability and the verifier, and label this as an algorithmic reference. This benchmark tests learned control and efficiency within the neural systems, not the existence of a solution unavailable to classical search.

Count verified diversity by distinct valid action/node trajectories, not differences in padding or malformed tokens. Report both the fraction of problems with an additional valid alternative and unconditional solve rate. Persistence or intermediate accuracy conditional on an eligible subset must include that subset's denominator and an all-problem companion metric.

Proposed confirmation allocation: 2,048 ordinary and 4,096 hard-goal instances per training seed, paired across methods within that seed, with independent locked generator seeds across training seeds. Fix these sizes and the primary cost budget from development measurements before opening test results. They are a starting allocation, not a power guarantee.

The primary endpoint is hard-goal success at the elapsed-time budget used by the deterministic eleven-candidate retry policy on development data, on fixed hardware and batch settings. Lock the budget numerically before confirmation. Methods may use as many complete attempts as fit the budget; the eleven-candidate cap applies to the separate attempt-count comparison. Record incomplete work as cost, never as an extra accepted answer.

### Proposed gates

These are V6 design thresholds, to be locked before viewing fresh test results.

| Gate | Requirement |
|---|---|
| G0: correctness | Causal timing, exact anchor, gradient reach, replay, verifier and final-update checks pass. |
| G1: deterministic transplant | NLL degradation versus its own foundation ≤0.02 nats in each domain; standard success loss ≤2 percentage points. |
| G2: exploration quality | Prior-draw NLL degradation versus the deterministic anchor ≤0.05 nats in each domain; format loss ≤2 points; multiple controls add hard-goal coverage over one deterministic answer. |
| G3: usable authority | Z interventions produce verified behavioral alternatives with other randomness fixed. A large swap rate alone cannot pass. |
| G4: repair direction | GradCuit beats norm-matched random updates by ≥3 hard-goal points, with positive paired uncertainty bounds and a consistent gain across confirmation seeds. |
| G5: deployment advantage | At preregistered end-to-end cost, GradCuit exceeds the strongest retry baseline by ≥3 hard-goal points, with positive uncertainty bounds and no material language/format regression. Publish the complete cost curve. |
| G6: persistence, optional | Holding Z changes intended behavior across decisions while preserving validity and task success; test relabelings and new compositions. Colour persistence alone is insufficient. |

For G1/G2, check upper confidence bounds on degradation, not just favorable means. If a bound crosses a margin, mark the result unresolved.

For prior NLL, average conditional token NLL over fixed independent prior schedules, not over the winning sample; also label any separately reported mixture NLL. Use identical token masks and prompt boundaries for the anchor and controlled arm. Report full-sequence text NLL, including uncontrolled prompt positions, alongside controlled-suffix NLL so a long prompt cannot hide poor continuation quality.

For G4/G5, report 95% paired item-bootstrap intervals within each seed, preserving the pairing of methods on each item. Separately compute the mean paired gain across training seeds and a 95% Student-t interval over those seed means. Require the ≥3-point mean gain, positive lower bounds for both the seed-level interval and each seed's paired interval, and consistent gain signs. With three seeds, the seed interval has only two degrees of freedom and is assumption-sensitive; call inconclusive results unresolved.

Use the corresponding upper-bound rule for G1/G2, with document-level paired resampling for text NLL. Lock one primary method/baseline/cost comparison before testing; other curves and ablations are secondary. If more seeds are needed, preregister a new fixed-size confirmation rather than repeatedly adding seeds until an interval becomes positive.

### Minimum experiment ladder

| Order | Arm | Question |
|---|---|---|
| 0 | NCP + deterministic replacement + token retries; first isolate the declared masking correction | Does the strong baseline reproduce on the new data protocol? |
| 1 | V5 candidate-selection-only control, then an otherwise matched no-epsilon version | What changes when guidance noise is removed? |
| 1a | No-epsilon full-core control: per-window selection versus complete-schedule selection, then learned prior versus fixed bounded prior | What do coherent selection and the prior change separately? |
| 1b | Same fixed prior, complete-schedule objective and adapters: core trainable versus frozen, both initialized from B0 | What does freezing the deterministic anchor change? |
| 1c | Frozen B0 with two direct bias-free Z lifts versus the context-conditioned adapters | Does context gating earn its extra parameters and computation? |
| 2 | Anchored adapters, candidate selection with beta=0 | Can the restricted interface learn useful exploration? |
| 3 | Same with beta=0.25 | Does average-candidate training improve calibration without losing coverage? |
| 4 | Shared-offset versus per-window repair on the selected model | Does temporal separation help? |
| 5 | KL penalty 0 versus 0.05 | Does distribution preservation improve repair per cost? |
| 6 | L=1 versus L=4, both trained | Is holding control useful? |
| 7 | Selected full system and strongest baseline across independent seeds | Does the result survive selection and training variation? |

Fix intermediate width r=16, d_z=8, write cap=0.1 and control RMS bound=2 for this ladder. If oracle reachability is weak, first check the optimizer and gradient path, then compare a declared d_z=16 capacity arm at the same write cap. Increasing r while leaving d_z=8 cannot increase the local control rank beyond eight. A wider or less-capped control family is a recorded design revision, promoted only using fresh development comparisons; the exact zero-control invariant remains mandatory.

Use the same B0 checkpoint, minibatch order, candidate draws, suffix masks and window distribution for paired B1 ablations where the intervention permits. Report both training steps and actual compute; equal steps with different backward graphs are not equal cost. The full-core-to-adapter interface change remains a bundled capacity/parameterization intervention. Do not attribute its effect to rank, context gating or injection location individually without further matched controls.

Run a small privileged reachability diagnostic on development failures before the expensive ladder: use a known valid trajectory to optimize Z by teacher-forced target NLL, with the same adapter, control bounds and window lengths. Predeclare a fixed diagnostic budget, such as four starts × 50 updates on 128 failures; report verified regeneration, not only fitted NLL. Labels must not enter the verifier-only repair arm, and this diagnostic's extra compute cannot count as ordinary GradCuit performance.

Implement the minimal path first: existing anchor → direct/gated adapters → prior coverage → verifier-only repair versus retries. Run the bridge controls to explain a promising result, and replicate the selected system before widening it. The experiment ladder is a decision sequence, not a requirement to launch every expensive arm concurrently.

## 10. Required checks before a future run

This is an acceptance specification. No training or test suite was launched to write the document.

1. Future-token mutation leaves earlier logits unchanged, including chunk boundaries and padded rows.
2. Future-window control mutation leaves earlier logits unchanged.
3. After arbitrary adapter training, Z=0 matches F_det logits and generation; bypassing adapters agrees with their zero-control execution.
4. Lambda=0 reproduces the foundation; lambda=1,Z=0 reproduces F_det, across both ports.
5. Adapter and Z gradients are nonzero where expected; every frozen weight remains unchanged. Z=0 retains a usable local derivative.
6. Initialization yields distinct small candidate perturbations, not an all-zero branch.
7. Candidate scores correspond to realizable complete schedules; no independent window-winner stitching.
8. Each target is counted exactly once. Include the first three positions for full-text likelihood; mask padding and document boundaries.
9. Replayed conditional probabilities match generation's control/noise/temperature/mask contract.
10. On a fixed rejected sequence, a sufficiently small unregularized descent step lowers its log probability when the gradient is nonzero. This does not assert better task success.
11. Every returned updated candidate was regenerated and verified; U=0 checks the original answer.
12. Accepted rows and their optimizer states remain fixed while other rows continue.
13. The verifier checks legal transitions and complete stopping/format semantics, including every token of [DONE, SEP, SEP, SEP]. V5's early DONE handling must not hide malformed trailing output; use the same checker for all baselines and distinguish generated content from unused buffer padding.
14. Changed response controls invalidate affected caches; prompt reuse remains equivalent.
15. Near-zero inputs and clipping have finite derivatives. Nonfinite loss/state/gradients cause a recorded numerical failure with the offending update rejected, not success; exhausted context without a complete terminator is failure.
16. Eager/compiled and cached/uncached outputs, losses and Z gradients match within declared tolerances.
17. Saved per-item records reproduce success/cost curves and count every attempt, replay, backward pass and verifier call.
18. Reused core blocks have distinct cache slots per visit; cached logits and gradients agree with full replay after an earlier window changes.
19. Codebook restarts clear exactly the replaced rows' moments, preserve other rows, and use valid training chunks only.
20. Packing and microbatch/distributed accumulation preserve the declared domain denominators; row permutation does not change per-problem updates.
21. A checkpoint/resume fixture reproduces the next data/control draws and update; incomplete checkpoints are never selected as latest.
22. FP32 versus proposed low precision includes zero-control gradients, cap saturation and small probabilities, with measured tolerances and greedy-margin disagreements reported.
23. Candidate score-then-replay agrees with retained-graph B1; no optimizer step occurs between winner selection and its gradient computation.
24. Full-size parameter ledgers exclude inactive teacher modules while resident-memory ledgers include every loaded copy and precision/optimizer buffer.

## 11. Failure interpretation and deferred mechanisms

| Observation | Interpretation and decision |
|---|---|
| Forecast-loss effect fails replication | Reassess foundation claims before attributing downstream gains to it. |
| B0 fails | Resolve replacement/interface behavior before adding search. |
| Oracle Z fitting fails | Adapter family or optimizer may be too restricted; diagnose before increasing attempts. |
| Oracle succeeds, random Z coverage is poor | Useful controls exist but exploration is poorly calibrated; revisit candidate training or a proposal model. |
| Random Z works, gradients add nothing | Use resampling; inspect parameterization and rejection credit. |
| GradCuit beats random but loses to token retries | Useful mechanism evidence, insufficient deployment advantage. |
| Z becomes nearly irrelevant | Check selection gradients, write caps and overregularization; low KL is not success. |
| Persistence rises while task success falls | Do not claim useful temporal abstraction. |
| Success requires hidden answers | Keep it in the privileged diagnostic table. |

No default learned switch, actor/value network, posterior, epsilon, adaptive halting, diffusion, larger PQ vocabulary, new retrieval block or decoder Engram is introduced.

Later work can train a proposal or shared controller from verified search traces. That introduces new training data and costs and needs a separate experiment. Per-problem GradCuit does not accumulate knowledge across tasks.

Repair claims initially apply to tasks with a trustworthy public verifier. A learned language verifier needs independent false-acceptance/false-rejection assessment, with the same feedback available to every competing search method.

## 12. Review after drafting

Completed after drafting and repeated for revision 2, checking the equations against the local implementation, older experiments and completed result JSON. The second review also addresses the requested scalability and stability. This is a design review; implementation checks in section 10 remain future work.

### Step 1 — does the evidence justify the direction?

Yes, with a narrower claim than “GradCuit won.” NCP forecasting, deterministic replacement and XM prior calibration each have useful local evidence. GradCuit beat random changes but lost to the strongest token-retry control. The design therefore preserves that strong deterministic reference and makes repair earn its extra compute. Single-seed and historical-verifier limits are explicit.

### Step 2 — can exploration destroy the starting model?

The algebraic zero-control invariant protects the complete deterministic path after B1: every added write contains a factor of Z, while all original weights are frozen. This protects Z=0 only. Nonzero controls can still reduce quality, and numerical corruption can still invalidate the invariant. G0/G2 address these different risks.

### Step 3 — can repair move away from that exact zero?

The adapter value vanishes at zero without requiring its Z derivative to vanish. The review corrected the scalar clipping norm to put epsilon inside the square root, and prohibited an inference bypass during repair differentiation. These avoid undefined zero-norm derivatives and an accidentally disconnected first update.

### Step 4 — is every prediction causal?

The chunk plan first affects the position predicting the next chunk. Future controls cannot affect earlier states. The review corrected prompt-cache reuse: the final prompt boundary can already depend on the first response control. Dependency-based invalidation and replay agreement are required.

### Step 5 — does the training objective describe a deployable computation?

Every candidate now has one complete causal control schedule, scored across its suffix. A whole-schedule winner avoids mixing independently selected windows with incompatible histories. Training remains teacher-forced: free-generation coverage and prior quality must still be measured. The mean-candidate term and K=2 are proposed choices, not proven solutions.

### Step 6 — are loss names hiding different gradients?

The review distinguished codebook-only VQ gradients from NTP/NCP gradients that also reach Q in Phase A. It also recorded that V5 VQ includes headers and padding-only chunks, and made the V6 masking correction a separately measured shared baseline change. Frozen B1 weights still transmit gradients into adapters.

### Step 7 — does inference optimize the intended object?

The rejection term decreases the rejected sequence's likelihood. It does not identify the right alternative. KL only constrains the replayed prefixes, and a bounded Z is not a guarantee of bounded task damage. Each update therefore needs fresh generation and verification. The review specified full terminator scoring, per-problem clipping and moment masking so batching does not change the logical algorithm.

### Step 8 — is the claimed benefit attributable?

The review added bridge controls for complete-schedule selection, fixed priors and freezing. The remaining full-core-versus-adapter comparison is explicitly a bundle of interface and capacity changes. A larger gain cannot be credited to one of those ingredients without a further control.

### Step 9 — is the comparison fair?

All new search policies share the first deterministic answer and verifier. Attempt-count and compute-budget comparisons are separate. Extra backward passes, KL references and online norm-matching work are charged. The historical V5 scores are context, not substitutes for rerun baselines.

### Step 10 — does the budget add up?

The corrected analytic deployed count is 2,063,904 parameters, including 8,480 new adapter parameters. Revision 1 omitted removal of the inactive 16,384-parameter teacher A projection. B1 evaluates two candidates per row and differentiates through frozen modules, so its small trainable count does not imply a proportionately small training cost. Instantiated counts and timings remain unverified.

### Step 11 — what could still make the proposal fail?

The main unresolved risk is **insufficient useful control authority**: a frozen core with small capped adapters may preserve competence but fail to expose better alternatives. The opposite risk is poor prior calibration despite useful oracle controls. Further risks are weak gradients near zero, selection collapse, teacher-forcing mismatch and rejection updates that favor another failure. Revision 2 adds direct-lift controls, local-gradient checks, failure categories and a declared control-dimension escalation. These diagnose the risk; they do not resolve it without experiments.

### Step 12 — what is the decision after review?

Proceed to the small mechanism pilot with the declared defaults and gates when implementation is requested. Promote to fresh tiny confirmation, then width/context scaling only after the relevant gates pass. The design preserves the strongest observed starting behavior, gives exploration an explicit training objective, removes an unsuccessful switching dependency, and makes each claimed improvement falsifiable.

### Step 13 — is the architecture scalable?

It now has an explicit 2.06M→7.63M→29.29M width family, separate context experiments, corrected parameter formulas and compute/cache estimates. The audit also states the limit: six full token-attention blocks remain quadratic in context. Memory-efficient execution, score-then-replay training and stage-specific budgets make scaling testable; they do not establish a scaling advantage over the deterministic or ordinary-transformer baselines.

### Step 14 — is the stability plan sufficient?

The draft now covers FP32 reductions, safe zero-norm derivatives, width calibration, loss-unit changes, full gradient paths, codebook moment resets, separate recursive caches and atomic reproducible recovery. Numerical failures cannot silently count as successful continuations. Larger-model and long-run stability remain empirical gates. Post-normalization and input caps alone were insufficient grounds for the earlier stability implication.

### Step 15 — what did the repository-wide review change?

It added the original concept/decoder counterexamples, V2 gradient-conflict lessons, V3 resume/thermal evidence, V4/V4.1 attribution and stopping lessons, and the completed V5 controller/search results. It corrected the active parameter total, added a simpler control interface, separated mathematical invariants from empirical gates, and made the scaling protocol concrete. Evidence from papers remains motivation; local model performance is claimed only where a saved experiment supports it.

## 13. Source and implementation map

- [model_v5.py](v5_tiny/code/model_v5.py): foundation, PQ, recurrence and control injection.
- [system_v5.py](v5_tiny/code/system_v5.py): frozen skeleton, control schedule and port blending.
- [phase_a.py](v5_tiny/code/phase_a.py): foundation objective and optimizer.
- [phase_b.py](v5_tiny/code/phase_b.py): replacement, posterior and XM training.
- [phase_d.py](v5_tiny/code/phase_d.py): switching and trained-controller experiments.
- [phase_e.py](v5_tiny/code/phase_e.py), [v2_bestof.py](v5_tiny/code/v2_bestof.py): repair and retry protocols.
- [Original five-phase design](NCP_GRAM_Z_Five_Phase_Architecture.md), [mathematical review](review_v5.md).
- [Methods](v5_tiny/METHODS.md), [original preregistration](v5_tiny/PRE_REGISTRATION.md).
- [V4 draft](draft_v4.md), [V4.1 draft](draft_v4_1.md), [V4.1 assessment](v41_tiny/ASSESSMENT.md).
- [V3 trainer](run_v3_19M/code/train.py): complete-chunk codebook updates, moment reset, exact accumulation and checkpoint recovery.
- [V3 operational evidence](run_v3_19M/DECISIONS.md): gradient/caching checks, memory guards and thermal timing confounds.
- [V4 codebook implementation](v4_tiny/code/model_v4.py): row-specific optimizer-state reset.
- [V5 data and verifier](v5_tiny/code/data_v5.py): document-stream convention, graph generation, stopping semantics and header-only verification.

Future implementation should use a separate v6_tiny directory and preserve all V5 checkpoints/results. This deliverable specifies the architecture only.

## 14. Repository-review coverage and limits

Revision 2 inventoried 3,061 files across 348 subdirectories, including hidden/dependency/cache entries. The audit read/scanned 1,735 project text files, parsed all 164 JSONL files (32,490 records), parsed JSON results and Python syntax, and grouped duplicate text by content. It extracted text from all 49 PDFs (1,359 pages including duplicates), the DOCX design conversation, and inspected the two RTF conversations and decoder archive. Detailed semantic review focused on the architecture/training/evaluation paths and evidence bearing on V6.

Binary inspection covered the storage metadata of 120 checkpoints and headers of 39 NPY/100 NPZ files. Checkpoint tensors and full training corpora were not deserialized or rerun. The 978 dependency/cache entries were inventoried; vendored packages and bytecode are not architecture evidence. Paper extraction is coverage of text, not a claim of a new line-by-line technical validation of all 1,359 pages.

| Repository area | Review contribution |
|---|---|
| Encoder Design, Concept Design, Decoder Design | Information boundaries, control injection, isolated versus end-to-end results, archived conversation/spec lineage |
| run_m2_25M, run_v2_19M, run_v3_19M | Whole-model wiring, data/gradient paths, recurrence, parameter conventions, recovery and timing evidence |
| audit_v2_2026-09-28, v2_underperformance_review_2026-09-28 | Dataset overlap, objective conflict, routing bias and gradient truncation |
| v3_design and both V3 review folders | Reference implementations, frozen snapshots, causal controls and prior acceptance-test gaps |
| v4_tiny, v41_tiny | Pooling/recurrence controls, fact retrieval, calibrated plan interventions, long-horizon stopping and per-item evidence |
| v5_tiny | Current NCP/replacement/control implementation, completed B–E JSON, verifier and runtime reductions |
| Root drafts/reviews and REPO_WALKTHROUGH | Version mapping and historical assumptions cross-checked against code/results |
| Loop Papers, Imp Paper, Emergence in Residual Stream, root PDFs | Mechanism and scaling background; particularly NCP, GRAM, XM, GradCuit, temporal abstraction, DeepLoop and compute-matched looping |
| Dashboards, launch scripts, archived copies, .claude and generated artifacts | Inventory and provenance; presentation/old operational instructions do not supersede the current task |

Only this architecture draft was revised in the repository. No model implementation, checkpoint, training process or experiment result was changed.
