# V15-XHRM-Atlas — Generalized Architecture Specification: Two-Stage Frozen Residual Codec, Explorative HRM Concept World Model, and GradCuit–ATLAS Wake–Sleep Learning

**Status:** Generalized architecture specification (Tiny **6.79M** -> Mid **7B-A1.4B** -> Frontier **2.8T-A115B**); revised **16 October 2026, IST** from completed V13/V14 empirical audits[span_0](start_span)[span_0](end_span)[span_1](start_span)[span_1](end_span)[span_2](start_span)[span_2](end_span) and primary literature on *Next-Concept Modeling* (NCM)[span_3](start_span)[span_3](end_span), *HRM-Text*[span_4](start_span)[span_4](end_span), *Explorative Modeling* (XM)[span_5](start_span)[span_5](end_span), *GradCuit*[span_6](start_span)[span_6](end_span), *Titans*[span_7](start_span)[span_7](end_span), and *ATLAS*[span_8](start_span)[span_8](end_span).

---

## 0. Executive Summary & Plain-English Mental Model

**Design in one sentence:** Pretrain and permanently freeze a generalized multi-resolution "Concept Tokenizer" (Stage 1 Codec) so future concept labels never drift[span_9](start_span)[span_9](end_span); train a causal Local/Global backbone whose dual-timescale **HRM** Concept Core explores $K$ discrete future modes via **Explorative Modeling** and feeds its top predictions to a confined decoder via **PlanAttention** (Stage 2)[span_10](start_span)[span_10](end_span)[span_11](start_span)[span_11](end_span)[span_12](start_span)[span_12](end_span); and adapt at test/continual time by optimizing mid-core latents via **GradCuit** during Wake, storing fast weights in a fixed-size **ATLAS** memory, and amortizing discoveries into bounded LoRA + replay during Sleep (Stage 3)[span_13](start_span)[span_13](end_span)[span_14](start_span)[span_14](end_span)[span_15](start_span)[span_15](end_span).

### Why This Architecture Exists (In Plain English)
Across versions V1 through V14, five hard lessons emerged[span_16](start_span)[span_16](end_span)[span_17](start_span)[span_17](end_span)[span_18](start_span)[span_18](end_span):
1. **The Shortcut Lesson (V1–V13):** If the token decoder can look directly at far-away past words, it ignores the Concept Core[span_19](start_span)[span_19](end_span). Bounding the decoder's direct token reach ($R_D$) forces all long-range memory to flow through the Concept Core[span_20](start_span)[span_20](end_span)[span_21](start_span)[span_21](end_span).
2. **The Moving Dictionary Lesson (V13–V14):** In V13 and V14, the model tried to invent its Concept IDs (via online k-means/EMA) *at the exact same time* it was learning to predict them[span_22](start_span)[span_22](end_span)[span_23](start_span)[span_23](end_span)[span_24](start_span)[span_24](end_span). Even with V14's EMA twin, ridge regression $P$, and SIGReg, **96.36%** of the global codes changed their meaning mid-training, and teacher/online agreement hit only **40.96%**[span_25](start_span)[span_25](end_span). **Fix:** Build and freeze the "Concept Tokenizer" *first* in Stage 1 (**0% code churn**)[span_26](start_span)[span_26](end_span).
3. **The Unused Forecast Lesson (V13–V14):** V13 and V14 trained an NCP head to predict future codes, but the token decoder *never read those predictions*[span_27](start_span)[span_27](end_span)[span_28](start_span)[span_28](end_span)[span_29](start_span)[span_29](end_span). **Fix:** Add a detached **`PlanAttention` bridge** so the decoder actively consults the top predicted future plans when writing tokens[span_30](start_span)[span_30](end_span).
4. **The Shallow Concept Core Lesson (V13–V14 Diagnostic V3):** In both V13 and V14, a probe on the Concept Core's output ($u_g$) failed to beat a probe on its pooled input ($p_g$) by 5 points across all four reasoning families[span_31](start_span)[span_31](end_span)[span_32](start_span)[span_32](end_span), and held-out reasoning rungs 8/9 scored **0%**[span_33](start_span)[span_33](end_span). **Fix:** Replace static 4-block depth attention with **`HRM-Text`'s dual-timescale recurrent core** (fast $L$-module + slow $H$-module with `MagicNorm` and warmup credit assignment)[span_34](start_span)[span_34](end_span)[span_35](start_span)[span_35](end_span)[span_36](start_span)[span_36](end_span) paired with **$K$ Explorative Modeling (XM) mode embeddings** so $C$ performs deep, multimodal deliberation ($E \ge K$)[span_37](start_span)[span_37](end_span)[span_38](start_span)[span_38](end_span)[span_39](start_span)[span_39](end_span).
5. **The Continual Memory Bloat Lesson (V14 Stage 3):** V14's growing RBF kernel bank lost to simple replay (**0.3033 vs. 0.3341**), learned less than half as much on new tasks, and spiked MPS driver memory to **17.01 GiB**[span_40](start_span)[span_40](end_span). **Fix:** Use **GradCuit** (test-time gradient search at the 50% $L \to H$ circuit midpoint)[span_41](start_span)[span_41](end_span)[span_42](start_span)[span_42](end_span) + **ATLAS** (a fixed-size 2-layer Gated MLP memory updated via the sliding-window Omega rule and Newton-Schulz `Muon`)[span_43](start_span)[span_43](end_span)[span_44](start_span)[span_44](end_span)[span_45](start_span)[span_45](end_span) during Wake, then distill into bounded Rank-4 LoRA + 20% Replay during Sleep[span_46](start_span)[span_46](end_span).

### Component Map: `TPH-X-GradAtlas`

| Symbol | Component Name | What Job It Does | Source Lineage |
| :--- | :--- | :--- | :--- |
| **`T`** | **Generalized Frozen Codec ($\mathcal{C}_{\text{gen}}$)** | Pre-builds a stationary hierarchical residual dictionary of future spans (`Coarse Average + Fine Differences`), then freezes forever[span_47](start_span)[span_47](end_span). | NCM[span_48](start_span)[span_48](end_span) + Generalized Tree RVQ |
| **`P`** | **Top-$B_0$ `PlanAttention` Bridge** | Extracts the top-$K_{\text{plan}}$ joint future plans (plus a null tail-mass fallback) and hands them detached to the token decoder[span_49](start_span)[span_49](end_span). | V15 §8[span_50](start_span)[span_50](end_span) + Beam Extension |
| **`H`** | **HRM Dual-Timescale Core** | Splits $C$ into a fast local $L$-module and slow strategic $H$-module with `MagicNorm` and $K_{\text{bptt}} = 2 \to 5$ credit warmup[span_51](start_span)[span_51](end_span)[span_52](start_span)[span_52](end_span)[span_53](start_span)[span_53](end_span). | `HRM-Text`[span_54](start_span)[span_54](end_span)[span_55](start_span)[span_55](end_span)[span_56](start_span)[span_56](end_span) |
| **`X`** | **Explorative Mode Bank ($Z_{1:K}$)** | Injects $K$ discrete latent embeddings $e_{1:K}$ at the 50% $L \to H$ handoff, trained via Smooth Forward XM to capture $K$ distinct future modes[span_57](start_span)[span_57](end_span)[span_58](start_span)[span_58](end_span)[span_59](start_span)[span_59](end_span). | `Explorative Modeling`[span_60](start_span)[span_60](end_span)[span_61](start_span)[span_61](end_span)[span_62](start_span)[span_62](end_span) |
| **`Grad`** | **GradCuit Circuit Optimizer** | Optimizes a continuous offset $\Delta z_g$ at the 50% $L \to H$ interface at test/wake time via direct self-attention gradients to steer reasoning connectors[span_63](start_span)[span_63](end_span)[span_64](start_span)[span_64](end_span)[span_65](start_span)[span_65](end_span). | `GradCuit`[span_66](start_span)[span_66](end_span)[span_67](start_span)[span_67](end_span)[span_68](start_span)[span_68](end_span) |
| **`Atlas`** | **ATLAS-MAG Fast-Weight Memory** | Fixed-size 2-layer Gated MLP with degree-$p$ polynomial features $\phi_p$, Omega-rule context surprise, and `NS-5` Muon updates[span_69](start_span)[span_69](end_span)[span_70](start_span)[span_70](end_span)[span_71](start_span)[span_71](end_span)[span_72](start_span)[span_72](end_span). | `Titans`[span_73](start_span)[span_73](end_span) & `ATLAS`[span_74](start_span)[span_74](end_span) |

---

## 1. End-to-End Architecture Diagram

~~~mermaid
flowchart TD
  subgraph Stage1["STAGE 1 (Stage A): Generalized Hierarchical Residual Codec (Frozen Before Stage 2)"]
    SpanY["Future Span Y_g (W = M*k tokens)"] --> Stem["Stage-1A Shared Causal Stem (N_stem layers, 1x/token)"]
    Stem --> WinBi["Stage-1B Bidirectional Window Aggregator (N_span = 2 layers)"]
    WinBi --> TreeVQ["L-Level Dyadic Subspace Residual VQ (Codebooks E_0 ... E_{L-1})<br/>Coarse Root c* + Conditional Fine Residuals r*"]
    TreeVQ --> WeakDec["Shallow Bottleneck Decoder (N_dec = 1..4 layers, ZERO skip connections)<br/>Trained with Nested Level Dropout -> Thrown Away After Stage 1"]
  end

  subgraph Stage2["STAGE 2 (Stage B): Causal Backbone + XHRM World Model + PlanAttention Bridge"]
    TokX["Observed Prefix x_{<=4g+3}"] --> EncLG["Causal Role-Masked Encoder (E1 + Engram, E2 GDN-2, E3)<br/>Strict One-Way Valve: L -> G only, G never -> L"]
    EncLG -- "3-Source Pool (x, sg[L_enc], G_enc)" --> PoolP["Completed Group Input p_g"]
    PoolP --> FastL["Fast L-Module (n_L blocks, S_L=3 steps, MagicNorm)<br/>Reads p_g + 16-token sg[L_enc] window + Persistent Tokens P"]
    
    FastL --> MidCircuit["50% Mid-Core Circuit Interface<br/>z_tilde_{g,k} = z_H + z_L + e_k + Delta_z_g"]
    XMBank["K Learned Discrete XM Modes {e_1..e_K}"] --> MidCircuit
    Router["Mode Router pi_phi(k | z_L)"] --> MidCircuit

    MidCircuit --> SlowH["Slow H-Module (n_H blocks, C_H=2 cycles, MagicNorm)<br/>Warmup Deep Credit Assignment (K_bptt = 2 -> 5)"]
    SlowH --> Prior["Tree-Conditioned Prior Head + Auto-Guidance<br/>Smooth Forward XM Loss: -log(1/K sum_k exp(-l_k))"]
    TreeVQ -. "Frozen Target Codes (0% Churn)" .-> Prior

    Prior --> Beam["Detached Top-B0 Beam Joint Selection<br/>Top-K_plan Candidates + Tail-Mass Null Vector"]
    EncLG -- "Fixed Local K/V (L_enc, Reach R_D)" --> DecD["Confined Token Decoder D (N_D blocks)"]
    SlowH -- "Router-Weighted Historical U_g" --> DecD
    DecD --> PlanAttn["Gated PlanAttention Bridge"]
    Beam --> PlanAttn
    PlanAttn --> Logits["Next-Token Logits"]
  end

  subgraph Stage3["STAGE 3 (Stage C): Wake-Sleep Test-Time & Continual Learning"]
    Logits --> Verifier["Task / Self-Reward Verifier R(x, c)"]
    Verifier -- "Wake Step 1: If R = -1, Backprop through Self-Attn Circuit" --> GradOpt["GradCuit Adam Update on Delta_z_g at 50% Midpoint"]
    GradOpt --> MidCircuit
    GradOpt -- "Wake Step 2: Commit Winning Transition (k_g, v*_g)" --> AtlasMem["ATLAS-MAG 2-Layer Gated MLP Memory<br/>Degree-p Polynomial phi_p(k) + Omega Rule (c=4) + NS-5 Muon"]
    AtlasMem --> DecD
    AtlasMem -- "Sleep Step: Distill Delta_z*_g & M_ATLAS into System 1" --> SleepLoRA["Bounded Rank-r LoRA + 20% Replay + Tuple-EBM Anchor"]
  end
~~~

---

## 2. Foundational Indexing, Causality, and Reach Invariants

Every tier of `V15-XHRM-Atlas` enforces four strict structural contracts[span_75](start_span)[span_75](end_span)[span_76](start_span)[span_76](end_span):

1. **Document-Relative Zero-Based Indexing:** Within any packed sequence of length $T$, token row $t \in \{0, \dots, T-1\}$ consumes token $x_t$ and predicts $x_{t+1}$[span_77](start_span)[span_77](end_span). All attentions, short convolutions, recurrent GDN-2 states, and concept pools reset strictly at document boundaries[span_78](start_span)[span_78](end_span)[span_79](start_span)[span_79](end_span).
2. **Concept Stride and Release Timing:** Tokens are grouped with base stride $k = 4$[span_80](start_span)[span_80](end_span)[span_81](start_span)[span_81](end_span). Group $g \in \{0, \dots, T/k - 1\}$ spans tokens $x_{kg : kg+k}$ and completes at **release row** $e_g = kg + k - 1 = 4g + 3$[span_82](start_span)[span_82](end_span). At decoder row $t$, the newest legally completed concept group is[span_83](start_span)[span_83](end_span)[span_84](start_span)[span_84](end_span):
   $$g(t) = \left\lfloor \frac{t+1}{4} \right\rfloor - 1$$
   If $g(t) < 0$ (the first 3 tokens of any document), historical concept cross-attention routes to its learned null entry and the `PlanAttention` bridge outputs identically zero[span_85](start_span)[span_85](end_span)[span_86](start_span)[span_86](end_span).
3. **One-Way Role Valve ($L \to G$, Never $G \to L$):** The encoder maintains two local branches ($L_1, L_2$) and two global branches ($G_1, G_2$)[span_87](start_span)[span_87](end_span)[span_88](start_span)[span_88](end_span). Every sublayer that writes to $L$ reads *only* $L$ (or token IDs), and every scalar gate, write scale, and within-role Sinkhorn matrix touching $L$ is computed exclusively from $L$[span_89](start_span)[span_89](end_span).
4. **Confined Reach Ratio ($\rho = R_D / T_{\text{train,min}} \le 1/8$):** Because the decoder's local self-attention takes keys and values from the fixed encoder exit state $L_{\text{enc}}$ within window $w_D$, local reach never compounds across decoder depth[span_90](start_span)[span_90](end_span)[span_91](start_span)[span_91](end_span):
   $$R_L = n_{S, E1}(w_E - 1) + (n_{\max} - 1) + (\kappa - 1)\delta, \qquad R_D = R_L + (w_D - 1)$$
   * **Tiny Tier ($T = 1,024$):** $n_{S,E1} = 6, w_E = 8, n_{\max} = 3, \kappa = 4, \delta = 3 \implies R_L = 51$, and with $w_D = 16$, $R_D = 66$ ($\rho = 0.064$)[span_92](start_span)[span_92](end_span)[span_93](start_span)[span_93](end_span).
   * **3T Frontier Tier ($T_{\text{train,min}} = 4,096$):** $n_{S,E1} = 6, w_E = 32 \implies R_L = 197$, and with $w_D = 128$, $R_D = 324$ ($\rho = 0.079 \le 1/8$)[span_94](start_span)[span_94](end_span).
   Consequently, information from $> R_D$ tokens in the past can reach the output logits **only** through the Concept Core[span_95](start_span)[span_95](end_span)[span_96](start_span)[span_96](end_span).

---

## 3. Stage 1 (Stage A): Generalized Hierarchical Residual Codec ($\mathcal{C}_{\text{gen}}$)

### 3.1 First-Principles Design: Why an Asymmetric Autoencoder?
Stage 1 builds a permanent "Concept Tokenizer" over future windows $Y_g = x[e_g + 1 : e_g + 1 + W]$ of length $W = M \cdot k$ tokens ($M$ groups of $k=4$ tokens)[span_97](start_span)[span_97](end_span).
* **Why the Codec Encoder Scales Up:** As vocabulary $V$ and task diversity grow from Tiny ($V = 2,048$)[span_98](start_span)[span_98](end_span) to 3T ($V = 160,000$)[span_99](start_span)[span_99](end_span), the encoder needs deep representation capacity to extract syntactic, mathematical, and algorithmic structure before quantizing.
* **Why the Codec Decoder Stays Shallow ($N_{\text{dec}} \in [1, 4]$):** If the throwaway reconstruction decoder is too deep, it acts as a standalone language model that guesses fluent text while ignoring the discrete codes[span_100](start_span)[span_100](end_span). Keeping $N_{\text{dec}} \ll N_{\text{enc}}$ with **zero skip connections** acts as an information funnel that forces the encoder to pack maximum semantic content into the codebooks[span_101](start_span)[span_101](end_span). Once Stage 1 finishes, the Codec Decoder is discarded[span_102](start_span)[span_102](end_span).

### 3.2 Factored Stage-1A Stem + Stage-1B Window Encoder
Because a new $W$-token window starts every $k = 4$ tokens, running a deep encoder independently on every window would recompute every token $W/k$ times ($4\times$ at $W=16$; $8\times$ at $W=32$)[span_103](start_span)[span_103](end_span). $\mathcal{C}_{\text{gen}}$ eliminates this redundancy by splitting the encoder into two parts:
1. **Stage-1A Shared Causal Stem ($N_{\text{stem}}$ blocks, run $1\times$ per token):**
   Given document tokens $x_{0:T-1}$ and standalone codec embedding $E_{\text{codec}} \in \mathbb{R}^{V \times d_c}$, run $N_{\text{stem}}$ causal PreLN Transformer blocks (RoPE + SwiGLU of width $2d_c$) once across the sequence to produce cached features $h^{\text{stem}}_{0:T-1} \in \mathbb{R}^{T \times d_c}$. *(At Tiny scale, $N_{\text{stem}} = 0$, so $h^{\text{stem}}_t = E_{\text{codec}}[x_t]$, matching V15-tiny byte-for-byte[span_104](start_span)[span_104](end_span); at 7B/3T, $N_{\text{stem}} = 8\text{ to }16$).*
2. **Stage-1B Bidirectional Window Aggregator ($N_{\text{span}} = 2$ blocks):**
   For each valid anchor $g$, slice $H_g^{(0)} = h^{\text{stem}}[e_g+1 : e_g+1+W] + P_{\text{enc}} \in \mathbb{R}^{W \times d_c}$ and run $N_{\text{span}} = 2$ bidirectional PreLN blocks masked strictly to same-document valid tokens $v_{g, \tau} \in \{0, 1\}$[span_105](start_span)[span_105](end_span).
3. **Stride-$k$ Group Pooling:**
   Pool each of the $M$ 4-token groups over its valid tokens, project with bias-free $W_{\text{pool}} \in \mathbb{R}^{d_c \times d_c}$, and apply learned RMSNorm[span_106](start_span)[span_106](end_span):
   $$z_m = \text{RMSNorm}\!\left( W_{\text{pool}} \left( \frac{\sum_{\tau = mk}^{(m+1)k - 1} v_{g,\tau}\, H_{g,\tau}^{(N_{\text{span}})}}{\max\!\left(\sum_{\tau = mk}^{(m+1)k - 1} v_{g,\tau},\, 1\right)} \right) \right) \in \mathbb{R}^{d_c}, \quad m \in \{0, \dots, M-1\}$$

### 3.3 $L$-Level Dyadic Subspace Residual Quantization (`Average + Differences`)
Let the hierarchy have $L \ge 2$ levels with slot counts $1 = M_0 < M_1 < \dots < M_{L-1} = M$, where level $l$ partitions the $M$ groups into $M_l$ spans of $s_l = M / M_l$ groups each ($\mathcal{I}_{l, i} = \{i s_l, \dots, (i+1)s_l - 1\}$). Each level $l$ splits $\mathbb{R}^{d_c}$ into $G_q \ge 1$ subspaces of width $d_{\text{sub}} = d_c / G_q$ with codebooks $E_{l, j} \in \mathbb{R}^{N_l \times d_{\text{sub}}}$.

Starting from $\hat{z}^{(-1)}_m = \mathbf{0} \in \mathbb{R}^{d_c}$, iterate top-down for $l = 0, \dots, L-1$:
1. **Subtract Coarser Reconstruction (The Residual Step):**
   $$u^{(l)}_m = z_m - \hat{z}^{(l-1)}_m, \qquad \bar{u}^{(l)}_i = \frac{\sum_{m \in \mathcal{I}_{l,i}} \nu_{g,m}\, u^{(l)}_m}{\max\!\left(\sum_{m \in \mathcal{I}_{l,i}} \nu_{g,m},\, 1\right)} \in \mathbb{R}^{d_c}$$
2. **Quantize Each Subspace $j \in \{1, \dots, G_q\}$ Against Pre-Update Codebooks:**
   $$c^*_{l, i, j} = \arg\min_{v \in \{0, \dots, N_l - 1\}} \left\| \bar{u}^{(l)}_{i, j} - E^{\text{pre}}_{l, j}[v] \right\|_2^2, \qquad e^*_{l, i} = \bigoplus_{j=1}^{G_q} E^{\text{pre}}_{l, j}\big[c^*_{l, i, j}\big] \in \mathbb{R}^{d_c}$$
3. **Accumulate Quantized Approximation:**
   $$\hat{z}^{(l)}_m = \hat{z}^{(l-1)}_m + e^*_{l, i} \quad \forall m \in \mathcal{I}_{l, i}$$
*(In the Tiny preset: $W=16, M=4, L=2, M_0=1, M_1=4, G_q=1, N_l=32, d_c=64$, yielding 1 coarse root $c^* \in E_0$ and 4 fine group residuals $r^*_{1:4} \in E_1$[span_107](start_span)[span_107](end_span)).*

### 3.4 Nested Level Dropout, EMA Updates, and Readiness Freeze
* **Nested Level Dropout Objective:** On each window, sample a cutoff level $l_{\text{cut}} \sim \mathbf{p}_{\text{drop}}$ (Tiny: $25\%$ coarse-only $l_{\text{cut}}=0$, $75\%$ full $l_{\text{cut}}=1$)[span_108](start_span)[span_108](end_span). Pass the straight-through vector $\tilde{z}^{(l_{\text{cut}})}_m = \bar{z}^{(l_{\text{cut}})}_m + \text{sg}\big(\hat{z}^{(l_{\text{cut}})}_m - \bar{z}^{(l_{\text{cut}})}_m\big)$ repeated $k$ times plus decoder positional embeddings $P_{\text{dec}}$ through the shallow $N_{\text{dec}}$-block bidirectional decoder and tied head $E_{\text{codec}}^T / \sqrt{d_c}$[span_109](start_span)[span_109](end_span):
  $$\mathcal{L}_{\text{Stage1}} = \text{CE}_{\text{recon}}\!\left(\tilde{z}^{(l_{\text{cut}})},\, Y_g\right) + 0.25 \sum_{l=0}^{l_{\text{cut}}} \text{mean}_{i}\!\left( \frac{1}{d_c}\left\| \bar{u}^{(l)}_i - \text{sg}(e^*_{l,i}) \right\|_2^2 \right)$$
* **Post-Step EMA Update & Two-Strike Dead-Code Reset:** After `optimizer.step()`, update all codebooks $E_{l, j}$ via EMA cluster means ($\gamma = 0.99$) using the saved detached **pre-step** residuals $\text{sg}(\bar{u}^{(l)}_{i, j})$[span_110](start_span)[span_110](end_span). Every 100 steps, any entry whose EMA count falls below $1\%$ of uniform expected usage on **two consecutive checks** is reseeded to the valid residual vector in the current batch with the highest quantization error[span_111](start_span)[span_111](end_span).
* **Readiness Gate & SHA-256 Freeze:** Held-out coarse-only reconstruction CE must beat a training-fitted phase-position unigram baseline by $\ge 0.05\text{ nats/token}$ (equal-family mean), and each finer level must improve CE by another $\ge 0.05\text{ nats/token}$ (positive in $\ge 3$ of 5 families)[span_112](start_span)[span_112](end_span). The winning checkpoint computes per-level scalar RMS values $s_l = \max(\text{RMS}(E_l), 10^{-4})$ and is **permanently frozen and hashed**[span_113](start_span)[span_113](end_span).

---

## 4. Stage 2 (Stage B): Causal Backbone, `XHRM` Concept Core, and Smooth Forward XM

### 4.1 Inherited Causal Local/Global Encoder ($E_1, E_2, E_3$) and Pooler
Stage 2 retains the proven V14/V15 role-masked encoder[span_114](start_span)[span_114](end_span)[span_115](start_span)[span_115](end_span):
* **$E_1$:** Two `[S, S, S, F]` blocks (Tiny $w_E = 8$; Scale $w_E = 32$) with the **Engram** conditional bigram/trigram lookup module after the first `S` layer[span_116](start_span)[span_116](end_span)[span_117](start_span)[span_117](end_span).
* **$E_2$:** One `[M, M, M, F]` Gated DeltaNet-2 workspace pass ($R=1$ at Tiny; looped MoE $R=2$ at 7B/3T where sparse routing differs across passes) initialized from $s_0$ ($\Vert{}s_0\Vert{} \approx \sqrt{d}$) and writing exclusively to $G$ at exit[span_118](start_span)[span_118](end_span)[span_119](start_span)[span_119](end_span).
* **$E_3$:** `[S', F]` refining $G$ only[span_120](start_span)[span_120](end_span), followed by Gated Residual exit reads producing $L_{\text{enc}}, G_{\text{enc}} \in \mathbb{R}^{B \times T \times d}$[span_121](start_span)[span_121](end_span)[span_122](start_span)[span_122](end_span).
* **3-Source Group Pooler:** Every $k=4$ tokens, channel-wise attention pools over $x$, $\text{sg}[L_{\text{enc}}]$, and $G_{\text{enc}}$ are RMS-normalized, mixed via learned softmax weights $\beta$, and projected to the concept input $p_g = W_p m_g \in \mathbb{R}^d$[span_123](start_span)[span_123](end_span)[span_124](start_span)[span_124](end_span).

### 4.2 The `XHRM` Concept Core ($C$): Dual-Timescale Recurrence + Explorative Modes
To fix V14's failed V3 diagnostic and 0% hard-rung accuracy[span_125](start_span)[span_125](end_span), $C$ replaces single-pass static blocks with a dual-timescale **HRM** core[span_126](start_span)[span_126](end_span)[span_127](start_span)[span_127](end_span) partitioned at its 50% midpoint by $K$ **Explorative Modeling (XM)** discrete latent modes[span_128](start_span)[span_128](end_span)[span_129](start_span)[span_129](end_span):

1. **Persistent Memory Prefix (`Titans` §3.3):** Prepend $N_p = 4$ (Scale: $16$) learnable, data-independent **Persistent Memory tokens** $P = [p_1, \dots, p_{N_p}] \in \mathbb{R}^{N_p \times d}$ to the concept sequence to encode task-level priors and absorb initial-token attention sinks[span_130](start_span)[span_130](end_span).
2. **Fast $L$-Module ($n_L = 2$ blocks, $S_L = 3$ steps per cycle, Parameterless `MagicNorm`):**
   * Initialized with slow state $z_{H,g}^{(0)} = p_g$ and fast state $z_{L,g}^{(0)} = z_L^{\text{init}}$[span_131](start_span)[span_131](end_span).
   * Each block runs `[Parameterless RMSNorm -> Gated Causal Concept Self-Attn -> Gated 16-Token Local Cross-Attn to sg[L_enc] -> Parameterless RMSNorm -> SwiGLU]`[span_132](start_span)[span_132](end_span)[span_133](start_span)[span_133](end_span).
   * **MagicNorm (`HRM-Text` §2.1.1):** Each recurrent step $s \in \{1, 2, 3\}$ caps the sum of internal PreNorm block deltas with a module-exit parameterless RMSNorm[span_134](start_span)[span_134](end_span)[span_135](start_span)[span_135](end_span):
     $$z_{L,g}^{(s)} = \text{RMSNorm}\!\left(z_{L,g}^{(s-1)} + \sum_{l=1}^{n_L} \text{Block}_{L,l}\big(z_{L,g}^{(s-1)}, z_{H,g}^{(c-1)}, \text{sg}[L_{\text{enc}}]\big)\right)$$
3. **The 50% Mid-Core Circuit Interface (XM Mode Bank + Mode Router + GradCuit Hook):**
   * At the handoff from the fast $L$-module to the slow $H$-module (50% core depth, optimal for latent steering per `GradCuit` Figure 5[span_136](start_span)[span_136](end_span)), maintain $K$ learned discrete mode embeddings $Z_{\text{XM}} = \{e_1, \dots, e_K\} \subset \mathbb{R}^d$ (Tiny: $K = 4$; 7B: $K = 8$; 3T: $K = 16$)[span_137](start_span)[span_137](end_span)[span_138](start_span)[span_138](end_span) and an instance-specific continuous offset $\Delta z_g \in \mathbb{R}^d$ ($\Delta z_g = \mathbf{0}$ during Stage 2 pretraining and standard 1-pass inference)[span_139](start_span)[span_139](end_span).
   * A lightweight **Mode Router** predicts mode probabilities from the fast state:
     $$\pi_\phi(k \mid z_{L,g}^{(S_L)}) = \text{softmax}\!\left(W_r\,\text{RMSNorm}\big(z_{L,g}^{(S_L)}\big) + b_r\right) \in \Delta^{K-1}$$
   * On the final outer cycle $c = C_H$, branch the state into $K$ parallel candidates folded into the batch dimension[span_140](start_span)[span_140](end_span):
     $$\tilde{z}_{H,g,k}^{(C_H)} = z_{H,g}^{(C_H - 1)} + z_{L,g}^{(S_L)} + e_k + \Delta z_g, \quad k \in \{1, \dots, K\}$$
4. **Slow $H$-Module ($n_H = 2$ blocks, $C_H = 2$ cycles, `MagicNorm` + Credit Warmup):**
   * Runs `[Parameterless RMSNorm -> Gated Causal Concept Self-Attn -> Parameterless RMSNorm -> SwiGLU]` capped by exit `MagicNorm` (omitting the 16-token local token read since local context is already integrated by $L$, saving $131,072$ parameters at Tiny)[span_141](start_span)[span_141](end_span)[span_142](start_span)[span_142](end_span)[span_143](start_span)[span_143](end_span).
   * Cycle 1 updates $z_{H,g}^{(0)} \to z_{H,g}^{(1)}$ ($K=1$); Cycle 2 evaluates the $K$ branched states in parallel to produce mode-conditioned concept outputs $u_{g,k} = z_{H,g,k}^{(2)} \in \mathbb{R}^d$ for $k \in \{1, \dots, K\}$[span_144](start_span)[span_144](end_span)[span_145](start_span)[span_145](end_span).
   * **Warmup Deep Credit Assignment (`HRM-Text` §2.1.2):** To prevent multiplicative Jacobian spikes across recurrent unrolls[span_146](start_span)[span_146](end_span)[span_147](start_span)[span_147](end_span), backpropagate through only the final $K_{\text{bptt}} = 2$ recurrent module calls during early student training ($s \in [0, 0.2]$), warming up linearly to $K_{\text{bptt}} = 5$ steps by $s = 0.5$[span_148](start_span)[span_148](end_span)[span_149](start_span)[span_149](end_span).

### 4.3 Tree-Conditioned Prior & Smooth Forward XM Objective
For each explored mode $k \in \{1, \dots, K\}$, normalize $v_{g,k} = \text{RMSNorm}(u_{g,k})$[span_150](start_span)[span_150](end_span):
1. **Hierarchical Coarse-to-Fine Prior Head:**
   * **Level 0 (Coarse Root):** $q_0(c_0 \mid u_{g,k}) = \text{softmax}\big(\text{MLP}_0(v_{g,k})\big)$[span_151](start_span)[span_151](end_span).
   * **Level $l > 0$ (Root/Ancestor-Conditioned Residuals):** For slot $i \in \{0, \dots, M_l - 1\}$ with horizon embedding $\text{hor}_{l, i}$ and learned parent-code address tables $\text{addr}_\lambda$[span_152](start_span)[span_152](end_span):
     $$a_{l, i, k} = W_{u,l} v_{g,k} + b_{u,l} + \text{hor}_{l, i} + \sum_{\lambda = 0}^{l-1} \text{addr}_\lambda\big[c_{\lambda, \text{anc}(\lambda, i)}\big]$$
     $$q_{l, i}\big(r \;\big\vert{}\; u_{g,k}, c_{<l}\big) = \text{softmax}\big(W_{f,l}\,\text{GELU}(a_{l, i, k}) + b_{f,l}\big)$$
2. **Per-Mode Loss $\ell_k(g)$ with Counterfactual Root Retargeting:**
   Using frozen Stage-1 labels $(c^*, r^*)$ and horizon weights $w = (0.4, 0.3, 0.2, 0.1)$[span_153](start_span)[span_153](end_span):
   $$\ell_{k, \text{clean}}(g) = \frac{1}{L}\text{CE}\big(q_0(\cdot \mid u_{g,k}), c^*\big) + \frac{L-1}{L}\sum_{l=1}^{L-1} \sum_{i=0}^{M_l - 1} \tilde{w}_{l,i}\,\text{CE}\big(q_{l,i}(\cdot \mid u_{g,k}, c^*_{<l}), r^*_{l,i}\big)$$
   Draw one counterfactual root $c' \sim \text{sg}(q_0(\cdot \mid u_{g,k}))$, recompute its exact residual label against the frozen Stage-1 features $r_{l,i}(c') = \text{nearest}\big(E_l,\, \text{sg}(z_{l,i}) - E_0[c']\big)$, and set $\ell_k(g) = \ell_{k,\text{clean}}(g) + 0.1\,\ell_{k,\text{alt}}(g, c')$[span_154](start_span)[span_154](end_span).
3. **Smooth Forward XM Mixture Objective (`Explorative Modeling` Prop. 1 & 3):**
   $$\mathcal{L}_{\text{XM-Prior}} = \text{mean}_{\text{valid } g}\!\left[ -\log \left( \frac{1}{K} \sum_{k=1}^K \exp\big(-\ell_k(g)\big) \right) \right]$$
   $$\mathcal{L}_{\text{router}} = \text{mean}_{\text{valid } g}\!\left[ -\sum_{k=1}^K w_{g,k}^* \log \pi_\phi\big(k \mid z_{L,g}^{(S_L)}\big) \right], \qquad w_{g,k}^* = \text{sg}\!\left(\frac{\exp(-\ell_k(g))}{\sum_{j=1}^K \exp(-\ell_j(g))}\right)$$
   By `Explorative Modeling` Proposition 1 and 3, $\mathcal{L}_{\text{XM-Prior}}$ is the exact negative log-likelihood of the future span under the $K$-mode mixture, raising generative expressivity to $E \ge K$ and capturing cross-horizon dependencies across all future groups[span_155](start_span)[span_155](end_span)[span_156](start_span)[span_156](end_span)[span_157](start_span)[span_157](end_span).

---

## 5. Stage 2 Decoder Interface: Top-$B_0$ Beam `PlanAttention` & Auto-Guidance

1. **Zero-Cost `HRM-Text` Auto-Guidance (Inference Option):**
   Before forming the horizon-1 joint plan table at inference, extrapolate prior logits between the shallow Cycle-1 $H$-state $u'_g = z_{H,g}^{(1)}$ and the final Cycle-2 $H$-state $u_{g,k}$ (`HRM-Text` Appendix D)[span_158](start_span)[span_158](end_span)[span_159](start_span)[span_159](end_span):
   $$\text{logits}^{(w)} = (1 + w)\,\text{logits}(u_{g,k}) - w\,\text{logits}(u'_g), \quad w \in \{0, 0.1, 0.5\}$$
2. **Scalable Top-$B_0$ Conditional Beam Selection:**
   To avoid materializing an $N_0 \times N_1$ table when $N_l = 8,192$ at 3T:
   * Marginalize Level-0 root probabilities over the router: $\bar{q}_0(c) = \sum_{k=1}^K \pi_\phi(k)\, q_0(c \mid u_{g,k})$ and select the **top-$B_0$ roots** $\{c_1, \dots, c_{B_0}\}$ (at Tiny, $B_0 = 32 = N_0$, which evaluates the exact $32 \times 32 = 1,024$ table of V15 §8[span_160](start_span)[span_160](end_span); at 3T, $B_0 = 16$).
   * Evaluate the horizon-1 fine distribution $\bar{q}_1(r \mid c_b)$ only on those $B_0$ roots ($B_0 \times N_1$ cells) and extract the **top-$K_{\text{plan}}$ joint pairs** $(c_i, r_i)_{i=1}^{K_{\text{plan}}}$ with probabilities $\pi_i$, retained mass $m = \sum_{i=1}^{K_{\text{plan}}} \pi_i$, and tail mass $\max(1 - m, 10^{-6})$ assigned to a learned null key/value[span_161](start_span)[span_161](end_span).
3. **Detached `PlanAttention` Bridge:**
   Form candidate vectors $b_i = \text{concat}\big(\gamma_0 \odot E_0[c_i]/s_0,\; \gamma_1 \odot E_1[r_i]/s_1\big) \in \mathbb{R}^d$[span_162](start_span)[span_162](end_span). **Detach (`sg`) the entire plan package** (vectors, log-masses, retained mass $m$, and normalized root entropy $H(\bar{q}_0)/\log N_0$)[span_163](start_span)[span_163](end_span). After the $N_D$-block confined decoder produces $h_t$ (reading fixed local $L_{\text{enc}}$ and router-weighted historical $U_g = \sum_k \pi_\phi(k) u_{g,k}$), apply `PlanAttention` conditioned on offset $t - e_{g(t)} \in \{0, 1, 2, 3\}$[span_164](start_span)[span_164](end_span):
   $$a_t = \text{PlanAttention}\big(h_t,\, \text{sg}[\text{bank}_{g(t)}],\, \text{sg}[\log \text{masses}],\, \text{offset}\big)$$
   $$\text{gate}_t = \sigma\!\left(w_g^T \big[\text{RMSNorm}(h_t);\; \text{sg}[m];\; \text{sg}[H(\bar{q}_0)/\log N_0];\; \text{onehot}(\text{offset})\big] + b_g\right)$$
   $$h'_t = h_t + \text{ramp}(s) \cdot \text{gate}_t \cdot a_t, \qquad \text{logits}_t = \frac{\text{RMSNorm}(h'_t)\, E_{\text{student}}^T}{\sqrt{d}}$$
4. **Stage-2 Schedule & Disjoint Clipping:**
   Student progress $s \in [0, 1]$ (80% of total budget = **8.8 hours** on M2)[span_165](start_span)[span_165](end_span):
   * $s \in [0, 0.1)$: $\mathcal{L}_{\text{NTP}}$ only ($\lambda = 0, \text{ramp} = 0$)[span_166](start_span)[span_166](end_span).
   * $s \in [0.1, 0.2)$: Ramp $\lambda(s): 0 \to 0.05$ on $\mathcal{L}_{\text{XM-Prior}} + 0.1\,\mathcal{L}_{\text{router}}$[span_167](start_span)[span_167](end_span).
   * $s \in [0.3, 0.4)$: Ramp $\text{ramp}(s): 0 \to 1$[span_168](start_span)[span_168](end_span).
   * Clip prior + router parameters to norm $1.0$ **separately** from the student trunk norm-$1.0$ clip[span_169](start_span)[span_169](end_span).

---

## 6. Stage 3 (Stage C): Wake–Sleep Test-Time & Continual Learning (`GradAtlas`)

### 6.1 The `ATLAS-MAG` Fast-Weight Memory Module
At the exit of the slow $H$-module, attach a fixed-size **ATLAS Memory-as-Gate (MAG)** branch (`Titans` §4.2, `ATLAS` §3.1, §3.2, §5)[span_170](start_span)[span_170](end_span)[span_171](start_span)[span_171](end_span)[span_172](start_span)[span_172](end_span)[span_173](start_span)[span_173](end_span)[span_174](start_span)[span_174](end_span):
* **Degree-2 Polynomial Kernel Lifting ($\phi_2$):** Project $z_{L,g}^{(S_L)}$ via linear + 1D depthwise conv (kernel 4) + $L_2$-norm to query $q_g$ and key $k_g$, and lift via degree-2 polynomial features $\phi_2(x)$ with learnable Taylor coefficients $a_0, a_1, a_2$ initialized at $1, 1, \frac{1}{2}$ (ATLAS Eq. 5), raising memory capacity from $O(d_k)$ to $O(d_k^2)$[span_175](start_span)[span_175](end_span)[span_176](start_span)[span_176](end_span)[span_177](start_span)[span_177](end_span)[span_178](start_span)[span_178](end_span).
* **2-Layer Gated MLP Memory ($\mathcal{M}_{\text{ATLAS}}$):** Parameterized by fast weights $\theta_{\mathcal{M}} = \{W_1, W_2, W_3\}$[span_179](start_span)[span_179](end_span):
  $$\mathcal{M}(\tilde{k}) = \tilde{k} + W_1\big(\sigma(W_2 \tilde{k}) \otimes W_3 \tilde{k}\big)$$
* **MAG Gated Readout:**
  $$u_g^{\text{final}} = u_g + \sigma\big(W_{\text{mag}}\,\text{RMSNorm}(u_g)\big) \odot \text{RMSNorm}\big(\mathcal{M}^*(\phi_2(q_g))\big)$$

### 6.2 Phase 3A — WAKE: System 1 $\to$ `GradCuit` Search $\to$ `ATLAS` Omega Write
1. **Step 1 (System 1 Fast Decode):** Generate greedily with $\Delta z_g = \mathbf{0}$ and check verifier $R(x, c) \in \{0, -1\}$[span_180](start_span)[span_180](end_span)[span_181](start_span)[span_181](end_span). If $R(x, c) = 0$, return $x$[span_182](start_span)[span_182](end_span).
2. **Step 2 (Discrete XM Mode Sweep):** If $R(x, c) = -1$, evaluate the remaining $k \in \{1, \dots, K\}$ discrete XM modes $e_k$[span_183](start_span)[span_183](end_span)[span_184](start_span)[span_184](end_span).
3. **Step 3 (`GradCuit` Mid-Core Circuit Optimization):** If discrete modes fail, initialize trainable $\Delta z_g \leftarrow \mathbf{0}$ at the **50% $L \to H$ circuit interface** ($\tilde{z}_{H,g}^{(C_H)} = z_{H,g}^{(C_H-1)} + z_{L,g}^{(S_L)} + e_{k^*} + \Delta z_g$) and take up to $4$ Adam steps ($\eta = 10^{-3}$) minimizing the teacher-forced continuation objective while all model weights stay frozen[span_185](start_span)[span_185](end_span)[span_186](start_span)[span_186](end_span)[span_187](start_span)[span_187](end_span):
   $$\mathcal{L}_{\text{opt}}(\Delta z_g) = -R(x, c) \sum_{t=1}^T \log \pi_\theta\big(x_t \;\big\vert{}\; x_{<t},\, \tilde{z}_{H,g}^{(C_H)}(\Delta z_g),\, c\big)$$
   Because every continuation token attends directly to $U_g$ and the plan bridge, gradients flow cleanly through the self-attention circuit to $\Delta z_g$, concentrating on **reasoning-connector tokens** (`therefore`, `because`, `then`) without corrupting surface text[span_188](start_span)[span_188](end_span)[span_189](start_span)[span_189](end_span)[span_190](start_span)[span_190](end_span).
4. **Step 4 (`ATLAS` Sliding-Window Omega-Rule + Muon Fast-Weight Write):**
   Once a winning trajectory $\Delta z_g^*$ is found (with target residual $v_g^* = u_g(\Delta z_g^*) - u_g(\mathbf{0})$), update $\theta_{\mathcal{M}}$ over the local window of $c = 4$ concept groups using input-dependent context-pruning gates $\gamma_i^{(g)} \in [0, 1]$, surprise momentum $\theta_g$, adaptive forget gate $\alpha_g$, and 5-step **Newton-Schulz (`Muon`)** orthogonalization (`ATLAS` Eq. 32–33, 57–58)[span_191](start_span)[span_191](end_span)[span_192](start_span)[span_192](end_span)[span_193](start_span)[span_193](end_span):
   $$S_g = \theta_g S_{g-1} - \nabla_{\theta_{\mathcal{M}}} \sum_{i=g-3}^g \gamma_i^{(g)} \left\Vert{} \mathcal{M}_{g-1}(\phi_2(k_i)) - v_i^* \right\Vert{}_2^2, \qquad \theta_{\mathcal{M}, g} = \alpha_g \theta_{\mathcal{M}, g-1} + \eta_g\,\text{NewtonSchulz-5}(S_g)$$

### 6.3 Phase 3B — SLEEP: Amortizing System 2 into System 1 (Bounded LoRA + Replay)
At the end of each continual phase chunk (Dataset_v13 S1–S6)[span_194](start_span)[span_194](end_span)[span_195](start_span)[span_195](end_span):
* Maintain V15 §12's bounded **Rank-4 LoRA adapter** (Scale: Rank-64) on $L/H$ concept blocks and decoder projections plus **20% uniform replay from a 1,000-row reservoir**[span_196](start_span)[span_196](end_span).
* Train the LoRA adapters, Mode Router $\pi_\phi$, and Persistent Memory tokens $P$ on the wake + replay batches with $\Delta z_g = \mathbf{0}$ to absorb the discovered `GradCuit` + `ATLAS` targets $u_g^*$ while anchoring to the frozen Stage-1 codec over observed batch/replay code tuples $\mathcal{V}$[span_197](start_span)[span_197](end_span)[span_198](start_span)[span_198](end_span):
  $$\mathcal{L}_{\text{Sleep}} = \mathcal{L}_{\text{NTP}} + 0.05\,\mathcal{L}_{\text{XM-Prior}}^{(\mathcal{V})} + \beta_{\text{amort}} \left\Vert{} u_g^{\text{LoRA}}(\Delta z_g = \mathbf{0}) - \text{sg}(u_g^*) \right\Vert{}_2^2$$
* Reset transient $\Delta z_g = \mathbf{0}$ so subsequent tasks run at full 1-pass speed.

---

## 7. Generalized Scaling Blueprint: M2 Tiny -> 7B -> 3T Frontier

| Architectural Dimension | Tiny Tier (M2 Air) | Mid Scale Tier (7B-A1.4B) | Frontier K3 Tier (2.8T-A115B) |
| :--- | :--- | :--- | :--- |
| **Student Width $d$ / Heads / Vocab $V$** | $128$ / $4 \times 32$ / $2,048$[span_199](start_span)[span_199](end_span) | $1,536$ / $12 \times 128$ / $65,536$[span_200](start_span)[span_200](end_span)[span_201](start_span)[span_201](end_span) | $7,168$ / $96 \times 128$ / $160,000$[span_202](start_span)[span_202](end_span) |
| **Stage-1 Codec Width $d_c$ & Depth** | $d_c = 64$; $N_{\text{stem}}=0, N_{\text{span}}=2, N_{\text{dec}}=1$[span_203](start_span)[span_203](end_span) | $d_c = 512$; $N_{\text{stem}}=8, N_{\text{span}}=2, N_{\text{dec}}=2$ | $d_c = 1,536$; $N_{\text{stem}}=16, N_{\text{span}}=2, N_{\text{dec}}=4$ |
| **Stage-1 Codec Params (% of Student)** | $260.6\text{K}$ + $4.1\text{K}$ EMA ($3.9\%$)[span_204](start_span)[span_204](end_span) | $42\text{M}$ ($0.61\%$ of stored) | $680\text{M}$ (**$0.024\%$ of stored**) |
| **Target Span $W$ & Tree Levels $L$** | $W = 16$, $L = 2$, slots $(1, 4)$[span_205](start_span)[span_205](end_span) | $W = 16$, $L = 3$, slots $(1, 2, 4)$ | $W = 32$, $L = 3$, slots $(1, 2, 8)$ |
| **Codebook Size $N_l \times G_q$ (Bits/Win)** | $32 \times 1$ (**25 bits**, $1.56\text{ b/tok}$)[span_206](start_span)[span_206](end_span) | $1,024 \times 1$ (**70 bits**, $4.38\text{ b/tok}$) | $8,192 \times 2$ (**286 bits**, $8.94\text{ b/tok}$) |
| **Encoder Layout ($E_1, E_2, E_3$)** | $2\times\text{E1}(w_E=8)$, $\text{E2}(R=1)$, $\text{E3}(2)$[span_207](start_span)[span_207](end_span) | $2\times\text{E1}(w_E=32)$, $\text{E2}(R=2\text{ MoE})$, $\text{E3}(4)$ | $2\times\text{E1}(w_E=32)$, $\text{E2}(R=2\text{ LatentMoE})$, $\text{E3}(4)$[span_208](start_span)[span_208](end_span) |
| **Engram Conditional Memory** | $16$ tables $\times 8,191 \times 8$ ($1.12\text{M}$)[span_209](start_span)[span_209](end_span) | $16$ tables $\times 1.19\text{M} \times 64$ ($1.22\text{B}$)[span_210](start_span)[span_210](end_span) | $16$ tables $\times 269\text{M} \times 128$ ($550\text{B}$ host)[span_211](start_span)[span_211](end_span) |
| **`XHRM` Core ($n_L, n_H$, Schedule)** | $n_L=2, n_H=2$, $H_2 L_3$, MagicNorm[span_212](start_span)[span_212](end_span)[span_213](start_span)[span_213](end_span) | $n_L=3, n_H=3$, $H_2 L_3$ MoE, MagicNorm | $n_L=8, n_H=8$, $H_2 L_3$ LatentMoE, MagicNorm |
| **XM Discrete Modes $K$ (at 50% Midpoint)** | $K = 4$ modes ($<5\%$ FLOPs)[span_214](start_span)[span_214](end_span) | $K = 8$ modes[span_215](start_span)[span_215](end_span) | $K = 16$ modes[span_216](start_span)[span_216](end_span) |
| **`PlanAttention` Beam $B_0 \to K_{\text{plan}}$** | $B_0 = 32$ (exact $32^2$) $\to K_{\text{plan}} = 4$[span_217](start_span)[span_217](end_span) | $B_0 = 8 \to K_{\text{plan}} = 8$ | $B_0 = 16 \to K_{\text{plan}} = 16$ |
| **Decoder $D$ Layers & Reach $R_D$ ($\rho$)** | $N_D = 4, w_D = 16 \implies R_D = 66$ ($0.06$)[span_218](start_span)[span_218](end_span) | $N_D = 16, w_D = 128 \implies R_D = 324$ ($0.08$) | $N_D = 56, w_D = 128 \implies R_D = 324$ ($0.08$)[span_219](start_span)[span_219](end_span) |
| **`ATLAS-MAG` Continual Fast Memory** | 2-layer Gated MLP, $\phi_2$, $c=4$, `NS-5` ($99.8\text{K}$) | 2-layer Gated MLP, $\phi_2$, $c=8$, `NS-5` ($14\text{M}$) | 2-layer Gated MLP, $\phi_2$, $c=16$, `NS-5` ($310\text{M}$) |
| **Total Student Stored / Active Params** | **6,785,003** / **5,660,443** | **6.82B** / **1.38B** | **2.78T** / **115B**[span_220](start_span)[span_220](end_span) |

---

## 8. Falsification Matrix, Ablation Ladder, and Implementation Gates

To guarantee every added mechanism earns its keep without repeating V14's bundled-confound mistake[span_221](start_span)[span_221](end_span), experiments execute in a strict factorial ladder under the **11.0-hour M2 wall-clock cap** ($\le 2.2\text{h}$ Stage 1 + $\le 8.8\text{h}$ Stage 2 for codec arms; $11.0\text{h}$ student training for non-codec arms)[span_222](start_span)[span_222](end_span):

| Arm ID | Frozen Codec (`T`) | Plan Bridge (`P`) | Concept Core ($C$) | XM Modes ($K$) | Stage 3 Continual / Test-Time | Scientific Question Isolated |
| :--- | :---: | :---: | :--- | :---: | :--- | :--- |
| **`B`** | Off | Off | 4-blk `Delta-AttnRes` | $K=1$ | Rank-4 LoRA + 20% Replay | Clean NTP baseline on inherited L/G shell[span_223](start_span)[span_223](end_span). |
| **`T`** | On | Off | 4-blk `Delta-AttnRes` | $K=1$ | Rank-4 LoRA + 20% Replay | Does frozen Stage-1 codec supervision help alone[span_224](start_span)[span_224](end_span)? |
| **`TP`** | On | On | 4-blk `Delta-AttnRes` | $K=1$ | Rank-4 LoRA + 20% Replay | Does consulting top-4 plans beat `T` at equal time[span_225](start_span)[span_225](end_span)? |
| **`TPD`** | On | On | 4-blk `V15-Δ` Memory | $K=1$ | Rank-4 LoRA + 20% Replay | Does within-pass delta depth memory beat `TP`[span_226](start_span)[span_226](end_span)? |
| **`TPH`** | On | On | **2L + 2H `HRM` MagicNorm** | $K=1$ | Rank-4 LoRA + 20% Replay | Does dual-timescale HRM recurrence beat `TP` and `TPD` on V3 and rungs 8/9[span_227](start_span)[span_227](end_span)[span_228](start_span)[span_228](end_span)? |
| **`TPH-X`** | On | On | **2L + 2H `HRM` MagicNorm** | **$K=4$ XM** | Rank-4 LoRA + 20% Replay | Does $K=4$ Forward XM mode exploration beat $K=1$ `TPH`[span_229](start_span)[span_229](end_span)[span_230](start_span)[span_230](end_span)? |
| **`TPH-X-GradAtlas`** | On | On | **2L + 2H `HRM` MagicNorm** | **$K=4$ XM** | **`GradCuit` + `ATLAS-MAG` + Sleep** | Does mid-core `GradCuit` + fixed-size `ATLAS` beat retry (equal-time) and plain replay[span_231](start_span)[span_231](end_span)[span_232](start_span)[span_232](end_span)[span_233](start_span)[span_233](end_span)? |
| **`TF-Engram`** | Off | Off | None (24-blk PreLN TF) | — | Rank-4 LoRA + 20% Replay | Table-matched ordinary Transformer control ($6,873,424$ params)[span_234](start_span)[span_234](end_span). |

### Mandatory G0–G4 Launch Gates Before Training
* **G0 (Float64 Causality & Reach):** Verify on CPU in `float64` that (a) suffix edits after row $t$ change row $t$ logits by $0.0$, (b) packed-document perturbations never cross document boundaries, (c) disabling historical $U$ and returning exact zero from the entire `PlanAttention` bridge restores exact local reach $R_L = 51, R_D = 66$, and (d) frozen Stage-1 codec assignments are bitwise identical before and after 1,000 student steps ($0.00\%$ churn)[span_235](start_span)[span_235](end_span).
* **G1 (Stage-1 Codec Readiness):** Held-out coarse-only CE improves over training-fitted phase-position unigram by $\ge 0.05\text{ nats/token}$, and full hierarchy improves over coarse-only by $\ge 0.05\text{ nats/token}$[span_236](start_span)[span_236](end_span).
* **G2 (M2 Memory & Soak Gate):** 1,500-step MPS soak of `TPH-X` stays $\le 8.0\text{ GiB}$ driver allocation with zero sustained swap growth; during Stage 3, `ATLAS-MAG` adds $< 1\text{ MiB}$ of state (compared to V14's $17.01\text{ GiB}$)[span_237](start_span)[span_237](end_span).
* **G3 (Promotion Rule):** At equal 11.0-hour wall-clock time, `TPH-X` improves registered Reasoning by $\ge 0.02$ absolute over the strongest matched baseline (paired 95% interval $>0$), passes Diagnostic V3 ($u_g$ beats $p_g$ probe by $\ge 5$ points in $\ge 2$ families), and `GradCuit` beats equal-time best-of-$N$ retry on held-out rungs 8/9[span_238](start_span)[span_238](end_span).