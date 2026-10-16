# HRM THINKER — Every paper behind the encoder, concept module and decoder

*Compiled 2026-09-27 from the design records: the encoder design conversation (`Encoder.md`, 8 source lists), the concept-module conversation (`Conversation.rtf`, 6 source lists), the decoder session (NextLat and HiLP PDFs), the frozen encoder spec, the decoder spec, and `ARCHITECTURES_AND_PAPERS.md` §5 (the papers carried in from the earlier programmes). Nothing here is new reading; each entry says what we took and what happened to it. arXiv ids are given only where they appear in those records; "—" means the record names the work without an id.*

---

## 0. How to read this

**Status** (what the paper did to the design):

| tag | meaning |
|---|---|
| **ADOPTED** | in the current design as the paper has it |
| **ADAPTED** | in the design, changed on first principles or by our own measurement |
| **TESTED, REJECTED** | we built it and measured it; it lost |
| **SCALE PATH** | reserved for the frontier-size version, not in the M2 model |
| **INFORMED** | shaped a decision or a rule without contributing a component |
| **READ, NOT ADOPTED** | considered and left out, with the reason |

**Reading depth** where the records state it: *full* (full text through the web reader or the PDF), *partial*, *abstract*, *blog* (a blog or overview, not a paper), *known* (not fetched; known from prior knowledge), *not recorded*.

Sections 1–3 go module by module; a paper used by several modules appears in each, with that module's use. §4 lists the earlier-programme papers that fed the design indirectly. §5 is the master index. §6 is the completeness review.

---

## 1. Encoder v1.0 (frozen 2026-09-25)

### 1.1 Layer pattern and attention

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **NCP-ArchPreview** (Next Concept Prediction, latent-space LMs) | 2609.10715 | full | the [SWA, SWA, SWA, FULL] shell pattern (3:1), 4-token mean pooling into concepts | **ADOPTED** (3:1 as `S S S F S S S F`; pooling initialised to NCP's mean) |
| **Gated Attention for LLMs** (Qwen; NeurIPS 2025 best paper) | 2505.06708 | not recorded | elementwise sigmoid gate after SDPA; removes the need for attention sinks | **ADOPTED** in every attention layer (encoder, core, decoder) |
| **Sliding-Window Beats Linear Attention** | 2608.28444 | not recorded (cited by the user) | SWA beats linear attention at equal cost | **INFORMED**: gated SWA for local layers, no linear attention in the encoder |
| **RNoPE-SWA** (hybrid attention: RoPE in local layers, NoPE in global) | 2501.18795 | not recorded | RoPE only in SWA layers, NoPE in FULL layers | **ADAPTED**: partial RoPE (16 of 64 dims) in SWA, NoPE + learned per-head temperature in FULL |
| **Kimi K3** technical report | 2607.24653 | not recorded | Gated MLA with NoPE, Muon, 3:1 hybrid at scale | **SCALE PATH** (FULL layers → Gated MLA; per-head Muon) |
| **DeepSeek-V4** | 2606.19348 | not recorded | compressed sparse attention (CSA/HCA) with a local window, partial RoPE, mHC, Engram, Muon | **SCALE PATH** (FULL layers → CSA/HCA); partial RoPE **ADOPTED** |
| CSA/HCA overview | — | blog | how V4's compressed attention works | **INFORMED** (scale path) |
| **Kimi Linear** | — | not recorded | linear attention in a hybrid stack | **READ, NOT ADOPTED** in the encoder: unbounded reach would break the L stream's bound; candidate inside the concept module at scale |
| **Gated DeltaNet-2** (NVIDIA, May 2026) | — | not recorded | two-gate delta rule; beat Mamba-2, Gated DeltaNet, KDA, Mamba-3 at 1.3B | **SCALE PATH** as a mixer candidate *inside the concept core*; excluded from encoder L stream and decoder (reach) |

### 1.2 Short convolutions and n-gram memory

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **Canon layers** (Physics of LMs 4.1; NeurIPS 2025) | 2512.17351 | overview | causal short convolutions give every token its predecessors; induction in one layer | **ADOPTED**: kernel-4 residual convs on q, k, v, zero-initialised, document-aware |
| **LFM2** (Liquid AI) technical report | 2511.23404 | not recorded | gated short convolutions (kernel 3) as cheap local mixers | **INFORMED** (supports convs) |
| Short convolutions in LFM2 ("Flipping the script") | — | blog | why short convs work (much of Mamba's conv1d effect) | **INFORMED** |
| **Inkling** architecture notes (Thinking Machines) | — | blog | Inkling keeps 4 convs per layer at 975B | **INFORMED**: keep convs on the scale path |
| ShortConv gallery (Raschka) | — | blog | survey of short-conv placements | **INFORMED** |
| **Engram** (conditional memory via n-gram lookup) | 2601.07372 | not recorded | hashed 2/3-gram tables, context gate, conv | **ADAPTED**: after S1 (so its gate has context), orders {2,3}, 4 heads, canonical token ids, W_V = 0 / tables ≠ 0 init |
| **Over-Tokenized Transformer** | — | not recorded | scaling the *input* vocabulary with n-gram embeddings | **INFORMED** (background for n-gram memory) |
| **Per-Layer Embeddings** (Gemma 3n, Gemma 4 E2B/E4B) and Raschka's note | — | blog | unigram lookup per layer gated by the hidden state | **READ, NOT ADOPTED**: Engram covers the role with n-grams |
| Superposition Yields Robust Neural Scaling | — | not recorded | listed in the encoder's research round; its role is not recorded beyond the source list | **INFORMED** (background) |

### 1.3 Residual streams and normalisation

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **Qwen3.8-Next** (gated residuals, zero-centred norms) | 2608.30320 | not recorded | gated residual read/write; zero-centred RMSNorm with decay on the deviation from 1 | **ADAPTED**: two role streams L and G with read/write masks, used in all three modules |
| Gated Residuals (Raschka) | — | blog | explanation of the gated residual | **INFORMED** |
| **mHC** (manifold-constrained hyper-connections) | 2512.24880 | not recorded | doubly stochastic mixing of several residual streams | **READ, NOT ADOPTED**: doubly stochastic mixing cannot express one-way flow between roles (L must never receive G) |
| **Delta Attention Residuals** | 2605.18855 | full | routing over layer/step deltas | **SCALE PATH** (Delta Block AttnRes within a role once depth ≥ 16); CDR form measured inert earlier (F2) |
| **Attention Residuals** (Kimi) | 2603.15031 | partial | softmax routing over depth | **TESTED, REJECTED** earlier (Dual-IRC inert, F2) |
| **HRM-Text** | 2605.20613 | full | BPTT spikes, truncation warm-up, MagicNorm | **INFORMED** the gated-residual/recurrence discussion; its recipe (constant LR 2.2e-4, EMA) still untried |

### 1.4 Concepts, chunking and hierarchy (encoder → concept interface)

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **NCP-ArchPreview** | 2609.10715 | full | pooling k = 4, concepts from the encoder, CRC | **ADAPTED**: CRC depth-mix of {e0, L, G} → concat-project pooling (init = mean) → sphere |
| **ConceptLM** | 2602.08984 | full | the original NCP: MSE on the codeword mix, VQ with commitment | **INFORMED** (encoder side: stop-gradient on targets; commitment term not adopted) |
| **Block Transformer** | 2406.02657 | full | a global model over blocks + a local token decoder reading one global state | **INFORMED** the local/global split and the one-read decoder contract |
| **H-Net** (dynamic chunking) | 2507.07955 | not recorded | learned, content-dependent chunk boundaries | **READ, NOT ADOPTED**: fixed k = 4 keeps causality and the plan index exact |
| **BLT** (Byte Latent Transformer) | 2412.09871 | not recorded | entropy-based patches | **READ, NOT ADOPTED** (same reason) |
| **Free Transformer** | — | full | "bound what bypasses the core" (KL budget on a latent) | **INFORMED** the bounded-bypass principle behind L/G |

### 1.5 Sources checked during the encoder research round

NeurIPS 2025 best-paper awards (Gated Attention; Canon layers), NeurIPS 2026 accepted-papers page and dates (the list was not public yet: 0 papers shown), Gated Attention code release.

---

## 2. Concept module (v0 → v2 → v2.1; not frozen)

### 2.1 The skeleton and the vocabulary

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **NCP-ArchPreview** | 2609.10715 | full | continuous pooled input, causal transformer over chunks, product-quantised vocabulary (32 × 128 × 128), codebook trained by VQ only, NCP = MSE(Σπe, sg(c_next)), shifted injection | **ADAPTED**: NCP loss changed to **cross-entropy on code indices** (MSE's optimum is the conditional mean, so π carries no meaning; derived); 64 × 128 × 8 vocabulary; distance head with τ₀ = 2d_s; MSE kept as an arm |
| **ConceptLM** | 2602.08984 | full | SimVQ codebook parameterisation; commitment term | **TESTED, REJECTED**: SimVQ 46/128 codes used (83/128 with a bias) vs k-means++ init + dead-code restarts 128/128, variance explained 0.94, also under drift |
| **VQ-VAE** | 1711.00937 | known | codebook + stop-gradient | **ADOPTED** (the VQ loss form) |

### 2.2 Recurrence: loop, H/L, two states

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **HRM** (Hierarchical Reasoning Model) | 2506.21734 | full | H/L two-timescale recurrence; H sees the input only through L | **INFORMED** (constraint C1); v2 chose a single looped layer instead of H/L |
| **TRM** ("Less is More", tiny recursive networks) | 2510.04871 | full | two states (scratch z, answer y) from one network; deep supervision | **ADAPTED** in v0 (Z/Y two-state loop, Y⁰ = 0); replaced in v2 by one state S with the input re-read every pass |
| **Huginn** (recurrent depth, "Scaling by thinking in continuous space") | 2502.05171 | abstract | prelude → recurrent block → coda; input injection; sampled depth | **ADOPTED** in v2: input X re-read every pass, S⁰ = 0 |
| **Loop, Think & Generalize** | 2604.07822 | full | sampling the number of passes during training gives hop-count extrapolation; overthinking at excess depth | **ADOPTED**: R ~ U{1..4} in training. Measured: v1's loop trained at fixed R = 2 learned "a 2-hop circuit" (a third pass solved ≤ 7% of 3-hop); v2 trained on 1–3 hops reached hop 6 at 8 passes. Sampled R held that at the end of training (77% vs 31% for fixed R = 4); both arms peaked at 97–99% mid-training, so checkpoint choice matters |
| **Diffusion as a Training Curriculum** (Google) | 2609.01449 | full | persistent state + noise, K ~ Unif[20,160] → anytime iterative solver | **INFORMED** sampled depth; noise injection not adopted |
| **DiscoLoop** | 2607.00341 | full | soft decode → re-encode channel between loop passes | **TESTED, REJECTED**: slowed the subset loop (+750 steps, 2 seeds), broke the pass-1 bridge representation (bridge decodability 35–62% → 2–5%); earlier Q-LoopNCP also found it unused |
| **ETD** (Recursive Latent Thoughts) | — | full | iterate only middle layers; split by angular distance | **INFORMED** the prelude/loop/coda placement (v1), superseded by v2's loop-only core |
| **SMELT** | 2609.01343 | abstract | loop the middle of an MoE model twice at matched FLOPs/params/KV | **INFORMED** (recurrence on the only path at scale) |
| **LoopCoder-v2** / Loop the Loopies | 2606.18023 | full | 2 loops best, ≥ 3 regress | **INFORMED** (trained pass count is what gets used) |
| **Nanbeige4.2** report | — | partial | 2-pass looped transformer; pass 2 always used | **INFORMED** (constraint C4) |
| **Qwen3.8-Next** | 2608.30320 | not recorded | gated residual | **ADOPTED** as the core's block grammar (read {X, S} → write S) |
| **Delta AttnRes**, **mHC** | 2605.18855, 2512.24880 | full / not recorded | depth routing, multi-stream mixing | **READ, NOT ADOPTED** in the core (routing follows use, F2) |
| **Gated DeltaNet-2**, **Kimi K3** | —, 2607.24653 | not recorded | linear mixer, Gated MLA | **SCALE PATH** (core mixer: 3:1 GDN-2:FULL; MLA/CSA over concepts; per-head Muon) |

### 2.3 Multiple hypotheses and test-time steering (v2.1)

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **GRAM** (Generative Recursive Reasoning) | 2605.19376 | full | stochastic latent guidance in a recursive core; decoder reads only h | **READ, NOT ADOPTED** as a VAE: the posterior sees the future and an AR decoder can ignore z (collapse); its multiple-hypothesis goal is met through XM instead |
| Bowman et al., posterior collapse in VAE language models | — | known | why an autoregressive decoder ignores a VAE latent | **INFORMED** the GRAM analysis |
| **Explorative Modeling (XM)** | 2607.27372 | full | K learned latents; train only the best candidate (min over k), no posterior, no KL | **ADAPTED** (hypothesis H2, tested): K = 4 latents as a read-only stream Z; chain coherence 59.0% → 86.6% (hard min) / 87.7% (relaxed WTA); coverage 30.7% → 92%; no collapse (cos 0.30, wins 23–27% each). Leak rule derived: the decoder reads a *random* k in training |
| Rupprecht et al., multiple hypotheses (relaxed winner-take-all) | — | known | winner weight 1−ε, losers share ε | **ADOPTED** ε = 0.05 (fixed hard-min's uncorrected losers: per-latent accuracy 84–93% → 94–99%) |
| **GradCuit** (credit-assigned gradient flow for test-time latent reasoning) | 2608.02585 | full | optimise mid-network latents against a verifier, ≤ 10 updates | **ADAPTED** (hypothesis H1, tested): offsets on the concept-stream slots; 2.2× better than matched-norm random directions, half the verifier calls of enumeration, intermediate hops kept correct 92–100%; policy: depth first, then steering |
| **I-JEPA** (JEPA-style prediction) | 2301.08243 | known | representation collapse in latent-prediction objectives | **INFORMED** (why the NCP target is stop-gradient on the encoder) |
| PRISM (our own campaign, not a paper) | — | — | K > 1 candidates collapsed to one vector | the contrary evidence XM had to overcome; XM's learned input embeddings broke the symmetry |

### 2.4 Output of the concept module

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **Block Transformer** | 2406.02657 | full | the local decoder reads one global state per block | **ADOPTED** as the single-read contract (the decoder reads at j*(t) only) |
| **NCP-ArchPreview** | 2609.10715 | full | ĉ added to the decoder state shifted by one chunk; decoder also reads core states through CRC | **ADAPTED**: target-aligned shift j*(t) = ⌊(t+1)/4⌋ − 1; two plan vectors (u, ĉ); post-loop states only; null plan at every document start (bug found by test) |

---

## 3. Decoder v1 (design candidate, 2026-09-27)

| paper | id | depth | what we took | status |
|---|---|---|---|---|
| **Next-Latent Prediction (NextLat)** — "Next-Latent Prediction Transformers Learn Compact World Models" (Microsoft Research) | 2511.05963 | full (PDF, incl. ablations and appendix) | ĥ_{t+1} = h_t + f_ψ(LN[h_t; E x_{t+1}]); SmoothL1 to sg(h_{t+1}) + KL through a frozen head; belief-state theorem; λ_h = 1, λ_KL = 1; both stop-gradients matter | **TESTED, kept OPTIONAL**: on the carry task no speed-up, loss +0.014 / +0.019 nats; the paper itself reports ≈ +0.03 nats perplexity at 1.3B with gains on planning tasks. Optional arm for the M2 run |
| **Hierarchical Latent Prediction (HiLP)** (Microsoft Research) | 2608.05806 | full (PDF, incl. hyperparameters and cost table) | upper latent = SWA over the last k = 4 lower latents, 4-step lookahead; combined NTP head dropped at inference; glu_cross fusion (NTP 3.942 vs concat 4.113) | **READ, NOT ADOPTED**: our concept module *is* HiLP's upper level, kept on the forward path; glu_cross not needed (SwiGLU already multiplies plan and state). Its 1B result (HumanEval 8.77 → 10.58 → 11.33) supports conditioning NTP on a 4-token latent |
| **NCP-ArchPreview** | 2609.10715 | full | shifted concept injection; CRC from encoder to decoder | **ADAPTED**: target-aligned shift; plan *added* to a private per-position state; encoder→decoder CRC dropped (`tok` already sums every L write; G forbidden) |
| **Block Transformer** | 2406.02657 | full | one global read per block | **ADOPTED** (the single-read rule, R2) |
| **DeepSeek-V4.1-Flash** (CED: decoder attention over a compressed global state) | 2609.19969 | full (CED part) | IHN adapted it into cross-attention over all past core states → F7 | **INFORMED** rule R7 (no position may read another position's plan) |
| **Gated Attention** (Qwen) | 2505.06708 | not recorded | output gate after SDPA | **ADOPTED** in the decoder layer |
| **Sliding-window / linear attention** evidence (2608.28444; Kimi Linear; Gated DeltaNet-2; Mamba, RetNet) | 2608.28444 | — | reach analysis | **INFORMED**: SWA only; linear attention excluded because Σ(Πα) k vᵀ keeps every past token (unbounded reach) |
| **MUDDFormer** | 2502.12170 | known | dense depth connections (NCP's IRC form) | **READ, NOT ADOPTED** (little dilution at 4 layers; IRC-type routers inert, F2) |
| **Attention Residuals** (Kimi), **Delta AttnRes** | 2603.15031, 2605.18855 | partial / full | depth routing | **READ, NOT ADOPTED** in the decoder (same reason) |
| **Multi-token prediction** (Gloeckle et al.) and **JTP** (Ahn et al.) | 2404.19737, 2503.21801 | via NextLat/HiLP | token-space multi-step objectives | **INFORMED** (NextLat's comparison: they hurt next-token perplexity more than NextLat) |
| **RoPE**, **GQA**, **GLU variants (SwiGLU)**, **RMSNorm** | 2104.09864, 2305.13245, 2002.05202, 1910.07467 | known | block primitives | **ADOPTED** (partial RoPE 16/64 on q and k; GQA 8/2; SwiGLU 1408; zero-centred RMSNorm) |

---

## 4. Carried in from the earlier programmes (fed the design indirectly)

These were read for Latent Thinker … DARWINMINDS (`ARCHITECTURES_AND_PAPERS.md` §5) and supplied the failure ledger (F1–F7) and constraints (C1–C9) that every step above was checked against. Papers already listed in §1–§3 are not repeated.

| paper | id | depth | measured outcome in our programme | where it constrains the new design |
|---|---|---|---|---|
| **Intern-S2-Mobius** (shared core FFN / knowledge store) | — | partial | fast weights unread in IHN | C5: memory must be exercised during pretraining |
| **In-Place Test-Time Training** | — | full | IHN's fast memory unread (1.153 on / off / absent) | C5 |
| **Language Models Need Sleep** (Lee et al.) | 2605.26099 | full | N = 0 and N = 2 sleep passes identical | C5 |
| **Language Models Need Sleep: Self-Modify and Consolidate** (Behrouz et al.) | 2606.03979 | full | consolidation never run | continual-learning "install" regime (future) |
| **Emergent temporal abstractions** ("AI inside AI", Google) | 2512.20605 | full | never run | continual-learning "compose" regime (future) |
| **Contrastive LMs (CLM)** (Stanford/NVIDIA blog + README) | — | partial / blog | InfoNCE never switched on | candidate selector/verifier after pretraining |
| **qlabs, "Computational depth is all you need"** | — | full | boundary operator off at IHN's depth | depth as a scaling axis |
| **Macaron-V1** | — | not recorded | never run | frozen base + LoRA specialists (future) |
| **Titans**, **Atlas**, **MIRAS** | — | not fetched | never run | memory update rules (future) |
| **Jev** (TypeSafe AI) | — | blog | below random as a long-horizon verifier (CLM blog) | calibration target for any selector |
| **Recurrent Looped Transformer** | — | not recorded | — | encoder/memory separation |
| **Energy-Based Transformers** | — | not fetched | — | alternative verifier |
| **Entropy → Epiplexity** | — | not fetched | — | data order / curriculum |
| **LoopedWM**, **Metis**, **SMT** (Pretraining RNNs without Recurrence), **BDH-CQ**, **In-Context Learning with Recurrent Latent Reasoning** | — | not recorded | — | background for memory and credit assignment |
| **DeepSeek-V3.2 / V4 / V4.1**, **Native Sparse Attention**, **IndexCache** | — | not recorded | — | scale path (KV compression, sparse attention) |
| **Hierarchical Latent Prediction** (earlier Better-Arch copy) | — | read for Q-LoopNCP, not used then | — | now read in full for the decoder (§3, HiLP) |

**Foundations cited in the specifications** (`ARCHITECTURES_AND_PAPERS.md` §5.4): Transformer (1706.03762), Universal Transformers (1807.03819), Perceiver (2103.03206), Perceiver IO (2107.14795), Funnel-Transformer (2006.03236), YOCO (2405.05254), MegaByte (2305.07185), Feedback Memory (2002.09402), DenseFormer (2402.02622), identity mappings in ResNets (1603.05027), DeepNet (2203.00555), sparse MoE (1701.06538), GShard (2006.16668), Switch (2101.03961), DeepSeekMoE (2401.06066), Chinchilla (2203.15556), Grokking (2201.02177), plus RMSNorm, GLU variants, RoPE, GQA, VQ-VAE and I-JEPA listed above.

---

## 5. Master index

E = encoder, C = concept module, D = decoder, ● = used or decided on, ○ = read/considered, · = not applicable.

| # | paper | id | E | C | D | status in the current design |
|---|---|---|---|---|---|---|
| 1 | NCP-ArchPreview | 2609.10715 | ● | ● | ● | adapted in all three modules |
| 2 | Gated Attention (Qwen) | 2505.06708 | ● | ● | ● | adopted |
| 3 | Qwen3.8-Next (gated residual, zc-norm) | 2608.30320 | ● | ● | ● | adapted (role streams) |
| 4 | Canon layers | 2512.17351 | ● | ● | ○ | adopted in encoder and core; not in decoder (would read neighbours' plans) |
| 5 | Engram | 2601.07372 | ● | · | ○ | adapted (encoder only) |
| 6 | RNoPE-SWA | 2501.18795 | ● | ● | ● | adapted (NoPE global, partial RoPE local) |
| 7 | Sliding-Window Beats Linear Attention | 2608.28444 | ● | · | ● | informed |
| 8 | Kimi K3 | 2607.24653 | ○ | ○ | · | scale path |
| 9 | DeepSeek-V4 | 2606.19348 | ○ | ○ | · | scale path; partial RoPE adopted |
| 10 | CSA/HCA overview | — | ○ | ○ | · | scale path |
| 11 | Kimi Linear | — | ○ | · | ○ | not adopted (reach) |
| 12 | Gated DeltaNet-2 | — | ○ | ○ | ○ | scale path in the core only |
| 13 | LFM2 | 2511.23404 | ○ | · | · | informed |
| 14 | Short convolutions (LFM2 blog) | — | ○ | · | · | informed |
| 15 | Inkling notes | — | ○ | · | · | informed (convs at scale) |
| 16 | ShortConv gallery | — | ○ | · | · | informed |
| 17 | Over-Tokenized Transformer | — | ○ | · | · | informed |
| 18 | Per-Layer Embeddings (Gemma 3n / 4) | — | ○ | · | · | not adopted |
| 19 | Superposition Yields Robust Neural Scaling | — | ○ | · | · | background |
| 20 | Gated Residuals (Raschka) | — | ○ | · | · | informed |
| 21 | mHC | 2512.24880 | ○ | ○ | · | not adopted |
| 22 | Delta AttnRes | 2605.18855 | ○ | ○ | ○ | scale path (depth ≥ 16) |
| 23 | Attention Residuals (Kimi) | 2603.15031 | ○ | · | ○ | rejected (F2) |
| 24 | MUDDFormer | 2502.12170 | · | · | ○ | not adopted |
| 25 | HRM-Text | 2605.20613 | ○ | ○ | · | informed |
| 26 | ConceptLM | 2602.08984 | ○ | ● | · | SimVQ tested, rejected |
| 27 | VQ-VAE | 1711.00937 | · | ● | · | adopted (VQ loss) |
| 28 | Block Transformer | 2406.02657 | ○ | ● | ● | adopted (single-read contract) |
| 29 | H-Net | 2507.07955 | ○ | · | · | not adopted |
| 30 | BLT | 2412.09871 | ○ | · | · | not adopted |
| 31 | Free Transformer | — | ○ | · | ○ | informed (bounded bypass) |
| 32 | HRM | 2506.21734 | · | ○ | · | informed (C1) |
| 33 | TRM | 2510.04871 | · | ● | · | adapted in v0, superseded in v2 |
| 34 | Huginn | 2502.05171 | · | ● | · | adopted (input re-read, S⁰ = 0) |
| 35 | Loop, Think & Generalize | 2604.07822 | · | ● | · | adopted (sampled depth) |
| 36 | Diffusion as a Training Curriculum | 2609.01449 | · | ○ | · | informed |
| 37 | DiscoLoop | 2607.00341 | · | ● | · | tested, rejected |
| 38 | ETD (Recursive Latent Thoughts) | — | · | ○ | · | informed |
| 39 | SMELT | 2609.01343 | · | ○ | · | informed |
| 40 | LoopCoder-v2 / Loop the Loopies | 2606.18023 | · | ○ | · | informed |
| 41 | Nanbeige4.2 | — | · | ○ | · | informed (C4) |
| 42 | GRAM | 2605.19376 | · | ○ | · | not adopted as VAE; goal met via XM |
| 43 | Bowman et al. (posterior collapse) | — | · | ○ | · | informed |
| 44 | Explorative Modeling (XM) | 2607.27372 | · | ● | ○ | adapted, tested (v2.1 latent Z; decoder reads random k) |
| 45 | Rupprecht et al. (relaxed WTA) | — | · | ● | · | adopted (ε = 0.05) |
| 46 | GradCuit | 2608.02585 | · | ● | · | adapted, tested (test-time steering) |
| 47 | I-JEPA | 2301.08243 | · | ○ | · | informed |
| 48 | NextLat | 2511.05963 | ○ | · | ● | tested; optional arm |
| 49 | HiLP | 2608.05806 | · | · | ● | read, not adopted |
| 50 | DeepSeek-V4.1-Flash (CED) | 2609.19969 | ○ | · | ● | informed R7 |
| 51 | Multi-token prediction (Gloeckle) | 2404.19737 | · | · | ○ | informed |
| 52 | JTP (Ahn et al.) | 2503.21801 | · | · | ○ | informed |
| 53 | RoPE / GQA / SwiGLU / RMSNorm | 2104.09864 / 2305.13245 / 2002.05202 / 1910.07467 | ● | ● | ● | adopted (primitives) |
| 54 | Intern-S2-Mobius | — | ○ | · | · | C5 |
| 55 | In-Place TTT | — | ○ | · | · | C5 |
| 56 | LMs Need Sleep (Lee et al.) | 2605.26099 | ○ | · | · | C5 |
| 57 | LMs Need Sleep (Behrouz et al.) | 2606.03979 | ○ | · | · | future (install) |
| 58 | Emergent temporal abstractions | 2512.20605 | ○ | · | · | future (compose) |
| 59 | CLM (Contrastive LMs) | — | ○ | · | · | future selector |
| 60 | qlabs depth | — | ○ | · | · | informed |
| 61 | Macaron-V1; Titans / Atlas / MIRAS; Jev; Recurrent Looped Transformer; EBT; Epiplexity; LoopedWM; Metis; SMT; BDH-CQ; ICL with recurrent latent reasoning; DeepSeek-V3.2; NSA; IndexCache | — | ○ | · | · | background (§4) |

---

## 6. Completeness review

### 6.1 Is every paper in? (name-by-name check)

Every source list in the design records was typed into `check_papers.py`, which searches this file for each name (or a fixed alias of it) and reports anything absent. Result:

| source | items | found |
|---|---|---|
| Encoder list 1 (after the ledger) | 28 | 28 |
| Encoder list 2 (chunking, n-grams, convs) | 8 | 8 |
| Encoder list 3 (frontier, NeurIPS) | 15 | 15 |
| Encoder list 4 (Engram, gated residual, linear attn) | 13 | 13 |
| Encoder list 5 (two streams, BPTT) | 6 | 6 |
| Encoder list 6 (final design) | 19 | 19 |
| Encoder list 7 (review fixes) | 5 | 5 |
| Concept list 1 (v0) | 5 | 5 |
| Concept list 2 (v1 layout) | 8 | 8 |
| Concept list 3 (three papers) | 5 | 5 |
| Concept list 4 (loop tests) | 5 | 5 |
| Concept list 5 (v2) | 6 | 6 |
| Concept list 6 (v2.1) | 4 | 4 |
| Concept body (named in text) | 4 | 4 |
| Decoder session | 21 | 21 |
| Ledger §5.1 (trained and measured) | 15 | 15 |
| Ledger §5.2 (specified, never run) | 14 | 14 |
| Ledger §5.3 (read, not adopted) | 21 | 21 |
| Ledger §5.4 (foundations) | 28 | 28 |


**Missing: none.** 230 name checks across 19 lists (many papers appear in several lists). The first run reported one miss, "Hierarchical Reasoning Model"; it was a false alarm from the check itself (the bold markers in `**HRM** (Hierarchical Reasoning Model)` broke the alias) and the alias was corrected.

### 6.2 Is every entry correct? (read against the records)

Each entry's numbers were checked against the result files (`RESULTS.txt`, `RESULTS_v2.txt`, `RESULTS_H1_H2.txt`, `RESULTS_DECODER.txt`, `vq_check2/3`) and each reading depth against the reading notes at the start of the encoder design ("full text / partial / abstract only / not fetched").

| entry | check | result |
|---|---|---|
| Loop, Think & Generalize (sampled depth) | `RESULTS_v2.txt` | **corrected**: the first version credited hop-6 extrapolation to sampled depth alone; the fixed-R arm also reached 99% at its best checkpoint. Sampled depth mattered at the final checkpoint (77% vs 31%) and against v1's fixed R = 2 |
| DiscoLoop | `RESULTS.txt` | +750 steps in both SUB_D seeds; bridge decodability 34.9% → 2.2% (SUB) and 62.4% → 4.7% (FULL) ✓ |
| ConceptLM / SimVQ | `vq_check2.py` results in the concept record | 46/128 and 83/128 vs 128/128, VE 0.94 ✓ |
| Explorative Modeling, relaxed WTA | `RESULTS_H1_H2.txt` | 59.0 → 86.6 / 87.7% coherent; coverage 30.7 → 91.9 / 92.0%; per-latent 84.5–92.9 → 93.6–98.7% ✓ |
| GradCuit | `RESULTS_H1_H2.txt` | 2.2× matched-norm random; intermediates 92–100% ✓ |
| NextLat | `RESULTS_DECODER.txt`, paper Table 2 | +0.014 / +0.019 nats here; ≈ +0.03 nats at 1.3B (10.83 vs 10.52 FineWeb-Edu ppl) ✓ |
| HiLP | paper Tables 1, 4, 6 | HumanEval 8.77 / 10.58 / 11.33; glu_cross 3.942 vs concat 4.113; window 4, lookahead 4 ✓ |
| reading depths | encoder record, lines on reading depth | full / partial / abstract / not fetched copied as recorded; papers the record does not grade are marked "not recorded" rather than guessed |
| arXiv ids | the records | only ids that appear in the records are given; Bowman et al., Rupprecht et al., Kimi Linear, Gated DeltaNet-2, ETD, Free Transformer, Nanbeige4.2 and the blogs have "—" because no id was recorded |

### 6.3 Gaps that remain (stated, not filled)

- **Reading depth is unknown** for the papers of the encoder's frontier round (Qwen3.8-Next, Gated Attention, RNoPE-SWA, Kimi K3, DeepSeek-V4, mHC, Engram, H-Net, BLT, LFM2, Sliding-Window Beats Linear Attention). The record lists them as sources without grading the depth.
- **Superposition Yields Robust Neural Scaling** and **Over-Tokenized Transformer** appear only in source lists; the record does not say what was taken from them, so they are marked background.
- **Titans / Atlas / MIRAS, EBT, Epiplexity, MUDDFormer** were never fetched; their entries rest on prior knowledge.
- Papers that the *earlier* programmes read but that never touched the encoder, concept or decoder decisions are kept in §4 only as background, as the request was for the papers behind these three modules.
