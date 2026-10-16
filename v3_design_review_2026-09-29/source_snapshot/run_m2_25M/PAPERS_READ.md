# Papers read for this run (depth stated honestly)

Text extracted with PyMuPDF from the PDFs already on this machine (`~/Desktop/HRM THINKER/Papers`, `~/Desktop/Papers`,
`~/Desktop/RL/Papers Direct`, `~/Downloads`). The design is frozen (PROMPT ground rule 1), so papers were read for
implementation details and for interpreting the gates, not for redesign.

**Depth legend**: *method* = abstract, method/architecture section and the parts that fix an implementation or recipe
detail; *abstract* = abstract and conclusion only; *not found* = no local PDF (not fetched; the design records'
summaries in `HRM_THINKER_PAPERS.md` are relied on).

| paper | local PDF | depth | what it confirmed for this run |
|---|---|---|---|
| NCP-ArchPreview (2609.10715) | yes | method (§1–4.6) | mean pooling k = 4, PQ vocabulary (32 × 128 × 128 at 8.9B), VQ with sg on the encoder (Eq. 21), NCP as MSE on Σπe (Eq. 22; our CE variant is the concept spec's deliberate change), shift Δ = k (Eq. 11; our j*(t) is the decoder spec's target-aligned version), IRC/CRC (+0.032 loss at 1B; not used here, F2), per-head QK-norm suppresses logit growth under Muon (we use per-head QK-norm) |
| Engram (2601.07372) | yes | method (§2) | canonical-id compression (NFKC, lowercase), multi-head multiplicative-XOR hashing into prime-sized tables, gate σ(RMS(h)·RMS(k)/√d), short causal conv (kernel 4) + SiLU; matches `hrm_encoder.Engram` (ours: conv dilation 1, placed after S1) |
| Gated Attention (NeurIPS 2025) | yes | method (§1–2) | elementwise sigmoid gate after SDPA (G1) removes attention sinks and stabilises training; used in all attention layers |
| Loop, Think & Generalize (2604.07822) | yes | method (§1–5) | dynamic (sampled) recurrence per batch gives depth extrapolation; overthinking at excess depth; three-stage grokking. Basis of R ~ U{1..4} and of the loop-depth sweep R ∈ {1,2,3,4,6,8} |
| HRM-Text (2605.20613) | yes | method (recipe) | constant LR 2.2e-4, EMA 0.9999, 196k-token batches, no clipping, MagicNorm, warm-up deep credit assignment. Not used (PROMPT fixes AdamW 6e-4 cosine); listed as a next experiment |
| Qwen3.8-Flash-Next ("Qwen 4" PDF) | yes | method (GR section) | Gated Residual: per-branch RMSNorm, elementwise sigmoid read gate from a low-rank bottleneck, per-branch scalar write s = 2σ(·); our GRRead/GRWrite follow this form with role masks |
| Next-Latent Prediction (2511.05963) | yes | abstract + objective | belief-state argument; optional arm, off in the main run (PROMPT 2.4) |
| Hierarchical Latent Prediction (2608.05806) | yes | abstract + method | upper latent over k = 4 lower latents; the concept module plays this role on the forward path |
| Sliding-Window Beats Linear Attention (2608.28444) | yes | abstract | SWA (with sinks) ≥ post-trained linear attention; our SWA has no sinks (gated attention replaces them) |
| TRM, "Less is More" (2510.04871) | yes | abstract + intro | single tiny network, carried latent, deep supervision; v2 keeps one network + carried state + full BPTT |
| DiscoLoop (2607.00341) | yes | abstract + conclusion | decoded-embedding channel between loops; rejected in the concept design (co-adapted away) |
| Explorative Modeling (2607.27372) | yes | abstract + conclusion | K-candidate best-of training; XM latent is screen S4 only |
| GradCuit (2608.02585) | yes | abstract + conclusion | test-time latent steering; not used in training |
| GRAM (2605.19376) | yes | abstract + conclusion | stochastic recursive latents; not adopted |
| LoopCoder-v2 (2606.18023) | yes | abstract + conclusion | 2 loops best, ≥ 3 regress under CLP offsets; relevant to reading the R-sweep |
| Nanbeige4.2 report | yes | abstract + conclusion | looped transformer at 3B; pass 2 used (C4) |
| Delta Attention Residuals (2605.18855) | yes | abstract + conclusion | routing over deltas; scale path only |
| Attention Residuals (2603.15031) | yes | abstract + conclusion | softmax depth routing; measured inert earlier (F2) |
| Free Transformer | yes | abstract + conclusion | bounded-bypass principle (informs L/G) |
| Encode-Think-Decode (Recursive Latent Thoughts) | yes | abstract + conclusion | iterate a middle subset of layers |
| Diffusion as a Training Curriculum (2609.01449) | yes | abstract + conclusion | anytime iterative solver with persistent state |
| Kimi K3, DeepSeek-V4.1-Flash (CED) | yes | abstract + conclusion | scale path; CED cross-attention motivated rule R7 (no position reads another's plan) |
| In-Place TTT, Intern-S2-Mobius, LLM Needs Sleep (both), AI inside AI, Recurrent Looped Transformer, Loop the Loopies, SMT (Pretraining RNNs without Recurrence), BDH-CQ, Titans, Atlas, MIRAS, Metis, LoopedWM, EBT, Epiplexity, Macaron-V1 | yes | abstract (+ conclusion where present) | background for memory / continual learning / looping (constraints C5, C6); none enters this run |
| Canon layers, RNoPE-SWA, H-Net, BLT, Block Transformer, Huginn, ConceptLM, MUDDFormer, mHC, LFM2, VQ-VAE, I-JEPA, Over-Tokenized Transformer, Gated DeltaNet-2, Kimi Linear | not found locally | not read here | relied on the design records' summaries (`HRM_THINKER_PAPERS.md`, `ARCHITECTURES_AND_PAPERS.md`) |
