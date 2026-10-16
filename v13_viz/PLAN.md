# V13 Anatomy: a 3D explorer of the V13 architecture (plan, 14 Oct 2026, 13:50 IST)

User request (13:15): a web app that renders V13 in 3D with detailed explanations and an exploding animation; a
separate 3D view of one token's journey from embedding to next-token prediction, including the HiLP-derived concept
path between the encoder sections, concept pooling and so on; WebGPU and three.js; extrapolation from 7-8M to 10T
parameters; the forward pass and the gradient flowing backward from the loss; more than 20 advisor consultations.

## Sources (every number in the app comes from one of these)
- `v13_tiny/model.py`: the model training now (V13-tiny-v2, 7,744,181 parameters, d = 128). Exact per-module counts are
  dumped by `build.py` from PyTorch, not typed by hand.
- `v13_NOTES.md`: §2-§9 mechanisms, §9.1 HiLP mapping, §11.2 tiers (7B-A1.4B and the 2.8T class), §13.1 evidence
  labels, §14.1-§14.6 forward pass, backward routes, barriers and gradient junctions.
- `v13_design/accounting.py`: the parameter, FLOP and cache formulas, ported to JS and checked against its own output.
- `v13_tiny/runs/base/log.jsonl`: a dated snapshot of learned values (β pool weights, τ temperatures).

## Views (one canvas, three modes)
1. **Architecture**: the whole model as a 3D stack. Four encoder rails (L1, L2 blue; G1, G2 orange) pass through E1
   (2 x [S S S F], Engram after the first S), E2 ([M M M F] looped x2 with the workspace rail W from s0), E3
   ([S' S' S' F], G only), the exit reads, then two towers: the L tower (decoder, 6 x [local, concept read, FFN],
   tied head, L_NTP) and the G tower (pool of x, sg L_enc and G_enc, W_P, concept module 4 x [SA, window read, FFN],
   u_g, NCP head, PQ codebook 4 x 32, L_NCP). Cross links: L -> G reads, u_g -> every decoder concept read, the two
   stop-gradient taps (grey dashed), the identity tap, the tied head.
   - Exploded view: one parameter e in [0, 1] interpolates assembled and exploded positions (stages separate along
     depth, sublayers within a layer, towers apart); button and slider.
   - Hover and click: a side panel with what the part does, its reads and writes, formula, exact parameters,
     section reference and the §13.1 evidence label.
   - Forward pass: stage-by-stage glow plus particles along the active edges, E2 shown looping twice, narration.
   - Backward pass: red particles from L_NTP and/or L_NCP along §14.3/§14.4's routes; they stop at barriers (the
     two stop-gradient taps, the codebook and targets, "NCP never reaches the decoder"); junctions (L^enc, u_g,
     embedding, S/Engram weights, E2 shared weights) light up with their incoming route counts (§14.6).
2. **Token journey** (position level): 16 real tokens (our own sentence, tokenised by `v13_tiny/tok.py`) as columns,
   L and G sheets, sliding windows (w_E = 8, w_E3 = 16), causal F arcs into G only, Engram n-gram lookup, the E2
   workspace swept twice by the GDN-2 state, pooling 4 tokens -> 1 concept, concept self-attention and the window read,
   the HiLP part (u_g predicts the next group's PQ codes without its tokens; the target comes through the codebook,
   no gradient), the release staircase j*(t) = floor((t+1)/4) - 1, the decoder's 16-token local window, next-token
   bars. A reach overlay (R_L = 53, R_D = 68) on a longer row shows that beyond R_D only the concept path reaches the
   logits. A HiLP panel maps HiLP's three parts to V13's and states the two differences (CE over codes, not SmoothL1;
   the concept path is kept at inference).
3. **Scale**: a log slider from 7.74M to 10T with snaps at the tiny-v2 model (exact), 6.84B and 2.78T (§11.2, design
   accounting) and 10T (extrapolated, labelled as beyond the design doc). The stack morphs (layer counts, widths,
   MoE expert grids, Engram table volume); a table shows stored and active parameters, FLOPs per token, cache at 1M
   tokens, bf16 weight memory and reach.

## Graph model (drives the layout and both animations; `src/logic.js`)
The model is a residual computation graph, so the animations are derived from it, not scripted.
- **Lanes** are residual streams: L (branches L1, L2), G (G1, G2), W (E2's workspace), C (the concept stream),
  D (the decoder stream). A lane is a chain of **joints** (the stream's value between two sublayers).
- **Sublayer nodes** (E1 8 layers x {attention, FFN} plus a within-role mix, Engram after layer 1; E2 4 layers x 2
  sublayers, unrolled into loop 1 and loop 2 that share one 3D slab; E2's exit write; E3 4 x 2 plus a G mix; exit
  reads; pool of x, pool of sg L_enc, pool of G_enc, the beta mix and W_P; C 4 x {SA, window read, FFN} and u_g;
  NCP head, target, codebook, L_NCP; D 6 x {local attention, concept read, FFN}; final norm and tied head; L_NTP).
- **Edges**, each with `grad` true or false:
  - rail: joint to the next joint on a lane (the residual identity path);
  - read: joint to sublayer, only for lanes in the sublayer's read set (§2.2: S reads L; F, S' read L and G; M reads
    L, G and W; the E2 exit gate reads L, G and W);
  - write: sublayer to the next joint, only for lanes in its write set (S writes L and G; F, S' write G; M and E2's F
    write W; the E2 exit writes G; Engram writes L and G);
  - taps: identity tap emb -> pool of x (grad); L_enc -> pool of L (stop-gradient, grad false); G_enc -> pool of G;
    L_enc -> every C window read (stop-gradient, grad false); p_g -> every C sublayer (Delta-AttnRes block 0);
    L_enc -> every D sublayer (block 0, NTP route 1) and -> every D local attention (keys and values, route 2);
    u_g -> every D concept read; tied head <-> embedding (tiny model);
  - targets: m (mix) -> target sg[m_{g+1}] -> codebook (EMA) -> L_NCP, all grad false; token ids -> embedding and
    -> Engram (ids, not differentiable).
- **Forward animation** = topological order from the token ids. **Backward animation** = breadth-first search from a
  loss over reversed edges with grad true; a reversed edge with grad false is a **barrier** (particles stop and the
  barrier flashes). Nodes never reached stay dim ("the decoder is never reached by NCP", "NTP never reaches the NCP
  head or the codebook" fall out of the graph). A node's count of incoming gradient edges marks it as a **junction**
  (§14.6: L^enc, u_g, the embedding, S and Engram weights, E2's shared weights).
- Checks in node: NTP and NCP reach exactly the module sets of §9.3's table; the barriers hit are exactly §14.3/§14.4's.

## Honesty labels shown in the app
- exact (PyTorch count of the model training now); design accounting (§11.2); interpolated; extrapolated.
- Animations are schematic: particle counts, speeds and glow are illustrative, not measured magnitudes.
- Learned values (β, τ) are a dated snapshot from the training log.

## Technology (revised after advisor #1)
- three.js 0.180.0 `build/three.webgpu.js` imported by absolute jsdelivr URL (no import map; its only import is the
  relative `./three.core.js`). WebGPURenderer falls back to WebGL 2 when WebGPU is unavailable; the active backend is
  shown and logged. Own orbit controls and DOM label projection (no addons). No post-processing.
- Sources: `src/page.html` (markup, CSS), `src/logic.js` (pure: graph, routes, layout, scaling; node-tested),
  `src/content.js` (explanations), `src/app.js` (scene, UI, animation), `data/*.json` (generated by `extract.py`).
  `build.py` inlines them into `dist/v13_anatomy.html` (the published file, no skeleton) and `dist/local.html` (with a
  doctype, for opening on this Mac).

## Delivery: three publishes, one look each
1. Architecture view, explode, detail panel, forward and backward routes.
2. Token journey and the HiLP panel.
3. Scale (7.74M to 10T).
No looks or app use on this Mac between 20:15 and 20:25 (the continual budget probe is timed then).

## Training safety (the 24-h run uses this Mac's GPU until ~04:00)
- Render on demand only (no continuous loop when idle), Eco mode by default (pixel ratio <= 1.25, 30 fps cap during
  animations), pause when the tab is hidden.
- My own checks: node tests of `logic.js` (no GPU); one rendered look at the end (headless Chrome at background QoS,
  console captured), one pass of fixes, then publish.

## Verification
- `scale_check.mjs`: the JS port reproduces accounting.py for its three tiers (stored, active, FLOPs) and the tiny-v2
  PyTorch total within the formula's stated omissions (norms, biases, scalar gates).
- Graph check: every §14.3/§14.4 route edge exists; barriers block exactly the routes the doc says.

## v2 scene spec: token journey + HiLP (written 14 Oct 15:45, before code)

A second view ("Token journey"), toggled from the header, sharing the renderer, controls, panel and narration. The
x-axis is token position, one column per token; the y-axis is the pipeline stage, bottom to top. Walked step by step
(Next / Back / Play) with narration; every step names its section of v13_NOTES.

Document: tokens.json, 16 tokens (`<text>` + 15 pieces), so 4 groups; the next piece is " He".

Steps:
1. **Ids → x**: 16 token cards; each rises into an embedding column (x_t = E[id_t]).
2. **Four branches**: x copied to L₁ L₂ (blue) and G₁ G₂ (orange) sheets above the columns.
3. **E1 S layers**: for a chosen row t, an 8-token window (t−7 … t) highlighted on the L sheet; arrows into L and G at t.
4. **E1 F layers**: causal arcs from every earlier row into G only (none into L).
5. **Engram**: 2- and 3-gram brackets under the cards (t−1..t, t−2..t) feeding a lookup, gated into each branch.
6. **E2**: a workspace sheet (amber) starting from s₀; two left-to-right sweeps (loop 1, loop 2) of the DeltaNet state,
   then the exit write into G.
7. **E3**: 16-token windows reading L and G, writing G; L sheet unchanged (dimmed "L is final after E1").
8. **Exit reads**: L^enc and G^enc ribbons.
9. **Pool**: each group of 4 columns funnels into one concept bead (x, sg[L], G sources, β weights at the bead).
10. **Concept module**: 4 beads; self-attention arcs to earlier beads; each bead's 16-token window read of sg[L^enc]
    (4g−12 … 4g+3, clipped at 0) drawn as a dashed grey bracket.
11. **NCP (HiLP's upper level)**: from u_g a prediction arrow to the codes of group g+1 (4 segments × 32), and a dashed
    stop-gradient arrow down from m_{g+1} through the codebook (target). Last group has no target.
12. **Release staircase**: decoder row t may read concepts j ≤ ⌊(t+1)/4⌋ − 1. A step-shaped mask over (row × concept).
13. **Decoder**: for the chosen row, a 16-token window over L^enc as keys and values, plus the released concept reads.
14. **Next token**: the last row's top-5 next-token bars from the tied head.
15. **Reach overlay**: on a longer ruler (100 positions, schematic, not this document): R_L = 51, R_D = 66; anything
    further back reaches the logits only through concepts.

HiLP panel (in the reading panel, opened from step 11 and from the NCP node in the architecture view): a three-row
mapping table, HiLP term → V13 component → difference (§9.1): abstract latent = pooling over k = 4 → m_g/p_g; upper
dynamics = concept module + NCP head (CE over PQ codes, not SmoothL1, C2); combined head = the decoder (V13 keeps the
concept path at inference).

Measured or illustrative: decide at consultation 5 whether a CPU-only, background-QoS forward of `latest.pt` on this
16-token document supplies measured values (pool softmax weights per group, β, the NCP's top code per segment against
the target code, concept-read attention for the chosen row, next-token top-5). If yes, `extract.py --journey` writes
`data/journey.json` with the checkpoint step, and every measured number in the view carries "step N, this document".
