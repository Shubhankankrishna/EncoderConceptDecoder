// Architecture data for the explorer: V9 → V10.1 → V10.2 → V10.3 → V11 → V12.
// Geometry conventions: y = depth (tokens at the bottom, logits at the top); lane 'trunk' = token rate (x = 0);
// lane 'concept' = segment rate (+x, narrower: it runs once per segment); lane 'train' = training-only (behind, −z,
// translucent); lane 'side' = memory and continual learning (−x); lane 'post' = post-training / inference-time modules
// (−x, high). Component ids are shared across versions so the diff view is one function.

export const STATUS = {
  measured: { label: 'Measured here', hint: 'Run in this project; number shown is from its report.' },
  paper:    { label: 'Paper evidence', hint: 'Supported by a paper at 1–9B scale; not reproduced here.' },
  untested: { label: 'Untested', hint: 'Designed, never run.' },
  notrun:   { label: 'Arm not run', hint: 'Implemented and queued, not run in the 12-hour budget.' },
  failed:   { label: 'Measured: failed', hint: 'Run; the pre-registered check failed.' },
  design:   { label: 'Design only', hint: 'This version was never trained.' },
  inherited:{ label: 'Inherited', hint: 'Measured in V10.3 at ~0.5M parameters; not re-run in V12.' },
};

// Every V12 figure lives here so the final update is one object.
export const V12_NUMBERS = {
  usage8k: 'pinpad +0.42 nats/action, math +0.18 (u removed); text +0.03',
  info8k: 'pinpad: u beats its own input by 7.7 points on next-segment codes (math, text inside the 2–4-point noise floor)',
  cost8k: '+0.28 text, +0.27 math vs the same topology without NCP',
  competence: '0.28 (TF7), 0.23 (ECD), 0.16 (ECDe)',
  meta: 'switches land worse than chance (pair F1 0.012 vs 0.265 at random placement): one switch right after the first step, then a held episode code that carries no subgoal identity; β collapses to zero switches at α = 0.5',
  rl: 'Internal RL gated off (switch F1 far below 0.8)',
};

// Shared component vocabulary. `lane`, `level` and size set the rest geometry; versions may override `variant`,
// `label`, `text`, `status`, `level`.
export const COMPONENTS = {
  tokens:   { lane: 'trunk', level: 0, w: 4.2, h: 0.18, label: 'Tokens', group: 'io',
              text: 'The observed stream. In the V12 test: pinpad episodes (60%), text (25%), math answers (15%); joint vocabulary 1,090.' },
  emb:      { lane: 'trunk', level: 1, w: 4.0, h: 0.3, label: 'Embedding', group: 'io',
              text: 'Token embedding, tied with the output head.' },
  engram:   { lane: 'side', level: 1.6, w: 1.6, h: 0.5, label: 'Engram n-gram table', group: 'mem',
              text: 'Hashed n-gram lookup after E1 block 1; 20M sparse parameters on CPU at the 10M reference. A local-pattern memory, not preloaded facts.' },
  enc:      { lane: 'trunk', level: 2, w: 3.8, h: 1.2, label: 'Encoder E', group: 'E', blocks: 2,
              text: 'Token-rate causal blocks. E is the lower part of an ordinary trunk.' },
  enc_twostream: { lane: 'trunk', level: 2.9, w: 3.8, h: 0.9, label: 'E2 two-stream ×2 visits', group: 'E',
              text: 'Two-stream gated blocks executed twice with shared weights (V10.1–V11).' },
  enc3:     { lane: 'trunk', level: 3.6, w: 3.8, h: 0.6, label: 'E3', group: 'E',
              text: 'Exit reader then independent encoder blocks.' },
  pool:     { lane: 'concept', level: 4.2, w: 2.0, h: 0.32, label: 'Segment pooling', group: 'C',
              text: 'Completed groups of k tokens are summarised into one state per segment.' },
  codes_selector: { lane: 'concept', level: 4.75, w: 1.8, h: 0.32, label: 'K codes + selector', group: 'C',
              text: 'K discrete strategy codes and a selector that predicts which code will win.' },
  core:     { lane: 'concept', level: 5.4, w: 2.0, h: 1.1, label: 'Concept module C', group: 'C', blocks: 2,
              text: 'Segment-rate causal blocks over completed segments. It costs 1/k̄ per token.' },
  visit2:   { lane: 'concept', level: 6.35, w: 2.0, h: 0.55, label: 'Visit 2 (shared weights)', group: 'C',
              text: 'A second pass through the same concept blocks.' },
  ncp_head: { lane: 'train', level: 6.2, w: 1.8, h: 0.6, label: 'Next-concept head', group: 'aux',
              text: 'Predicts the next segment from the concept state.' },
  forecast_to_dec: { lane: 'concept', level: 7.0, w: 1.8, h: 0.35, label: 'Soft-PQ forecast → D', group: 'C',
              text: 'A soft product-quantised forecast of the next group, shifted and fed into the decoder input.' },
  reader:   { lane: 'concept', level: 7.55, w: 1.8, h: 0.35, label: 'Concept-history reader', group: 'C',
              text: 'Content-addressed reader over released concept states, read by the decoder through cross-attention.' },
  port_a:   { lane: 'concept', level: 7.0, w: 1.4, h: 0.3, label: 'Continuous port A', group: 'C',
              text: 'A continuous concept port shifted into the decoder.' },
  crc:      { lane: 'trunk', level: 4.6, w: 0.5, h: 2.6, label: 'Cross-module routes (CRC)', group: 'route', x: 2.3,
              text: 'Encoder→concept and encoder→decoder shortcut routes.' },
  dec:      { lane: 'trunk', level: 8.2, w: 3.8, h: 1.2, label: 'Decoder D', group: 'D', blocks: 2,
              text: 'Token-rate causal blocks. D is the upper part of an ordinary trunk.' },
  head:     { lane: 'trunk', level: 9.4, w: 4.0, h: 0.3, label: 'Output head → next token', group: 'io',
              text: 'Tied vocabulary projection; next-token loss (NTP).' },
  rbf:      { lane: 'side', level: 5.4, w: 1.6, h: 0.6, label: 'RBF correction bank', group: 'mem',
              text: 'Fast memory: exact-gradient corrections keyed by the concept state, weights frozen.' },
  lora_gate:{ lane: 'side', level: 7.2, w: 1.6, h: 0.7, label: 'Gated LoRA (sleep)', group: 'mem',
              text: 'Slow memory: the bank is consolidated into a gated low-rank adapter, then reset.' },
  bridge:   { lane: 'train', level: 7.8, w: 1.8, h: 0.6, label: 'NextLat bridge (8 steps)', group: 'aux',
              text: 'From the concept state and the actual next tokens, roll out up to 8 decoder states without re-grounding.' },
  probe:    { lane: 'train', level: 7.0, w: 1.4, h: 0.35, label: 'Detached forecast probe', group: 'aux',
              text: 'Training-only probe of the forecast; no gradient into the trunk.' },
  nextlat:  { lane: 'train', level: 8.9, w: 1.8, h: 0.5, label: 'NextLat on D', group: 'aux',
              text: 'p([d_t; E x_{t+1}]) → d̂_{t+1}; SmoothL1 to stopped d_{t+1} plus KL through a frozen head view; one step, dropped at inference.' },
  metacontroller: { lane: 'post', level: 5.6, w: 1.9, h: 1.6, label: 'Metacontroller', group: 'post',
              text: 'Post-training, on the frozen base: reads the residual at the control point, chooses abstract action z, switches with β, writes a low-rank controller U.' },
  policy:   { lane: 'post', level: 7.6, w: 1.6, h: 0.5, label: 'Internal RL policy', group: 'post',
              text: 'A causal policy over z replaces the acausal encoder; PPO on sparse reward.' },
  mcts:     { lane: 'post', level: 7.0, w: 1.8, h: 0.8, label: 'Decoder-free concept search', group: 'post',
              text: 'Receding-horizon search over next-concept codes without the decoder; decode and execute one step, re-encode, search again.' },
  lsl:      { lane: 'side', level: 3.0, w: 1.6, h: 0.6, label: 'LSL continual learning', group: 'mem',
              text: 'Local Support Learning gates: a new skill trains where its inputs are separable from earlier phases.' },
};

// Sub-explode parts (clicked component splits into these).
export const PARTS = {
  core: [
    { label: 'Visit embedding', text: 'Added at each visit so shared blocks know which pass they are on.' },
    { label: 'C block 1', text: 'Causal attention over completed segments + MLP.' },
    { label: 'Control point (mid-C)', text: 'Where the metacontroller writes e ← e + U e. Probes show this state is the least decodable in C (parity 0.74 vs 0.89 at u), the same decode-vs-control split the paper reports by depth.' },
    { label: 'C block 2', text: 'Computes the steered concept that D reads.' },
    { label: 'RMSNorm → u', text: 'The released concept state. D sees segment g from its last token on.' },
  ],
  ncp_head: [
    { label: 'Target: next segment summary', text: 'sg[EMA-normalised mean state of segment g+1]: encoder side (NCP) or decoder side (HiLP upper level).' },
    { label: 'Product codebook 4 × 32', text: 'Updated by EMA k-means with dead-code reset, never by gradient. Usage perplexity stayed 29–30 of 32 on every family.' },
    { label: 'MLP head → 4 softmaxes', text: 'Cross-entropy over code indices: mode-preserving, unlike the squared error V9–V11 used, whose optimum is the blurred conditional mean.' },
  ],
  metacontroller: [
    { label: 'Sequence embedder s(e₁:T)', text: 'Non-causal bi-GRU over the whole episode (training only; replaced by the policy in RL).' },
    { label: 'History GRU', text: 'Causal summary h_{t−1} of the residuals so far.' },
    { label: 'Encoder → N(μ, Σ), n_z = 8', text: 'Proposes a new abstract action z̃_t. KL to N(0, I) weighted by α sets the rate.' },
    { label: 'Switch β_t', text: 'z_t = β z̃_t + (1−β) z_{t−1}. Sparse switching emerges because KL is paid every step.' },
    { label: 'Hypernetwork → U (rank 16)', text: 'U_t = A_t B_t applied as e ← e + U_t e. With binarised β, U is emitted only at switches: identical output, 13.9× fewer stored values on a 64-episode batch (untrained switch).' },
  ],
  rbf: [
    { label: 'Keys = concept states', text: 'Each write stores a corrective vector at the concept state where the loss occurred.' },
    { label: 'Gaussian read', text: 'Nearby concept states retrieve the correction; far ones see none.' },
    { label: 'Weights frozen', text: 'V10.3: −1.55 nats on an unseen math module with frozen weights (88% of LoRA\'s gain).' },
  ],
  lora_gate: [
    { label: 'LSL gate (GMM)', text: 'Opens only on inputs that look like the new phase; reference covers every earlier phase.' },
    { label: 'Unlocked low-rank expert', text: 'Pre-allocated, masked until sleep; at scale a new MoE expert.' },
    { label: 'Distil, then reset the bank', text: 'V10.3: 90% of LoRA\'s gain kept, zero forgetting on dissimilar families; related add/sub still forgets (+0.26 to +0.46).' },
  ],
  enc: [
    { label: 'Block 1', text: 'Causal self-attention (RoPE) + GELU MLP, pre-norm.' },
    { label: 'Block 2', text: 'Same shape. At scale: E is about the lower 40% of layers, every FFN an MoE.' },
  ],
  dec: [
    { label: 'Cross-block 1', text: 'Causal self-attention, then cross-attention to released u, then MLP.' },
    { label: 'Cross-block 2', text: 'Same. D never receives the NCP gradient (verified: exactly 0).' },
  ],
  pool: [
    { label: 'Mean of k = 4 tokens', text: 'Fixed k is a tiny-test convenience. At scale: a trained H-Net-style boundary detector with a ratio loss to k̄ ≈ 4–8.' },
    { label: 'RMSNorm + linear', text: 'Projects the summary into C\'s input space.' },
  ],
};

PARTS.forecast_to_dec = [
  { label: 'Soft-PQ expectation', text: 'A probability-weighted average of codewords: the blended conditional mean, not a committed choice.' },
  { label: 'Fusion map', text: 'V10.1 passed it through a per-coordinate gain: it carried 3–14% of the target variance. V10.3 used a full learned map (keeps the scale).' },
  { label: 'Shift into the decoder', text: 'Repeated over the next group\'s rows and added to the decoder input; still predicted the next group worse than the core\'s own input.' },
];
PARTS.bridge = [
  { label: 'Anchor: concept state u₁', text: 'Starts from the visit-1 concept state instead of the decoder\'s own previous state.' },
  { label: '8 token-conditioned steps', text: 'Rolls out decoder states using the true next tokens, with no re-grounding between steps.' },
  { label: 'SmoothL1 + KL via frozen head', text: 'Measured: 2.6–2.9× NTP\'s gradient into the core, +0.41 nats on math, compressed pooled states.' },
];
PARTS.enc_twostream = [
  { label: 'Two residual streams', text: 'Read, compute and write gates move information between two streams.' },
  { label: 'Two visits, shared weights', text: 'The same 4 blocks run twice: 12 distinct blocks, 16 executed applications in the encoder.' },
];
// Which versions show each part set (V12 numbers stay on V12).
export const PARTS_VERSIONS = {
  core: ['v12'], ncp_head: ['v12'], metacontroller: ['v12'], enc: ['v12'], dec: ['v12'], pool: ['v12'],
  rbf: ['v10_1', 'v10_2', 'v10_3', 'v11', 'v12'], lora_gate: ['v10_3', 'v11', 'v12'],
  forecast_to_dec: ['v10_1', 'v10_2', 'v10_3', 'v11'], bridge: ['v11'], enc_twostream: ['v10_1', 'v10_2', 'v10_3', 'v11'],
};

// Versions. `has` lists component ids present, with per-version overrides.
export const VERSIONS = [
  {
    id: 'v9', name: 'V9', date: '6–10 Oct 2026',
    title: 'Committed next concepts and decoder-free search',
    designSource: 'draft_v9.md §0–1',
    measured: 'v9_tiny Q5 (GraphNav, standard / hard10): reference 83.2% / 34.0%, expert iteration 85.2% / 33.0% (+2.0 / −1.0 vs reference), GRPO 0% / 0% (a baseline failure, not a V9 win). Single seed, unequal budgets; the report makes no equal-compute search superiority claim.',
    idea: 'Give the concept module K committed hypotheses and a selector, so search and RL can choose among discrete next concepts.',
    wrong: 'NCP was still squared error on a soft-PQ expectation (its optimum is the conditional mean), the stages were trained separately against a frozen V8 teacher, and the test bed had segments of about 4 hops whose latent was copyable four tokens back.',
    has: {
      tokens: {}, emb: {},
      enc: { label: 'Encoder E3', blocks: 3, text: 'Three token-rate blocks (V8-D foundation, 2.45M parameters).' },
      pool: { label: 'Pool k = 4' },
      core: { label: 'Concept stack C4', blocks: 4, text: 'Four concept blocks; in stage 2 a student core (dense or HRM-recurrent) is distilled from the frozen stage-1 teacher.' },
      codes_selector: { label: 'H-NCP: K codes + selector', status: 'design', text: 'K codes trained winner-take-all against the true next concept; the selector is both the search prior and the RL policy.' },
      ncp_head: { label: 'Soft-PQ head, MSE NCP', status: 'failed', text: 'Squared error between the soft-PQ expectation and the next pooled concept. Its minimiser is the conditional mean: one blended vector, not branchable alternatives.' },
      port_a: {},
      dec: { label: 'Decoder D3', blocks: 3 },
      head: {},
      mcts: { status: 'measured', text: 'Receding-horizon search over next concepts with the decoder removed from the lookahead. Expert iteration on the selector reached 85.2% / 33.0% against the reference 83.2% / 34.0% (+2.0 / −1.0): no clear gain. GRPO scored 0%.' },
      lsl: { status: 'design' },
    },
  },
  {
    id: 'v10_1', name: 'V10 / V10.1', date: '11 Oct 2026',
    title: 'Recurrent concept core with a correction bank (RCDL)',
    designSource: 'V10_1_RCDL_Architecture.md §3',
    measured: 'V10 10M run (54k updates, 115M tokens): text dev NLL 4.88; strict math 3/16; Sudoku 0/4; concept-off +0.0007 nats with an interval including zero (the deployed concept path never beat its control).',
    idea: 'Deepen the encoder (Engram, two-stream recurrence), pool 8 tokens into 2 slots, and add an RBF bank keyed by the concept state for fast learning.',
    wrong: 'The soft forecast reached the decoder through a per-coordinate gain and carried 3–14% of the target variance; the codebook shrank under weight decay; the forecast predicted worse than its own input.',
    has: {
      tokens: {}, emb: {}, engram: {},
      enc: { label: 'E1 (4 blocks)', blocks: 4, h: 0.9, level: 2 },
      enc_twostream: {}, enc3: {},
      pool: { label: 'Pooler 8 → 2 slots' },
      codes_selector: { label: 'K = 8 codes + prefix selector' },
      core: { label: 'Anchored core 4 × 2 visits', blocks: 4 },
      visit2: { status: 'failed', text: 'Second visit of the same core. Measured later as inert (≤ 0.011 nats in V10.3).' },
      rbf: { status: 'measured' },
      forecast_to_dec: { status: 'failed' },
      crc: {},
      ncp_head: { label: 'Latent MSE + next-chunk readout', status: 'failed', text: 'Latent regression of the next group plus a training-only next-chunk readout.' },
      dec: { label: 'Decoder (4 blocks)', blocks: 4 },
      head: {},
    },
  },
  {
    id: 'v10_2', name: 'V10.2', date: '12 Oct 2026',
    title: 'Predictive concepts with causal decoder retrieval',
    designSource: 'V10_2_RCDL_Architecture.md',
    measured: 'Design only: never implemented or trained (5.94M dense + 20M Engram by count).',
    idea: 'Replace the pointwise concept→decoder interface with a content-addressed reader over released concept history, and supervise the forecast head directly on future token ids.',
    wrong: 'A token-space concept objective still reached the trunk, and the forecast still regressed a multimodal target.',
    has: {
      tokens: {}, emb: {}, engram: {},
      enc: { label: 'E1 (4 blocks)', blocks: 4, h: 0.9 }, enc_twostream: {}, enc3: {},
      pool: { label: 'Pooler 8 → 2 slots' }, codes_selector: { label: 'K = 8 codes + selector' },
      core: { label: 'Core 4 × 2 visits', blocks: 4 }, visit2: {},
      rbf: {}, crc: {},
      forecast_to_dec: { label: 'Forecast-only token head', status: 'design', text: 'Compact forecast head supervised on future token ids instead of latent regression.' },
      reader: { status: 'design' },
      dec: { label: 'Decoder (4 blocks)', blocks: 4 }, head: {},
    },
  },
  {
    id: 'v10_3', name: 'V10.3', date: '12 Oct 2026',
    title: 'Calibrated concept prediction, mixture exploration, gated sleep',
    designSource: 'V10_3_RCDL_Architecture.md; V10_3_tiny_REPORT.md',
    measured: 'Tiny (~0.5M): concept arm X8 worse than the concept-free C0 on text (+0.018 seed 1, +0.005 seed 2) and math (+0.029 / +0.119); forecast worse than its input; visit 2 inert; codes barely matter; Sudoku gain was a period-8 layout artefact. Sleep: gated LoRA passes G4; RBF −1.55 nats frozen; consolidation keeps 90%; add/sub unprotected.',
    idea: 'Fix the scale of what reaches the decoder (full learned forecast map), exempt the codebook from decay, train exploration as a mixture likelihood with the selector as prior, and add a gated sleep layer.',
    wrong: 'The concept path cost more than it returned, and exploration had nothing to explore: the data had no persistent latent for codes to choose between.',
    has: {
      tokens: {}, emb: {}, engram: {},
      enc: { label: 'E1', blocks: 2, h: 0.9 }, enc_twostream: {}, enc3: {},
      pool: { label: 'Pooler 8 → 2 slots' },
      codes_selector: { label: 'K = 8 mixture, selector as prior', status: 'failed', text: 'Mixture, uniform and committed likelihoods agreed within 0.002 nats/token: the code choice barely mattered.' },
      core: { label: 'Core × 2 visits', blocks: 2 }, visit2: { status: 'failed' },
      forecast_to_dec: { label: 'Soft-PQ forecast, full learned map', status: 'failed', text: 'Full learned map keeps the target scale (N2), but the forecast still predicts the next group worse than the core input (math FVU 0.694 vs 0.637).' },
      reader: { status: 'measured' },
      ncp_head: { label: 'Forecast token CE + mixture', status: 'failed' },
      crc: {}, rbf: { status: 'measured' }, lora_gate: { status: 'measured' },
      dec: { blocks: 2 }, head: {},
    },
  },
  {
    id: 'v11', name: 'V11', date: '13 Oct 2026 (morning)',
    title: 'Latent next-concept objective and a NextLat bridge',
    designSource: 'v11.md §1, §3.2; v11_tiny/REPORT.md',
    measured: 'Tiny: every pre-registered gate fails on K1. The bridge pushed 2.6–2.9× NTP\'s gradient into the core and cost +0.41 nats on math; the latent objective made the concept route live but added nothing on NLL.',
    idea: 'Keep V10.3\'s deployed graph; train the concept state to predict the next normalised pooled state, move the strategy code to visit 2, and add a training-only concept-anchored NextLat bridge.',
    wrong: 'No metacontroller was tested (frozen-base meta-control was listed as optional, co-training as rejected). The bridge was neither NextLat (which anchors on the network\'s own previous state, one step) nor HiLP (an upper predictor without token input). The code at visit 2 still fed a regression target.',
    has: {
      tokens: {}, emb: {}, engram: {},
      enc: { label: 'E1', blocks: 2, h: 0.9 }, enc_twostream: {}, enc3: {},
      pool: { label: 'Pooler 8 → 2 slots' },
      core: { label: 'Visit 1 (code-free belief)', blocks: 2 },
      visit2: { label: 'Visit 2 + code from prior(sg[u₁])', status: 'failed' },
      codes_selector: { label: 'Prior π(k | sg[u₁])', status: 'failed' },
      forecast_to_dec: { label: 'PQ forecast → fusion map', status: 'failed' },
      reader: {}, crc: {}, rbf: {}, lora_gate: {},
      ncp_head: { label: 'Latent Gaussian NLL (next pooled state)', status: 'failed', text: 'Regression of the next normalised, stopped pooled encoder state, mixture over K visit-2 branches.' },
      bridge: { status: 'failed' }, probe: {},
      dec: { blocks: 2 }, head: {},
    },
  },
  {
    id: 'v12', name: 'V12', date: '13 Oct 2026',
    title: 'Encoder · Concept · Decoder with a frozen-base metacontroller',
    designSource: 'v12_Architecture.md §2; v12_tiny/DESIGN_STEPS.md',
    measured: `Tiny (~1.08M), matched step 8k. D reads C: ${V12_NUMBERS.usage8k}. Information: ${V12_NUMBERS.info8k}. The NTP-only control never uses C. NCP's cost persists: ${V12_NUMBERS.cost8k}. Ground-truth controllers carry the label, but nothing composes closed-loop at base competence ${V12_NUMBERS.competence}. Metacontroller: ${V12_NUMBERS.meta}. ${V12_NUMBERS.rl}.`,
    idea: 'E is the lower trunk, D the upper trunk, C sits at the split, which is mid-depth: the same place the metacontroller controls. Next-concept prediction becomes classification over a discrete codebook. The metacontroller is trained afterwards on the frozen base.',
    wrong: null,
    has: {
      tokens: {}, emb: {},
      enc: { label: 'Encoder E (2 blocks)', status: 'measured' },
      pool: { label: 'Segment pooling (k = 4)', status: 'measured' },
      core: { label: 'Concept module C (mid-depth)', status: 'measured' },
      visit2: { label: 'Loop: visit 2 = post-control depth', status: 'notrun', text: 'What a second visit uniquely buys is depth after the controller at 1/k̄ cost. Every earlier measurement found visit 2 inert; the ECDL2e arm was not run.' },
      ncp_head: { label: 'NCP: CE over codes (4 × 32)', status: 'measured', text: 'Cross-entropy over product-codebook indices of the stopped next-segment summary, λ = 0.1. Reaches C and E, never D (gradient exactly 0).' },
      dec: { label: 'Decoder D (2 cross-blocks)', status: 'measured', text: 'Reads only the released concept states u by cross-attention.' },
      head: {},
      nextlat: { status: 'notrun' },
      metacontroller: { status: 'measured', text: `Paper-faithful (ETA D.2), trained on the frozen base. Measured: ${V12_NUMBERS.meta}. ${V12_NUMBERS.rl}.` },
      policy: { status: 'untested' },
      rbf: { status: 'inherited', text: 'Inherited from V10.3 (tested at 0.5M), keyed by u. Not re-run in V12.' },
      lora_gate: { label: 'Sleep → unlocked expert', status: 'untested', text: 'V12 additions, untested: on-policy distillation on self-generated dreams, dreams diversified by the metacontroller prior, replay anchored to the frozen pre-sleep model.' },
    },
  },
];

// The first-principles chain for V12 (Design Steps 2–9, in order).
export const STEPS = [
  { title: 'Start from what the data must contain', show: ['tokens', 'emb', 'enc', 'dec', 'head'], focus: ['tokens'],
    text: 'Concepts, exploration and a metacontroller pay only where the data has a latent that persists over many tokens, cannot be copied from a recent token, is multimodal given the prefix, and composes. Pinpad has one (which pair of colours next, no task token). Text and math test that the hierarchy costs nothing where no such latent exists.',
    status: 'measured' },
  { title: 'E and D are one ordinary trunk, split', show: ['tokens', 'emb', 'enc', 'dec', 'head'], focus: ['enc', 'dec'],
    text: 'Encoder E is the lower part of a standard trunk, decoder D the upper part. At frontier scale every FFN is an MoE. Nothing about E or D is exotic.',
    status: 'paper' },
  { title: 'Insert C at the split: mid-depth', show: ['tokens', 'emb', 'enc', 'pool', 'core', 'dec', 'head'], focus: ['pool', 'core'],
    text: 'C runs once per segment (1/k̄ of the token cost) over completed segments only. The split is mid-depth, where the ETA paper finds control works best, so the concept module and the control point coincide by construction. Measured here: mid-C was the weakest controlled point of the four tested (action NLL −3.5% vs −12% at TF e_5).',
    status: 'paper' },
  { title: 'Two clocks: segments and subgoals', show: ['tokens', 'emb', 'enc', 'pool', 'core', 'dec', 'head'], focus: ['pool'],
    text: 'The segmentation boundary sets C\'s clock (fixed k = 4 in the tiny test; a trained H-Net-style detector at scale). The subgoal boundary β is learned later by the metacontroller and spans several segments. Dynamic chunking is the largest untested item.',
    status: 'untested' },
  { title: 'Predict the next concept as a choice, not an average', show: ['tokens', 'emb', 'enc', 'pool', 'core', 'ncp_head', 'dec', 'head'], focus: ['ncp_head'],
    text: 'Squared error on a multimodal target returns the conditional mean, which is why V9–V11 forecasts blurred. V12 classifies the next segment\'s code in a product codebook updated by EMA k-means. Measured: without this loss D ignores C entirely; with it, on pinpad u beats its own input by 7.7 points (math and text gains sit inside the noise floor). It costs +0.28 nats on text at this budget.',
    status: 'measured' },
  { title: 'D reads only the released concept', show: ['tokens', 'emb', 'enc', 'pool', 'core', 'ncp_head', 'dec', 'head'], focus: ['dec', 'core'],
    text: 'D attends to u through cross-attention, from each segment\'s last token on. The NCP head is a training-time readout, not an input to D. Removing u costs +0.42 nats per action on pinpad and +0.18 on math at 8k.',
    status: 'measured' },
  { title: 'NextLat on D; HiLP\'s upper level is NCP on D\'s states', show: ['tokens', 'emb', 'enc', 'pool', 'core', 'ncp_head', 'dec', 'nextlat', 'head'], focus: ['nextlat', 'ncp_head'],
    text: 'Each method goes to the level it was designed for. NextLat predicts D\'s own next state one step ahead. HiLP\'s upper level is the concept state predicting the next segment\'s decoder latent as codes. V11\'s 8-step bridge was neither. Both arms were not run.',
    status: 'notrun' },
  { title: 'Loop of 2 = depth after the controller', show: ['tokens', 'emb', 'enc', 'pool', 'core', 'visit2', 'ncp_head', 'dec', 'head'], focus: ['visit2'],
    text: 'A second visit with the same weights only adds depth. Its one unique job: the controller writes between visits, and visit 2 computes the steered concept. Carried as untested, with every earlier measurement saying inert.',
    status: 'notrun' },
  { title: 'Metacontroller on the frozen base', show: ['tokens', 'emb', 'enc', 'pool', 'core', 'dec', 'head', 'metacontroller', 'policy'], focus: ['metacontroller'],
    text: 'Trained after pretraining with the base frozen (co-training collapses). It reads the residual, proposes z, switches with β, and writes U = A B. Here ground-truth controllers carry the label but nothing composes closed-loop: the base completes only 28% of its own tasks. The unsupervised metacontroller: ' + V12_NUMBERS.meta + '.',
    status: 'measured' },
  { title: 'Fast memory, then sleep', show: ['tokens', 'emb', 'enc', 'pool', 'core', 'dec', 'head', 'rbf', 'lora_gate'], focus: ['rbf', 'lora_gate'],
    text: 'An RBF bank keyed by the concept state learns a new skill with frozen weights (−1.55 nats in V10.3). Sleep distils it into a gated low-rank expert and resets the bank. Related skills are still unprotected.',
    status: 'measured' },
];

// Scale anchors. `fact` fields are vendor figures; everything else is the V12 schedule (illustrative).
export const ANCHORS = [
  { n: 1.08e6, name: 'V12 tiny (measured)', fact: '1,075,712 dense + 66,048 NCP head; d 128; E2/C2/D2', measured: true },
  { n: 1e8, name: '100M' },
  { n: 1e9, name: '1B' },
  { n: 8.9e9, name: 'NCP-Arch scale', fact: 'NCP-ArchPreview 8.9B: 16 encoder / 8 concept / 16 decoder layers; 5.73T tokens' },
  { n: 7.44e11, name: 'GLM-5 class', fact: 'GLM-5: 744B total, ~40B active; 256 experts, 8 routed + 1 shared (vendor figures, unverified)' },
  { n: 2.8e12, name: 'Kimi K3 class', fact: 'Kimi K3: 2.8T total; 16 of 896 experts active; active count undisclosed (vendor figures, unverified)' },
  { n: 1e13, name: '10T (extrapolated)' },
];

// First principles per component: why it exists, what the evidence says, what would show it wrong.
export const PRINCIPLES = {
  v12: {
    enc: { why: 'Token-level features must exist before anything can be summarised into segments.', evidence: 'Same blocks as any trunk; the ECD control learns faster than TF7 early (dev pinpad 0.642 vs 0.691 at 4k).', falsifier: 'An encoder this shallow loses to a deeper plain trunk at matched compute.' },
    pool: { why: 'C should run once per segment, so its cost is 1/k̄ per token and its clock is coarser than tokens.', evidence: 'Fixed k = 4 here; Sudoku showed fixed groups can align with data layout (V10.3), so Sudoku was excluded.', falsifier: 'A trained boundary detector at equal compute beats fixed k; or C\'s gains vanish when k is shifted.' },
    core: { why: 'An abstract state that persists across tokens is what the metacontroller needs to switch between, and mid-depth is where control works.', evidence: 'Used heavily by D at 8k (pinpad +0.42 nats removed); without an NCP objective it is never used.', falsifier: 'Removing C costs nothing at matched compute; or a plain trunk\'s mid-depth is as controllable.' },
    ncp_head: { why: 'The next segment is multimodal given the prefix. Regression returns the conditional mean; a categorical over codes keeps the modes.', evidence: 'Makes C used and informative (pinpad +7.7 points over its input); costs +0.28 text, +0.27 math at λ = 0.1; codes drift 37–64% between 4k and 8k.', falsifier: 'A λ schedule removes the cost and the gain together; or an MSE head matches it on the info test.' },
    dec: { why: 'Token prediction stays an ordinary job; D should only read what C has finished computing.', evidence: 'NCP gradient into D measured exactly 0; release mask tested (no read before a segment completes).', falsifier: 'D routes around u (usage ≈ 0), as it does in the NTP-only control.' },
    visit2: { why: 'If the controller writes between visits, visit 2 is post-control depth at 1/k̄ cost.', evidence: 'Every earlier measurement found visit 2 inert (≤ 0.011 nats). The arm was not run.', falsifier: 'ECDL2e is no more controllable than ECDe controlled at C\'s output.' },
    nextlat: { why: 'One-step latent prediction on D\'s own state makes it a compact belief state (NextLat Thm 3.2).', evidence: 'Paper evidence at 1.3B; arm not run here.', falsifier: 'TFNL shows no gain in mid-depth probes or GT-controller success over TF7.' },
    metacontroller: { why: 'Exploration should happen in abstract-action space on a frozen base; co-training collapses.', evidence: `Here: ${V12_NUMBERS.meta}.`, falsifier: 'On a base that completes its own tasks (≥ 0.8), switches still fail to align with subgoals.' },
    policy: { why: 'Replacing the acausal encoder with a causal policy over z turns RL into choosing abstract actions.', evidence: 'Gated off: switch F1 far below 0.8.', falsifier: 'Internal RL does not beat raw-token RL on a novel composition once the gate passes.' },
    rbf: { why: 'A new skill can be learned in fast memory without touching weights, keyed where the concept state says it applies.', evidence: 'V10.3: −1.55 nats on an unseen math module with frozen weights.', falsifier: 'Bank corrections interfere with unrelated families at the same keys.' },
    lora_gate: { why: 'Fast memory is consolidated into slow weights during sleep, then cleared.', evidence: 'V10.3: 90% of the gain kept, zero forgetting on dissimilar skills; related add/sub forgets.', falsifier: 'Dreams from the metacontroller prior fail to protect related skills.' },
  },
  v11: {
    bridge: { why: 'Intended to make the concept state predictive of the decoder\'s future.', evidence: '2.6–2.9× NTP\'s gradient into the core; +0.41 nats on math.', falsifier: 'Already falsified: the predicted text gain did not appear.' },
    ncp_head: { why: 'Predict the next pooled state in normalised units instead of tokens.', evidence: 'Made the route live, added nothing on NLL.', falsifier: 'Regression of a multimodal target blurs to the mean (as measured).' },
  },
  v10_3: {
    forecast_to_dec: { why: 'Give the decoder a forecast of the next group at the right scale.', evidence: 'Forecast predicts the next group worse than the core\'s own input (math FVU 0.694 vs 0.637).', falsifier: 'Already falsified at tiny scale.' },
    codes_selector: { why: 'Let K codes represent different futures and a prior choose among them.', evidence: 'Mixture, uniform and committed likelihoods agree within 0.002 nats/token.', falsifier: 'Already falsified: the data had nothing to explore.' },
  },
  v9: {
    ncp_head: { why: 'Predict the next concept so search can run without the decoder.', evidence: 'Squared error on a soft-PQ expectation: its optimum is one blended vector.', falsifier: 'K codes trained winner-take-all collapse to the same mode.' },
    mcts: { why: 'Search over discrete next concepts, decoding only the chosen step.', evidence: 'Expert iteration +2.0 / −1.0 points vs the reference on GraphNav (single seed).', falsifier: 'No gain over the reference at equal compute.' },
  },
};

// Gradient routes (v12_Architecture.md §2.9): which loss trains which module.
export const TRAINS = [
  { loss: 'NTP (next token)', color: '--g-D', reaches: 'head, D; C through D\'s cross-attention read; E through D and through C' },
  { loss: 'NCP (next segment code), λ 0.1', color: '--g-C', reaches: 'NCP head, C, E. Into D: exactly 0 (verified). Codebook: EMA only' },
  { loss: 'NextLat (D\'s next state)', color: '--g-aux', reaches: 'D, and C/E through D (arm not run)' },
  { loss: 'Metacontroller (post-training)', color: '--g-post', reaches: 'metacontroller only; E, C, D, codebook frozen' },
];

// Step chains for the versions before V12: kept → changed → broke (with the number) → measured.
const ALL = null;   // null = show every component of that version
export const STEPS_BY_VERSION = {
  v12: STEPS,
  v10_3: [
    { title: 'Kept from V10.1', show: ALL, focus: ['enc', 'enc_twostream', 'pool', 'core', 'visit2', 'dec'],
      text: 'The same graph: a two-stream encoder, a pooler of 8 tokens into 2 slots, a core visited twice with shared weights, and a decoder that reads the core\'s forecast. The concept path is still trained by predicting a pooled summary of the next group.', status: 'measured' },
    { title: 'Changed: scale, codebook, exploration, sleep', show: ALL, focus: ['forecast_to_dec', 'codes_selector', 'lora_gate', 'rbf'],
      text: 'A full learned forecast map keeps the target\'s scale; the codebook is exempt from decay; exploration becomes a mixture likelihood over K = 8 codes with the selector as prior; a gated LoRA sleep layer consolidates the RBF correction bank.', status: 'measured' },
    { title: 'What broke, with the number', show: ALL, focus: ['forecast_to_dec', 'codes_selector', 'visit2'],
      text: 'The forecast predicts the next group worse than the core\'s own input (math FVU 0.694 vs 0.637). Mixture, uniform and committed likelihoods agree within 0.002 nats/token, so the code choice barely matters. Visit 2 is inert. The concept arm loses to the concept-free C0 on text and math in both seeds.', status: 'failed' },
    { title: 'What held: fast memory and sleep', show: ALL, focus: ['rbf', 'lora_gate'],
      text: 'With weights frozen, the RBF bank removes 1.55 nats on an unseen math module; gated LoRA consolidation keeps 90% of the gain with zero forgetting on dissimilar skills. Related add/sub skills are not protected. These carry into V12 unchanged.', status: 'measured' },
    { title: 'First-principles reading', show: ALL, focus: ['codes_selector', 'ncp_head'],
      text: 'Exploration needs a latent to explore: a persistent, multimodal choice that cannot be copied from a recent token. Text and math at this scale have none, so codes had nothing to choose between. V12 adds pinpad, which has one, and keeps text and math as the cost check.', status: 'measured' },
  ],
  v11: [
    { title: 'Kept V10.3\'s deployed graph', show: ALL, focus: ['enc', 'pool', 'core', 'dec'],
      text: 'Same encoder, pooler, two-visit core and decoder; RBF and gated LoRA unchanged.', status: 'measured' },
    { title: 'Changed: a latent concept objective', show: ALL, focus: ['ncp_head', 'visit2', 'codes_selector'],
      text: 'The concept state predicts the next normalised, stopped pooled encoder state (Gaussian NLL, a mixture over K visit-2 branches). The strategy code moves to visit 2, chosen by a prior on the code-free visit-1 belief.', status: 'failed' },
    { title: 'Added: the concept-anchored bridge', show: ALL, focus: ['bridge'],
      text: 'A training-only rollout of 8 decoder states anchored on the visit-1 concept state, with no re-grounding. It is neither NextLat (which anchors on the network\'s own previous state, one step) nor HiLP (an upper predictor without token input).', status: 'failed' },
    { title: 'What broke, with the number', show: ALL, focus: ['bridge', 'core'],
      text: 'The bridge pushed 2.6–2.9× NTP\'s gradient into the core and cost +0.41 nats on math. The latent objective made the concept route live but added nothing on NLL. Every pre-registered gate failed. No metacontroller was tested.', status: 'failed' },
    { title: 'First-principles reading → V12', show: ALL, focus: ['ncp_head', 'bridge'],
      text: 'Regressing a multimodal target returns its mean, so NCP becomes classification over discrete codes. Auxiliary losses must not dominate the trunk, so NCP is stopped from reaching D and NextLat runs on D\'s own state. Control belongs on a frozen base, so the metacontroller is trained afterwards.', status: 'measured' },
  ],
};
