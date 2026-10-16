/* Plain-language content: glossary, journey, figure intros, guide. */
const GLOSS = [
  ["Token", "data", "A piece of text from the tokenizer (a word, part of a word, a digit or punctuation). This model uses 8,192 different tokens; each training row is 1,024 tokens long."],
  ["Concept", "model", "One vector (384 numbers) that summarises 4 consecutive tokens. A 1,024-token row becomes 256 concepts. Concepts sit on a sphere of radius √384."],
  ["L stream (local)", "model", "The encoder's local working memory. Only sliding-window layers write to it, so it can only see 63 tokens back. It is the only encoder output the decoder reads directly (as “tok”)."],
  ["G stream (global)", "model", "The encoder's global working memory. Full-attention layers write to it, so it can see the whole document, but it only feeds the concepts, never the decoder directly."],
  ["Reach", "model", "How many tokens back can influence a prediction. Encoder L stream: 63. Decoder: 70. Anything beyond 70 tokens must reach the decoder through the concept stream."],
  ["Plan", "model", "What the decoder receives from the concept module at each position: P_t = W_u·u + W_c·ĉ_pred + offset. Position t reads the newest complete chunk, so the plan never contains the token being predicted."],
  ["Confined decoder", "model", "A decoder whose positions never read each other's plans; each position sees only its own plan plus the last 70 tokens. The alternative (“stack”) lets plans mix between neighbouring positions."],
  ["Loop depth R", "model", "How many times the concept module applies its single shared layer. Random 1–4 during training, 4 at evaluation. More passes = more thinking per concept."],
  ["Codebook (PQ vocabulary)", "model", "48 small dictionaries of 128 entries each (8 numbers per entry). A concept is described by picking one entry from each dictionary: its 48 “codes”."],
  ["Engram", "model", "A large memory of hashed 2- and 3-token combinations (50.3M numbers, looked up, not computed). A gate decides how much of the looked-up vector to use."],
  ["CRC depth-mix", "model", "Per token, a small router decides how much of the raw embedding, the local state and the global state goes into the concept."],
  ["SWA / FULL layer", "model", "Sliding-window attention looks at the last 8 tokens; full attention looks at everything earlier in the same document."],
  ["GQA", "model", "Grouped-query attention: 6 query heads share 2 key/value heads, which saves parameters and memory."],
  ["RoPE / NoPE", "model", "Rotary position encoding (on 16 of 64 dims in sliding-window layers) vs no position encoding (full layers), which use a learned temperature τ per head instead."],
  ["NTP loss", "training", "Next-token prediction: cross-entropy of the true next token, in nats. Lower is better. ln(8,192) = 9.01 would be random guessing."],
  ["NCP loss", "training", "Next-concept prediction: how well the concept module predicts the 48 codes of the next concept. Uniform guessing = ln 128 = 4.85 nats per segment. Weighted 0.1 in the total loss."],
  ["VQ loss", "training", "How far concepts are from their nearest codebook entries. It only updates the codebook, keeping the codes close to the concepts they describe."],
  ["Total loss", "training", "NTP + 0.1 × NCP + VQ. Because VQ is on a different scale, compare NTP (or validation loss), not the total, with other models."],
  ["EMA", "training", "Exponential moving average, a smoothed version of a noisy curve (here 98 % previous value + 2 % new value per step)."],
  ["Learning rate schedule", "training", "Warm up linearly for 300 steps to 6e-4, then decay along a cosine to 10 % of the peak at step 4,050. Training loss usually drops faster as the rate decays."],
  ["Gradient norm", "training", "Size of the update direction before clipping at 1.0. Stable values (≈ 0.3 here) mean healthy training; spikes or growth signal instability."],
  ["Step", "training", "One optimizer update = 32,768 tokens (2 rows × 16 accumulation × 1,024 tokens). The main run has 4,050 steps ≈ 132.7M tokens."],
  ["Validation loss", "evaluation", "NTP loss on held-out TinyStories and WikiText-103 text never seen in training, in nats per token. The most reliable measure of language ability here."],
  ["Bits per byte (bpb)", "evaluation", "Validation loss converted to bits per character byte, so it does not depend on the tokenizer. Lower is better."],
  ["Long-range probes", "evaluation", "Synthetic tests where the answer is 128–420 tokens back, beyond the decoder's reach: recall1, carry3, multi4, hop, bind."],
  ["Plan swap", "evaluation", "Evaluation trick: give the decoder another document's plan. If a probe drops to chance, the decoder was using the concept stream for it."],
  ["Chance level", "evaluation", "Accuracy of random guessing. For recall1 it is about 1/125 = 0.008."],
  ["Code entropy H", "evaluation", "How unpredictable the codes are if you only know how often each is used. NCP loss below H means the model predicts the next concept better than frequency alone."],
  ["Variance explained (VE)", "evaluation", "Fraction of the concepts' variation captured by the codebook. ≥ 0.6 was the expectation (gate G-E2)."],
  ["Codes used", "evaluation", "Share of the 128 entries per dictionary that are actually used; ≥ 95 % in every dictionary is gate G-C1. Unused entries are re-seeded every 100 steps."],
  ["Perplexity (codebook)", "evaluation", "Effective number of codes in use per dictionary (of 128). Higher = the codebook is used more evenly."],
  ["Screen", "process", "A short 265-step training run used to compare design choices before committing 24 hours."],
  ["Gate", "process", "A pass/fail criterion written down before training (PROMPT.md §6)."],
  ["Checkpoint", "process", "A saved copy of the model and optimizer, written every hour and at pauses; training resumes from it exactly."],
  ["Heartbeat", "process", "A tiny file (live.json) the trainer writes after every step so this page can show per-step values. Active after the trainer's next restart."],
  ["MPS", "process", "Apple's GPU backend for PyTorch. Training uses the M2 GPU; anything else heavy on the GPU (video, 3D) slows it down."],
];

const JOURNEY = [
  ["Tokenise", "The text is cut into tokens (8,192 kinds) and packed into rows of 1,024.", "overview"],
  ["Embed", "Each token id looks up a 384-number vector in the embedding table. The same table produces the output at the end (tied).", "overview"],
  ["Encode", "8 layers read the tokens with a local and a global stream; an n-gram memory (Engram) adds facts about 2–3 token combinations.", "enc"],
  ["Compress", "Every 4 tokens are mixed and pooled into one concept vector on a sphere.", "crc"],
  ["Think", "The concept module applies one shared attention layer R times over the concepts, then predicts the next concept's codes.", "core"],
  ["Plan", "Each position receives the plan of the newest complete chunk, never containing the token it must predict.", "plan"],
  ["Decode", "4 confined layers combine the plan with the last 70 tokens and predict the next token through the tied output head.", "dec"],
];

const FIG_INTRO = {
  overview: { b: ["Read left column bottom→top, middle column top→bottom, right column bottom→top.", "Thin arrows read a stream, thick arrows write to it.", "Only two things reach the decoder: tok (the local stream, 63 tokens of reach) and the plan (u and ĉ_pred)."],
    k: [["compute params", "25.05M"], ["decoder reach", "70 tokens"], ["concepts", "1 per 4 tokens"]] },
  enc: { b: ["Both layer types share the same attention block (middle).", "S layers (×6) read only L and write L and G; F layers (×2) read L and G but write only G.", "That rule is what keeps the local stream's reach bounded at 63 tokens."],
    k: [["S layer", "1.81M"], ["F layer", "1.86M"], ["attention block", "543K"]] },
  reach: { b: ["Each sliding-window layer adds 10 tokens of reach (window 8 + convolution 4).", "Full layers do not add reach to L, because they never write to it.", "The decoder adds 7 more, for 70 in total, whatever its depth."],
    k: [["L stream", "63"], ["decoder (confined)", "70"], ["stack control", "103"]] },
  engram: { b: ["Token pairs and triples are hashed into 8 big tables; one row is read from each.", "A gate α compares the looked-up key with the current state and admits only what agrees.", "The table is 50.3M numbers but it is a lookup, so it costs almost no compute."],
    k: [["tables", "8 × 131,071 × 48"], ["lookup params", "50.3M"], ["compute params", "0.32M"]] },
  crc: { b: ["Each token mixes three sources: raw embedding, local state, global state.", "4 consecutive mixed tokens are concatenated and projected to one 384-number concept.", "Concepts are normalised to the same length, so only their direction carries information."],
    k: [["pool", "1,536 → 384"], ["radius", "√384 = 19.6"], ["live mix", "shown at right"]] },
  core: { b: ["One layer, applied R times with the same weights, reads the concepts X and a running state S.", "The result u is the concept module's view of each chunk.", "A distance head predicts which of the 128 codes each of the 48 dictionaries will pick for the next concept."],
    k: [["loop layer", "1.86M"], ["codebook", "48 × 128 × 8"], ["NCP weight", "0.1"]] },
  dec: { b: ["Left (used): keys and values come only from tok, so a position never sees another position's plan.", "Right (control): one mixed stream, so plans leak to later positions and reach grows with depth.", "Both have the same parameter count; the screen found them within 0.011 nats."],
    k: [["confined reach", "70"], ["stack reach", "103"], ["screen gap", "−0.011"]] },
  plan: { b: ["Position t predicts token t+1.", "It reads chunk j* = ⌊(t+1)/4⌋ − 1, the newest chunk fully seen at t, so the answer is never inside the plan.", "The first 3 positions of each document get a learned “null” plan."],
    k: [["j*(t)", "⌊(t+1)/4⌋ − 1"], ["offset r(t)", "(t+1) mod 4"], ["null plan", "3 per document"]] },
  obj: { b: ["What the model is trained to minimise, with what optimiser, on what data.", "The long-range probes are the part of the data that only the concept stream can solve."],
    k: [["tokens / step", "32,768"], ["steps", "4,050"], ["tokens total", "132.7M"]] },
};

const GUIDE_READ = `
  <ol>
    <li><b>Top panel</b> of Live training refreshes every 5 s: state, current step, time left, GPU, power and memory. The box on the right
      says whether everything is healthy and lists anything amber or red; click an item to jump to its chart.</li>
    <li><b>Every chart has a verdict</b>: green Healthy, amber Watch (worth a look, not broken), red Problem. The sentence beside it says why;
      the <b>?</b> button explains what the chart measures.</li>
    <li><b>Drag across any chart</b> to compare two points: the change and % change appear under it. Click to clear.</li>
    <li><b>Loss curves</b> get a new point every 50 steps (≈ 15 min); <b>evaluation</b> charts every hour.</li>
    <li>The two most important numbers: <b>validation loss</b> (should keep falling) and <b>recall1 vs plan swap</b> (a gap = the concept stream works).</li>
  </ol>`;

const GUIDE_CONTROL = `
  <div style="font-size:13.5px;color:var(--ink2)">Run these in Terminal from <code>~/Desktop/EncoderConceptDecoder/run_m2_25M</code>.
  <div style="margin-top:8px"><b style="color:var(--ink)">Pause safely</b> (saves a checkpoint, wait for “SAFE TO CLOSE LID”)</div>
  <code class="cmd">bash code/pause.sh</code>
  <b style="color:var(--ink)">Resume</b> (plug in first)<code class="cmd">bash code/resume.sh</code>
  <b style="color:var(--ink)">Restart this dashboard</b> after a reboot
  <code class="cmd">nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py &gt; webapp/server.log 2&gt;&amp;1 &amp;</code>
  <div style="margin-top:4px"><b style="color:var(--ink)">Continuation (+24 h, D-041)</b> starts by itself after the main run's final evaluation (≈ 08:00 Mon).
  Cancel it before it starts: <code class="cmd">touch runs/NO_CONT</code>
  Do not pause between the main run's end (≈ 06:40) and “chain done” (≈ 08:00): the final evaluation cannot pause. pause.sh / resume.sh pick the active run by themselves.</div>
  Paused time does not count against the 24-hour budget. A battery guard pauses automatically at 7 % when unplugged.
  Keep the lid open and avoid heavy GPU use (video, games, the 3D “Animate flow”) for full training speed.</div>`;
