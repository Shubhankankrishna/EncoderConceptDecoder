import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.webgpu.js';
// V13 Anatomy: pure logic (no DOM, no three.js). The residual computation graph of V13-tiny-v2 (v13_tiny/model.py),
// its forward order, the backward breadth-first search from a loss, and the 3D layout. Tested in node by
// test/graph_check.mjs. build.py strips the `export` keywords when it inlines this file into the page.

// ------------------------------------------------------------------------------------------------ graph
// Lanes are residual streams: L (branches L1, L2), G (G1, G2), W (E2's workspace), C (the concept stream), D (the
// decoder stream). A lane is a chain of joints (the stream's value between sublayers). A sublayer reads the current
// joint of each lane in its read set and writes a new joint on each lane in its write set (rail = identity path).
function buildGraph(cfg) {
  const nodes = new Map(), edges = [];
  const add = (id, o) => { if (nodes.has(id)) throw new Error('duplicate node ' + id); nodes.set(id, { id, ...o }); return id; };
  const edge = (from, to, kind, grad = true, extra = {}) => {
    if (!nodes.has(from) || !nodes.has(to)) throw new Error(`edge ${from} -> ${to}: unknown node`);
    edges.push({ id: `${from}>${to}:${kind}`, from, to, kind, grad, ...extra });   // two edges may join the same pair (block 0 and keys/values)
  };
  const cur = {}, nJ = {};
  const joint = (lane, stage) => add(`${lane}#${(nJ[lane] = (nJ[lane] || 0) + 1)}`, { kind: 'joint', lane, stage });
  // reads: lanes read through the GR read; writes: lanes written; extra reads (taps) are added by the caller
  const sub = (id, o, reads, writes) => {
    add(id, { kind: 'sub', reads, writes, ...o });
    for (const l of reads) edge(cur[l], id, 'read', true, { lane: l });
    for (const l of writes) {
      const j = joint(l, o.stage);
      if (cur[l]) edge(cur[l], j, 'rail', true, { lane: l });        // a lane that starts here has no rail in
      edge(id, j, 'write', true, { lane: l });
      cur[l] = j;
    }
    return id;
  };

  // inputs and the embedding
  add('tok', { kind: 'input', stage: 'input', label: 'token ids' });
  add('emb', { kind: 'param', stage: 'emb', label: 'embedding E', type: 'emb' });
  add('x', { kind: 'joint', lane: 'X', stage: 'emb' });
  edge('tok', 'emb', 'input', false);
  edge('emb', 'x', 'lookup', true);
  for (const l of ['L', 'G']) { const j = joint(l, 'emb'); edge('x', j, 'init', true, { lane: l }); cur[l] = j; }

  // E1: n_e1_blocks x [S, S, S, F]; Engram after the first layer; within-role mixes after every layer
  const e1kinds = cfg.kinds.e1;
  e1kinds.forEach((k, i) => {
    const isS = k === 'S', layer = `e1.${i}`, st = 'E1';
    const rd = isS ? ['L'] : ['L', 'G'], wr = isS ? ['L', 'G'] : ['G'];
    sub(`${layer}.att`, { stage: st, layer, type: isS ? 'S' : 'F', part: 'att', block: Math.floor(i / 4) }, rd, wr);
    sub(`${layer}.ffn`, { stage: st, layer, type: isS ? 'S.ffn' : 'F.ffn', part: 'ffn', block: Math.floor(i / 4) }, rd, wr);
    sub(`${layer}.mixL`, { stage: st, layer, type: 'mixL', part: 'mix' }, ['L'], ['L']);
    sub(`${layer}.mixG`, { stage: st, layer, type: 'mixG', part: 'mix' }, ['G'], ['G']);
    if (i === 0) {   // Engram: one table, a gate per branch computed from that branch only, so it splits per role
      sub('eng.L', { stage: 'Engram', layer: 'eng', type: 'engL', part: 'eng', slab: 'eng' }, ['L'], ['L']);
      sub('eng.G', { stage: 'Engram', layer: 'eng', type: 'engG', part: 'eng', slab: 'eng' }, ['G'], ['G']);
      edge('tok', 'eng.L', 'input', false); edge('tok', 'eng.G', 'input', false);
    }
  });
  const L_after_E1 = cur.L;

  // E2: [M, M, M, F] looped R times over the workspace W (starts at s0, plus a per-loop embedding e_r).
  // Every E2 sublayer reads {L, G, W} and writes W only; the loop is unrolled, the two instances share one slab.
  add('s0', { kind: 'param', stage: 'E2', label: 's₀', type: 's0' });
  add('er', { kind: 'param', stage: 'E2', label: 'e_r', type: 'er' });
  for (let r = 1; r <= cfg.R; r++) {
    const j = joint('W', 'E2');
    if (r === 1) edge('s0', j, 'param', true, { lane: 'W' });
    else edge(cur.W, j, 'rail', true, { lane: 'W', loop: true });
    edge('er', j, 'param', true, { lane: 'W' });
    cur.W = j;
    cfg.kinds.e2.forEach((k, i) => {
      const t = k === 'M2' ? 'M' : 'F2';
      sub(`e2.r${r}.${i}.att`, { stage: 'E2', layer: `e2.${i}`, type: t, part: 'att', loop: r, slab: `e2.${i}.att` }, ['L', 'G', 'W'], ['W']);
      sub(`e2.r${r}.${i}.ffn`, { stage: 'E2', layer: `e2.${i}`, type: t + '.ffn', part: 'ffn', loop: r, slab: `e2.${i}.ffn` }, ['L', 'G', 'W'], ['W']);
    });
  }
  sub('e2.exit', { stage: 'E2', layer: 'e2.exit', type: 'e2exit', part: 'exit' }, ['L', 'G', 'W'], ['G']);

  // E3: [S', S', S', F] reads {L, G}, writes G only; a mix within G after every layer
  cfg.kinds.e3.forEach((k, i) => {
    const layer = `e3.${i}`, t = k === 'S3' ? 'S3' : 'F3';
    sub(`${layer}.att`, { stage: 'E3', layer, type: t, part: 'att' }, ['L', 'G'], ['G']);
    sub(`${layer}.ffn`, { stage: 'E3', layer, type: t + '.ffn', part: 'ffn' }, ['L', 'G'], ['G']);
    sub(`${layer}.mixG`, { stage: 'E3', layer, type: 'mixG', part: 'mix' }, ['G'], ['G']);
  });

  // exit reads: one stream per role from here on
  sub('exitL', { stage: 'exit', layer: 'exit', type: 'exitL', part: 'exit' }, ['L'], ['LE']);
  sub('exitG', { stage: 'exit', layer: 'exit', type: 'exitG', part: 'exit' }, ['G'], ['GE']);
  const Lenc = cur.LE, Genc = cur.GE;
  nodes.get(Lenc).label = 'L^enc'; nodes.get(Genc).label = 'G^enc';

  // pool: three sources, per-channel softmax over each 4-token group, beta mix, W_P
  add('poolX', { kind: 'sub', stage: 'pool', layer: 'pool', type: 'poolX', part: 'pool', reads: [], writes: [] });
  add('poolL', { kind: 'sub', stage: 'pool', layer: 'pool', type: 'poolL', part: 'pool', reads: [], writes: [] });
  add('poolG', { kind: 'sub', stage: 'pool', layer: 'pool', type: 'poolG', part: 'pool', reads: [], writes: [] });
  edge('x', 'poolX', 'tap', true, { tap: 'identity' });
  edge(Lenc, 'poolL', 'tap', false, { tap: 'sg' });                 // Pool(sg[L_enc]): the pool's weights still train
  edge(Genc, 'poolG', 'tap', true, { tap: 'G' });
  add('mixB', { kind: 'sub', stage: 'pool', layer: 'pool', type: 'mixB', part: 'mix', reads: [], writes: [] });
  for (const p of ['poolX', 'poolL', 'poolG']) edge(p, 'mixB', 'pool', true);
  add('m', { kind: 'joint', lane: 'M', stage: 'pool', label: 'm_g' });
  edge('mixB', 'm', 'write', true);
  add('wp', { kind: 'sub', stage: 'pool', layer: 'pool', type: 'wp', part: 'proj', reads: [], writes: [] });
  edge('m', 'wp', 'read', true);
  const p = joint('C', 'pool'); nodes.get(p).label = 'p_g';
  edge('wp', p, 'write', true, { lane: 'C' }); cur.C = p;

  // concept module: n_c x [self-attention over concepts <= g, window read of sg L_enc, FFN]; Delta-AttnRes, p_g block 0
  for (let c = 0; c < cfg.n_c; c++) {
    for (const [part, t] of [['sa', 'c.sa'], ['win', 'c.win'], ['ffn', 'c.ffn']]) {
      const id = `c.${c}.${part}`;
      sub(id, { stage: 'C', layer: `c.${c}`, type: t, part }, ['C'], ['C']);
      if (c + part !== '0sa') edge(p, id, 'block0', true);
      if (part === 'win') edge(Lenc, id, 'tap', false, { tap: 'sg' });
    }
  }
  const u = cur.C; nodes.get(u).label = 'u_g';

  // NCP head and its target (no gradient into the target or the codebook)
  sub('ncp', { stage: 'NCP', layer: 'ncp', type: 'ncp', part: 'head' }, ['C'], ['N']);
  add('tgt', { kind: 'sub', stage: 'NCP', layer: 'ncp', type: 'tgt', part: 'target', reads: [], writes: [] });
  add('cb', { kind: 'sub', stage: 'NCP', layer: 'ncp', type: 'cb', part: 'codebook', reads: [], writes: [] });
  add('lncp', { kind: 'loss', stage: 'NCP', type: 'lncp', label: 'L_NCP' });
  edge('m', 'tgt', 'target', false);                               // sg[m_{g+1}]: the next group's concept input
  edge('tgt', 'cb', 'target', false);                              // EMA k-means update and assignment, no gradient
  edge('cb', 'lncp', 'target', false);                             // target codes
  edge(cur.N, 'lncp', 'loss', true);

  // decoder: continues L (h0 = L_enc); n_d x [local attention (keys and values from L_enc), concept read, FFN]
  const d0 = joint('D', 'D'); edge(Lenc, d0, 'rail', true, { lane: 'D', route: 1 }); cur.D = d0;
  for (let d = 0; d < cfg.n_d; d++) {
    for (const [part, t] of [['loc', 'd.loc'], ['read', 'd.read'], ['ffn', 'd.ffn']]) {
      const id = `d.${d}.${part}`;
      sub(id, { stage: 'D', layer: `d.${d}`, type: t, part }, ['D'], ['D']);
      if (d + part !== '0loc') edge(Lenc, id, 'block0', true, { route: 1 });
      if (part === 'loc') edge(Lenc, id, 'kv', true, { route: 2 });
      if (part === 'read') edge(u, id, 'concept', true);
    }
  }
  sub('head', { stage: 'head', layer: 'head', type: 'head', part: 'head' }, ['D'], ['O']);
  edge('emb', 'head', 'tied', true);                               // tied output head (tiny model)
  add('lntp', { kind: 'loss', stage: 'head', type: 'lntp', label: 'L_NTP' });
  edge(cur.O, 'lntp', 'loss', true);

  return { nodes, edges, Lenc, Genc, u, p, L_after_E1, cfg };
}

// ------------------------------------------------------------------------------------------------ traversal
function forwardWaves(g) {
  // longest-path depth from the sources: each node's wave index in the forward animation
  const indeg = new Map([...g.nodes.keys()].map(k => [k, 0])), out = new Map([...g.nodes.keys()].map(k => [k, []]));
  for (const e of g.edges) { indeg.set(e.to, indeg.get(e.to) + 1); out.get(e.from).push(e); }
  const depth = new Map(), q = [...g.nodes.keys()].filter(k => indeg.get(k) === 0);
  q.forEach(k => depth.set(k, 0));
  for (let i = 0; i < q.length; i++) {
    const k = q[i];
    for (const e of out.get(k)) {
      depth.set(e.to, Math.max(depth.get(e.to) || 0, depth.get(k) + 1));
      indeg.set(e.to, indeg.get(e.to) - 1);
      if (indeg.get(e.to) === 0) q.push(e.to);
    }
  }
  if (q.length !== g.nodes.size) throw new Error('forward graph has a cycle');
  return depth;
}

function backward(g, losses) {
  // breadth-first search from the losses over reversed edges that carry gradient. A reversed edge with grad false
  // (other than a token-id input) is a barrier: the route ends there.
  const into = new Map([...g.nodes.keys()].map(k => [k, []]));
  for (const e of g.edges) into.get(e.to).push(e);
  const depth = new Map(), barriers = [], edgeDepth = new Map(), traversed = [];
  const q = [];
  for (const l of losses) { depth.set(l, 0); q.push(l); }
  for (let i = 0; i < q.length; i++) {
    const k = q[i];
    for (const e of into.get(k)) {
      if (e.kind === 'input') continue;
      if (!e.grad) { if (!barriers.find(b => b.id === e.id)) barriers.push({ ...e, depth: depth.get(k) + 1 }); continue; }
      if (!edgeDepth.has(e.id)) { edgeDepth.set(e.id, depth.get(k) + 1); traversed.push(e); }
      if (!depth.has(e.from)) { depth.set(e.from, depth.get(k) + 1); q.push(e.from); }
    }
  }
  return { depth, barriers, edgeDepth, traversed };
}

// §14.6's junctions, named. Their incoming gradient edges are counted from backward()'s traversed edges, edges inside
// the junction excluded, and broken down by route so the panel can show where the gradient arrives from.
const JUNCTIONS = [
  { key: 'Lenc', label: 'L^enc (L after E1)', match: (n, g) => n.id === g.Lenc || n.id === g.L_after_E1 || n.id === 'exitL' },
  { key: 'Genc', label: 'G^enc', match: (n, g) => n.id === g.Genc },
  { key: 'u', label: 'u_g', match: (n, g) => n.id === g.u },
  { key: 'emb', label: 'Embedding', match: n => n.id === 'emb' || n.id === 'x' },
  { key: 'sEng', label: 'S layers and Engram (one output, two writes)', match: n => ['S', 'S.ffn', 'engL', 'engG'].includes(n.type) },
  { key: 'e2', label: "E2's shared weights (both loops)", match: n => n.stage === 'E2' && n.kind === 'sub' && !!n.loop },
  { key: 'C', label: 'Concept module', match: n => n.stage === 'C' && n.kind === 'sub' },
];

function routeName(e, g) {
  if (e.kind === 'block0' && e.route === 1 || e.kind === 'rail' && e.route === 1) return 'route 1: block-0 residual';
  if (e.kind === 'kv') return 'route 2: local keys and values';
  if (e.kind === 'concept') return 'concept read';
  if (e.kind === 'read' && e.lane === 'L') return 'route 3: a G-writer reads L';
  if (e.kind === 'write') return `${e.lane} write` + (g.nodes.get(e.from).loop ? ` (loop ${g.nodes.get(e.from).loop})` : '');
  if (e.kind === 'init') return `${e.lane} starts at x`;
  if (e.kind === 'tap') return e.tap === 'identity' ? 'identity tap' : 'pool of G';
  if (e.kind === 'tied') return 'tied output head';
  if (e.kind === 'read' && e.to === 'ncp') return 'NCP head';
  return e.kind;
}

function junctionCounts(g, bw) {
  return JUNCTIONS.map(J => {
    const inside = id => J.match(g.nodes.get(id), g);
    const by = {}; let total = 0;
    for (const e of bw.traversed) if (inside(e.from) && !inside(e.to)) { const r = routeName(e, g); by[r] = (by[r] || 0) + 1; total++; }
    return { key: J.key, label: J.label, edges: total, by };
  });
}

// ------------------------------------------------------------------------------------------------ layout
// Positions and sizes of every drawn object for an explode factor e in [0, 1] (0 assembled, 1 exploded).
// Y is depth through the model (up = later). Drawn: sublayer slabs (E2's two loop instances share one slab),
// parameter leaves, losses, joints (as points on the rails).
const lerp = (a, b, t) => a + (b - a) * t;

function layout(g, e) {
  const H = 0.12;                                        // slab thickness
  const gapSub = lerp(0.035, 0.15, e), gapLayer = lerp(0.09, 0.27, e), gapStage = lerp(0.34, 0.85, e);   // exploded: wider more than taller
  const lane = { L1: -lerp(1.15, 2.2, e), L2: -lerp(0.55, 1.0, e), G1: lerp(0.55, 1.0, e), G2: lerp(1.15, 2.2, e), W: lerp(2.05, 3.6, e) };
  const encL = lane.L1 - 0.32, encR = lane.G2 + 0.32;
  const towerL = -lerp(0.85, 2.3, e), towerG = lerp(0.85, 2.3, e);
  const pos = new Map(), size = new Map(), slabOf = n => n.slab || n.id;
  let y = 0;
  const put = (id, x, yy, z, w, h, d) => { pos.set(id, [x, yy, z]); size.set(id, [w, h, d]); };
  const step = h => { y += h; };

  // input and embedding
  put('tok', 0, y, 0, encR - encL, 0.05, 0.8); step(H + gapStage * 0.6);
  put('emb', 0, y, 0, encR - encL, H, 1.0); step(H + gapStage);
  const yEmb = pos.get('emb')[1];

  // encoder slabs in forward order
  const encLayer = (layer, parts, spanR) => {
    for (const part of parts) {
      const id = `${layer}.${part}`;
      if (part === 'mixL') { put(id, (lane.L1 + lane.L2) / 2, y, 0, lane.L2 - lane.L1 + 0.34, 0.05, 0.34); continue; }
      if (part === 'mixG') { put(id, (lane.G1 + lane.G2) / 2, y, 0, lane.G2 - lane.G1 + 0.34, 0.05, 0.34); step(0.05 + gapLayer); continue; }
      const w = (spanR || encR) - encL; put(id, encL + w / 2, y, 0, w, H, 1.0); step(H + gapSub);
    }
  };
  const nE1 = g.cfg.kinds.e1.length;
  const stageY = {};
  stageY.E1 = [y];
  for (let i = 0; i < nE1; i++) {
    encLayer(`e1.${i}`, ['att', 'ffn', 'mixL', 'mixG']);
    if (i === 0) {
      put('eng', 0, y, 0, encR - encL, H, 1.0);
      put('eng.table', 0, y, -lerp(1.45, 2.4, e), encR - encL, 0.46, 0.5);
      step(H + gapLayer);
    }
    if (i % 4 === 3 && i < nE1 - 1) step(gapStage * 0.45);           // between E1 blocks
  }
  stageY.E1.push(y); step(gapStage);
  // E2 (slabs shared by both loops) with the workspace lane W
  stageY.E2 = [y];
  put('s0', lane.W, y - 0.02, 0, 0.16, 0.16, 0.16); put('er', lane.W + 0.36, y - 0.02, 0, 0.12, 0.12, 0.12);
  step(0.2 + gapSub);
  const wR = lane.W + 0.32;
  g.cfg.kinds.e2.forEach((_, i) => encLayer(`e2.${i}`, ['att', 'ffn'], wR));
  put('e2.exit', (lane.G1 + lane.W) / 2, y, 0, lane.W - lane.G1 + 0.34, 0.06, 0.4); step(0.06 + gapLayer);
  stageY.E2.push(y); step(gapStage);
  // E3
  stageY.E3 = [y];
  g.cfg.kinds.e3.forEach((_, i) => encLayer(`e3.${i}`, ['att', 'ffn', 'mixG']));
  stageY.E3.push(y); step(gapStage * 0.6);
  // exits
  put('exitL', (lane.L1 + lane.L2) / 2, y, 0, lane.L2 - lane.L1 + 0.4, H, 0.6);
  put('exitG', (lane.G1 + lane.G2) / 2, y, 0, lane.G2 - lane.G1 + 0.4, H, 0.6);
  const yExit = y; step(H + gapStage);

  // G tower: pool, W_P, concept module, u_g, NCP head
  const tw = 1.15, td = 0.9;
  stageY.pool = [y];
  put('poolX', towerG - 0.42, y, 0, 0.36, H, 0.6); put('poolL', towerG, y, 0, 0.36, H, 0.6); put('poolG', towerG + 0.42, y, 0, 0.36, H, 0.6);
  step(H + gapSub);
  put('mixB', towerG, y, 0, tw, 0.06, td * 0.8); step(0.06 + gapSub);
  put('wp', towerG, y, 0, tw, H, td); step(H + gapStage * 0.7);
  stageY.pool.push(y);
  stageY.C = [y];
  for (let c = 0; c < g.cfg.n_c; c++) {
    for (const part of ['sa', 'win', 'ffn']) { put(`c.${c}.${part}`, towerG, y, 0, tw, H, td); step(H + gapSub); }
    step(gapLayer - gapSub);
  }
  stageY.C.push(y);
  const yU = y + 0.08; put('u', towerG, yU, 0, 0.22, 0.22, 0.22); step(0.3 + gapSub);
  put('ncp', towerG + lerp(0.15, 0.35, e), y, 0, tw * 0.8, H, td * 0.8); step(H + gapSub);
  put('lncp', towerG + lerp(0.15, 0.35, e), y + 0.12, 0, 0.3, 0.3, 0.3);
  const xCb = towerG + lerp(1.45, 2.35, e);
  put('cb', xCb, pos.get('ncp')[1] - 0.05, 0, 0.9, 0.5, 0.12);
  put('tgt', xCb, pos.get('ncp')[1] - 0.62, 0, 0.36, 0.1, 0.36);
  const yTopG = y + 0.5;

  // L tower: the L_enc rail rises to the decoder, which starts above u_g (it reads the released concepts)
  let yd = Math.max(yU + gapStage, pos.get('wp')[1] + 0.3);
  stageY.D = [yd];
  const kvX = towerL - lerp(0.78, 1.15, e);
  for (let d = 0; d < g.cfg.n_d; d++) {
    for (const part of ['loc', 'read', 'ffn']) { put(`d.${d}.${part}`, towerL, yd, 0, tw, H, td); yd += H + gapSub; }
    yd += gapLayer - gapSub;
  }
  stageY.D.push(yd);
  put('kvcol', kvX, (stageY.D[0] + yd) / 2 - 0.05, 0, 0.2, yd - stageY.D[0], 0.2);   // L_enc read as keys and values
  put('head', towerL, yd + 0.05, 0, tw, H, td); yd += H + gapSub;
  put('lntp', towerL, yd + 0.17, 0, 0.3, 0.3, 0.3);
  const yTop = Math.max(yd + 0.5, yTopG);

  // joints: points on their lanes (used by rails, glow and particles)
  for (const n of g.nodes.values()) {
    if (n.kind !== 'joint' || pos.has(n.id)) continue;
    pos.set(n.id, null);                                   // filled below from neighbouring slabs
  }
  return { pos, size, lane, stageY, yEmb, yExit, yU, yTop, towerL, towerG, kvX, encL, encR, slabOf };
}

// Where a joint sits: on its lane, just above the slab that wrote it (or at the lane start).
function jointPos(g, L, id) {
  const n = g.nodes.get(id);
  const writer = g.edges.find(e => e.to === id && (e.kind === 'write' || e.kind === 'param' || e.kind === 'init' || e.kind === 'lookup'));
  const laneX = { L: (L.lane.L1 + L.lane.L2) / 2, G: (L.lane.G1 + L.lane.G2) / 2, W: L.lane.W, LE: L.towerL, GE: L.towerG,
                  C: L.towerG, D: L.towerL, N: L.towerG, O: L.towerL, X: 0, M: L.towerG }[n.lane] ?? 0;
  if (id === 'x') return [0, L.yEmb + 0.1, 0];
  if (writer && writer.from === 'x') return [laneX, L.yEmb + 0.09, 0];          // the lanes start at the embedding
  if (!writer) return n.lane === 'D' ? [L.towerL, L.stageY.D[0] - 0.12, 0] : [laneX, 0, 0];   // h0 = L_enc enters the decoder
  const src = g.nodes.get(writer.from);
  const slab = L.pos.get(src.slab || src.id) || L.pos.get(writer.from);
  if (!slab) return [laneX, 0, 0];
  const sz = L.size.get(src.slab || src.id) || [0, 0.12, 0];
  return [laneX, slab[1] + sz[1] / 2 + 0.02, 0];
}

// Axis-aligned boxes of every slab at explode factor e (for the overlap test)
function boxes(L) {
  const out = [];
  for (const [id, p] of L.pos) {
    if (!p) continue;
    const s = L.size.get(id);
    out.push({ id, min: [p[0] - s[0] / 2, p[1] - s[1] / 2, p[2] - s[2] / 2], max: [p[0] + s[0] / 2, p[1] + s[1] / 2, p[2] + s[2] / 2] });
  }
  return out;
}

function overlaps(bs, eps = 1e-6) {
  const hits = [];
  for (let i = 0; i < bs.length; i++) for (let j = i + 1; j < bs.length; j++) {
    const a = bs[i], b = bs[j];
    if ([0, 1, 2].every(k => a.min[k] < b.max[k] - eps && b.min[k] < a.max[k] - eps)) hits.push([a.id, b.id]);
  }
  return hits;
}

// V13 Anatomy: the words. Every statement is from v13_NOTES.md (section given) or v13_tiny/DEVIATIONS.md (item
// given); numbers about the model training now come from data/*.json, filled in by app.js.
const SNAP = (typeof window !== 'undefined' && window.V13_DATA && window.V13_DATA.snapshot) || { beta: [0, 0, 0], tau: [0, 0, 0, 0], step: 0 };
const b3 = v => v.map(x => x.toFixed(3)).join(', ');

const CONTENT = {
  tok: { title: 'Token ids', tag: 'ids', stage: 'Input',
    body: 'One id per token from the 2,048-piece tokenizer. Documents are packed into 1,024-token rows and each starts at a multiple of 4, so a concept group never straddles two documents. The ids also feed Engram’s n-gram hashes.',
    fw: 'token ids enter; Engram hashes their 2- and 3-grams', ref: '§7.1, §12.2' },
  emb: { title: 'Embedding E', tag: 'E', stage: 'Embedding',
    rw: 'writes all four branches: L₁ = L₂ = G₁ = G₂ = x',
    body: 'x_t = E[id_t]. Every encoder branch starts at x, so G carries token identity from the first layer. x also feeds the pool directly (the identity tap), and the same matrix is the output head (tied in the tiny model).',
    formula: 'x_t = E[id_t]\nL₁ = L₂ = G₁ = G₂ = x_t',
    dev: 'Initialised N(0, 1); the tied head divides its logits by √d (DEVIATIONS 21).',
    fw: 'x = E[id]: all four branches start at the embedding',
    bw: 'the embedding collects four routes: L’s start, G’s start, the identity tap and the tied head', ref: '§3.1, §14.3' },
  S: { title: 'S · sliding-window attention', tag: 'S', stage: 'E1',
    rw: 'reads L · writes L and G',
    body: 'Gated attention over the last 8 tokens: RoPE on a quarter of each head’s dimensions, RMSNorm on queries and keys with a learned temperature, and a sigmoid gate on the output. Its read gate and write scales are computed from the L branches only, so it may write both roles. Information crosses from L to G here and never back.',
    formula: 'x = mean_b g_b ⊙ RMSNorm(L_b)        read gate g = σ(·) from L only\no = W_O[ SDPA(q, k, v) ⊙ σ(x W_θ) ]\nR_b ← R_b + s_b · o,   s_b = 2σ(·) ∈ (0, 2)',
    evidence: 'Gated attention: measured elsewhere (Gated Attention, 15B MoE and 1.7B). Role masks on widened streams: untested in this composition.',
    dev: 'Window 8 with two E1 blocks instead of one block at 16: §8.2’s trade of more, shallower L layers inside the same reach (DEVIATIONS 15).',
    fw: 'S layers read L in an 8-token window and write both L and G',
    bw: 'an S layer’s one output feeds two writes, so gradients from L and G meet in its weights', ref: '§2.2, §3.2–§3.4' },
  'S.ffn': { title: 'SwiGLU FFN after S', tag: 'FFN', stage: 'E1', rw: 'reads L · writes L and G',
    body: 'A SwiGLU FFN (hidden 256) with the same read and write sets as the attention before it: it reads L only and writes both roles.',
    formula: 'FFN(x) = W₂( SiLU(W₁x) ⊙ W₃x )', fw: 'the FFN after S reads L and writes L and G', ref: '§2.2, §3.3' },
  F: { title: 'F · full attention, no positions', tag: 'F', stage: 'E1', rw: 'reads L and G · writes G only',
    body: `Causal attention over the whole document with no position encoding and a learned per-head temperature τ. It sees unbounded context, so it may write only G: a layer of unbounded reach that wrote L would break the guarantee. Learned τ at step ${SNAP.step.toLocaleString('en-US')}: ${SNAP.tau.slice(0, 2).map(x => x.toFixed(3)).join(' and ')} (the two E1 F layers, mean over heads; τ starts at 1).`,
    evidence: 'NoPE global layers: measured elsewhere (Kimi K3, to 1M tokens). The endless-generation rate after fine-tuning is untested.',
    fw: 'F layers read L and G over the whole prefix and write G only', ref: '§2.2, §3.2–§3.3' },
  'F.ffn': { title: 'FFN after F', tag: 'FFN', stage: 'E1', rw: 'reads L and G · writes G only',
    body: 'A SwiGLU FFN with F’s read and write sets: it reads both roles and writes G only.', fw: 'the FFN after F writes G only', ref: '§2.2' },
  mixL: { title: 'Within-role mix of L₁ and L₂', tag: 'mix', stage: 'E1', rw: 'reads L₁, L₂ · writes L₁, L₂',
    body: 'A 2 × 2 doubly stochastic matrix [[a, 1−a], [1−a, a]] from a Sinkhorn projection, predicted from the L branches only, mixes L₁ and L₂ after every E1 layer. Its spectral norm is 1, so it cannot amplify. Across roles the mix is block diagonal: mHC’s full mix would carry G’s writes into L.',
    evidence: 'mHC at 27B (no roles): measured elsewhere. With roles at n = 2: untested; “mix off” is an ablation.', fw: 'the L branches mix with each other only', ref: '§2.1, §3.4' },
  mixG: { title: 'Within-role mix of G₁ and G₂', tag: 'mix', stage: 'E1 / E3', rw: 'reads G₁, G₂ · writes G₁, G₂',
    body: 'The same 2 × 2 doubly stochastic mix for the G branches, predicted from G only. E3 has it too; E2 writes a single workspace branch, so it has nothing to mix.',
    fw: 'the G branches mix with each other only', ref: '§3.4' },
  engL: { title: 'Engram · n-gram memory', tag: 'Engram', stage: 'Engram (after E1’s first layer)', rw: 'reads token ids and each branch’s own state · writes L and G',
    body: 'Suffix 2- and 3-grams of the token ids, hashed by 8 heads per order into 16 prime-sized tables of about 8,191 rows × 8 dims, are concatenated into e_t (128 dims). Each branch has its own gate, computed from that branch’s state, then a short causal convolution (kernel 4, dilation 3). Because an L branch’s gate reads only L, its write depends on L only. Engram takes static local patterns off the backbone.',
    formula: 'α_t⁽ᵇ⁾ = σ( RMSNorm(h_t⁽ᵇ⁾)ᵀ RMSNorm(W_K⁽ᵇ⁾ e_t) / √d )\nu_t⁽ᵇ⁾ = α_t⁽ᵇ⁾ · W_V e_t   then a causal conv, then R_b += u⁽ᵇ⁾',
    evidence: 'Measured elsewhere (Engram-27B: MMLU +3.4, BBH +5.0 at iso-parameters).',
    dev: 'Exact reach R_L = 51 (bound 53): Engram follows the first S layer, so its n-gram and that layer’s window are parallel paths (DEVIATIONS 16).',
    fw: 'Engram looks up n-gram memory and gates it into each branch',
    bw: 'Engram’s one module writes L and G, so both roles’ gradients meet in it', ref: '§3.5' },
  engG: null,
  engTable: { title: 'Engram table', tag: 'table', stage: 'Engram',
    body: 'Sixteen hash tables (orders 2 and 3 × 8 heads): the largest single tensor in the model. In the design it trains with Adam at 5× the base learning rate and no weight decay, and at the 2.8T class it holds 550B parameters in host memory, prefetched while E1’s first layer runs.',
    ref: '§3.5, §11.4' },
  M: { title: 'M · Gated DeltaNet-2', tag: 'M', stage: 'E2 (looped twice)', rw: 'reads L, G and W · writes W only',
    body: 'Linear attention with a matrix state per head, a channel-wise decay and separate erase and write gates. Its state sees the whole prefix, so it writes only the workspace W. The state resets at every document start, in both loops.',
    formula: 'S_t = (I − k_t (b_t ⊙ k_t)ᵀ) Diag(α_t) S_{t−1} + k_t (w_t ⊙ v_t)ᵀ\no_t = S_tᵀ q_t',
    evidence: 'GDN-2 at 1.3B: measured elsewhere. Its erase step is not norm-bounded (spectral norm up to 2/√3); looping is untested here and was inert at tiny scale before (C4).',
    dev: 'Exact chunkwise form with the log decay bounded below at −5 (DEVIATIONS 17).',
    fw: 'Gated DeltaNet-2 sweeps the sequence and updates the workspace W',
    bw: 'gradient re-enters the shared E2 weights once per loop', ref: '§3.6' },
  'M.ffn': { title: 'FFN in E2', tag: 'FFN', stage: 'E2', rw: 'reads L, G and W · writes W only',
    body: 'A SwiGLU FFN that reads all three streams and writes the workspace. At scale E2’s FFNs are mixture-of-experts and the router sees the loop index, the setting where looping has paid in published work.', fw: 'the E2 FFN writes the workspace', ref: '§3.6, §11.4' },
  F2: { title: 'F · full attention inside E2', tag: 'F', stage: 'E2', rw: 'reads L, G and W · writes W only',
    body: 'The fourth E2 layer: full attention with no positions and a temperature, writing the workspace only. At scale it becomes MLA, because a quadratic layer inside a loop is paid once per loop.',
    fw: 'E2’s full-attention layer writes the workspace', ref: '§3.6' },
  'F2.ffn': { title: 'FFN after E2’s F', tag: 'FFN', stage: 'E2', rw: 'reads L, G and W · writes W only', body: 'A SwiGLU FFN that writes the workspace.', ref: '§3.6' },
  s0: { title: 's₀ · the workspace start', tag: 's₀', stage: 'E2',
    body: 'A learned vector at full scale (‖s₀‖ ≈ √d, no weight decay). It carries no information about the input, so everything in W that depends on the text was computed by E2. Constraint C5: a recurrent workspace must start at full scale with read-only inputs.',
    bw: 's₀ receives gradient through both loops', ref: '§1 (C5), §3.6' },
  er: { title: 'e_r · loop embeddings', tag: 'e_r', stage: 'E2',
    body: 'One learned vector per loop, added to W at the start of each pass, so the shared weights know which pass they are in: W⁽ʳ⁾ = Block(W⁽ʳ⁻¹⁾ + e_r ; L, G).', ref: '§3.6' },
  e2exit: { title: 'E2 exit: the workspace enters G', tag: 'exit', stage: 'E2', rw: 'reads L, G and W · writes G',
    body: 'After the second loop each G branch adds a scaled copy of the workspace, with the scale predicted from L, G and W. L may inform a write into G; the reverse is forbidden. W is then discarded.',
    formula: 'G_b ← G_b + s_b · W⁽²⁾,   s = 2σ(·) from {L, G, W}', fw: 'the workspace is written into G; W is discarded', ref: '§3.6' },
  S3: { title: 'S′ · sliding-window attention in E3', tag: 'S′', stage: 'E3', rw: 'reads L and G · writes G only',
    body: 'A 16-token window over both roles that writes G only. E3 folds the workspace into G. It never writes L, because every local layer that writes L adds its window to the decoder’s reach.',
    dev: 'Window 16 in E3 (DEVIATIONS 15).', fw: 'E3 refines G from L and G; L is untouched', ref: '§3.7, §2.4' },
  'S3.ffn': { title: 'FFN after S′', tag: 'FFN', stage: 'E3', rw: 'reads L and G · writes G only', body: 'A SwiGLU FFN that writes G only.', ref: '§3.7' },
  F3: { title: 'F · full attention in E3', tag: 'F', stage: 'E3', rw: 'reads L and G · writes G only',
    body: `Full attention, no positions, writing G only. Learned τ at step ${SNAP.step.toLocaleString('en-US')}: ${(SNAP.tau[3] || 0).toFixed(3)}.`, ref: '§3.7' },
  'F3.ffn': { title: 'FFN after E3’s F', tag: 'FFN', stage: 'E3', rw: 'reads L and G · writes G only', body: 'A SwiGLU FFN that writes G only.', ref: '§3.7' },
  exitL: { title: 'Exit read: L → L^enc', tag: 'L^enc', stage: 'Encoder exit', rw: 'reads L₁, L₂',
    body: 'One more gated read collapses the two L branches into one stream. L is final after E1: E2 and E3 read it but never write it, so L^enc’s reach is E1’s, 51 tokens exactly (tests_g0.py). L^enc feeds the decoder three ways: its starting state, the keys and values of every local attention, and (behind a stop-gradient) the pool and the concept window reads.',
    bw: 'L^enc is the widest junction: every decoder sublayer reads it, and every G-writer read L', ref: '§3.8, §14.6' },
  exitG: { title: 'Exit read: G → G^enc', tag: 'G^enc', stage: 'Encoder exit', rw: 'reads G₁, G₂',
    body: 'Collapses the two G branches into G^enc, the only stream with unbounded reach. It feeds nothing but the pool.', ref: '§3.8' },
  poolX: { title: 'Pool of x · identity tap', tag: 'x', stage: 'Pool',
    body: 'A per-channel softmax over the 4 tokens of each group, initialised to the exact mean, applied to the embedding. Token identity stays linearly available in the concept, which a long-range copy through the concept path needs.',
    formula: 'a = softmax over the 4 tokens of H_i W_Z + B,  Pool(H)_g = Σ_i a_i ⊙ (H_i W_C)', ref: '§4.1–§4.2' },
  poolL: { title: 'Pool of sg[L^enc] · local tap', tag: 'L', stage: 'Pool',
    body: 'The same pool over L^enc, behind a stop-gradient: the concept path reads L, but NCP’s gradient does not pull on L through this tap. The pool’s own weights still train.',
    dev: 'Pool(sg[L_enc]) rather than sg[Pool(L_enc)], so the pool’s weights train (DEVIATIONS 18).', ref: '§4.2' },
  poolG: { title: 'Pool of G^enc · global source', tag: 'G', stage: 'Pool',
    body: 'The main source: the only stream with unbounded reach, carrying E2’s workspace.', ref: '§4.2' },
  mixB: { title: 'β mix → m_g', tag: 'β', stage: 'Pool',
    body: `Each pooled source is RMS-normalised, then mixed with β = softmax(b), initialised to (0.107, 0.107, 0.787). Learned β at step ${SNAP.step.toLocaleString('en-US')}: (${b3(SNAP.beta)}) for (x, L, G). m_g is also what the codebook quantises: the NCP target for group g is m_{g+1}.`,
    formula: 'm_g = β₁·RMSNorm(Pool(x)) + β₂·RMSNorm(Pool(sg L^enc)) + β₃·RMSNorm(Pool(G^enc))', fw: 'every 4 tokens become one concept input m_g', ref: '§4.2' },
  wp: { title: 'W_P → p_g', tag: 'W_P', stage: 'Pool',
    body: 'p_g = W_P m_g, the concept input of group g. Group g is complete after token 4g+3, the row at which it is released to the decoder.', ref: '§4.2–§4.3' },
  'c.sa': { title: 'Concept self-attention', tag: 'SA', stage: 'Concept module', rw: 'reads concepts ≤ g of the same document',
    body: 'Gated causal self-attention over the concepts up to g, with RoPE on the group index and normalised queries and keys. Every sublayer of the concept module reads a softmax over the earlier sublayers’ changes, with p_g as source 0 (Delta-AttnRes). No loop: every measured second pass over a concept core was inert (C4).',
    dev: 'No key-side gain on this attention (DEVIATIONS 19).', fw: 'concepts attend to earlier concepts', ref: '§5.1' },
  'c.win': { title: 'Window read of sg[L^enc]', tag: 'Win', stage: 'Concept module', rw: 'reads L^enc at tokens 4g−12 … 4g+3 (stop-gradient)',
    body: 'Concept g’s query attends to the 16 tokens ending at its release row: its own group and the three before it. Pooling averages four tokens; this read can pick one. It replaces a router over depths, which was inert here, with a content-based read over positions.',
    evidence: 'Untested; switching it off is a pre-registered ablation.', fw: 'each concept reads a 16-token window of L^enc', ref: '§5.2' },
  'c.ffn': { title: 'Concept FFN', tag: 'FFN', stage: 'Concept module', body: 'A SwiGLU FFN at concept rate: it runs once per four tokens.', ref: '§5.1' },
  u: { title: 'u_g · the concept state', tag: 'u_g', stage: 'Concept module',
    body: 'The concept module’s output, one vector per group, at a quarter of the token rate. Three readers normalise their own copy: the decoder’s concept read, the NCP head, and (with frozen weights only) the continual-learning correction bank.',
    bw: 'u_g collects gradient from every decoder layer’s concept read and from the NCP head', ref: '§5, §7.1' },
  ncp: { title: 'NCP head: HiLP’s upper level', tag: 'NCP', stage: 'Concept head', rw: 'reads u_g',
    body: 'RMSNorm, a GELU MLP and 4 × 32 logits: one categorical per product-quantised segment, predicting the codes of the next group’s input m_{g+1}, before that group’s tokens exist. This is HiLP’s upper dynamics model with a cross-entropy over discrete codes instead of HiLP’s SmoothL1 regression, which would predict the conditional mean (C2).',
    formula: 'L_NCP = mean_g Σ_s CE( logits_s(u_g), code_s(sg[m_{g+1}]) )',
    evidence: 'Measured here in V12 (u beat its input by 7.7 points on pinpad, one seed). The λ ramp is untested. Interim read in this run: the ≥ 5-point probe test failed at step ≈ 2,400 (all four families); the pre-registered read is at the final checkpoint.',
    fw: 'u_g predicts the next group’s codes', bw: 'NCP’s gradient enters at the head', ref: '§6.2, §9.1' },
  tgt: { title: 'Target: sg[m_{g+1}]', tag: 'target', stage: 'Concept head',
    body: 'The next group’s concept input, normalised by running per-coordinate mean and variance (EMA 0.99) and split into 4 segments of 32 dimensions. Stop-gradient: NCP never trains its own target.', ref: '§4.2, §6.1' },
  cb: { title: 'Product codebook, 4 × 32', tag: 'codebook', stage: 'Concept head',
    body: '4 segments × 32 words name 32⁴ ≈ 1.05M concepts with 128 stored words. The words move by EMA k-means (decay 0.99), never by gradient, and a word used less than 2% of uniform is re-seeded from a current target. One codebook serves every data family.',
    evidence: 'Measured here (V12’s PQCodebook, including its dead-code fix).', ref: '§6.1' },
  lncp: { title: 'L_NCP', tag: 'L_NCP', stage: 'Loss',
    body: 'Next-concept loss, weighted λ(t), which ramps from 0 to 0.1 over the first 10% of training time. Its gradient trains the head, the concept module and the whole G side, and reaches L only through the G-writers’ reads of L. It never reaches the decoder.',
    bw: 'NCP’s gradient starts here', ref: '§6.2, §9.2, §14.4' },
  'd.loc': { title: 'Local attention, confined', tag: 'Loc', stage: 'Decoder', rw: 'queries: the decoder state · keys and values: L^enc',
    body: 'Queries come from the decoder state; keys and values are projections of the fixed L^enc in a 16-token window. No row reads another row’s decoder state, so the decoder’s reach is R_D = R_L + 15 = 66 tokens however deep it is.',
    dev: 'R_D = 66 exactly (design bound 68).', fw: 'local attention reads L^enc in a 16-token window', ref: '§8.1–§8.2' },
  'd.read': { title: 'Concept read', tag: 'Read', stage: 'Decoder', rw: 'reads u_j for j ≤ ⌊(t+1)/4⌋ − 1',
    body: 'Cross-attention from row t to the released concepts of its document. Concept g is readable from row 4g+3 on, so the newest concept a row can read forecasts the group it is writing. A null entry and a sink logit let a row read nothing. This is the decoder’s only long-range input.',
    dev: 'The null entry is scored with the unrotated query (DEVIATIONS 20); no key-side gain (DEVIATIONS 19).', fw: 'each row reads the released concepts', ref: '§7.1' },
  'd.ffn': { title: 'Decoder FFN', tag: 'FFN', stage: 'Decoder', body: 'A SwiGLU FFN. Every decoder sublayer reads a softmax over earlier sublayer changes with L^enc as source 0 (Delta-AttnRes).', ref: '§8.1' },
  head: { title: 'Final norm and tied output head', tag: 'head', stage: 'Output',
    body: 'RMSNorm, then 2,048 next-token logits from the embedding matrix itself: logits = h Eᵀ / √d. The 128 parameters here are the norm; the head’s 262,144 weights are the embedding’s.', fw: 'the decoder’s final state becomes next-token logits', ref: '§3.1, §8.3' },
  lntp: { title: 'L_NTP', tag: 'L_NTP', stage: 'Loss',
    body: 'Next-token cross-entropy. Its gradient reaches L by three routes: the decoder’s starting state h⁰ = L^enc (route 1), the keys and values of every local attention (route 2), and the concept read, back through every G-writer that read L (route 3).',
    bw: 'NTP’s gradient starts here', ref: '§9.3, §14.3' },
};
CONTENT.engG = CONTENT.engL;

// Stage labels (positions are functions of the layout)
const STAGES = [
  { label: 'Token ids', minE: 0.3, at: L => [L.encR + 0.3, L.pos.get('tok')[1], 0] },
  { label: 'Embedding · tied with the head', at: L => [L.encR + 0.3, L.pos.get('emb')[1], 0] },
  { label: 'E1 · 2 × [S S S F]', at: L => [L.encR + 0.3, (L.stageY.E1[0] + L.stageY.E1[1]) / 2 + 0.5, 0] },
  { label: 'Engram', at: L => [L.encR + 0.3, L.pos.get('eng')[1], 0] },
  { label: 'E2 · [M M M F] ×2 loops', at: L => [L.lane.W + 0.95, (L.stageY.E2[0] + L.stageY.E2[1]) / 2, 0] },
  { label: 'E3 · [S′ S′ S′ F]', at: L => [L.encR + 0.3, (L.stageY.E3[0] + L.stageY.E3[1]) / 2, 0] },
  { label: 'Exit reads → L^enc · G^enc', at: L => [L.encR + 0.3, L.yExit, 0] },
  { label: 'Pool → m_g → p_g', at: L => [L.towerG + 0.95, (L.stageY.pool[0] + L.stageY.pool[1]) / 2, 0] },
  { label: 'Concept module ×4', at: L => [L.towerG + 0.95, (L.stageY.C[0] + L.stageY.C[1]) / 2, 0] },
  { label: 'u_g', minE: 0.3, at: L => [L.towerG + 0.3, L.yU, 0] },
  { label: 'NCP head · HiLP’s upper level', minE: 0.3, at: L => [L.pos.get('ncp')[0] + 0.7, L.pos.get('ncp')[1] + 0.25, 0] },
  { label: 'Codebook 4 × 32 · EMA', minE: 0.3, at: L => [L.pos.get('cb')[0], L.pos.get('cb')[1] + 0.45, 0] },
  { label: 'L_NCP', minE: 0.3, at: L => [L.pos.get('lncp')[0] + 0.3, L.pos.get('lncp')[1], 0] },
  { label: 'Decoder · 6 layers, confined', at: L => [L.kvX - 0.25, (L.stageY.D[0] + L.stageY.D[1]) / 2, 0], cls: 'left' },
  { label: 'L^enc as keys and values', minE: 0.3, at: L => [L.kvX - 0.2, L.stageY.D[0] - 0.25, 0], cls: 'left' },
  { label: 'L_NTP', at: L => [L.pos.get('lntp')[0] - 0.3, L.pos.get('lntp')[1], 0], cls: 'left' },
];

// V13 Anatomy: the 3D app (three.js WebGPURenderer, WebGL 2 fallback). build.py inlines logic.js, content.js and the
// data before this file and strips the module imports below, so the published page is one file.

const DATA = window.V13_DATA;
const $ = s => document.querySelector(s);
const showErr = msg => { const el = document.createElement('div'); el.id = 'err'; el.setAttribute('role', 'alert');
  el.textContent = 'The 3D view could not start: ' + msg + '. Try a recent Chrome, Edge or Safari.'; document.querySelector('main').appendChild(el); console.error('[v13-anatomy] ' + msg); };
window.addEventListener('error', ev => showErr(ev.message || String(ev.error)));
window.addEventListener('unhandledrejection', ev => showErr(String(ev.reason && ev.reason.message || ev.reason)));
$('#nparams').textContent = DATA.tiny.total.toLocaleString('en-US');
const cfg = { ...DATA.tiny.cfg, kinds: DATA.tiny.kinds };
const G = buildGraph(cfg);
const FW = forwardWaves(G);

// ------------------------------------------------------------------------------------------------ theme colours
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ROLE_VARS = { L: '--role-l', G: '--role-g', W: '--role-w', C: '--role-c', D: '--role-d', loss: '--role-loss', nograd: '--role-nograd', norm: '--role-norm', emb: '--role-emb' };
let PAL = {};
function readPalette() {
  PAL = Object.fromEntries(Object.entries(ROLE_VARS).map(([k, v]) => [k, new THREE.Color(css(v) || '#888')]));
  PAL.bg = new THREE.Color(css('--scene') || '#0b0f14'); PAL.slab = new THREE.Color(css('--slab') || '#1b2430');
  PAL.glowF = new THREE.Color(css('--glow-fw') || '#ffffff'); PAL.glowB = new THREE.Color(css('--role-loss') || '#ff4040');
}
readPalette();

// role of a node, for its tint
function roleOf(n) {
  if (!n) return 'nograd';
  if (n.kind === 'loss') return 'loss';
  const t = n.type || '';
  if (t === 'emb' || n.id === 'tok') return 'emb';
  if (t.startsWith('mix') || t === 'mixB') return 'norm';
  if (t === 'S' || t === 'S.ffn' || t === 'engL' || t === 'exitL') return 'L';
  if (['F', 'F.ffn', 'S3', 'S3.ffn', 'F3', 'F3.ffn', 'engG', 'exitG', 'e2exit'].includes(t)) return 'G';
  if (n.stage === 'E2') return 'W';
  if (['poolX', 'poolL', 'poolG', 'wp', 'ncp'].includes(t) || n.stage === 'C') return 'C';
  if (n.stage === 'D' || t === 'head') return 'D';
  if (t === 'cb' || t === 'tgt') return 'nograd';
  return 'nograd';
}

// ------------------------------------------------------------------------------------------------ renderer
const host = $('#stage');
// WebGPU first. init() can throw (no adapter) or never settle (seen in headless Chrome on macOS), so it gets a time
// limit; on failure the same renderer class runs on WebGL 2.
const withTimeout = (p, ms) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`no answer in ${ms} ms`)), ms))]);
$('#boot').textContent = 'Starting WebGPU…';
let renderer = new THREE.WebGPURenderer({ antialias: true });
try { await withTimeout(renderer.init(), 3000); }
catch (e) {
  console.warn('[v13-anatomy] WebGPU did not start (' + (e && e.message) + '); using WebGL 2');
  $('#boot').textContent = 'WebGPU did not start here; using WebGL 2…';
  renderer = new THREE.WebGPURenderer({ antialias: true, forceWebGL: true });
  await withTimeout(renderer.init(), 8000);
}
let ECO = true;
const setDPR = () => renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, ECO ? 1.25 : 2));
setDPR();
host.appendChild(renderer.domElement);
renderer.domElement.setAttribute('aria-label', 'Interactive 3D model of the V13 architecture');
renderer.domElement.tabIndex = 0;
const BACKEND = renderer.backend && renderer.backend.isWebGPUBackend ? 'WebGPU' : 'WebGL 2';
console.log('[v13-anatomy] backend=' + BACKEND);
$('#backend').textContent = BACKEND;
$('#backend').title = BACKEND === 'WebGPU' ? 'Rendering with WebGPU' : 'WebGPU is not available in this browser or frame: rendering with the WebGL 2 fallback';

const scene = new THREE.Scene();
scene.background = PAL.bg.clone();
const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 400);
scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 3.0); sun.position.set(6, 14, 9); scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 1.0); fill.position.set(-8, 4, -6); scene.add(fill);
const world = new THREE.Group(); scene.add(world);

// ------------------------------------------------------------------------------------------------ helpers
const UP = new THREE.Vector3(0, 1, 0);
const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 10, 1, false);
const unitBox = new THREE.BoxGeometry(1, 1, 1);
const unitSph = new THREE.SphereGeometry(1, 20, 14);
function mat(color, opts = {}) {
  const op = opts.opacity ?? 1;
  const m = new THREE.MeshStandardMaterial({ color, roughness: 0.62, metalness: 0.04, emissive: new THREE.Color(0x000000), transparent: op < 1, opacity: op, depthWrite: op >= 1 });
  m.userData.base = m.color.clone();
  return m;
}
function placeCyl(mesh, a, b, r) {
  const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b), d = vb.clone().sub(va), len = Math.max(d.length(), 1e-4);
  mesh.position.copy(va).addScaledVector(d, 0.5); mesh.scale.set(r, len, r);
  mesh.quaternion.setFromUnitVectors(UP, d.normalize());
}
// a dashed tube along a curve, merged into one geometry
function tubeGeom(points, r, dashed) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)), false, 'centripetal');
  if (!dashed) return new THREE.TubeGeometry(curve, 32, r, 6, false);
  const pos = [], nor = [], idx = []; const n = 14;
  for (let i = 0; i < n; i++) {
    const t0 = i / n, t1 = t0 + 0.55 / n;
    const sub = new THREE.CatmullRomCurve3([0, 0.5, 1].map(f => curve.getPointAt(t0 + (t1 - t0) * f)));
    const gg = new THREE.TubeGeometry(sub, 3, r, 6, false), base = pos.length / 3;
    pos.push(...gg.attributes.position.array); nor.push(...gg.attributes.normal.array);
    for (const k of gg.index.array) idx.push(k + base);
    gg.dispose();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3)); geo.setIndex(idx);
  return geo;
}

// ------------------------------------------------------------------------------------------------ build the scene
const meshes = new Map();       // drawn id -> mesh
const pickables = [];
const nodeMeshes = new Map();   // graph node id -> [mesh]
const edgeMeshes = new Map();   // graph edge id -> [mesh]
const curveDefs = [];           // {mesh, edgeId, pts: L => points, r, dashed}
const linkNode = (gid, m) => { if (!nodeMeshes.has(gid)) nodeMeshes.set(gid, []); nodeMeshes.get(gid).push(m); };
const linkEdge = (eid, m) => { if (!edgeMeshes.has(eid)) edgeMeshes.set(eid, []); edgeMeshes.get(eid).push(m); };

const drawnFor = new Map();     // drawn id -> graph node ids it represents
for (const n of G.nodes.values()) {
  if (n.kind === 'joint') continue;
  const did = n.slab || n.id;
  if (!drawnFor.has(did)) drawnFor.set(did, []);
  drawnFor.get(did).push(n.id);
}
drawnFor.set('u', [G.u]); drawnFor.set('eng.table', ['eng.L', 'eng.G']); drawnFor.set('kvcol', [G.Lenc]);

let L0 = layout(G, 0);
for (const [did, gids] of drawnFor) {
  if (!L0.pos.get(did)) continue;
  const n = G.nodes.get(gids[0]);
  const role = did === 'kvcol' ? 'L' : did === 'eng.table' ? 'norm' : roleOf(n);
  let m;
  if (did === 'u' || n.kind === 'loss' || did === 's0' || did === 'er') m = new THREE.Mesh(unitSph, mat(PAL[role]));
  else if (did === 'kvcol') m = new THREE.Mesh(unitCyl, mat(PAL.L, { opacity: 0.35 }));
  else if (did === 'cb') {
    m = new THREE.Group();
    const words = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat(PAL.nograd), cfg.pq_S * cfg.pq_N);
    const M4 = new THREE.Matrix4(); let k = 0;
    for (let s = 0; s < cfg.pq_S; s++) for (let w = 0; w < cfg.pq_N; w++) {
      M4.makeScale(0.8 / cfg.pq_N, 0.8 / cfg.pq_S, 0.6).setPosition((w + 0.5) / cfg.pq_N - 0.5, 0.5 - (s + 0.5) / cfg.pq_S, 0);
      words.setMatrixAt(k++, M4);
    }
    m.add(words); m.userData.inner = words;
    const frame = new THREE.Mesh(unitBox, mat(PAL.nograd, { opacity: 0.12 })); m.add(frame);
  } else if (did === 'eng.table') {
    m = new THREE.Group();
    for (let i = 0; i < 16; i++) { const s = new THREE.Mesh(unitBox, mat(PAL.norm, { opacity: 0.55 })); s.position.x = (i + 0.5) / 16 - 0.5; s.scale.set(0.8 / 16, 1, 1); m.add(s); }
  } else m = new THREE.Mesh(unitBox, mat(did.includes('mix') ? PAL.norm : PAL.slab.clone().lerp(PAL[role], 0.38)));
  m.userData = { ...m.userData, did, gids, role };
  world.add(m); meshes.set(did, m);
  if (did !== 'kvcol') pickables.push(m);
  for (const gid of gids) linkNode(gid, m);
  // role bands on top of encoder slabs: one per written lane
  if (n.kind === 'sub' && ['E1', 'E2', 'E3'].includes(n.stage) && n.part !== 'mix' && did !== 'e2.exit') {
    m.userData.bands = (n.writes || []).map(l => { const b = new THREE.Mesh(unitBox, mat(PAL[l])); world.add(b); return { lane: l, mesh: b }; });
  }
}

// rails: one cylinder per rail edge (two per edge on the two-branch lanes)
const branchX = (L, lane, b) => lane === 'L' ? (b ? L.lane.L2 : L.lane.L1) : (b ? L.lane.G2 : L.lane.G1);
// drawn as rails: residual identity paths, lane starts, parameter leaves, the pool's inputs, the losses, and the last read
// of each lane (so L runs read-only through E2 and E3 to its exit read, and C and D reach their heads)
const READ_RAIL = ['exitL', 'exitG', 'ncp', 'head'];
const railEdges = G.edges.filter(e => ['rail', 'init', 'param', 'loss', 'pool'].includes(e.kind) || (e.kind === 'read' && READ_RAIL.includes(e.to)));
for (const e of railEdges) {
  const lane = e.lane || G.nodes.get(e.to).lane;
  const two = lane === 'L' || lane === 'G';
  const color = PAL[e.kind === 'pool' ? 'C' : e.kind === 'loss' ? 'loss' : e.from === G.Lenc ? 'L' : { L: 'L', G: 'G', W: 'W', C: 'C', D: 'D', LE: 'L', GE: 'G', N: 'C', O: 'D' }[lane] || 'nograd'];
  if (e.loop) {   // E2's loop-back: an arc beside the W lane
    const mm = new THREE.Mesh(new THREE.BufferGeometry(), mat(color));
    world.add(mm); linkEdge(e.id, mm);
    curveDefs.push({ mesh: mm, r: 0.022, dashed: true, pts: L => {
      const top = jointPos(G, L, e.from), bot = jointPos(G, L, e.to), x = L.lane.W + 0.55;
      return [top, [x, top[1] + 0.1, 0], [x + 0.25, (top[1] + bot[1]) / 2, 0], [x, bot[1] - 0.1, 0], bot];
    } });
    continue;
  }
  for (const b of two ? [0, 1] : [0]) {
    const mm = new THREE.Mesh(unitCyl, mat(color));
    mm.userData = { edge: e, branch: b, lane, two };
    world.add(mm); linkEdge(e.id, mm);
  }
}

// taps and long links: curves
const curveEdge = (e, r, dashed, pts, color) => {
  const mm = new THREE.Mesh(new THREE.BufferGeometry(), mat(color));
  world.add(mm); linkEdge(e.id, mm); curveDefs.push({ mesh: mm, r, dashed, pts });
};
const top = (L, id) => { const p = L.pos.get(id), s = L.size.get(id); return [p[0], p[1] + s[1] / 2, p[2]]; };
const bot = (L, id) => { const p = L.pos.get(id), s = L.size.get(id); return [p[0], p[1] - s[1] / 2, p[2]]; };
const side = (L, id, dx) => { const p = L.pos.get(id), s = L.size.get(id); return [p[0] + dx * s[0] / 2, p[1], p[2]]; };
for (const e of G.edges) {
  if (e.kind === 'tap' && e.tap === 'identity') curveEdge(e, 0.02, false, L => {
    const a = top(L, 'emb'), b = bot(L, 'poolX'), x = L.encR + 0.25, z = -0.75;
    return [[a[0] + 0.6, a[1], z], [x, a[1] + 1.0, z], [x, b[1] - 1.0, z], [b[0], b[1] - 0.3, z * 0.5], b];
  }, PAL.emb);
  else if (e.kind === 'tap' && e.tap === 'sg') curveEdge(e, 0.016, true, L => {
    const a = [L.towerL + 0.1, 0, 0], b = side(L, e.to, -1); a[1] = b[1];
    const startY = Math.min(b[1], L.pos.get('d.0.loc') ? b[1] : b[1]);
    return [[L.towerL, startY, 0], [(L.towerL + b[0]) / 2, b[1] + 0.06, 0.18], b];
  }, PAL.nograd);
  else if (e.kind === 'tap' && e.tap === 'G') curveEdge(e, 0.03, false, L => [top(L, 'exitG'), [L.towerG + 0.42, L.pos.get('exitG')[1] + 0.4, 0], bot(L, 'poolG')], PAL.G);
  else if (e.kind === 'concept') curveEdge(e, 0.018, false, L => {
    const a = L.pos.get('u'), b = side(L, e.to, 1);
    return [a, [a[0] - 0.3, (a[1] + b[1]) / 2 + 0.25, 0.35], [b[0] + 0.25, b[1], 0.15], b];
  }, PAL.C);
  else if (e.kind === 'kv') curveEdge(e, 0.014, false, L => { const b = side(L, e.to, -1); return [[L.kvX, b[1], 0], [(L.kvX + b[0]) / 2, b[1] + 0.03, 0], b]; }, PAL.L);
  else if (e.kind === 'target') curveEdge(e, 0.014, true, L => {
    if (e.from === 'm') { const a = L.pos.get('mixB'), b = L.pos.get('tgt'); return [a, [b[0], a[1] + 0.2, 0.4], b]; }
    if (e.from === 'tgt') return [top(L, 'tgt'), bot(L, 'cb')];
    const a = side(L, 'cb', -1), b = L.pos.get('lncp'); return [a, [(a[0] + b[0]) / 2, b[1] + 0.1, 0.2], b];
  }, PAL.nograd);
  else if (e.kind === 'tied') curveEdge(e, 0.014, true, L => {
    const a = side(L, 'emb', -1), b = side(L, 'head', -1), x = Math.min(L.encL, L.kvX) - 0.6;
    return [a, [x, a[1] + 1.2, 0.3], [x, b[1] - 1.2, 0.3], b];
  }, PAL.D);
}

{ // where the keys and values come from: L^enc, beside the decoder (decoration, not a graph edge)
  const mm = new THREE.Mesh(new THREE.BufferGeometry(), mat(PAL.L));
  world.add(mm); curveDefs.push({ mesh: mm, r: 0.022, dashed: false, pts: L => {
    const y = L.stageY.D[0] - 0.12, yb = L.pos.get('kvcol')[1] - L.size.get('kvcol')[1] / 2;
    return [[L.towerL, y, 0], [(L.towerL + L.kvX) / 2, y - 0.05, 0], [L.kvX, yb, 0]]; } });
}

// ------------------------------------------------------------------------------------------------ labels
const labelHost = $('#labels');
const labels = [];
function addLabel(text, at, cls = '', minE = 0) {
  const el = document.createElement('div'); el.className = 'lbl ' + cls; el.textContent = text; labelHost.appendChild(el);
  const o = { el, at, minE }; labels.push(o); return o;
}
for (const s of STAGES) addLabel(s.label, s.at, 'stage ' + (s.cls || ''), s.minE || 0);
for (const n of G.nodes.values()) {
  if (n.kind !== 'sub' || (n.loop && n.loop !== 1)) continue;
  const did = n.slab || n.id; if (did === 'eng' && n.id !== 'eng.L') continue;
  const t = CONTENT[n.type]; if (!t) continue;
  addLabel(t.tag || t.title, L => { const p = L.pos.get(did), s = L.size.get(did); return p ? [p[0] + s[0] / 2 + 0.06, p[1], 0] : null; }, 'sub', 0.35);
}
const juncLabels = [];

// ------------------------------------------------------------------------------------------------ apply a layout
let E = 0, Ecur = layout(G, 0);
function apply(e) {
  const L = layout(G, e); Ecur = L;
  for (const [did, m] of meshes) {
    const p = L.pos.get(did), s = L.size.get(did); if (!p) continue;
    m.position.set(...p);
    if (m.isGroup) m.scale.set(...s);
    else if (m.geometry === unitSph) m.scale.setScalar(s[0] / 2);
    else if (did === 'kvcol') { m.scale.set(s[0] / 2, s[1], s[2] / 2); }
    else m.scale.set(...s);
    if (m.userData.bands) for (const b of m.userData.bands) {
      const x0 = b.lane === 'L' ? L.lane.L1 - 0.2 : b.lane === 'G' ? L.lane.G1 - 0.2 : L.lane.W - 0.2;
      const x1 = b.lane === 'L' ? L.lane.L2 + 0.2 : b.lane === 'G' ? L.lane.G2 + 0.2 : L.lane.W + 0.2;
      b.mesh.position.set((x0 + x1) / 2, p[1] + s[1] / 2 + 0.006, p[2]); b.mesh.scale.set(x1 - x0, 0.012, s[2] * 0.92);
    }
  }
  for (const e of railEdges) {
    if (e.loop) continue;
    for (const mm of edgeMeshes.get(e.id) || []) {
      const { branch, two } = mm.userData;
      let a = e.from === 'x' ? [0, L.yEmb + 0.07, 0] : jointPos(G, L, e.from);
      let b = jointPos(G, L, e.to);
      if (G.nodes.get(e.from).kind !== 'joint' && e.from !== 'x') { const p = L.pos.get(G.nodes.get(e.from).slab || e.from); if (p) a = [p[0], p[1], p[2]]; }
      if (e.kind === 'read') b = bot(L, e.to);
      if (e.kind === 'rail' && e.lane === 'D' && e.from === G.Lenc) b = [L.towerL, L.stageY.D[0] - 0.12, 0];
      if (e.kind === 'loss') { b = L.pos.get(e.to); }
      if (e.kind === 'pool') { a = L.pos.get(e.from); b = L.pos.get('mixB'); }
      if (two) { const x = branchX(L, mm.userData.lane, branch); a = [x, a[1], a[2]]; b = [x, b[1], b[2]]; if (e.kind === 'init') a = [x * 0.5, L.yEmb + 0.06, 0]; }
      if (e.from === G.Lenc && e.lane === 'D') a = top(L, 'exitL');
      placeCyl(mm, a, b, two ? 0.03 : 0.035);
    }
  }
  for (const c of curveDefs) { const old = c.mesh.geometry; c.mesh.geometry = tubeGeom(c.pts(L), c.r, c.dashed); old.dispose(); }
  needRender = true;
}

// ------------------------------------------------------------------------------------------------ camera controls
const ctl = { target: new THREE.Vector3(0, 8, 0), r: 30, theta: 0.55, phi: 1.28, vt: 0, vp: 0, vr: 0, pan: new THREE.Vector3() };
const fitR = L => Math.max(12, L.yTop * 1.62);
function frameAll() {
  const L = Ecur; ctl.target.set(0, L.yTop * 0.48, 0); ctl.r = fitR(L); ctl.theta = 0.55; ctl.phi = 1.32; needRender = true;
}
// centre the model in the space the floating panels leave free (desktop): shift the projection, not the orbit centre
function viewShift() {
  const w = host.clientWidth, h = host.clientHeight;
  if (w < 760) { camera.clearViewOffset(); return; }
  const c = $('#controls').getBoundingClientRect(), d = $('#detail').getBoundingClientRect(), m = host.getBoundingClientRect();
  const left = c.right - m.left, right = d.hidden ? w : d.left - m.left;
  camera.setViewOffset(w, h, -((left + right) / 2 - w / 2), 0, w, h);
}
function updateCamera() {
  ctl.phi = Math.min(Math.max(ctl.phi, 0.12), Math.PI - 0.12); ctl.r = Math.min(Math.max(ctl.r, 2.5), 160);
  const s = Math.sin(ctl.phi);
  camera.position.set(ctl.target.x + ctl.r * s * Math.sin(ctl.theta), ctl.target.y + ctl.r * Math.cos(ctl.phi), ctl.target.z + ctl.r * s * Math.cos(ctl.theta));
  camera.lookAt(ctl.target);
}
const ptrs = new Map(); let dragMode = null, lastPinch = 0, moved = 0;
const cv = renderer.domElement;
cv.style.touchAction = 'none';
cv.addEventListener('contextmenu', ev => ev.preventDefault());
cv.addEventListener('pointerdown', ev => { cv.setPointerCapture(ev.pointerId); ptrs.set(ev.pointerId, { x: ev.clientX, y: ev.clientY }); moved = 0;
  dragMode = ptrs.size === 2 ? 'pinch' : (ev.button === 2 || ev.shiftKey ? 'pan' : 'rot');
  if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; lastPinch = Math.hypot(a.x - b.x, a.y - b.y); } });
cv.addEventListener('pointermove', ev => {
  const p = ptrs.get(ev.pointerId);
  if (!p) { hover(ev); return; }
  const dx = ev.clientX - p.x, dy = ev.clientY - p.y; moved += Math.abs(dx) + Math.abs(dy); p.x = ev.clientX; p.y = ev.clientY;
  if (dragMode === 'pinch' && ptrs.size === 2) {
    const [a, b] = [...ptrs.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
    if (lastPinch) ctl.r *= lastPinch / Math.max(d, 1); lastPinch = d;
  } else if (dragMode === 'pan') panBy(dx, dy);
  else if (dragMode === 'rot') { ctl.vt = -dx * 0.0055; ctl.vp = -dy * 0.0055; ctl.theta += ctl.vt; ctl.phi += ctl.vp; }
  kick();
});
const endPtr = ev => { ptrs.delete(ev.pointerId); if (!ptrs.size) dragMode = null; };
cv.addEventListener('pointerup', ev => { if (moved < 5 && ptrs.size === 1) pick(ev); endPtr(ev); });
cv.addEventListener('pointercancel', endPtr);
cv.addEventListener('wheel', ev => { ev.preventDefault(); ctl.r *= Math.exp(ev.deltaY * 0.0012); kick(); }, { passive: false });
function panBy(dx, dy) {
  const k = ctl.r * 0.0016, right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
  ctl.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
}
window.addEventListener('keydown', ev => {
  if (ev.target.closest('input,select,textarea,button')) return;
  const k = { ArrowLeft: [0.08, 0], ArrowRight: [-0.08, 0], ArrowUp: [0, 0.06], ArrowDown: [0, -0.06] }[ev.key];
  if (k) { ctl.theta += k[0]; ctl.phi += k[1]; kick(); ev.preventDefault(); }
  if (ev.key === '+' || ev.key === '=') { ctl.r *= 0.9; kick(); } if (ev.key === '-') { ctl.r *= 1.1; kick(); }
});

// ------------------------------------------------------------------------------------------------ picking and the panel
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
function hit(ev) {
  const r = cv.getBoundingClientRect(); ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const h = ray.intersectObjects(pickables, true)[0];
  if (!h) return null;
  let o = h.object; while (o && !o.userData.did) o = o.parent; return o;
}
let hovered = null, selected = null, lastHover = 0;
function hover(ev) {
  const now = performance.now(); if (now - lastHover < 50) return; lastHover = now;
  const o = hit(ev); if (o === hovered) return;
  hovered = o; cv.style.cursor = o ? 'pointer' : 'grab';
  const tip = $('#tip');
  if (o) { const c = CONTENT[G.nodes.get(o.userData.gids[0])?.type] || CONTENT[o.userData.did] || {}; tip.textContent = c.title || o.userData.did; tip.hidden = false; tip.style.left = ev.clientX + 14 + 'px'; tip.style.top = ev.clientY + 12 + 'px'; }
  else tip.hidden = true;
  needRender = true;
}
function pick(ev) { const o = hit(ev); select(o ? o.userData.did : null); }
const fmtN = x => x == null ? '—' : x.toLocaleString('en-US');
function paramsOf(did) {
  const M = DATA.tiny.modules;
  const m = did.match(/^(e1|e2|e3)\.(\d+)\.(att|ffn|mixL|mixG)$/);
  if (m) { const L = M[m[1]].children[m[2]].children;
    if (m[3] === 'att') return L.att.total + L.gr1.total; if (m[3] === 'ffn') return L.ffn.total + L.gr2.total;
    return L.mix ? (m[1] === 'e1' ? L.mix.total / 2 : L.mix.total) : null; }
  const c = did.match(/^(c|d)\.(\d+)\.(\w+)$/);
  if (c) { const st = c[1] === 'c' ? M.cm : M.dec, parts = c[1] === 'c' ? ['sa', 'win', 'ffn'] : ['loc', 'read', 'ffn'];
    return st.children.subs.children[String(+c[2] * 3 + parts.indexOf(c[3]))].total; }
  const direct = { emb: M.emb.total, eng: M.engram.total, 'eng.table': M.engram.children.table.total, s0: M.s0.total, er: M.e_r.total, 'e2.exit': M.e2_exit.total,
    exitL: M.exit_L.total, exitG: M.exit_G.total, poolX: M.pool_x.total, poolL: M.pool_L.total, poolG: M.pool_G.total, mixB: M.src_norm.total + M.beta.total,
    wp: M.wp.total, ncp: M.ncp.total, head: M.final.total, cb: 0, tgt: 0, u: 0, lncp: 0, lntp: 0, tok: 0 };
  return direct[did];
}
function select(did) {
  selected = did;
  if (did) document.body.classList.add('sheet-open');
  for (const m of meshes.values()) m.userData.sel = m.userData.did === did;
  const panel = $('#detail');
  if (!did) { panel.innerHTML = $('#intro-tpl').innerHTML; needRender = true; return; }
  const m = meshes.get(did), n = G.nodes.get(m.userData.gids[0]);
  const c = CONTENT[did === 'u' ? 'u' : did === 'eng.table' ? 'engTable' : n?.type] || CONTENT[did] || { title: did, body: '' };
  const p = paramsOf(did);
  const loops = m.userData.gids.length > 1 && n?.stage === 'E2' ? '<p class="note">One slab, run twice: loop 1 and loop 2 share these weights, so their gradients add.</p>' : '';
  panel.innerHTML = `
    <p class="eyebrow">${c.stage || n?.stage || ''}</p>
    <h2>${c.title}</h2>
    ${c.rw ? `<p class="rw">${c.rw}</p>` : ''}
    <p>${c.body}</p>
    ${c.formula ? `<pre class="formula">${c.formula}</pre>` : ''}
    ${loops}
    <dl class="facts">
      <dt>Parameters</dt><dd>${p === 0 ? 'none (a buffer or a value, not a weight)' : p == null ? 'counted with its stack' : fmtN(p) + ' <span class="src">exact, PyTorch</span>'}</dd>
      ${c.evidence ? `<dt>Evidence</dt><dd>${c.evidence}</dd>` : ''}
      ${c.dev ? `<dt>In this build</dt><dd>${c.dev}</dd>` : ''}
      ${c.ref ? `<dt>Read more</dt><dd>v13_NOTES.md ${c.ref}</dd>` : ''}
    </dl>`;
  needRender = true;
}

// ------------------------------------------------------------------------------------------------ animation: waves
let anim = null, needRender = true, animating = false;
const SPEED = { slow: 0.26, normal: 0.13, fast: 0.06 };
function startWaves(kind, losses) {
  stopWaves(false);
  let nodeDepth, edgeDepth, maxD, bw = null;
  if (kind === 'fw') {
    nodeDepth = FW; edgeDepth = new Map(G.edges.map(e => [e.id, FW.get(e.from)])); maxD = Math.max(...FW.values());
  } else {
    bw = backward(G, losses); nodeDepth = bw.depth; edgeDepth = bw.edgeDepth; maxD = Math.max(...bw.depth.values(), ...bw.barriers.map(b => b.depth));
  }
  const dt = SPEED[$('#speed').value] || SPEED.normal;
  anim = { kind, losses, nodeDepth, edgeDepth, maxD, bw, dt, t0: performance.now() / 1000, done: false };
  document.body.dataset.anim = kind;
  for (const j of juncLabels) j.el.remove(); juncLabels.length = 0;
  $('#narr').hidden = false; kick();
}
function stopWaves(clear = true) {
  anim = null; delete document.body.dataset.anim;
  for (const m of [...meshes.values(), ...[...edgeMeshes.values()].flat()]) setGlow(m, 0, null, 1);
  if (clear) { $('#narr').hidden = true; for (const j of juncLabels) j.el.remove(); juncLabels.length = 0; }
  needRender = true;
}
function setGlow(m, k, color, dim) {                // dim 1 = normal; lower blends the colour toward the background
  const apply = mm => { const M = mm.material; if (!M) return; M.emissive.copy(color || PAL.glowF).multiplyScalar(k);
    M.color.copy(M.userData.base).lerp(PAL.bg, 1 - dim); };
  if (m.isGroup) m.children.forEach(apply); else apply(m);
  if (m.userData.bands) m.userData.bands.forEach(b => apply(b.mesh));
}
const pulse = x => x < 0 ? 0 : x < 0.12 ? x / 0.12 : Math.max(0.28, Math.exp(-(x - 0.12) * 2.2));
function stepWaves(t) {
  const a = anim; if (!a) return false;
  const T = (t - a.t0) / a.dt, col = a.kind === 'fw' ? PAL.glowF : PAL.glowB;
  const dimUnreached = a.kind === 'bw';
  for (const [did, m] of meshes) {
    let k = 0, reached = false;
    for (const gid of m.userData.gids) { const d = a.nodeDepth.get(gid); if (d == null) continue; reached = true; k = Math.max(k, pulse((T - d) * a.dt * 3)); }
    if (did === 'u') { const d = a.nodeDepth.get(G.u); reached = d != null; k = reached ? pulse((T - d) * a.dt * 3) : 0; }
    setGlow(m, k * (a.kind === 'fw' ? 0.85 : 1.1), col, dimUnreached && !reached ? 0.18 : 1);
  }
  for (const [eid, ms] of edgeMeshes) {
    const d = a.edgeDepth.get(eid); const barrier = a.bw && a.bw.barriers.find(b => b.id === eid);
    for (const m of ms) {
      if (barrier) { const x = (T - barrier.depth) * a.dt * 3; setGlow(m, x > 0 ? 0.6 + 0.4 * Math.sin(t * 9) : 0, PAL.nograd.clone().lerp(col, 0.5), 1); continue; }
      setGlow(m, d == null ? 0 : pulse((T - d) * a.dt * 3) * 1.2, col, dimUnreached && d == null ? 0.15 : 1);
    }
  }
  // narration: the stage of the newest wave
  const cur = Math.floor(T);
  if (cur !== a.lastD && cur <= a.maxD) {
    a.lastD = cur;
    const ids = [...a.nodeDepth].filter(([, d]) => d === cur).map(([id]) => G.nodes.get(id)).filter(n => n && n.kind !== 'joint');
    const n = ids[0];
    if (n) $('#narr-text').textContent = narrate(a, n);
    $('#narr-bar').style.width = Math.min(100, 100 * cur / a.maxD) + '%';
  }
  if (T > a.maxD + 3 && !a.done) { a.done = true; finishWaves(a); }
  return !a.done || T < a.maxD + 6;
}
function narrate(a, n) {
  const c = CONTENT[n.type] || {};
  const loop = n.loop ? ` · loop ${n.loop}` : '';
  if (a.kind === 'fw') return `${c.stage || n.stage}${loop} — ${c.fw || c.title || n.id}`;
  return `${c.stage || n.stage}${loop} — ${c.bw || 'gradient arrives at ' + (c.title || n.id)}`;
}
function finishWaves(a) {
  if (a.kind === 'fw') { $('#narr-text').textContent = 'Forward pass complete: next-token logits from the decoder, next-concept logits from the NCP head.'; return; }
  const js = junctionCounts(G, a.bw);
  const names = a.losses.map(l => l === 'lntp' ? 'L_NTP' : 'L_NCP').join(' and ');
  const unreached = [...meshes.values()].filter(m => !m.userData.gids.some(gid => a.bw.depth.has(gid)) && !['u', 'tok'].includes(m.userData.did)).map(m => m.userData.did);
  $('#narr-text').textContent = `Backward from ${names}: ${a.bw.barriers.length} barrier${a.bw.barriers.length === 1 ? '' : 's'} hit (stop-gradients and targets); ${unreached.length} parts never reached.`;
  const anchor = { Lenc: L => top(L, 'exitL'), Genc: L => top(L, 'exitG'), u: L => L.pos.get('u'), emb: L => side(L, 'emb', 1),
                   sEng: L => side(L, 'e1.0.att', 1), e2: L => side(L, 'e2.0.att', 1), C: L => side(L, 'c.0.sa', 1) };
  js.filter(j => j.edges).forEach((j, i) => {
    j.pin = i + 1;
    const lab = addLabel(String(i + 1), anchor[j.key], 'junc', 0);
    lab.el.title = Object.entries(j.by).map(([k, v]) => `${k}: ${v}`).join('\n');
    lab.el.title = `${j.label}: ${j.edges} gradient edges in`;
    juncLabels.push(lab); labels.splice(labels.indexOf(lab), 1);
  });
  renderJunctionTable(js, a);
}
function renderJunctionTable(js, a) {
  const rows = js.filter(j => j.edges).map(j => `<tr><th scope="row"><span class="pinref">${j.pin}</span> ${j.label}</th><td class="num">${j.edges}</td><td>${Object.entries(j.by).map(([k, v]) => `${k} ${v}`).join(' · ')}</td></tr>`).join('');
  const nm = id => id === G.Lenc ? 'L^enc' : id === 'poolL' ? 'pool of L' : id === 'cb' ? 'codebook' : id === 'lncp' ? 'L_NCP'
    : /^c\.(\d)\.win$/.test(id) ? `window read, concept layer ${+id[2] + 1}` : id;
  const bar = a.bw.barriers.map(b => `<li>${nm(b.from)} → ${nm(b.to)}: ${b.kind === 'target' ? 'target codes, no gradient' : 'stop-gradient'}</li>`).join('');
  $('#detail').innerHTML = `<p class="eyebrow">Backward pass</p><h2>Where the gradient went</h2>
    <p>Derived by a breadth-first search from the loss over the model's edges; an edge without gradient ends a route. Counts are edges in this graph (one per sublayer read or write), not gradient magnitudes.</p>
    <table class="jt"><thead><tr><th>Junction (§14.6)</th><th class="num">Edges in</th><th>From</th></tr></thead><tbody>${rows}</tbody></table>
    <h3>Barriers hit</h3><ul class="bar">${bar}</ul>
    <p class="note">Dimmed parts were never reached: ${a.losses.includes('lntp') && !a.losses.includes('lncp') ? 'NTP never reaches the NCP head or the codebook.' : a.losses.includes('lncp') && !a.losses.includes('lntp') ? 'NCP never reaches the decoder or the output head.' : 'with both losses only the codebook and the target stay dark.'}</p>`;
}

// ------------------------------------------------------------------------------------------------ explode tween
let tween = null;
function explodeTo(target) {
  const Lt = layout(G, target);                         // the camera follows, so the taller model stays in frame
  tween = { from: E, to: target, t0: performance.now() / 1000, dur: matchMedia('(prefers-reduced-motion: reduce)').matches ? 0.01 : 0.9,
            r0: ctl.r, r1: fitR(Lt), y0: ctl.target.y, y1: Lt.yTop * 0.48 };
  $('#explode').value = Math.round(target * 100); kick();
}
const ease = x => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;

// ------------------------------------------------------------------------------------------------ render loop (on demand)
let raf = 0, lastFrame = 0;
function kick() { needRender = true; if (!raf) raf = requestAnimationFrame(frame); }
const v3 = new THREE.Vector3();
function placeLabels() {
  const w = host.clientWidth, h = host.clientHeight;
  const all = [...labels, ...juncLabels];
  for (const o of all) {
    const p = typeof o.at === 'function' ? o.at(Ecur) : o.at;
    const show = p && E >= o.minE - 1e-6;
    if (!show) { o.el.style.display = 'none'; continue; }
    v3.set(...p).project(camera);
    if (v3.z > 1 || v3.x < -1.2 || v3.x > 1.2 || v3.y < -1.2 || v3.y > 1.2) { o.el.style.display = 'none'; continue; }
    o.el.style.display = '';
    o.el.style.transform = `translate(${((v3.x + 1) / 2 * w).toFixed(1)}px, ${((1 - v3.y) / 2 * h).toFixed(1)}px)`;
    if (o.minE > 0) o.el.style.opacity = Math.min(1, (E - o.minE) / 0.25 + 0.0) * (ctl.r < 60 ? 1 : 0.0);
  }
}
function frame(tms) {
  raf = 0;
  const t = tms / 1000;
  if (ECO && t - lastFrame < 1 / 31 && (anim || tween)) { raf = requestAnimationFrame(frame); return; }
  lastFrame = t;
  let more = false;
  if (tween) {
    const x = Math.min(1, (t - tween.t0) / tween.dur), k = ease(x); E = tween.from + (tween.to - tween.from) * k; apply(E);
    ctl.r = tween.r0 + (tween.r1 - tween.r0) * k; ctl.target.y = tween.y0 + (tween.y1 - tween.y0) * k;
    if (x >= 1) tween = null; else more = true;
  }
  if (Math.abs(ctl.vt) + Math.abs(ctl.vp) > 1e-4 && !dragMode) { ctl.theta += ctl.vt; ctl.phi += ctl.vp; ctl.vt *= 0.86; ctl.vp *= 0.86; more = true; }
  else if (!dragMode) { ctl.vt = ctl.vp = 0; }
  if (anim) more = stepWaves(t) || more;
  for (const m of meshes.values()) if (m.userData.sel || m === hovered) { const k = m.userData.sel ? 0.55 : 0.3; if (!anim) setGlow(m, k, PAL.glowF, 1); }
  updateCamera();
  renderer.render(scene, camera);
  placeLabels();
  if (!anim && !tween) for (const m of meshes.values()) if (!m.userData.sel && m !== hovered) setGlow(m, 0, null, 1);
  needRender = false;
  if (more && !document.hidden) raf = requestAnimationFrame(frame);
  if (!window.__v13first) { window.__v13first = true; console.log('[v13-anatomy] first frame'); document.body.dataset.ready = '1'; }
}
function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); viewShift(); camera.updateProjectionMatrix(); kick();
}
new ResizeObserver(resize).observe(host);
document.addEventListener('visibilitychange', () => { if (!document.hidden) kick(); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', retheme);
new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
function retheme() {
  readPalette(); scene.background.copy(PAL.bg);
  for (const m of meshes.values()) {
    const { did, role } = m.userData;
    const base = did.includes('mix') ? PAL.norm : role && PAL[role] ? (m.geometry === unitBox ? PAL.slab.clone().lerp(PAL[role], 0.38) : PAL[role]) : PAL.nograd;
    if (!m.isGroup && m.material) { m.material.color.copy(did === 'kvcol' ? PAL.L : base); m.material.userData.base.copy(m.material.color); }
    if (m.userData.bands) m.userData.bands.forEach(b => { b.mesh.material.color.copy(PAL[b.lane]); b.mesh.material.userData.base.copy(PAL[b.lane]); });
  }
  kick();
}

// ------------------------------------------------------------------------------------------------ controls wiring
$('#explode').addEventListener('input', ev => { tween = null; E = ev.target.value / 100; apply(E); kick(); });
$('#btn-explode').addEventListener('click', () => explodeTo(E < 0.5 ? 1 : 0));
$('#btn-fw').addEventListener('click', () => startWaves('fw'));
$('#btn-bw-ntp').addEventListener('click', () => startWaves('bw', ['lntp']));
$('#btn-bw-ncp').addEventListener('click', () => startWaves('bw', ['lncp']));
$('#btn-bw-both').addEventListener('click', () => startWaves('bw', ['lntp', 'lncp']));
$('#btn-stop').addEventListener('click', () => stopWaves(true));
$('#sheet-toggle').addEventListener('click', () => { document.body.classList.toggle('sheet-open'); viewShift(); kick(); });
$('#btn-reset').addEventListener('click', () => { frameAll(); select(null); kick(); });
$('#eco').addEventListener('change', ev => { ECO = ev.target.checked; setDPR(); resize(); });
document.addEventListener('click', ev => { const b = ev.target.closest('[data-goto]'); if (b) { select(b.dataset.goto); kick(); } });

apply(0); frameAll(); select(null); resize();
document.body.classList.add('live');
const HASH = (location.hash || '').slice(1).split('.');
if (HASH.includes('exploded')) { E = 1; apply(1); $('#explode').value = 100; const Lt = layout(G, 1); ctl.r = fitR(Lt); ctl.target.y = Lt.yTop * 0.48; }
if (HASH.includes('backward')) { startWaves('bw', ['lntp', 'lncp']); anim.t0 -= (anim.maxD + 4) * anim.dt; }
if (HASH.includes('forward')) startWaves('fw');
kick();
