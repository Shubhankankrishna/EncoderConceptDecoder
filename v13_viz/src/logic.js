// V13 Anatomy: pure logic (no DOM, no three.js). The residual computation graph of V13-tiny-v2 (v13_tiny/model.py),
// its forward order, the backward breadth-first search from a loss, and the 3D layout. Tested in node by
// test/graph_check.mjs. build.py strips the `export` keywords when it inlines this file into the page.

// ------------------------------------------------------------------------------------------------ graph
// Lanes are residual streams: L (branches L1, L2), G (G1, G2), W (E2's workspace), C (the concept stream), D (the
// decoder stream). A lane is a chain of joints (the stream's value between sublayers). A sublayer reads the current
// joint of each lane in its read set and writes a new joint on each lane in its write set (rail = identity path).
export function buildGraph(cfg) {
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
export function forwardWaves(g) {
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

export function backward(g, losses) {
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
export const JUNCTIONS = [
  { key: 'Lenc', label: 'L^enc (L after E1)', match: (n, g) => n.id === g.Lenc || n.id === g.L_after_E1 || n.id === 'exitL' },
  { key: 'Genc', label: 'G^enc', match: (n, g) => n.id === g.Genc },
  { key: 'u', label: 'u_g', match: (n, g) => n.id === g.u },
  { key: 'emb', label: 'Embedding', match: n => n.id === 'emb' || n.id === 'x' },
  { key: 'sEng', label: 'S layers and Engram (one output, two writes)', match: n => ['S', 'S.ffn', 'engL', 'engG'].includes(n.type) },
  { key: 'e2', label: "E2's shared weights (both loops)", match: n => n.stage === 'E2' && n.kind === 'sub' && !!n.loop },
  { key: 'C', label: 'Concept module', match: n => n.stage === 'C' && n.kind === 'sub' },
];

export function routeName(e, g) {
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

export function junctionCounts(g, bw) {
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

export function layout(g, e) {
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
export function jointPos(g, L, id) {
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
export function boxes(L) {
  const out = [];
  for (const [id, p] of L.pos) {
    if (!p) continue;
    const s = L.size.get(id);
    out.push({ id, min: [p[0] - s[0] / 2, p[1] - s[1] / 2, p[2] - s[2] / 2], max: [p[0] + s[0] / 2, p[1] + s[1] / 2, p[2] + s[2] / 2] });
  }
  return out;
}

export function overlaps(bs, eps = 1e-6) {
  const hits = [];
  for (let i = 0; i < bs.length; i++) for (let j = i + 1; j < bs.length; j++) {
    const a = bs[i], b = bs[j];
    if ([0, 1, 2].every(k => a.min[k] < b.max[k] - eps && b.min[k] < a.max[k] - eps)) hits.push([a.id, b.id]);
  }
  return hits;
}
