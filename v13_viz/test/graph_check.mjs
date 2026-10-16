// Checks the V13 graph against v13_NOTES.md: the §2.2 invariant, §9.3's reach table, §14.3/§14.4's barriers and the
// NCP route into L through every kind of G-writer; and the layout (no slab overlaps, finite positions).
//   node v13_viz/test/graph_check.mjs
import { readFileSync } from 'node:fs';
import { buildGraph, forwardWaves, backward, junctionCounts, layout, jointPos, boxes, overlaps } from '../src/logic.js';

const data = JSON.parse(readFileSync(new URL('../data/tiny_v2.json', import.meta.url)));
const cfg = { ...data.cfg, kinds: data.kinds };
const g = buildGraph(cfg);
let fails = 0;
const ok = (cond, msg) => { console.log((cond ? 'PASS ' : 'FAIL ') + msg); if (!cond) fails++; };
const subs = [...g.nodes.values()].filter(n => n.kind === 'sub');
console.log(`graph: ${g.nodes.size} nodes (${subs.length} sublayers), ${g.edges.length} edges`);

// §2.2 invariant: nothing reaches an L joint from G, W or anything downstream of them (forward edges)
const out = new Map([...g.nodes.keys()].map(k => [k, []]));
for (const e of g.edges) out.get(e.from).push(e.to);
const reach = src => { const s = new Set(src), q = [...src]; while (q.length) for (const t of out.get(q.pop())) if (!s.has(t)) { s.add(t); q.push(t); } return s; };
const gSide = [...g.nodes.values()].filter(n => n.kind === 'joint' && ['G', 'W', 'GE'].includes(n.lane)).map(n => n.id);
const fromG = reach(gSide);
const lJoints = [...g.nodes.values()].filter(n => n.kind === 'joint' && (n.lane === 'L' || n.lane === 'LE')).map(n => n.id);
ok(lJoints.every(j => !fromG.has(j)), `§2.2: no L joint (or L^enc) is reachable from G, W or G^enc (${lJoints.length} L joints)`);
ok(fromG.has('d.0.read') && fromG.has('lntp'), 'the G side does reach the decoder and L_NTP (through the concept path, checked next)');
const gNoU = new Set(); { const s = new Set(gSide), q = [...gSide]; while (q.length) { const k = q.pop(); for (const e of g.edges) if (e.from === k && e.kind !== 'concept' && !s.has(e.to)) { s.add(e.to); q.push(e.to); } } s.forEach(v => gNoU.add(v)); }
ok(![...gNoU].some(id => id.startsWith('d.') || id === 'head' || id === 'lntp'), 'without the concept-read edges, nothing on the G side reaches the decoder or L_NTP (the central invariant)');

// forward order exists (acyclic) and loop 2 runs after loop 1
const fw = forwardWaves(g);
ok(fw.get('e2.r2.0.att') > fw.get('e2.r1.3.ffn'), 'E2 loop 2 starts after loop 1 ends');
ok(fw.get('d.0.read') > fw.get(g.u), 'the decoder concept read runs after u_g exists');

// §9.3 reach table
const stagesOf = ids => new Set(ids.map(id => g.nodes.get(id).stage));
const ntp = backward(g, ['lntp']), ncp = backward(g, ['lncp']);
const R = (bw, id) => bw.depth.has(id);
const allSubs = st => subs.filter(n => n.stage === st).map(n => n.id);
const S_nodes = subs.filter(n => n.type === 'S' || n.type === 'S.ffn').map(n => n.id), F_nodes = subs.filter(n => n.type === 'F' || n.type === 'F.ffn').map(n => n.id);
ok(['eng.L', 'eng.G', ...S_nodes, ...F_nodes].every(id => R(ntp, id)), 'NTP reaches E1 L-writes, Engram and E1 G-writes');
ok([...allSubs('E2'), 's0', 'er', ...allSubs('E3'), 'poolX', 'poolL', 'poolG', 'mixB', 'wp', ...allSubs('C')].every(id => R(ntp, id)), 'NTP reaches E2 (both loops, s0, e_r), E3, the pool and C');
ok(allSubs('D').every(id => R(ntp, id)) && R(ntp, 'head') && R(ntp, 'emb'), 'NTP reaches the decoder, the head and the embedding');
ok(!R(ntp, 'ncp') && !R(ntp, 'cb') && !R(ntp, 'tgt'), 'NTP never reaches the NCP head, the codebook or the target');
ok(['eng.L', 'eng.G', ...S_nodes, ...F_nodes, ...allSubs('E2'), 's0', 'er', ...allSubs('E3'), 'poolX', 'poolG', 'mixB', 'wp', ...allSubs('C'), 'ncp', 'emb'].every(id => R(ncp, id)), 'NCP reaches the G side, Engram, the S layers, C, the NCP head and the embedding');
ok(!allSubs('D').some(id => R(ncp, id)) && !R(ncp, 'head') && !R(ncp, 'lntp'), 'NCP never reaches the decoder or the head');
ok(!R(ncp, 'cb') && !R(ncp, 'tgt'), 'NCP never reaches the codebook or its own target');
const loopsReached = bw => [1, 2].every(r => allSubs('E2').filter(id => g.nodes.get(id).loop === r).every(id => R(bw, id)));
ok(loopsReached(ntp) && loopsReached(ncp), 'both E2 loop instances receive gradient under NTP and NCP');

// barriers (§14.3, §14.4)
const bset = bw => bw.barriers.map(b => b.id).sort();
const winB = [0, 1, 2, 3].map(c => `${g.Lenc}>c.${c}.win:tap`);
const expNTP = [...winB, `${g.Lenc}>poolL:tap`].sort(), expNCP = [...expNTP, 'cb>lncp:target'].sort();
ok(JSON.stringify(bset(ntp)) === JSON.stringify(expNTP), `NTP barriers = window reads x4 + pool L tap (got ${bset(ntp).join(', ')})`);
ok(JSON.stringify(bset(ncp)) === JSON.stringify(expNCP), `NCP barriers = those + codebook -> L_NCP (got ${bset(ncp).join(', ')})`);

// NCP's route into L through all three kinds of G-writer (§14.4)
const trav = new Set(ncp.traversed.map(e => e.id));
const lRead = (pred) => g.edges.some(e => e.kind === 'read' && e.lane === 'L' && trav.has(e.id) && pred(g.nodes.get(e.to)));
ok(lRead(n => n.type === 'S' || n.type === 'S.ffn') && g.edges.some(e => e.kind === 'write' && e.lane === 'G' && trav.has(e.id) && g.nodes.get(e.from).type === 'S'),
   'NCP reaches L through the S layers\' shared output (G write in, L read out)');
ok(lRead(n => n.type === 'F' || n.type === 'F.ffn'), 'NCP reaches L through E1\'s F layers\' reads of L');
ok(lRead(n => n.stage === 'E2') && lRead(n => n.stage === 'E3'), 'NCP reaches L through E2\'s and E3\'s reads of L');
ok(lRead(n => n.id === 'e2.exit'), 'NCP reaches L through the E2 exit gate\'s read of L');

// junctions
const jn = junctionCounts(g, ntp), jc = junctionCounts(g, ncp), jb = junctionCounts(g, backward(g, ['lntp', 'lncp']));
for (const j of jb) console.log(`  junction ${j.label}: ${j.edges} incoming gradient edges`, JSON.stringify(j.by));
const jl = jb.find(j => j.key === 'Lenc').by;
ok(jl['route 1: block-0 residual'] === 3 * cfg.n_d && jl['route 2: local keys and values'] === cfg.n_d, `L^enc collects route 1 from all ${3 * cfg.n_d} decoder sublayers and route 2 from all ${cfg.n_d} local attentions`);
ok(jn.find(j => j.key === 'emb').edges === 4, 'the embedding is a junction of 4 routes in the tiny model (L start, G start, identity tap, tied head)');
ok(jb.find(j => j.key === 'u').by['NCP head'] === 1 && jb.find(j => j.key === 'u').by['concept read'] === cfg.n_d, 'u_g collects the NCP head and every decoder layer\'s concept read');
ok(Object.keys(jb.find(j => j.key === 'e2').by).some(k => k.includes('loop 1')) && Object.keys(jb.find(j => j.key === 'e2').by).some(k => k.includes('loop 2')), 'E2\'s shared weights collect gradient from both loops');

// layout
for (const e of [0, 1]) {
  const L = layout(g, e), bs = boxes(L), hits = overlaps(bs);
  ok(hits.length === 0, `layout e=${e}: no slab overlaps (${bs.length} boxes)${hits.length ? ' ' + JSON.stringify(hits.slice(0, 6)) : ''}`);
  const bad = [...L.pos].filter(([, p]) => p && !p.every(Number.isFinite));
  const jbad = [...g.nodes.values()].filter(n => n.kind === 'joint').map(n => jointPos(g, L, n.id)).filter(p => !p.every(Number.isFinite));
  ok(bad.length === 0 && jbad.length === 0, `layout e=${e}: every position finite`);
  const drawn = subs.map(n => n.slab || n.id).filter(id => !L.pos.get(id));
  ok(drawn.length === 0, `layout e=${e}: every sublayer has a slab${drawn.length ? ' (missing ' + drawn.slice(0, 5) + ')' : ''}`);
  console.log(`  e=${e}: height ${L.yTop.toFixed(2)}, L tower x ${L.towerL.toFixed(2)}, G tower x ${L.towerG.toFixed(2)}`);
}
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
