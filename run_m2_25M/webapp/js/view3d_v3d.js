/* 3D model of the v3 DRAFT (draft_v3.md), derived from js/view3d_v2.js (same engine). Reads data/arch_v3d.json. Ids suffixed "c". */
import * as THREE from "three";
import { OrbitControls } from "../vendor/jsm/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "../vendor/jsm/geometries/RoundedBoxGeometry.js";
import { RoomEnvironment } from "../vendor/jsm/environments/RoomEnvironment.js";
import { CSS2DRenderer, CSS2DObject } from "../vendor/jsm/renderers/CSS2DRenderer.js";
import { EffectComposer } from "../vendor/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "../vendor/jsm/postprocessing/RenderPass.js";
import { OutlinePass } from "../vendor/jsm/postprocessing/OutlinePass.js";
import { OutputPass } from "../vendor/jsm/postprocessing/OutputPass.js";

let built = false, renderer, labelRenderer, composer, outline, scene, camera, controls, A, grid, floorMat;
let pending = false, camTween = null, lastT = 0;
const parts = {}, pickables = [], labelObjs = [], explodables = [], flowTubes = [];
let explode = 0, explodeTarget = 0, flowOn = false, labelMode = "auto", focus = null, selected = null, hovered = null;
let tourIdx = -1, tourTimer = null, passAnim = null, passTimer = null, passIdx = -1;

/* colour system shared with the 2D figures */
const C = {
  tok: 0xd8dce3, tokB: 0xc2c8d2, emb: 0xa996de, swa: 0x86b4ea, full: 0xf2a56f, mem: 0xf4cd5c, gate: 0xc4aaea, norm: 0xa3d69c,
  conc: 0x4fbcad, plan: 0x2e9c8c, loss: 0xef8fa1, dec: 0x9cc2ee, head: 0xa996de, L: 0x3b78c2, G: 0xe07b39, P: 0x2e9c8c, tie: 0x8f7cc0,
};
const CAT = {   // category chips for the info panel
  input: ["Input", "#a0a8b5"], stream: ["Stream", "#3b78c2"], enc: ["Encoder layer", "#6aa3e6"], full: ["Full attention", "#e8914f"],
  mem: ["Memory", "#e0b73a"], concept: ["Concepts", "#2e9c8c"], loop: ["Concept module", "#2e9c8c"], code: ["Codebook", "#e0b73a"],
  loss: ["Loss", "#e2728a"], plan: ["Plan", "#2e9c8c"], dec: ["Decoder layer", "#6aa3e6"], out: ["Output", "#8f7cc0"],
};
const GROUPS = { enc: "Encoder", core: "Concept module", dec: "Decoder" };

const fp = n => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n);
const fi = n => Number(n).toLocaleString("en-US");

/* ------------------------------------------------------------------ render loop (on demand) */
function requestRender() { if (pending) return; pending = true; requestAnimationFrame(frame); }
function frame(t) {
  window.__v3cframes = (window.__v3cframes || 0) + 1;          // frame counter (verifies on-demand rendering)
  pending = false;
  const dt = Math.min(0.05, (t - (lastT || t)) / 1000); lastT = t;
  let again = false;
  if (camTween) {
    const k = Math.min(1, (performance.now() - camTween.t0) / camTween.ms), e = 1 - Math.pow(1 - k, 3);
    camera.position.lerpVectors(camTween.p0, camTween.p1, e); controls.target.lerpVectors(camTween.q0, camTween.q1, e);
    if (k >= 1) camTween = null; else again = true;
  }
  if (controls.update()) again = true;                      // damping
  if (explode !== explodeTarget) {
    explode += (explodeTarget - explode) * 0.16;
    if (Math.abs(explodeTarget - explode) < 0.002) explode = explodeTarget; else again = true;
    explodables.forEach(o => { o.obj.position.y = o.base + o.k * explode; });
  }
  if (flowOn) { flowTubes.forEach(f => { f.tex.offset.x -= dt * f.speed; }); again = true; }
  composer.render();
  labelRenderer.render(scene, camera);
  updateLabelLOD();
  if (again) requestRender();
}

/* ------------------------------------------------------------------ builders */
function pmat(color, o = {}) {
  const m = new THREE.MeshPhysicalMaterial({ color, roughness: o.rough ?? 0.42, metalness: 0, clearcoat: o.clearcoat ?? 0.35, clearcoatRoughness: 0.35,
    transparent: (o.opacity ?? 1) < 1, opacity: o.opacity ?? 1, depthWrite: (o.opacity ?? 1) >= 1, emissive: o.emissive ?? 0x000000, emissiveIntensity: o.ei ?? 0 });
  m.userData.baseOpacity = o.opacity ?? 1;
  return m;
}
function tag(obj, group, extra = {}) { obj.userData.group = group; Object.assign(obj.userData, extra); return obj; }
function label(parent, text, o = {}) {
  const el = document.createElement("div");
  el.className = "l3 " + (o.level || "minor");
  el.innerHTML = o.sub ? `<b>${text}</b><span>${o.sub}</span>` : text;
  if (o.go) { el.dataset.go = o.go; el.classList.add("click"); }
  const obj = new CSS2DObject(el);
  obj.position.set(...(o.pos || [0, 0, 0]));
  obj.center.set(o.cx ?? 0.5, o.cy ?? 0.5);
  parent.add(obj);
  labelObjs.push({ obj, el, level: o.level || "minor", group: o.group || parent.userData.group });
  return obj;
}
function block(o) {
  const r = Math.min(o.r ?? 0.09, o.h / 2.2, o.w / 2.2, o.d / 2.2);
  const m = new THREE.Mesh(new RoundedBoxGeometry(o.w, o.h, o.d, 3, r), pmat(o.color, o));
  m.position.set(o.x, o.y, o.z || 0); m.castShadow = o.cast ?? true; m.receiveShadow = true;
  tag(m, o.g); scene.add(m);
  if (o.id) parts[o.id] = m;
  if (o.info) { m.userData.info = o.info; m.userData.id = o.id; pickables.push(m); }
  if (o.label) label(m, o.label, { level: o.level || "minor", sub: o.sub, pos: o.lpos || [0, 0, o.d / 2 + 0.02], group: o.g });
  if (o.explode != null) explodables.push({ obj: m, base: o.y, k: o.explode });
  return m;
}
function glass(o) {                                            // translucent stream wall with crisp edges
  const geo = new THREE.BoxGeometry(o.w, o.h, o.d);
  const m = new THREE.Mesh(geo, pmat(o.color, { opacity: 0.16, rough: 0.15, clearcoat: 0.6, emissive: o.color, ei: 0.25 }));
  m.position.set(o.x, o.y, o.z || 0); m.renderOrder = 2; tag(m, o.g); scene.add(m);
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: o.color, transparent: true, opacity: 0.7 }));
  e.material.userData.baseOpacity = 0.7; tag(e, o.g); m.add(e);
  if (o.id) parts[o.id] = m;
  if (o.info) { m.userData.info = o.info; m.userData.id = o.id; pickables.push(m); }
  if (o.label) label(m, o.label, { level: o.level || "major", sub: o.sub, pos: [0, o.h / 2 + 0.35, o.d / 2], group: o.g });
  return m;
}
let chevron = null;
function chevronTex() {                                        // direction marks on every flow tube (scrolls when "Animate flow" is on)
  if (chevron) return chevron;
  const c = document.createElement("canvas"); c.width = 128; c.height = 32; const x = c.getContext("2d");
  x.fillStyle = "#ffffff"; x.fillRect(0, 0, 128, 32);
  x.strokeStyle = "rgba(255,255,255,0.0)"; x.fillStyle = "#b9c0c9";
  x.beginPath(); x.moveTo(40, 0); x.lineTo(70, 16); x.lineTo(40, 32); x.lineTo(54, 32); x.lineTo(84, 16); x.lineTo(54, 0); x.closePath(); x.fill();
  chevron = new THREE.CanvasTexture(c); chevron.colorSpace = THREE.SRGBColorSpace; chevron.wrapS = THREE.RepeatWrapping; chevron.anisotropy = 4;
  return chevron;
}
function flow(points, color, o = {}) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)), false, "centripetal");
  const len = curve.getLength(), rad = o.r ?? 0.06;
  const tex = chevronTex().clone(); tex.needsUpdate = true; tex.repeat.set(Math.max(1, Math.round(len / (rad * 9))), 1);
  const mat = new THREE.MeshStandardMaterial({ color, map: tex, roughness: 0.35, metalness: 0, emissive: color, emissiveIntensity: 0.18, transparent: (o.opacity ?? 1) < 1, opacity: o.opacity ?? 1 });
  mat.userData.baseOpacity = o.opacity ?? 1;
  const m = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(24, Math.round(len * 10)), rad, 10, false), mat);
  m.castShadow = rad > 0.03; tag(m, o.g, { groups: o.groups }); scene.add(m);
  flowTubes.push({ tex, speed: o.speed ?? 0.8 });
  if (o.arrow !== false && rad > 0.03) {
    const end = curve.getPointAt(1), dir = curve.getTangentAt(1);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(rad * 2.6, rad * 6, 16), pmat(color, { ei: 0.2, emissive: color }));
    cone.position.copy(end).addScaledVector(dir, rad * 1.5); cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    cone.castShadow = true; tag(cone, o.g, { groups: o.groups }); scene.add(cone);
  }
  return m;
}
function anchor(pos, g) { const o = new THREE.Object3D(); o.position.set(...pos); tag(o, g); scene.add(o); return o; }
function zonePlate(x, w, d, color, g, title, sub, num) {
  const s = new THREE.Shape(), r = 0.8, hw = w / 2, hd = d / 2;
  s.moveTo(-hw + r, -hd); s.lineTo(hw - r, -hd); s.quadraticCurveTo(hw, -hd, hw, -hd + r); s.lineTo(hw, hd - r); s.quadraticCurveTo(hw, hd, hw - r, hd);
  s.lineTo(-hw + r, hd); s.quadraticCurveTo(-hw, hd, -hw, hd - r); s.lineTo(-hw, -hd + r); s.quadraticCurveTo(-hw, -hd, -hw + r, -hd);
  const geo = new THREE.ShapeGeometry(s, 12); geo.rotateX(-Math.PI / 2);
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.09, depthWrite: false }));
  m.material.userData.baseOpacity = 0.09; m.position.set(x, -0.58, 0); m.renderOrder = -1; tag(m, g); scene.add(m);
  const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(s.getPoints(12).map(p => new THREE.Vector3(p.x, 0, -p.y))),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 }));
  line.material.userData.baseOpacity = 0.45; line.position.set(x, -0.575, 0); tag(line, g); scene.add(line);
  label(anchor([x, -0.55, hd + 0.2], g), title, { level: "zone", sub, go: "zone:" + g, group: g, cy: 0 });
  const n = anchor([x - hw + 0.9, -0.55, hd - 0.9], g);
  label(n, String(num), { level: "znum", group: g }).element.style.background = "#" + color.toString(16).padStart(6, "0");
}

/* ------------------------------------------------------------------ scene */
function build() {
  A = window.ARCH3D;
  const stage = document.getElementById("v3c-stage"), cv = document.getElementById("v3c");
  renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: false, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.NeutralToneMapping; renderer.toneMappingExposure = 1.0;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(34, 1, 0.1, 500);
  labelRenderer = new CSS2DRenderer(); labelRenderer.domElement.className = "v3-labels"; stage.appendChild(labelRenderer.domElement);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; scene.environmentIntensity = 0.85;
  scene.add(new THREE.HemisphereLight(0xffffff, 0xd9d4ca, 0.55));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(10, 26, 16); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); Object.assign(sun.shadow.camera, { left: -26, right: 26, top: 22, bottom: -14, near: 1, far: 70 });
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02; scene.add(sun);
  floorMat = new THREE.ShadowMaterial({ opacity: 0.13 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), floorMat); floor.rotation.x = -Math.PI / 2; floor.position.y = -0.6; floor.receiveShadow = true; scene.add(floor);
  grid = new THREE.GridHelper(64, 64); grid.position.y = -0.595; grid.material.transparent = true; grid.material.opacity = 0.35; scene.add(grid);

  composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { samples: 4, type: THREE.HalfFloatType }));
  composer.addPass(new RenderPass(scene, camera));
  outline = new OutlinePass(new THREE.Vector2(1, 1), scene, camera);
  Object.assign(outline, { edgeStrength: 5, edgeGlow: 0.25, edgeThickness: 1.4, pulsePeriod: 0 });
  composer.addPass(outline); composer.addPass(new OutputPass());

  controls = new OrbitControls(camera, cv);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.09, rotateSpeed: 0.7, zoomSpeed: 0.9, minDistance: 6, maxDistance: 90, maxPolarAngle: Math.PI * 0.49, screenSpacePanning: true });
  controls.addEventListener("change", requestRender);

  buildModel();
  applyTheme();
  wireInput(stage, cv);
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(() => { applyTheme(); requestRender(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { applyTheme(); requestRender(); });
  setupBar(); setupTour(); showInfo(null);
  resize(); frameView("all", true);
  built = true;
}

function buildModel() {
  const e = A.ecfg, P = A.params, T = A.table, c3 = A.core;
  const XE = -10, XC = 0, XD = 10, W = 3.2, D = 8.4, zs = i => -3.75 + i * 0.5;
  zonePlate(XE, 12.5, 12, 0x3b78c2, "enc", "Encoder", `tokens → concepts · ${fp(T.encoder_non_embedding)} + ${fp(T.engram_table)} memory`, 1);
  zonePlate(XC, 8.2, 12, 0x2e9c8c, "core", "Concept module", `thinks in up to 3 passes · ${fp(T.concept_module)}`, 2);
  zonePlate(XD, 8.2, 12, 0xe07b39, "dec", "Decoder", `writes the next token · ${fp(T.decoder)}`, 3);

  /* ---------- tokens + embedding */
  for (let i = 0; i < 16; i++) block({ g: "enc", x: XE, y: -0.3, z: zs(i), w: 0.38, h: 0.38, d: 0.38, color: Math.floor(i / 4) % 2 ? C.tokB : C.tok, r: 0.07, id: i ? null : "tokens",
    info: i ? null : { t: "Tokens", cat: "input", fig: "v3whole", kv: [["drawn", "16 of 1,024 positions"], ["vocabulary", fi(e.vocab_size)], ["chunks", "4 tokens → 1 concept"]], d: "Token ids arrive at the bottom of the encoder. Alternating shades mark the 4-token chunks that become one concept each." } });
  label(anchor([XE + 2.2, -0.3, 3.9], "enc"), "Tokens in", { level: "major", sub: "16 of 1,024 drawn", cx: 0 });
  block({ g: "enc", id: "emb", x: XE, y: 0.5, w: W, h: 0.26, d: D, color: C.emb, label: "Token embedding", sub: "8,192 × 256 · tied",
    info: { t: "Token embedding E", cat: "input", fig: "v3whole", params: T.token_embedding, kv: [["shape", `${fi(e.vocab_size)} × ${e.d_model}`], ["params", fi(T.token_embedding) + " (not counted)"], ["use", "e0 = rms(E[x]); tied with the output head"]], d: "The same matrix E produces the logits at the top of the decoder." } });

  /* ---------- encoder streams + layers (S0, Engram read 1, S1, S2, F3, S4, Engram read 2, S5, S6, F7) */
  const rows = [["S", 0], ["E", 1], ["S", 1], ["S", 2], ["F", 3], ["S", 4], ["E", 2], ["S", 5], ["S", 6], ["F", 7]];
  const layerY = i => 1.4 + i * 0.86, streamTop = layerY(rows.length) + 0.7;
  glass({ g: "enc", id: "Lstream", x: XE - 1.95, y: (1.0 + streamTop) / 2, w: 0.12, h: streamTop - 1.0, d: D, color: C.L, label: "L", sub: "local · reach 63",
    info: { t: "L stream (local)", cat: "stream", fig: "v3enc", kv: [["reach", `${A.reach_enc} tokens`], ["written by", "S layers, both Engram reads"], ["read by", "S and F layers, CRC, decoder (tok)"]], d: "The only encoder stream the decoder may read. Its reach stays 63 because only sliding-window layers and conv-free Engram read 2 write to it." } });
  glass({ g: "enc", id: "Gstream", x: XE + 1.95, y: (1.0 + streamTop) / 2, w: 0.12, h: streamTop - 1.0, d: D, color: C.G, label: "G", sub: "global",
    info: { t: "G stream (global)", cat: "stream", fig: "v3enc", kv: [["reach", "unbounded (FULL layers)"], ["written by", "S layers, F layers, Engram"], ["read by", "F layers, CRC → concepts"]], d: "Carries document-wide information; it reaches the decoder only through the concepts and the concept module." } });
  rows.forEach((r, i) => {
    let info, color, lab, sub;
    if (r[0] === "S") {
      color = C.swa; lab = `S${r[1]}`; sub = "sliding window · 4 heads";
      info = { t: `Encoder layer ${r[1]} · S (sliding window)`, cat: "enc", fig: "v3enc", params: P.enc_layer_S, kv: [["params", fi(P.enc_layer_S)], ["reads", "{L}"], ["writes", "{L, G}"], ["attention", `SWA w = ${e.window}, partial RoPE ${e.rope_dims}/${e.head_dim}, 4 heads, GQA kv 2`], ["FFN", `SwiGLU 256 → ${e.d_ff} → 256`], ["reach added", "(w−1)+(k−1) = 10"]], d: "Gated-residual read of L → gated attention → per-stream write scales; then the same for the FFN." };
    } else if (r[0] === "F") {
      color = C.full; lab = `F${r[1]}`; sub = "full attention · 6 heads";
      info = { t: `Encoder layer ${r[1]} · F (full attention)`, cat: "full", fig: "v3enc", params: P.enc_layer_F, kv: [["params", fi(P.enc_layer_F)], ["reads", "{L, G}"], ["writes", "{G} only"], ["attention", "6 × 64 heads, GQA repeat 3, NoPE, τ_h"], ["FFN", `SwiGLU 256 → ${e.d_ff} → 256`]], d: "Global layer; 6 heads where facts enter G. Because it never writes L, the decoder's token path keeps its 63-token reach." };
    } else {
      color = C.mem; lab = `Engram read ${r[1]}`; sub = r[1] === 1 ? "with conv" : "no conv · W_v = 0";
      const p = r[1] === 1 ? P.engram_read1 : P.engram_read2;
      info = { t: `Engram read ${r[1]} (shared table)`, cat: "mem", fig: "v3engram", params: p, kv: [["compute", fi(p)], ["position", r[1] === 1 ? "after layer 0" : "after layer 4"], ["gate sees", r[1] === 1 ? "10 tokens of L" : "43 tokens of L (reads L only)"], ["output", r[1] === 1 ? "SiLU(conv₄(α v)) → L, G" : "SiLU(α v) → L, G (no conv: reach stays 63)"], ["init", r[1] === 1 ? "W_v zero-init" : "W_v zero-init ⇒ bit-identical to one read"]], d: "Both reads gather the same rows of one 80M table (the n-gram key is depth-invariant); each has its own projections, gate and write." };
    }
    block({ g: "enc", id: r[0] === "E" ? `engram${r[1]}` : `enc${r[1]}`, x: XE, y: layerY(i), w: W, h: 0.42, d: D, color, label: lab, sub, explode: i * 0.5, info });
  });
  const tY = (layerY(1) + layerY(6)) / 2, tH = layerY(6) - layerY(1) + 0.9;
  for (let t = 0; t < 8; t++) block({ g: "enc", id: t ? null : "engramTables", x: XE - 4.3 - (t % 4) * 0.36, y: tY, z: -2.2 + Math.floor(t / 4) * 4.4, w: 0.24, h: tH, d: 1.6, color: C.mem, r: 0.06, explode: 1.75,
    info: t ? null : { t: "Engram table (one, shared)", cat: "mem", fig: "v3engram", params: P.engram_table, kv: [["count", "8 (orders 2, 3 × 4 heads)"], ["rows × dim", `${fi(e.engram_rows)} × ${e.engram_dim}`], ["params", fi(P.engram_table) + " (lookup, not counted)"], ["reads", "2 (after layers 0 and 4), rows gathered once"]], d: "One row per table per token; a lookup, not compute. Four separate tables would store the same key → value map four times." } });
  const engA = anchor([XE - 4.85, tY + tH / 2 + 0.5, 0], "enc"); explodables.push({ obj: engA, base: engA.position.y, k: 1.75 });
  label(engA, "Engram memory", { level: "major", sub: "80M n-gram table · read twice" });
  flow([[XE - 3.95, layerY(1), 0], [XE - 1.7, layerY(1), 0]], C.mem, { g: "enc", r: 0.06 });
  flow([[XE - 3.95, layerY(6), 0], [XE - 1.7, layerY(6), 0]], C.mem, { g: "enc", r: 0.06 });
  const yCRC = streamTop + 0.5, yPool = yCRC + 0.75, yC = yPool + 0.95;
  block({ g: "enc", id: "crc", x: XE, y: yCRC, w: W, h: 0.32, d: D, color: C.gate, label: "CRC depth-mix", sub: "e0 · L · G", explode: 10 * 0.5,
    info: { t: "CRC depth-mix", cat: "concept", fig: "v3crc", params: P.crc_router, kv: [["params", fi(P.crc_router)], ["mix", "softmax(W rms(G)) over {e0, rms L, rms G}"], ["init", "0.107 / 0.107 / 0.787"], ["gate E5", "G share ≥ 0.5 at step 1,217"]], d: "A per-token router decides how much embedding, local and global state goes into the concept." } });
  block({ g: "enc", id: "pool", x: XE, y: yPool, w: W, h: 0.32, d: D, color: C.conc, label: "Concat-pool 4 → 1", sub: "1,024 → 256", explode: 10.3 * 0.5,
    info: { t: "Concept pooling", cat: "concept", fig: "v3crc", params: P.pool, kv: [["W_pool", "1,024 → 256"], ["params", fi(P.pool)], ["init", "mean pooling"], ["output", "ĉ = √256 · c/‖c‖ (radius 16)"]], d: "Four mixed token states are concatenated and projected, then put on the sphere: 4 : 1 compression, exactly v1's." } });
  const cz = [-3, -1, 1, 3];
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 3), pmat(C.conc, { clearcoat: 0.8, rough: 0.25, emissive: C.conc, ei: 0.12 }));
    c.position.set(XE, yC, z); c.castShadow = true; tag(c, "enc"); scene.add(c); explodables.push({ obj: c, base: yC, k: 10.6 * 0.5 });
    if (!j) { parts.concepts = c; c.userData.id = "concepts"; c.userData.info = { t: "Concepts C = ĉ", cat: "concept", fig: "v3crc", kv: [["count", "1 per 4 tokens (256 per row)"], ["dim", e.d_concept], ["norm", "√256 = 16"]], d: "Concepts live on a sphere: only direction carries information. This sequence C is the evidence the concept module re-reads." }; pickables.push(c); }
    for (let k = 0; k < 4; k++) flow([[XE, yPool + 0.2, zs(4 * j + k)], [XE, yPool + 0.5, zs(4 * j + k) * 0.6 + z * 0.4], [XE, yC - 0.4, z]], C.P, { g: "enc", r: 0.018, opacity: 0.6, arrow: false });
  });
  const cA = anchor([XE, yC + 0.85, 0], "enc"); explodables.push({ obj: cA, base: cA.position.y, k: 10.6 * 0.5 });
  label(cA, "Concepts C", { level: "major", sub: "1 per 4 tokens · 256-d" });

  /* ---------- concept module v3 (reads top → bottom) */
  const yX = yC, yIn = yX - 1.25, yLay = l => yIn - 1.2 - l * 0.95, yU = yLay(3) - 1.0, yHead = yU - 1.0, yCB = yHead - 1.15, yPred = yCB - 1.2;
  flow([[XE + 0.6, yC, 0], [-5, yC + 1.0, 0], [XC - 2.1, yX, 0]], C.P, { g: "core", groups: ["enc", "core"], r: 0.075 });
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.4, 3), pmat(C.conc, { clearcoat: 0.8, rough: 0.25, emissive: C.conc, ei: 0.12 }));
    c.position.set(XC, yX, z); c.castShadow = true; tag(c, "core"); scene.add(c);
    if (!j) { parts.X = c; c.userData.id = "X"; c.userData.info = { t: "C (evidence)", cat: "loop", fig: "v3core", kv: [["shape", "(B, 256, 256)"], ["role", "re-read by every sublayer of every pass"], ["gradient", "flows back to the encoder from both trajectories"]], d: "No input projection at d_c = d: the core reads the concepts directly." }; pickables.push(c); }
  });
  block({ g: "core", id: "cin", x: XC, y: yIn, w: W, h: 0.3, d: D, color: C.tok, label: "S₀ = 0 · working memory", sub: "C is re-read, S is written",
    info: { t: "Working memory S", cat: "loop", fig: "v3core", kv: [["S₀", "exact zeros"], ["written by", "every sublayer of every pass"], ["carried", "raw between passes (no carry norm)"], ["read as", "rms(S) inside the gated read"]], d: "S₀ = 0 means nothing has been computed yet, not that there is no input: the first read still sees C." } });
  const layerMeshes = [];
  for (let l = 0; l < 4; l++) {
    const m = block({ g: "core", id: `core${l + 1}`, x: XC, y: yLay(l), w: W, h: 0.5, d: D, color: C.full, label: `Core layer ${l + 1}`, sub: `6-head full attention · own FFN ${c3.own_ff}`, level: "minor", explode: -l * 0.35,
      info: { t: `Core layer ${l + 1} (distinct weights)`, cat: "loop", fig: "v3core", params: P.core_layer, kv: [["params", fi(P.core_layer)], ["attention", `${fi(P.core_attn)} · full causal over 256 concepts, 6 × 64, GQA kv 2, NoPE, τ_h`], ["own FFN", `SwiGLU 256 → ${c3.own_ff} → 256 (${fi(P.core_own_ffn)})`], ["shared FFN", `parallel branch, SwiGLU ${c3.shared_ff}`], ["reads", "{C, S} with (1+γ_C), (1+γ_S) from the table"], ["writes", "S, with (1+α) from the table"]], d: "One of four distinct layers; one pass = all four. The (t, Δ) row of the pass sets 7 gain vectors for this layer." } });
    m.material = m.material.clone(); m.material.userData.baseOpacity = 1; layerMeshes.push(m);
    if (l < 3) flow([[XC, yLay(l) - 0.27, 0], [XC, yLay(l + 1) + 0.27, 0]], C.P, { g: "core", r: 0.045 });
  }
  flow([[XC, yIn - 0.17, 0], [XC, yLay(0) + 0.27, 0]], C.P, { g: "core", r: 0.05 });
  // shared FFN: one block beside all four layers
  const shared = block({ g: "core", id: "shared", x: XC + 2.45, y: (yLay(0) + yLay(3)) / 2, w: 0.75, h: yLay(0) - yLay(3) + 0.6, d: 2.6, color: 0xf2a56f, r: 0.08, explode: -1.2,
    label: "Shared FFN", sub: `SwiGLU ${c3.shared_ff} · all layers`, level: "major", lpos: [0, (yLay(0) - yLay(3)) / 2 + 0.6, 1.32],
    info: { t: `Shared SwiGLU ${c3.shared_ff} (Mobius-style store)`, cat: "loop", fig: "v3core", params: P.core_shared, kv: [["params", fi(P.core_shared)], ["used by", "all 4 layers in every pass (≈ 18 applications / step)"], ["placement", "parallel branch on each FFN read"], ["init", "w2 zero-init: writes 0 at step 0"], ["write gain", "(1+α_sh) per (pair, layer)"], ["test", "param-matched arm: own FFN 992, no shared"]], d: "Layer 4 can query the same memory layer 1 used. The owner's hypothesis, tested by an exact parameter-matched control arm." } });
  shared.material = shared.material.clone(); shared.material.userData.baseOpacity = 1;
  for (let l = 0; l < 4; l++) flow([[XC + 2.05, yLay(l), 1.0], [XC + 1.65, yLay(l), 1.0]], 0xdc8b56, { g: "core", r: 0.03 });
  // (t, Δ) table: 6 pairs (columns) × 4 layers (rows)
  const cells = [];
  for (let p = 0; p < 6; p++) { cells.push([]); for (let l = 0; l < 4; l++) {
    const m = block({ g: "core", id: p || l ? null : "table", x: XC - 3.75 + p * 0.3, y: yLay(l), z: -2.4, w: 0.24, h: 0.42, d: 0.24, color: 0xc4aaea, r: 0.04, explode: -l * 0.35,
      info: p || l ? null : { t: "(t, Δ) table · Φ(S; C, t, Δt)", cat: "loop", fig: "v3table", params: P.core_table, kv: [["shape", "6 pairs × 4 layers × 7 vectors × 256"], ["params", fi(P.core_table)], ["pairs", "(0,⅓) (⅓,⅓) (⅔,⅓) (0,⅔) (⅓,⅔) (0,1)"], ["sites", "γ_C, γ_S, α (attn) · γ_C, γ_S, α_own, α_sh (FFN)"], ["init", "0 ⇒ bit-identical to the unconditioned core"], ["decay", "none (an all-zero table is v1's fixed point)"]], d: "Each column is one (t, Δ) pair, each row one layer. The pass lights its column: that is how a pass knows where it is and how much work it owes." } });
    m.material = m.material.clone(); m.material.userData.baseOpacity = 1; cells[p].push(m);
  } }
  label(anchor([XC - 3.0, yLay(0) + 0.75, -2.4], "core"), "(t, Δ) table", { level: "major", sub: "tells each pass where it is" });
  // pass ring: 3 markers (main M = 3); a shortcut trajectory re-uses it with 1 or 2 markers
  const ringC = [XC - 2.55, (yLay(0) + yLay(3)) / 2, D / 2 + 0.9], ringR = 1.05;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(ringR, 0.045, 12, 96), pmat(0x2e9c8c, { emissive: 0x2e9c8c, ei: 0.2, clearcoat: 0.7, opacity: 0.7 }));
  ring.position.set(...ringC); tag(ring, "core"); scene.add(ring);
  const markers = [];
  for (let p = 0; p < 3; p++) {
    const a = Math.PI / 2 - (p + 0.5) * (2 * Math.PI / 3);
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.17, 20, 14), pmat(0x2e9c8c, { emissive: 0x2e9c8c, ei: 0.15, clearcoat: 0.8 }));
    s.position.set(ringC[0] + ringR * Math.cos(a), ringC[1] + ringR * Math.sin(a), ringC[2]); s.castShadow = true; tag(s, "core"); scene.add(s); markers.push(s);
    if (p === 0) { parts.ring = s; s.userData.id = "ring"; s.userData.info = { t: "Trajectories (M ≤ 3)", cat: "loop", fig: "v3traj", kv: [["main", "M = 3, Δ = (⅓, ⅓, ⅓) → u₃"], ["shortcut", "S ~ U{1, 2}: M 1 (0,1) 50 % · 2a (⅓,⅔) · 2b (⅔,⅓) 25 % each"], ["start", "S₀ = 0 for both, C re-read"], ["backprop", "full, through both trajectories"], ["supervision", "endpoints only (terminal)"], ["eval", "M = 3 (M 1, 2a, 2b reported as elasticity)"]], d: "Each marker is one pass through the 4 layers. Pick a trajectory in the toolbar and press “Step passes”." }; pickables.push(s); }
  }
  label(anchor([ringC[0], ringC[1] + ringR + 0.55, ringC[2]], "core"), "Up to 3 passes", { level: "major", sub: "4 core layers per pass" });
  flow([[XC - 1.0, yU - 0.1, D / 2 + 0.05], [ringC[0] + 0.6, ringC[1] - ringR - 0.25, ringC[2]], [ringC[0] - ringR - 0.3, ringC[1], ringC[2]], [ringC[0] + 0.6, ringC[1] + ringR + 0.25, ringC[2]], [XC - 1.0, yLay(0) + 0.1, D / 2 + 0.05]], C.P, { g: "core", r: 0.03, opacity: 0.55 });
  block({ g: "core", id: "readout", x: XC, y: yU, w: W, h: 0.3, d: D, color: C.norm, label: "u₃ = zcRMSNorm(S₃)", sub: "raw S carried between passes",
    info: { t: "Readout u", cat: "loop", fig: "v3core", kv: [["u₃", "zcRMSNorm(S₃), main trajectory"], ["u_s", "same norm on the shortcut endpoint"], ["to decoder", "u₃ only"], ["consistency", "L_scons compares rms(S), not u (keeps the gain safe)"]], d: "One out_norm and one NCP head are shared by both trajectories." } });
  flow([[XC, yLay(3) - 0.27, 0], [XC, yU + 0.17, 0]], C.P, { g: "core", r: 0.05 });
  block({ g: "core", id: "ncp", x: XC, y: yHead, w: W, h: 0.34, d: D, color: C.conc, label: "NCP head · endpoints", sub: "q = W_q u (256 → 256)",
    info: { t: "NCP head", cat: "loop", fig: "v3crc", params: P.ncp_head, kv: [["W_q", "256 → 256"], ["logits", "−‖q_s − sg(e_s,n)‖² / τ_s, 32 segments"], ["used on", "u₃ (0.1) and u_s (0.05)"], ["c_pred", "Σ_n π_n sg(e_n)"]], d: "Predicts the next chunk's 32 PQ codes: the look-ahead half of the plan." } });
  flow([[XC, yU - 0.17, 0], [XC, yHead + 0.2, 0]], C.P, { g: "core", r: 0.05 });
  const S = A.pq.S, N = A.pq.N, gem = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.03, 0), pmat(0xe0b73a, { rough: 0.3, clearcoat: 0.6 }), S * N);
  const m4 = new THREE.Matrix4(), col = new THREE.Color(); let gk = 0;
  for (let s = 0; s < S; s++) for (let n = 0; n < N; n++) {
    m4.makeTranslation(XC - W / 2 + (n + 0.5) * W / N, yCB + 0.06, -D / 2 + (s + 0.5) * D / S); gem.setMatrixAt(gk, m4);
    const v = 0.6 + 0.4 * Math.abs(Math.sin(s * 12.9898 + n * 78.233)); col.setRGB(0.88 * v + 0.1, 0.7 * v + 0.08, 0.2 * v); gem.setColorAt(gk++, col);
  }
  tag(gem, "core"); scene.add(gem);
  block({ g: "core", id: "codebook", x: XC, y: yCB, w: W + 0.2, h: 0.06, d: D + 0.2, color: C.mem, opacity: 0.35, cast: false, label: "PQ codebook", sub: `${S} × ${N} codes · 8 dims`, lpos: [0, 0.1, (D + 0.2) / 2 + 0.02],
    info: { t: "PQ codebook", cat: "code", fig: "v3crc", params: P.codebook, kv: [["segments × codes × dims", `${S} × ${N} × 8`], ["bits / chunk", A.pq.bits], ["trained by", "L_VQ only, detached ĉ"], ["restarts", "dead codes every 100 steps"]], d: "Follows d_c = 256 with v1's per-segment geometry, so per-segment metrics stay comparable." } });
  const lossB = (id, x, y, lab, info, col2) => block({ g: "core", id, x, y, w: 1.35, h: 0.45, d: 1.6, color: col2 || C.loss, label: lab, lpos: [0, 0.5, 0.8], info });
  lossB("lncp", XC - 3.05, yHead, "0.1 · NCP(u₃)", { t: "NCP loss, main endpoint", cat: "loss", fig: "v3loss", kv: [["form", "per-segment CE of chunk j+1's codes from u₃"], ["weight", "0.1 (v1's)"], ["pairs", "clean: complete, same-document targets"]], d: "v1's term, at the main trajectory's endpoint only." });
  lossB("lncps", XC + 3.05, yHead, "0.05 · NCP(u_s)", { t: "NCP loss, shortcut endpoint", cat: "loss", fig: "v3loss", kv: [["form", "same head on u_s (M 1 / 2a / 2b)"], ["weight", "0.05 (half of main)"]], d: "The shortcut's only task-anchored signal." }, 0xd6b3f0);
  lossB("lscons", XC - 3.05, yU, "0.1 · L_scons", { t: "Shortcut consistency", cat: "loss", fig: "v3loss", kv: [["form", "MSE(rms S_s, sg rms S₃)"], ["weight", "0.1 (LoopFormer's)"], ["why gain-free", "on u it would shrink out_norm's gain (verified, T4c)"]], d: "Pulls the short trajectory's endpoint toward the stop-gradient main endpoint: makes M < 3 usable." }, 0xd6b3f0);
  lossB("lvq", XC + 3.05, yCB, "L_VQ", { t: "VQ loss", cat: "loss", fig: "v3loss", kv: [["form", "‖sg(ĉ) − e‖² per 8-dim segment"], ["updates", "codebook only"]], d: "Keeps the codes near the concepts (one-sided, as in v1)." });
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.36, 3), pmat(C.plan, { clearcoat: 0.8, rough: 0.25, emissive: C.plan, ei: 0.15 }));
    c.position.set(XC, yPred, z); c.castShadow = true; tag(c, "core"); scene.add(c);
    if (!j) { parts.pred = c; c.userData.id = "pred"; c.userData.info = { t: "c_pred and u₃", cat: "plan", fig: "v3dec", kv: [["c_pred", "Σ_n π_n sg(e_n), 256-d"], ["u₃", "main readout, 256-d"]], d: "The two things the decoder's plan builder receives, per chunk." }; pickables.push(c); }
  });
  label(anchor([XC, yPred - 0.75, 0], "core"), "Plan inputs → decoder", { level: "major", sub: "u₃ and predicted next concept" });
  flow([[XC, yCB - 0.1, 0.9], [XC, yPred + 0.4, 0.9]], C.P, { g: "core", r: 0.04 });
  passAnim = { markers, cells, layers: layerMeshes, shared };
  pollLiveTable();

  /* ---------- decoder (reads bottom → top) */
  const yPlan = 1.2, yDL = i => 2.6 + i * 1.0, yON = yDL(3) + 1.1, yH = yON + 0.9, yLog = yH + 1.1, yNTP = yLog + 1.3;
  flow([[XE - 1.95, streamTop + 0.1, 0], [XE - 1.95, streamTop + 3.6, 0], [0, streamTop + 4.8, 0], [XD - 1.95, streamTop + 3.6, 0], [XD - 1.95, yDL(3) + 0.7, 0]], C.L, { g: "dec", groups: ["enc", "dec"], r: 0.075 });
  label(anchor([0, streamTop + 5.3, 0], "dec"), "Local token stream → decoder", { level: "major", sub: `tok = zcRMSNorm(L) · reach ${A.reach_enc} tokens`, group: "dec" });
  glass({ g: "dec", id: "tok", x: XD - 1.95, y: (yPlan + yDL(3) + 0.6) / 2, w: 0.12, h: yDL(3) + 0.6 - yPlan, d: D, color: C.L, label: "tok", sub: "keys & values", level: "minor",
    info: { t: "tok (keys and values)", cat: "stream", fig: "v3dec", kv: [["shape", "(B, 1024, 256)"], ["reach", A.reach_enc]], d: "Every decoder layer takes its keys and values from tok only (window 8)." } });
  block({ g: "dec", id: "plan", x: XD, y: yPlan, w: W, h: 0.42, d: D, color: C.plan, label: "Plan builder", sub: "P = W_u u₃ + W_c c_pred + o_r", level: "major",
    info: { t: "Plan builder", cat: "plan", fig: "v3dec", params: P.dec_plan, kv: [["params", fi(P.dec_plan)], ["index", "j*(t) = ⌊(t+1)/4⌋ − 1"], ["W_u, W_c", "256 → 256"], ["null plan", "first 3 tokens of each document"], ["D⁰", "tok + P (ungated add)"]], d: "Each position gets the plan of the newest complete chunk; the target token is never inside it." } });
  flow([[XC + 1.7, yU, 0], [4.6, yU, 0], [6.3, yPlan + 0.1, 0], [XD - 1.75, yPlan, 0]], C.P, { g: "dec", groups: ["core", "dec"], r: 0.065 });
  flow([[XC + 0.5, yPred, 0], [4.6, yPred - 0.2, 1.4], [XD - 1.75, yPlan - 0.05, 1.4]], C.P, { g: "dec", groups: ["core", "dec"], r: 0.065 });
  label(anchor([4.9, yU + 0.45, 0], "dec"), "u₃", { level: "minor", group: "dec" });
  label(anchor([4.5, yPred - 0.65, 1.4], "dec"), "c_pred", { level: "minor", group: "dec" });
  for (let i = 0; i < 4; i++) {
    block({ g: "dec", id: `dec${i + 1}`, x: XD, y: yDL(i), w: W, h: 0.52, d: D, color: C.dec, label: `Confined layer ${i + 1}`, sub: "keys/values from tok", explode: (i + 1) * 0.55,
      info: { t: `Decoder layer ${i + 1} · confined`, cat: "dec", fig: "v3dec", params: P.dec_layer, kv: [["params", fi(P.dec_layer)], ["queries", "from GR read of D (plan-aware)"], ["keys, values", "from tok only, SWA w = 8, 4 heads"], ["positions", "partial RoPE 16/64"], ["FFN", "SwiGLU 256 → 1024 → 256"], ["reach", `${A.reach_dec} at any depth`]], d: "D is private to its position: position t sees 70 tokens plus its own plan." } });
    for (let q = -1; q <= 1; q += 2) flow([[XD - 1.9, yDL(i), q * 2.5], [XD - 1.62, yDL(i), q * 2.5]], C.L, { g: "dec", r: 0.035, arrow: false });
  }
  block({ g: "dec", id: "onorm", x: XD, y: yON, w: W, h: 0.26, d: D, color: C.norm, label: "zcRMSNorm(D)", explode: 5 * 0.55 });
  block({ g: "dec", id: "head", x: XD, y: yH, w: W, h: 0.34, d: D, color: C.head, label: "Tied output head", sub: "Eᵀ · 0 extra params", explode: 5.5 * 0.55,
    info: { t: "Output head (tied)", cat: "out", fig: "v3dec", kv: [["matrix", "Eᵀ, 256 × 8,192"], ["extra params", "0"], ["watch", "E row norms, output entropy (width-256 risk)"]], d: "logits = zcRMSNorm(D) · Eᵀ" } });
  for (let i = 0; i < 16; i++) {
    const h = 0.15 + 0.95 * Math.pow(Math.abs(Math.sin(i * 1.7 + 0.3)), 3);
    block({ g: "dec", x: XD, y: yLog + h / 2 - 0.3, z: zs(i), w: 0.3, h, d: 0.3, color: i === 9 ? 0xe07b39 : C.tok, r: 0.05, explode: 6 * 0.55 });
  }
  label(anchor([XD + 2.0, yLog + 0.2, 4.2], "dec"), "Next-token scores", { level: "minor", sub: "8,192 per position", cx: 0 });
  block({ g: "dec", id: "lntp", x: XD, y: yNTP, w: 2.2, h: 0.5, d: 2.2, color: C.loss, label: "Next-token loss", sub: "L_NTP · main training signal", level: "major", explode: 6.5 * 0.55, lpos: [0, 0.6, 1.1],
    info: { t: "Next-token loss", cat: "loss", fig: "v3loss", kv: [["targets", "x_{t+1}, same document, not PAD"], ["weight", "1"], ["reaches core", "through u₃ and c_pred of the main trajectory"]], d: "The main training signal. NextLat is removed in v3 (at plan boundaries it duplicates NCP)." } });

  // weight tying
  const tie = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(XE + 1.6, 0.5, -4.7), new THREE.Vector3(XD + 1.6, 0.5, -4.7), new THREE.Vector3(XD + 1.6, yH, -4.7)]),
    new THREE.LineDashedMaterial({ color: C.tie, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.9 }));
  tie.material.userData.baseOpacity = 0.9; tie.computeLineDistances(); tag(tie, "shared"); scene.add(tie);
  label(anchor([0, 0.5, -4.7], "shared"), "weight tying: head = Eᵀ", { level: "minor", group: "shared" });
}


/* ------------------------------------------------------------------ theme */
function applyTheme() {
  const cs = getComputedStyle(document.documentElement), dark = cs.getPropertyValue("color-scheme").trim() === "dark";
  scene.background = new THREE.Color(dark ? 0x0f1217 : 0xf6f7f9);
  grid.material.color = new THREE.Color(dark ? 0x2a303a : 0xdfe3e8); grid.material.vertexColors = false; grid.material.needsUpdate = true;
  floorMat.opacity = dark ? 0.35 : 0.13;
  scene.environmentIntensity = dark ? 0.7 : 0.85;
  outline.visibleEdgeColor.set(dark ? 0x7fb6f0 : 0x2f6fb3); outline.hiddenEdgeColor.set(dark ? 0x3a5a80 : 0x9db8d8);
}

/* ------------------------------------------------------------------ focus, selection, hover */
function isDim(o) {
  if (!focus) return false;
  const gs = o.userData.groups || [o.userData.group];
  return !gs.includes(focus);
}
function applyFocus() {
  scene.traverse(o => {
    if (!o.userData.group || !o.material) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    const dim = isDim(o);
    mats.forEach(m => {
      const b = m.userData.baseOpacity ?? 1;
      m.opacity = dim ? b * 0.07 : b; m.transparent = dim || b < 1; m.depthWrite = !dim && b >= 1; m.needsUpdate = true;
    });
  });
  labelObjs.forEach(l => l.el.classList.toggle("dim", !!focus && l.group !== focus));
  document.querySelectorAll("#v3c-bar [data-view]").forEach(b => b.classList.toggle("on", (b.dataset.view === "all" && !focus) || b.dataset.view === focus));
  requestRender();
}
function setFocus(g) { focus = g || null; applyFocus(); }
function setOutline() { outline.selectedObjects = [selected, hovered].filter(Boolean); }
function select(obj, fly) {
  selected = obj || null; setOutline(); showInfo(selected ? selected.userData.info : null, selected);
  if (selected && fly) fitTo([selected], null, 1.4);
  requestRender();
}

/* ------------------------------------------------------------------ camera */
const DIR = { all: [0.08, 0.34, 1], enc: [-0.55, 0.3, 1], core: [0.35, 0.28, 1], dec: [0.6, 0.3, 1] };
function fitTo(objs, dir, k = 1, instant = false) {        // frame the bounding sphere of objs; k < 1 zooms in, > 1 out
  const box = new THREE.Box3(); objs.forEach(o => box.expandByObject(o));
  const sph = box.getBoundingSphere(new THREE.Sphere()), c = sph.center.clone();
  const vf = THREE.MathUtils.degToRad(camera.fov), hf = 2 * Math.atan(Math.tan(vf / 2) * camera.aspect);
  const dist = Math.max(5, sph.radius / Math.sin(Math.min(vf, hf) / 2) * k);
  c.y -= sph.radius * 0.03;                                     // room for the plate titles under the model
  const p = c.clone().addScaledVector(new THREE.Vector3(...(dir || DIR.core)).normalize(), dist);
  if (instant) { camera.position.copy(p); controls.target.copy(c); controls.update(); requestRender(); return dist; }
  camTween = { t0: performance.now(), ms: 900, p0: camera.position.clone(), p1: p, q0: controls.target.clone(), q1: c };
  requestRender();
  return dist;
}
function groupObjs(g) { const out = []; scene.traverse(o => { if (o.isMesh && o.userData.group === g && !o.userData.groups) out.push(o); }); return out; }   // includes the floor plate
function groupInfo(g) {
  const T = A.table, P = A.params, c = A.core;
  return {
    enc: { t: "Encoder (v1 grammar, d 256)", cat: "enc", fig: "v3enc", params: T.encoder_non_embedding, kv: [["params", `${fi(T.encoder_non_embedding)} + ${fi(P.engram_table)} lookup`], ["layers", "8 (6 S · 4 heads, 2 F · 6 heads), SwiGLU 1024"], ["Engram", "one 80M table, reads after layers 0 and 4"], ["output", "256-d concepts C + tok (reach 63)"]], d: "v1's two-stream encoder at width 256; the headroom goes to FFN ratio 4 and 6-head F layers." },
    core: { t: "Concept module v3", cat: "loop", fig: "v3core", params: T.concept_module, kv: [["params", fi(T.concept_module)], ["layers", "4 distinct, 6-head full attention, own FFN 736"], ["shared FFN", `SwiGLU ${c.shared_ff}, all layers`], ["conditioning", `(t, Δ) table, ${fi(P.core_table)} params`], ["passes", "M = 3 main + shortcut 1 / 2a / 2b (training)"], ["share", `${(100 * T.concept_module / T.compute_params).toFixed(1)} % of compute params`]], d: "Reads top → bottom and loops around the ring: each pass runs the 4 layers with its own (t, Δ) column of gains." },
    dec: { t: "Decoder (confined)", cat: "dec", fig: "v3dec", params: T.decoder, kv: [["params", fi(T.decoder)], ["layers", "plan builder + 4 confined layers, SwiGLU 1024"], ["reach", `${A.reach_dec} tokens at any depth`], ["NextLat", "removed"]], d: "v1's confinement: all long-range information must arrive through the plan." },
  }[g];
}
function frameView(v, instant) {
  if (v === "all") { setFocus(null); wholeDist = fitTo(["enc", "core", "dec"].flatMap(groupObjs), DIR.all, 0.68, instant); if (!selected && tourIdx < 0 && built) showInfo(null); }
  else { setFocus(v); fitTo(groupObjs(v), DIR[v], 0.8, instant); if (!selected && tourIdx < 0) showInfo(groupInfo(v)); }
}

/* ------------------------------------------------------------------ labels: level of detail */
/* Labels are placed in priority order (selected part → zone → major → minor, nearer first); a label that would overlap
   one already placed is hidden, so the view never shows a pile of overlapping tags. Minor labels appear only when a
   module is focused or the camera is zoomed in. "Labels: all" shows everything, "off" keeps only the module plates. */
let wholeDist = 30;
const RANK = { zone: 0, znum: 0, major: 1, minor: 2 };
function updateLabelLOD() {
  const d = camera.position.distanceTo(controls.target), detail = !!focus || d < wholeDist * 0.72;
  const cand = [], placed = [], pad = 3, tmp = new THREE.Vector3();
  labelObjs.forEach(l => {
    const fixed = l.level === "zone" || l.level === "znum";
    let show = l.el.style.display !== "none";                  // CSS2DRenderer hides labels outside the view
    if (!fixed) {
      if (labelMode === "off") show = false;
      if (focus && l.group !== focus) show = false;
      if (labelMode === "auto" && l.level === "minor" && !detail) show = false;
    }
    if (!show) { l.el.classList.add("hid"); return; }
    l.sel = !!selected && l.obj.parent === selected;
    l.dist = l.obj.getWorldPosition(tmp).distanceTo(camera.position);
    cand.push(l);
  });
  if (labelMode === "all") { cand.forEach(l => l.el.classList.remove("hid")); return; }
  cand.sort((a, b) => (b.sel - a.sel) || (RANK[a.level] - RANK[b.level]) || (a.dist - b.dist));
  cand.forEach(l => {
    const r = l.el.getBoundingClientRect();
    const hit = placed.some(q => r.left < q.right + pad && r.right > q.left - pad && r.top < q.bottom + pad && r.bottom > q.top - pad);
    const keep = !hit || l.level === "zone" || l.level === "znum";
    l.el.classList.toggle("hid", !keep);
    if (keep) placed.push(r);
  });
}

/* ------------------------------------------------------------------ info panel */
function catChip(c) { const [n, col] = CAT[c] || ["Part", "#888"]; return `<span class="v3-cat"><i style="background:${col}"></i>${n}</span>`; }
function showInfo(info, obj) {
  const el = document.getElementById("v3c-info"); if (tourIdx >= 0) return;
  const T = A.table;
  if (!info) {
    el.innerHTML = `<div class="v3p-kicker">How to read this model</div><h5>HRM Thinker v3 · 19M</h5>
      <p>Data moves left to right across three plates. Click a plate to zoom in, click any block for details.</p>
      <div class="v3-zones">
        <button data-zone="enc"><span class="zn" style="background:#3b78c2">1</span><b>Encoder</b><small>reads the tokens and compresses every 4 into one concept; an 80M n-gram memory is read twice</small></button>
        <button data-zone="core"><span class="zn" style="background:#2e9c8c">2</span><b>Concept module</b><small>thinks over the concepts in up to 3 passes of 4 layers; the (t, Δ) table tells each pass where it is</small></button>
        <button data-zone="dec"><span class="zn" style="background:#e07b39">3</span><b>Decoder</b><small>sees only the last 70 tokens plus a plan from the concept module, and writes the next token</small></button>
      </div>
      <div class="v3p-sub">Colour key</div>
      <div class="v3-key"><span><i style="background:#86b4ea"></i>sliding-window layer</span><span><i style="background:#f2a56f"></i>full attention / FFN</span>
        <span><i style="background:#f4cd5c"></i>memory / codebook</span><span><i style="background:#4fbcad"></i>concepts &amp; plan</span>
        <span><i style="background:#c4aaea"></i>(t, Δ) table</span><span><i style="background:#ef8fa1"></i>loss</span><span><i style="background:#d6b3f0"></i>shortcut loss (training only)</span>
        <span><i class="ln" style="background:#3b78c2"></i>L / tok stream</span></div>
      <div class="v3-kv" style="margin-top:16px"><span>compute</span><b>${fi(T.compute_params)}</b><span>stored</span><b>${fi(T.stored_total)}</b><span>width</span><b>${A.ecfg.d_model}</b><span>passes</span><b>M ≤ 3</b></div>
      <div id="v3c-live-side">${liveHTML}</div>
      <div class="v3p-tip">Press “▶ Play passes” to watch one trajectory run: the lit ring marker is the pass, the lit table column is its (t, Δ) pair. Passes: 3 is the main path; 2a, 2b and 1 are the shorter training shortcuts.</div>`;
    el.querySelectorAll("[data-zone]").forEach(b => b.onclick = () => frameView(b.dataset.zone));
    return;
  }
  const share = info.params ? info.params / T.compute_params : null;
  el.innerHTML = `${catChip(info.cat)}<h5>${info.t}</h5><p>${info.d || ""}</p>
    ${share != null ? `<div class="v3-share"><div class="v3-share-l"><span>share of compute parameters</span><b>${(100 * share).toFixed(1)} %</b></div><div class="v3-share-t"><div style="width:${Math.max(1.5, Math.min(100, 100 * share))}%"></div></div></div>` : ""}
    <div class="v3-kv">${(info.kv || []).map(([a, b]) => `<span>${a}</span><b>${b}</b>`).join("")}</div>
    <div class="v3p-actions">${obj ? `<button class="btn sm" id="v3c-fly">Zoom to it</button>` : ""}${info.fig ? `<a class="btn sm" href="#v3arch2d" data-fig3="${info.fig}">Open 2D figure →</a>` : ""}<button class="btn sm ghost" id="v3c-clear">Clear</button></div>`;
  const fly = document.getElementById("v3c-fly"); if (fly) fly.onclick = () => fitTo([obj], null, 1.4);
  document.getElementById("v3c-clear").onclick = () => select(null);
}

/* ------------------------------------------------------------------ guided tour */
const TOUR = [
  { n: "Tokens in", g: "enc", pick: "emb", objs: ["tokens", "emb"], dir: [-0.5, 0.45, 1], x: "Token ids enter at the bottom left; each looks up a 256-number vector. The same table is the output head at the end." },
  { n: "v1's encoder, wider FFNs", g: "enc", pick: "Lstream", objs: ["enc0", "enc7", "Lstream", "Gstream"], dir: [-0.6, 0.3, 1], x: "SSSFSSSF with the local stream L (reach 63) and the global stream G. FFN ratio 4 in every layer, 6 heads in the two F layers. 9.64M parameters." },
  { n: "One Engram table, two reads", g: "enc", pick: "engram2", objs: ["engram1", "engram2", "engramTables"], dir: [-0.7, 0.35, 1], x: "The 80M n-gram table is gathered once and read twice: after layer 0 (with conv) and after layer 4 (no conv, W_v zero-init, gate reads L only), so reach stays 63." },
  { n: "256-d concepts", g: "enc", pick: "pool", objs: ["crc", "pool", "concepts"], dir: [-0.45, 0.4, 1], x: "Every 4 tokens become one 256-dim concept on a sphere (4 : 1, v1's ratio). These concepts are the evidence C." },
  { n: "Evidence C, memory S", g: "core", pick: "cin", objs: ["X", "cin", "core1"], dir: [0.3, 0.35, 1], x: "C is re-read by every sublayer of every pass; S starts at exactly zero and is the only thing written. The decoder reads zcRMSNorm of the final S." },
  { n: "Four distinct layers", g: "core", pick: "core1", objs: ["core1", "core4"], dir: [0.35, 0.3, 1], x: "Each core layer: gated read of {C, S} → 6-head full attention over 256 concepts → write S; read again → own SwiGLU 736 → write S." },
  { n: "The shared FFN", g: "core", pick: "shared", objs: ["shared", "core2"], dir: [0.7, 0.3, 1], x: "One SwiGLU 1024 used by all four layers in every pass, as a parallel branch on each FFN read (the owner's Mobius-style store). It writes 0 at step 0." },
  { n: "Φ(S; C, t, Δt)", g: "core", pick: "table", objs: ["table", "core1", "core4"], dir: [-0.2, 0.3, 1], x: "The (t, Δ) table: 6 reachable pairs × 4 layers × 7 gain vectors, zero-init, no decay. Each pass uses its column, so passes are told where they are (t) and how much they owe (Δ). A tested remedy for v1's near-repetitive passes, not a guaranteed fix." },
  { n: "At most three passes", g: "core", pick: "ring", objs: ["ring", "core1", "readout"], dir: [0.2, 0.3, 1], x: "Main trajectory (⅓, ⅓, ⅓) feeds the decoder. A shortcut of 1 or 2 passes (training only) is pulled toward the main endpoint, so M = 1, 2 still work at inference." },
  { n: "Endpoints graded", g: "core", pick: "ncp", objs: ["ncp", "codebook", "pred", "lscons"], dir: [0.35, 0.55, 1], x: "The NCP head predicts the next chunk's 32 codes from u₃ (0.1) and from the shortcut endpoint (0.05); L_scons compares the gain-free states. u₃ and c_pred go to the decoder." },
  { n: "Confined decoder", g: "dec", pick: "dec1", objs: ["plan", "dec1", "dec4", "tok", "lntp"], dir: [0.6, 0.3, 1], x: "Unchanged confinement: each position sees the last 70 tokens plus its own plan. SwiGLU 1024, 4 heads, tied head. NextLat is removed." },
];
function goTour(i) {
  tourIdx = i; const S = TOUR[i];
  setFocus(S.g);
  const obj = parts[S.pick]; selected = obj || null; setOutline();
  fitTo(S.objs.map(id => parts[id]).filter(Boolean), S.dir, 1.05);
  document.querySelectorAll("#tourc [data-i]").forEach(b => { b.classList.toggle("on", +b.dataset.i === i); b.classList.toggle("done", +b.dataset.i < i); });
  const info = obj ? obj.userData.info : null;
  document.getElementById("v3c-info").innerHTML = `<div class="v3p-kicker">Guided tour · ${i + 1} of ${TOUR.length}</div>
    <div class="v3-prog"><div style="width:${100 * (i + 1) / TOUR.length}%"></div></div>
    <h5>${S.n}</h5><p>${S.x}</p>
    ${info ? `${catChip(info.cat)}<div class="v3-kv">${(info.kv || []).slice(0, 5).map(([a, b]) => `<span>${a}</span><b>${b}</b>`).join("")}</div>` : ""}
    <div class="v3p-actions"><button class="btn sm" id="tb-tour-prev" ${i ? "" : "disabled"}>← Back</button>
      <button class="btn sm primary" id="tb-tour-next">${i === TOUR.length - 1 ? "Finish" : "Next →"}</button>
      <button class="btn sm ghost" id="tb-tour-play">${tourTimer ? "❚❚ Pause" : "▶ Autoplay"}</button><button class="btn sm ghost" id="tb-tour-end">Exit</button></div>
    <div class="v3p-tip">Keyboard: ← → to move, Esc to exit.</div>`;
  document.getElementById("tb-tour-prev").onclick = () => { stopAuto(); i > 0 && goTour(i - 1); };
  document.getElementById("tb-tour-next").onclick = () => { stopAuto(); i < TOUR.length - 1 ? goTour(i + 1) : endTour(); };
  document.getElementById("tb-tour-play").onclick = () => { tourTimer ? stopAuto() : startAuto(); goTour(tourIdx); };
  document.getElementById("tb-tour-end").onclick = endTour;
}
function startAuto() { tourTimer = setInterval(() => { if (tourIdx < TOUR.length - 1) goTour(tourIdx + 1); else endTour(); }, 7000); }
function stopAuto() { if (tourTimer) clearInterval(tourTimer); tourTimer = null; }
function endTour() {
  stopAuto(); tourIdx = -1;
  document.querySelectorAll("#tourc [data-i]").forEach(b => b.classList.remove("on", "done"));
  select(null); frameView("all");
}
function setupTour() {
  const el = document.getElementById("tourc");
  el.innerHTML = `<button class="tour-start" id="tb-tour-start">▶ Start guided tour</button><div class="tour-steps">` +
    TOUR.map((t, i) => `<button data-i="${i}" title="${t.n}"><span>${i + 1}</span><em>${t.n}</em></button>`).join("") + `</div>`;
  el.querySelectorAll("[data-i]").forEach(b => b.onclick = () => { stopAuto(); goTour(+b.dataset.i); });
  document.getElementById("tb-tour-start").onclick = () => { stopAuto(); goTour(0); };
}

/* ------------------------------------------------------------------ input */
function wireInput(stage, cv) {
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2(), tipEl = document.getElementById("v3c-hover");
  const pick = ev => {
    const r = cv.getBoundingClientRect();
    ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    return ray.intersectObjects(pickables.filter(o => !isDim(o)), false)[0]?.object || null;
  };
  let down = null, dragging = false, raf = 0;
  cv.addEventListener("pointerdown", ev => { down = [ev.clientX, ev.clientY]; dragging = false; });
  cv.addEventListener("pointermove", ev => {
    if (down && Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4) dragging = true;
    if (dragging || raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const h = pick(ev);
      if (h !== hovered) { hovered = h; setOutline(); requestRender(); }
      cv.style.cursor = h ? "pointer" : "grab";
      if (h) {
        const i = h.userData.info, r = stage.getBoundingClientRect();
        tipEl.innerHTML = `<b>${i.t}</b>${i.params ? `<span>${fp(i.params)} params</span>` : ""}<em>click for details</em>`;
        tipEl.style.transform = `translate(${Math.min(ev.clientX - r.left + 14, r.width - 240)}px, ${ev.clientY - r.top + 14}px)`; tipEl.classList.add("on");
      } else tipEl.classList.remove("on");
    });
  });
  cv.addEventListener("pointerleave", () => { tipEl.classList.remove("on"); if (hovered) { hovered = null; setOutline(); requestRender(); } });
  cv.addEventListener("pointerup", ev => {
    const click = down && Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) < 5; down = null;
    if (!click) return;
    if (tourIdx >= 0) { stopAuto(); tourIdx = -1; document.querySelectorAll("#tourc [data-i]").forEach(b => b.classList.remove("on", "done")); }
    select(pick(ev));
  });
  cv.addEventListener("dblclick", ev => { const h = pick(ev); if (h) select(h, true); });
  labelRenderer.domElement.addEventListener("click", ev => {
    const z = ev.target.closest("[data-go]"); if (z && z.dataset.go.startsWith("zone:")) frameView(z.dataset.go.slice(5));
  });
  document.addEventListener("keydown", ev => {
    if (!document.getElementById("tab-v3arch3d").classList.contains("on") || /INPUT|TEXTAREA/.test(document.activeElement?.tagName)) return;
    if (ev.key === "ArrowRight") { stopAuto(); goTour(Math.min(TOUR.length - 1, tourIdx + 1)); }
    else if (ev.key === "ArrowLeft" && tourIdx > 0) { stopAuto(); goTour(tourIdx - 1); }
    else if (ev.key === "Escape") { if (tourIdx >= 0) endTour(); else { select(null); frameView("all"); } }
  });
}

/* ------------------------------------------------------------------ toolbar */
let traj = "3";
function setupBar() {
  document.querySelectorAll("#v3c-bar button").forEach(b => b.onclick = () => {
    const act = b.dataset.act;
    if (b.dataset.view) { if (tourIdx >= 0) { stopAuto(); tourIdx = -1; document.querySelectorAll("#tourc [data-i]").forEach(x => x.classList.remove("on", "done")); select(null); } frameView(b.dataset.view); }
    else if (b.dataset.traj) { traj = b.dataset.traj; document.querySelectorAll("#v3c-bar [data-traj]").forEach(x => x.classList.toggle("on", x === b)); showTraj(); if (passTimer) startPasses(); }
    else if (act === "explode") { b.classList.toggle("on"); explodeTarget = b.classList.contains("on") ? 1 : 0; requestRender(); }
    else if (act === "labels") { labelMode = { auto: "all", all: "off", off: "auto" }[labelMode]; b.textContent = { auto: "Labels: auto", all: "Labels: all", off: "Labels: off" }[labelMode]; b.classList.toggle("on", labelMode !== "off"); requestRender(); }
    else if (act === "flow") { b.classList.toggle("on"); flowOn = b.classList.contains("on"); requestRender(); }
    else if (act === "passes") { b.classList.toggle("on"); b.classList.contains("on") ? startPasses() : stopPasses(); }
    else if (act === "reset") { if (tourIdx >= 0) endTour(); else { select(null); frameView("all"); } }
    else if (act === "png") { composer.render(); const a = document.createElement("a"); a.download = "hrm-thinker-v3-draft-3d.png"; a.href = renderer.domElement.toDataURL("image/png"); a.click(); }
  });
  showTraj();
}

/* ------------------------------------------------------------------ live (t, Δ) table shading from the running v3 (data/v3_live.json)
   Each column's resting colour is scaled by that pair's mean ‖γ_C‖ + ‖γ_S‖ + ‖α‖ (averaged over layers) relative to the largest pair;
   pale = still near its zero init. Falls back to the plain colour when no run is live. */
let liveTable = null, liveHTML = "";
const PAIR_LIVE_KEY = () => A.core.pairs.map(([a, b]) => "t" + a + b);
function liveCellColor(ci) {
  const base = new THREE.Color(0xc4aaea);
  if (!liveTable || !A) return base;
  const keys = PAIR_LIVE_KEY(), v = k => liveTable[k] ? liveTable[k].a + liveTable[k].gC + liveTable[k].gS : 0;
  const mx = Math.max(1e-9, ...keys.map(v));
  return base.lerp(new THREE.Color(0x4b2aa8), Math.min(1, v(keys[ci]) / mx));
}
async function pollLiveTable() {
  try {
    const d = await fetchLive("data/v3_live.json");
    if (liveTable && d.step === pollLiveTable.last) return;          // nothing new since the last poll
    pollLiveTable.last = d.step;
    liveTable = d.table || null;
    if (liveTable) {
      const keys = PAIR_LIVE_KEY(), names = A.core.pairs.map(([a, b]) => `(${["0", "⅓", "⅔", "1"][a]}, ${["0", "⅓", "⅔", "1"][b]})`);
      const v = k => liveTable[k] ? liveTable[k].a + liveTable[k].gC + liveTable[k].gS : 0, mx = Math.max(1e-9, ...keys.map(v));
      liveHTML = `<div class="v3p-sub">Live from the v3 run · step ${fi(d.step)}</div>
        <p class="v3-live-p">How far each (t, Δ) column has moved from zero (it starts at exactly 0). Darker cells in the model = larger.</p>
        <div class="v3-live-bars">${keys.map((k, i) => `<span>${names[i]}</span><i><em style="width:${(100 * v(k) / mx).toFixed(1)}%"></em></i><b>${v(k).toFixed(2)}</b>`).join("")}</div>`;
      const box = document.getElementById("v3c-live-side"); if (box) box.innerHTML = liveHTML;
    }
    if (passAnim && passIdx < 0) setPass(-1);
  } catch (e) { liveTable = null; }
}
setInterval(() => { if (document.getElementById("tab-v3arch3d")?.classList.contains("on")) pollLiveTable(); }, 30000);

/* ------------------------------------------------------------------ pass animation (illustrative: which (t, Δ) column each pass uses) */
const PAIRKEY = () => A.core.pairs.map(([a, b]) => a + "," + b);
function showTraj() {                                            // show one ring marker per pass of the chosen trajectory
  if (!passAnim) return;
  const n = A.core.schedules[traj].length, main = traj === "3";
  passAnim.markers.forEach((m, i) => { m.visible = i < n; m.material.color.setHex(main ? 0x2e9c8c : 0xa07cc5); m.material.emissive.setHex(main ? 0x2e9c8c : 0xa07cc5); });
  setPass(-1);
}
function setPass(k) {                                            // k = pass * 4 + layer, or -1
  if (!passAnim) return;
  const sch = A.core.schedules[traj], keys = PAIRKEY(), p = k < 0 ? -1 : Math.floor(k / 4), l = k < 0 ? -1 : k % 4;
  const col = p >= 0 ? keys.indexOf(sch[p].join(",")) : -1;
  passAnim.markers.forEach((m, i) => { m.material.emissiveIntensity = i === p ? 1.1 : 0.15; m.scale.setScalar(i === p ? 1.45 : 1); });
  passAnim.cells.forEach((cc, ci) => cc.forEach((m, li) => {
    const on = ci === col, hot = on && li === l;
    if (hot || on) m.material.color.setHex(hot ? 0x7a5bd0 : 0xa98ae6); else m.material.color.copy(liveCellColor(ci));
    m.material.emissive.setHex(on ? 0x7a5bd0 : 0x000000); m.material.emissiveIntensity = hot ? 0.8 : on ? 0.3 : 0;
  }));
  passAnim.layers.forEach((m, li) => { m.material.emissive.setHex(li === l ? 0xe0662a : 0x000000); m.material.emissiveIntensity = li === l ? 0.45 : 0; });
  passAnim.shared.material.emissive.setHex(l >= 0 ? 0xe0662a : 0x000000); passAnim.shared.material.emissiveIntensity = l >= 0 ? 0.35 + 0.1 * l : 0;
  const tip = document.getElementById("v3c-hover");
  if (p >= 0 && tourIdx < 0) { const [a, b] = sch[p], th = x => ["0", "⅓", "⅔", "1"][x]; tip.innerHTML = `<b>${traj === "3" ? "Main path" : "Shortcut " + traj} · pass ${p + 1} of ${sch.length} · core layer ${l + 1}</b><span>(t = ${th(a)}, Δ = ${th(b)}) → table column ${col + 1}</span><em>illustrative, not trained activations</em>`; tip.style.transform = "translate(14px, 14px)"; tip.classList.add("on"); }
  else if (p < 0) tip.classList.remove("on");
  requestRender();
}
function startPasses() { if (passTimer) clearInterval(passTimer); passIdx = -1; const n = A.core.schedules[traj].length * 4 + 2; passTimer = setInterval(() => { passIdx = (passIdx + 1) % n; setPass(passIdx < n - 2 ? passIdx : -1); }, 650); }
function stopPasses() { if (passTimer) clearInterval(passTimer); passTimer = null; passIdx = -1; setPass(-1); const b = document.querySelector('#v3c-bar [data-act="passes"]'); if (b) b.classList.remove("on"); }
function resize() {
  const st = document.getElementById("v3c-stage"), w = st.clientWidth, h = st.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); composer.setPixelRatio(renderer.getPixelRatio()); composer.setSize(w, h); labelRenderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix(); requestRender();
}
function stopFlow() {
  stopAuto(); if (passTimer) stopPasses();
  if (!flowOn) return; flowOn = false;
  const b = document.querySelector('#v3c-bar [data-act="flow"]'); if (b) b.classList.remove("on");
}
async function ensureArch() { if (!window.ARCH3D) window.ARCH3D = await (await fetch("data/arch_v3d.json")).json(); }

window.addEventListener("tabshown", async e => {
  if (e.detail !== "v3arch3d") { if (built) stopFlow(); return; }
  await ensureArch();
  if (!built) build(); else { resize(); requestRender(); }
});
document.addEventListener("visibilitychange", () => { if (document.hidden && built) stopFlow(); });
if (document.getElementById("tab-v3arch3d").classList.contains("on")) ensureArch().then(() => { if (!built) build(); });
