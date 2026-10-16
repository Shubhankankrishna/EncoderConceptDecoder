/* 3D model of MODEL v2 (d 256 · Engram 80M · D6 core · decoder + NextLat), derived from js/view3d.js (the v1 scene engine).
   Reads data/arch_v2.json. Element ids are suffixed "b" so both scenes can coexist.
   Engine notes from v1 (D-040): Three.js r169 + add-ons (vendor/jsm): physically based materials with image-based lighting (RoomEnvironment),
   soft shadows, rounded geometry, an outline post-pass for hover / selection, CSS2D labels, focus modes that dim everything else,
   and a guided tour. Built lazily when the tab opens; renders on demand only (no idle GPU use while the model trains). */
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
  window.__v3bframes = (window.__v3bframes || 0) + 1;          // frame counter (verifies on-demand rendering)
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
  updateLabelLOD();
  composer.render();
  labelRenderer.render(scene, camera);
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
  A = window.ARCH2;
  const stage = document.getElementById("v3b-stage"), cv = document.getElementById("v3b");
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
  const e = A.ecfg, P = A.params, T = A.table;
  const XE = -10, XC = 0, XD = 10, W = 3.2, D = 8.4, zs = i => -3.75 + i * 0.5;
  zonePlate(XE, 12.5, 12, 0x3b78c2, "enc", "Encoder", `${fp(T.encoder_non_embedding)} + ${fp(T.engram_table)} memory`, 1);
  zonePlate(XC, 8.2, 12, 0x2e9c8c, "core", "Concept module v2 (D6)", `${fp(T.concept_module)} · ${A.core.ffn.E} experts`, 2);
  zonePlate(XD, 8.2, 12, 0xe07b39, "dec", "Decoder (confined) + NextLat", fp(T.decoder), 3);

  /* ---------- tokens + embedding */
  for (let i = 0; i < 16; i++) block({ g: "enc", x: XE, y: -0.3, z: zs(i), w: 0.38, h: 0.38, d: 0.38, color: Math.floor(i / 4) % 2 ? C.tokB : C.tok, r: 0.07, id: i ? null : "tokens",
    info: i ? null : { t: "Tokens", cat: "input", fig: "overview", kv: [["drawn", "16 of 1,024 positions"], ["vocabulary", fi(e.vocab_size)], ["chunks", "4 tokens → 1 concept"]], d: "Token ids arrive at the bottom of the encoder. Alternating shades mark the 4-token chunks that later become one concept each." } });
  label(anchor([XE + 2.2, -0.3, 3.9], "enc"), "tokens · 16 of 1,024", { level: "minor", cx: 0 });
  block({ g: "enc", id: "emb", x: XE, y: 0.5, w: W, h: 0.26, d: D, color: C.emb, label: "Token embedding", sub: "8,192 × 256 · tied",
    info: { t: "Token embedding E", cat: "input", fig: "overview", params: T.token_embedding, kv: [["shape", `${fi(e.vocab_size)} × ${e.d_model}`], ["params", fi(T.token_embedding)], ["use", "e0 = rms(E[x]); tied with the output head"]], d: "Byte-level BPE with 8,192 tokens; digits split. The same matrix E produces the logits at the top of the decoder." } });

  /* ---------- encoder streams + layers */
  const layerY = i => 1.5 + i * 0.95, streamTop = layerY(8) + 0.9;
  glass({ g: "enc", id: "Lstream", x: XE - 1.95, y: (1.0 + streamTop) / 2, w: 0.12, h: streamTop - 1.0, d: D, color: C.L, label: "L", sub: "local · reach 63",
    info: { t: "L stream (local)", cat: "stream", fig: "reach", kv: [["reach", `${A.reach_enc} tokens`], ["written by", "S layers, Engram"], ["read by", "S and F layers, CRC, decoder (tok)"]], d: "The only encoder stream the decoder may read. Its reach is bounded because only sliding-window layers write to it." } });
  glass({ g: "enc", id: "Gstream", x: XE + 1.95, y: (1.0 + streamTop) / 2, w: 0.12, h: streamTop - 1.0, d: D, color: C.G, label: "G", sub: "global",
    info: { t: "G stream (global)", cat: "stream", fig: "reach", kv: [["reach", "unbounded (FULL layers)"], ["written by", "S layers, F layers, Engram"], ["read by", "F layers, CRC → concepts"]], d: "Carries information from the whole document, but reaches the decoder only through the concept pooling and the concept module." } });
  const rows = [["S", 0], ["E"], ["S", 1], ["S", 2], ["F", 3], ["S", 4], ["S", 5], ["S", 6], ["F", 7]];
  rows.forEach((r, i) => {
    let info, color, lab, sub;
    if (r[0] === "S") {
      color = C.swa; lab = `S${r[1]}`; sub = "sliding window";
      info = { t: `Encoder layer ${r[1]} · S (sliding window)`, cat: "enc", fig: "enc", params: P.enc_layer_S, kv: [["params", fi(P.enc_layer_S)], ["reads", "{L}"], ["writes", "{L, G}"], ["attention", `SWA w = ${e.window}, partial RoPE ${e.rope_dims}/${e.head_dim}, GQA ${e.n_heads}:${e.n_kv_heads}`], ["q/k/v conv", `kernel ${e.conv_kernel}, doc-aware`], ["FFN", `SwiGLU ${e.d_model} → ${e.d_ff} → ${e.d_model}`], ["reach added", "(w−1)+(k−1) = 10"]], d: "Gated-residual read of L → gated attention → per-stream write scales; then the same for the FFN." };
    } else if (r[0] === "F") {
      color = C.full; lab = `F${r[1]}`; sub = "full attention";
      info = { t: `Encoder layer ${r[1]} · F (full attention)`, cat: "full", fig: "enc", params: P.enc_layer_F, kv: [["params", fi(P.enc_layer_F)], ["reads", "{L, G}"], ["writes", "{G} only"], ["attention", "causal within the document, NoPE, per-head τ"]], d: "Global layer. Because it never writes L, the decoder's token path keeps its 63-token reach." };
    } else {
      color = C.mem; lab = "Engram"; sub = "n-gram memory";
      info = { t: "Engram (hashed n-gram memory)", cat: "mem", fig: "engram", params: P.engram_proj, kv: [["table", `${fi(P.engram_table)} params (lookup)`], ["compute", fi(P.engram_proj)], ["orders", e.engram_orders.join(", ")], ["heads", e.engram_heads + " per order"], ["tables", `8 × ${fi(e.engram_rows)} × ${e.engram_dim}`], ["gate", "α = σ(rms(h)·rms(k)/√d)"], ["writes", "{L, G}"]], d: "Canonical-id n-grams are hashed into 8 tables; the looked-up vector is gated by agreement with the hidden state and added to both streams." };
    }
    block({ g: "enc", id: r[0] === "E" ? "engram" : `enc${r[1]}`, x: XE, y: layerY(i), w: W, h: 0.46, d: D, color, label: lab, sub, explode: i * 0.55, info });
  });
  for (let t = 0; t < 8; t++) block({ g: "enc", id: t ? null : "engramTables", x: XE - 4.3 - (t % 4) * 0.36, y: layerY(1) + 0.25, z: -2.2 + Math.floor(t / 4) * 4.4, w: 0.24, h: 3.8, d: 1.6, color: C.mem, r: 0.06, explode: 0.55,
    info: t ? null : { t: "Engram tables", cat: "mem", fig: "engram", params: P.engram_table, kv: [["count", "8 (orders 2, 3 × 4 heads)"], ["rows × dim", `${fi(e.engram_rows)} × ${e.engram_dim}`], ["params", fi(P.engram_table)]], d: "One row is read from each table per token; the table is a lookup, not compute, so its 80.0M numbers cost almost nothing to use. v2 raised the rows from 131,071 to 208,333." } });
  const engA = anchor([XE - 4.85, layerY(1) + 1.85, 0], "enc"); explodables.push({ obj: engA, base: engA.position.y, k: 0.55 });
  label(engA, "8 Engram tables", { level: "minor", sub: `${fp(P.engram_table)} lookups` });
  flow([[XE - 3.95, layerY(1), 0], [XE - 1.7, layerY(1), 0]], C.mem, { g: "enc", r: 0.06 });
  const yCRC = layerY(8) + 1.25, yPool = yCRC + 0.8, yC = yPool + 1.0;
  block({ g: "enc", id: "crc", x: XE, y: yCRC, w: W, h: 0.34, d: D, color: C.gate, label: "CRC depth-mix", sub: "e0 · L · G", explode: 9 * 0.55,
    info: { t: "CRC depth-mix", cat: "concept", fig: "crc", params: P.crc_router, kv: [["params", fi(P.crc_router)], ["mix", "softmax(W rms(G)) over {e0, rms L, rms G}"], ["init", "0.107 / 0.107 / 0.787"]], d: "A per-token router decides how much of the embedding, local and global state goes into the concept." } });
  block({ g: "enc", id: "pool", x: XE, y: yPool, w: W, h: 0.34, d: D, color: C.conc, label: "Concat-pool 4 → 1", sub: "1,024 → 384", explode: 9.3 * 0.55,
    info: { t: "Concept pooling", cat: "concept", fig: "crc", params: P.pool, kv: [["W_pool", `${e.chunk * e.d_model} → ${e.d_concept}`], ["params", fi(P.pool)], ["init", "mean pooling"], ["output", "ĉ = √384 · c/‖c‖"]], d: "Four consecutive mixed token states are concatenated and projected, then put on the sphere of radius √384." } });
  const cz = [-3, -1, 1, 3];
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 3), pmat(C.conc, { clearcoat: 0.8, rough: 0.25, emissive: C.conc, ei: 0.12 }));
    c.position.set(XE, yC, z); c.castShadow = true; tag(c, "enc"); scene.add(c); explodables.push({ obj: c, base: yC, k: 9.6 * 0.55 });
    if (!j) { parts.concepts = c; c.userData.id = "concepts"; c.userData.info = { t: "Concepts ĉ", cat: "concept", fig: "crc", kv: [["count", "1 per 4 tokens (256 per row)"], ["dim", e.d_concept], ["norm", `√${e.d_concept} = ${Math.sqrt(e.d_concept).toFixed(2)}`]], d: "Concepts live on a sphere (drawn as spheres): only their direction carries information. This sequence X is handed to the concept module." }; pickables.push(c); }
    for (let k = 0; k < 4; k++) flow([[XE, yPool + 0.2, zs(4 * j + k)], [XE, yPool + 0.55, zs(4 * j + k) * 0.6 + z * 0.4], [XE, yC - 0.4, z]], C.P, { g: "enc", r: 0.018, opacity: 0.6, arrow: false });
  });
  const cA = anchor([XE, yC + 0.85, 0], "enc"); explodables.push({ obj: cA, base: cA.position.y, k: 9.6 * 0.55 });
  label(cA, "ĉ · concepts", { level: "major", sub: "on a sphere of radius √384" });

  /* ---------- concept module v2 = D6 (reads top → bottom) */
  const c2 = A.core, NE = c2.ffn.E;
  const yX = yC, yIn = yX - 1.25, yLay = l => yIn - 1.15 - l * 0.8, yCar = yLay(3) - 0.95, yHead = yCar - 1.0, yCB = yHead - 1.2, yPred = yCB - 1.25;
  flow([[XE + 0.6, yC, 0], [-5, yC + 1.0, 0], [XC - 2.1, yX, 0]], C.P, { g: "core", groups: ["enc", "core"], r: 0.075 });
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.4, 3), pmat(C.conc, { clearcoat: 0.8, rough: 0.25, emissive: C.conc, ei: 0.12 }));
    c.position.set(XC, yX, z); c.castShadow = true; tag(c, "core"); scene.add(c);
    if (!j) { parts.X = c; c.userData.id = "X"; c.userData.info = { t: "ĉ (384-d concepts)", cat: "loop", fig: "v2crc", kv: [["shape", "(B, 256, 384)"], ["role", "input of the core"]], d: "The concept sequence from the encoder; projected twice: X̃ (re-read every pass) and S₀ (the start of the carry)." }; pickables.push(c); }
  });
  block({ g: "core", id: "cin", x: XC, y: yIn, w: W, h: 0.3, d: D, color: C.conc, label: "X̃ = W_in ĉ · S₀ = W_s ĉ", sub: "384 → 256 each",
    info: { t: "Core inputs", cat: "loop", fig: "v2core", params: P.core_io, kv: [["params", fi(P.core_io)], ["X̃", "clean input, re-read by every sublayer of every pass"], ["S₀", "distinct projection: no dead start"]], d: "Two projections of the concepts: the clean input X̃ and the initial carry S₀." } });
  const layerMeshes = [];
  for (let l = 0; l < 4; l++) {
    layerMeshes.push(block({ g: "core", id: `core${l + 1}`, x: XC, y: yLay(l), w: W, h: 0.5, d: D, color: C.full, label: `Core layer ${l + 1}`, sub: `full attention · top-2 of ${NE}`, level: l ? "minor" : "major", explode: -l * 0.35,
      info: { t: `Core layer ${l + 1} (distinct weights)`, cat: "loop", fig: "v2core", params: P.core_layer_dense, kv: [["dense params", fi(P.core_layer_dense)], ["reads", "[X̃, S] via gated read, rank 32"], ["attention", "full causal over 256 chunk positions, 4×64, GQA kv 2, NoPE, output gate"], ["FFN", `router 256 → ${NE} → top-2 experts (shared pool)`], ["writes", "S"], ["init", "W_o zero-init"]], d: "One of four distinct layers; one pass = all four. The router reads the state, so later passes can choose different experts." } }));
    if (l < 3) flow([[XC, yLay(l) - 0.27, 0], [XC, yLay(l + 1) + 0.27, 0]], C.P, { g: "core", r: 0.045 });
  }
  flow([[XC, yIn - 0.17, 0], [XC, yLay(0) + 0.27, 0]], C.P, { g: "core", r: 0.05 });
  // expert shelf: 16 experts in 2 columns
  const experts = [];
  for (let k = 0; k < NE; k++) {
    const col = k % 2, row = Math.floor(k / 2), ex = XC + 2.3 + col * 0.72, ey = yLay(3) - 0.15 + row * ((yLay(0) - yLay(3) + 0.3) / (NE / 2 - 1));
    const m = block({ g: "core", id: k ? `exp${k}` : "experts", x: ex, y: ey, w: 0.58, h: 0.24, d: 1.5, color: 0xf2a56f, r: 0.05, rough: 0.35, explode: -1.2,
      info: k ? null : { t: `Shared expert pool (${NE} experts)`, cat: "loop", fig: "v2moe", params: P.core_pool, kv: [["experts", `${NE} × SwiGLU 256 → ${c2.ffn.d_expert} → 256`], ["per expert", fi(P.core_expert)], ["pool", fi(P.core_pool)], ["routing", "top-2 per layer, per pass"], ["active width", c2.ffn.active_width], ["aux", "load balance 0.01, z-loss 0.001 (per window)"], ["dispatch", "capacity 1.25 batched; drops logged"]], d: "Shared by the 4 layers. Which 2 experts fire depends on the state S, so the same chunk gets a different computation on later passes (press “Play passes”)." } });
    m.material = m.material.clone(); m.material.userData.baseOpacity = 1; experts.push(m);
  }
  label(anchor([XC + 2.66, yLay(0) + 0.75, 0], "core"), `${NE} experts`, { level: "minor", sub: "shared pool · top-2" });
  for (let l = 0; l < 4; l++) flow([[XC + 1.62, yLay(l), 2.8], [XC + 1.95, yLay(l), 2.8]], 0xdc8b56, { g: "core", r: 0.03, arrow: false });
  // pass ring with 12 markers, windows of 4, detach marks
  const ringC = [XC - 2.55, (yLay(0) + yLay(3)) / 2, D / 2 + 0.9], ringR = 1.05;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(ringR, 0.045, 12, 96), pmat(0x2e9c8c, { emissive: 0x2e9c8c, ei: 0.2, clearcoat: 0.7, opacity: 0.7 }));
  ring.position.set(...ringC); tag(ring, "core"); scene.add(ring);
  const markers = [];
  for (let p = 0; p < 12; p++) {
    const a = Math.PI / 2 - (p + 0.5) * (2 * Math.PI / 12), win = Math.min(2, Math.floor(p / 4));
    const col = win === 2 ? 0xa07cc5 : win === 1 ? 0x3fb3a3 : 0x2e9c8c;
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.15, 20, 14), pmat(col, { emissive: col, ei: 0.15, clearcoat: 0.8 }));
    s.position.set(ringC[0] + ringR * Math.cos(a), ringC[1] + ringR * Math.sin(a), ringC[2]); s.castShadow = true; tag(s, "core"); scene.add(s); markers.push(s);
    if (p === 0) { parts.ring = s; s.userData.id = "ring"; s.userData.info = { t: "Pass schedule", cat: "loop", fig: "v2sched", kv: [["train", "K ~ U{8..12} passes"], ["windows", "4 passes; remainder merged into the last"], ["carry", "detached between windows (red marks)"], ["grading", "NCP on every pass"], ["final window", "+ LoopFormer consistency → u_K (purple)"], ["eval", "R = 6 (inside window 2)"]], d: "Each marker is one pass through the 4 layers. Red marks: the carry is detached, so each window must improve on the state it receives." }; pickables.push(s); }
  }
  [4, 8].forEach(p => {
    const a = Math.PI / 2 - p * (2 * Math.PI / 12);
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.5, 0.5), pmat(0xd0667a, { emissive: 0xd0667a, ei: 0.3 }));
    bar.position.set(ringC[0] + ringR * Math.cos(a), ringC[1] + ringR * Math.sin(a), ringC[2]); bar.rotation.z = a; tag(bar, "core"); scene.add(bar);
  });
  label(anchor([ringC[0], ringC[1] + ringR + 0.55, ringC[2]], "core"), "× K passes", { level: "major", sub: "8–12 train · 6 eval · detach every 4" });
  block({ g: "core", id: "carry", x: XC, y: yCar, w: W, h: 0.3, d: D, color: C.norm, label: "carry LN_h(S) · readout u_r",
    info: { t: "Carry and readout", cat: "loop", fig: "v2sched", kv: [["carry", "LN_h(S), own gain (bounded)"], ["readout", "u_r = separately normed S after pass r"], ["to decoder", "u_K of the final pass"]], d: "The carry goes around the ring; every pass's readout goes to the NCP head." } });
  flow([[XC, yLay(3) - 0.27, 0], [XC, yCar + 0.17, 0]], C.P, { g: "core", r: 0.05 });
  flow([[XC - 1.0, yCar - 0.1, D / 2 + 0.05], [ringC[0] + 0.6, ringC[1] - ringR - 0.25, ringC[2]], [ringC[0] - ringR - 0.3, ringC[1], ringC[2]], [ringC[0] + 0.6, ringC[1] + ringR + 0.25, ringC[2]], [XC - 1.0, yLay(0) + 0.1, D / 2 + 0.05]], C.P, { g: "core", r: 0.03, opacity: 0.55 });
  block({ g: "core", id: "ncp", x: XC, y: yHead, w: W, h: 0.34, d: D, color: C.conc, label: "NCP head · every pass", sub: "q = W_q u_r (256 → 384)",
    info: { t: "NCP head", cat: "loop", fig: "v2core", params: P.ncp_head, kv: [["W_q", "256 → 384"], ["logits", "−‖q_s − sg(e_s,n)‖² / τ_s"], ["used", "on every pass r (per-pass grounding)"]], d: "Every pass predicts the next chunk's 48 PQ codes: each pass has a grounded target." } });
  flow([[XC, yCar - 0.17, 0], [XC, yHead + 0.2, 0]], C.P, { g: "core", r: 0.05 });
  const S = A.S, N = A.N, gem = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.03, 0), pmat(0xe0b73a, { rough: 0.3, clearcoat: 0.6 }), S * N);
  const m4 = new THREE.Matrix4(), col = new THREE.Color(); let gk = 0;
  for (let s = 0; s < S; s++) for (let n = 0; n < N; n++) {
    m4.makeTranslation(XC - W / 2 + (n + 0.5) * W / N, yCB + 0.06, -D / 2 + (s + 0.5) * D / S); gem.setMatrixAt(gk, m4);
    const v = 0.6 + 0.4 * Math.abs(Math.sin(s * 12.9898 + n * 78.233)); col.setRGB(0.88 * v + 0.1, 0.7 * v + 0.08, 0.2 * v); gem.setColorAt(gk++, col);
  }
  tag(gem, "core"); scene.add(gem);
  block({ g: "core", id: "codebook", x: XC, y: yCB, w: W + 0.2, h: 0.06, d: D + 0.2, color: C.mem, opacity: 0.35, cast: false, label: "PQ codebook (unchanged)", sub: `${S} × ${N} codes · 8 dims`, lpos: [0, 0.1, (D + 0.2) / 2 + 0.02],
    info: { t: "PQ codebook", cat: "code", fig: "v2crc", params: P.codebook, kv: [["segments × codes × dims", `${S} × ${N} × 8`], ["bits / chunk", "336"], ["trained by", "L_VQ only"]], d: "Unchanged from v1 (measured codes_min 0.99): that is why d_c stays 384." } });
  const lossB = (id, x, y, lab, info) => block({ g: "core", id, x, y, w: 1.35, h: 0.45, d: 1.6, color: C.loss, label: lab, lpos: [0, 0.5, 0.8], info });
  lossB("lncp", XC - 3.05, yHead, "0.1 · L_NCP", { t: "NCP loss (every pass)", cat: "loss", fig: "v2loss", kv: [["form", "mean over K passes of CE(codes of chunk j+1)"], ["weight", "0.1"]], d: "Grounded per-pass target: no pass is rewarded for doing nothing." });
  lossB("lvq", XC + 3.05, yCB, "L_VQ", { t: "VQ loss", cat: "loss", fig: "v2loss", kv: [["form", "‖sg(ĉ) − e‖²"], ["updates", "codebook only"]], d: "Keeps the codes near the concepts." });
  lossB("lcons", XC - 3.05, yCar, "0.1 · consistency", { t: "LoopFormer consistency", cat: "loss", fig: "v2loss", kv: [["form", "‖sg(u_K) − u_r‖² / d"], ["scope", "passes of the final window only"], ["weight", "0.1"]], d: "The late passes are short routes distilled toward the endpoint." });
  lossB("llb", XC + 3.05, yHead, "LB + z", { t: "MoE auxiliary losses", cat: "loss", fig: "v2loss", kv: [["load balance", "0.01"], ["router z-loss", "0.001"], ["scope", "per window, with its backward"]], d: "Keep the 16 experts used (E15 check)." });
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.36, 3), pmat(C.plan, { clearcoat: 0.8, rough: 0.25, emissive: C.plan, ei: 0.15 }));
    c.position.set(XC, yPred, z); c.castShadow = true; tag(c, "core"); scene.add(c);
    if (!j) { parts.pred = c; c.userData.id = "pred"; c.userData.info = { t: "ĉ_pred and u_K", cat: "plan", fig: "v2dec", kv: [["ĉ_pred", "Σ_n π_n sg(e_n), 384-d"], ["u_K", "final-pass readout, 256-d"]], d: "The two things the decoder's plan builder receives." }; pickables.push(c); }
  });
  label(anchor([XC, yPred - 0.75, 0], "core"), "u_K · ĉ_pred → decoder", { level: "minor" });
  flow([[XC, yCB - 0.1, 0.9], [XC, yPred + 0.4, 0.9]], C.P, { g: "core", r: 0.04 });
  passAnim = { markers, experts, layers: layerMeshes };
  const yU = yCar;

  /* ---------- decoder (reads bottom → top) */
  const yPlan = 1.2, yDL = i => 2.6 + i * 1.0, yON = yDL(3) + 1.1, yH = yON + 0.9, yLog = yH + 1.1, yNTP = yLog + 1.3;
  flow([[XE - 1.95, streamTop + 0.1, 0], [XE - 1.95, streamTop + 3.4, 0], [0, streamTop + 4.6, 0], [XD - 1.95, streamTop + 3.4, 0], [XD - 1.95, yDL(3) + 0.7, 0]], C.L, { g: "dec", groups: ["enc", "dec"], r: 0.075 });
  label(anchor([0, streamTop + 5.1, 0], "dec"), "tok = zcRMSNorm(L)", { level: "major", sub: `the decoder's only view of tokens · reach ${A.reach_enc}`, group: "dec" });
  glass({ g: "dec", id: "tok", x: XD - 1.95, y: (yPlan + yDL(3) + 0.6) / 2, w: 0.12, h: yDL(3) + 0.6 - yPlan, d: D, color: C.L, label: "tok", sub: "keys & values",
    info: { t: "tok (keys and values)", cat: "stream", fig: "dec", kv: [["shape", "(B, 1024, 256)"], ["reach", A.reach_enc]], d: "Every decoder layer takes its keys and values from tok only (window 8)." } });
  block({ g: "dec", id: "plan", x: XD, y: yPlan, w: W, h: 0.42, d: D, color: C.plan, label: "Plan builder", sub: "P = W_u u_K + W_c ĉ_pred + o_r", level: "major",
    info: { t: "Plan builder", cat: "plan", fig: "plan", params: P.dec_plan, kv: [["params", fi(P.dec_plan)], ["index", "j*(t) = ⌊(t+1)/4⌋ − 1"], ["offset", "o_r(t), r = (t+1) mod 4"], ["null plan", "first 3 tokens of each document"], ["D⁰", "tok + P (ungated add)"]], d: "Each position gets the plan of the newest complete chunk; the target token is never inside it." } });
  flow([[XC + 1.7, yU, 0], [4.6, yU, 0], [6.3, yPlan + 0.1, 0], [XD - 1.75, yPlan, 0]], C.P, { g: "dec", groups: ["core", "dec"], r: 0.065 });
  flow([[XC + 0.5, yPred, 0], [4.6, yPred - 0.2, 1.4], [XD - 1.75, yPlan - 0.05, 1.4]], C.P, { g: "dec", groups: ["core", "dec"], r: 0.065 });
  label(anchor([4.9, yU + 0.45, 0], "dec"), "u", { level: "minor", group: "dec" });
  label(anchor([4.5, yPred - 0.65, 1.4], "dec"), "ĉ_pred", { level: "minor", group: "dec" });
  for (let i = 0; i < 4; i++) {
    block({ g: "dec", id: `dec${i + 1}`, x: XD, y: yDL(i), w: W, h: 0.52, d: D, color: C.dec, label: `Confined layer ${i + 1}`, sub: "keys/values from tok", explode: (i + 1) * 0.55,
      info: { t: `Decoder layer ${i + 1} · confined`, cat: "dec", fig: "dec", params: P.dec_layer, kv: [["params", fi(P.dec_layer)], ["queries", "from GR read of D (plan-aware)"], ["keys, values", "from tok only, SWA w = 8"], ["positions", "partial RoPE 16/64"], ["FFN", "SwiGLU 256 → 704 → 256"], ["reach", `${A.reach_dec} at any depth`]], d: "D is private to its position: nothing reads D at another position, so position t sees only its own plan." } });
    for (let q = -1; q <= 1; q += 2) flow([[XD - 1.9, yDL(i), q * 2.5], [XD - 1.62, yDL(i), q * 2.5]], C.L, { g: "dec", r: 0.035, arrow: false });
  }
  block({ g: "dec", id: "onorm", x: XD, y: yON, w: W, h: 0.26, d: D, color: C.norm, label: "zcRMSNorm(D)", explode: 5 * 0.55 });
  block({ g: "dec", id: "head", x: XD, y: yH, w: W, h: 0.34, d: D, color: C.head, label: "Tied output head", sub: "Eᵀ · 0 extra params", explode: 5.5 * 0.55,
    info: { t: "Output head (tied)", cat: "out", fig: "dec", kv: [["matrix", "Eᵀ, 256 × 8,192"], ["extra params", "0 (shares the embedding)"]], d: "logits = zcRMSNorm(D) · Eᵀ" } });
  for (let i = 0; i < 16; i++) {
    const h = 0.15 + 0.95 * Math.pow(Math.abs(Math.sin(i * 1.7 + 0.3)), 3);
    block({ g: "dec", x: XD, y: yLog + h / 2 - 0.3, z: zs(i), w: 0.3, h, d: 0.3, color: i === 9 ? 0xe07b39 : C.tok, r: 0.05, explode: 6 * 0.55 });
  }
  label(anchor([XD + 2.0, yLog + 0.2, 4.2], "dec"), "logits · 8,192 per position", { level: "minor", cx: 0 });
  block({ g: "dec", id: "lntp", x: XD, y: yNTP, w: 2.2, h: 0.5, d: 2.2, color: C.loss, label: "L_NTP", sub: "next-token loss", explode: 6.5 * 0.55, lpos: [0, 0.6, 1.1],
    info: { t: "Next-token loss", cat: "loss", fig: "obj", kv: [["targets", "x_{t+1}"], ["masked", "cross-document and PAD targets"], ["total", "sum of 8 weighted terms"]], d: "The main training signal; it reaches the core only through u_K and ĉ_pred of the final pass." } });
  /* NextLat (training only) beside the output norm */
  block({ g: "dec", id: "nextlat", x: XD + 2.85, y: yON + 0.2, w: 1.9, h: 0.9, d: 2.6, color: 0xb9a3ec, opacity: 0.55, explode: 5 * 0.55, label: "NextLat ψ", sub: "training only", lpos: [0, 0.75, 1.32],
    info: { t: "NextLat ψ (training only)", cat: "loss", fig: "v2nextlat", params: P.nextlat_psi, kv: [["input", "LN[h_t ; rms(sg E x_{t+1})]"], ["MLP", "512 → 512 → 512 → 256, GELU, zero-init last"], ["output", "ĥ_{t+1} = h_t + f_ψ"], ["loss", "1.0 · SmoothL1(sg h_{t+1}, ĥ) + 1.0 · KL (head frozen)"], ["horizon", "d = 1"], ["params", fi(P.nextlat_psi) + " (not used at inference)"]], d: "Predicts the decoder's next hidden state from the current one and the next token. In the confined decoder the plan is the only channel for older history, so this pressure lands on the concept stream." } });
  block({ g: "dec", id: "lnl", x: XD + 2.85, y: yNTP, w: 1.9, h: 0.5, d: 2.2, color: C.loss, label: "L_next-h + L_KL", explode: 6.5 * 0.55, lpos: [0, 0.6, 1.1],
    info: { t: "NextLat losses", cat: "loss", fig: "v2nextlat", kv: [["next-h", "SmoothL1 to sg(h_{t+1}), λ 1.0"], ["KL", "p(·|sg h_{t+1}) ‖ p(·|ĥ), λ 1.0, head frozen"], ["mask", "document boundaries"]], d: "Keep/kill rule at the screen: NTP within 0.02 nats AND a long-range metric better." } });
  flow([[XD + 1.62, yON, 0], [XD + 1.95, yON + 0.2, 0]], 0x7a5bd0, { g: "dec", r: 0.04, opacity: 0.8 });
  flow([[XD + 2.85, yON + 0.7, 0], [XD + 2.85, yNTP - 0.3, 0]], 0x7a5bd0, { g: "dec", r: 0.04, opacity: 0.8 });

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
  document.querySelectorAll("#v3b-bar [data-view]").forEach(b => b.classList.toggle("on", (b.dataset.view === "all" && !focus) || b.dataset.view === focus));
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
  c.y += sph.radius * 0.06;                                     // leave room for the toolbar at the top of the stage
  const p = c.clone().addScaledVector(new THREE.Vector3(...(dir || DIR.core)).normalize(), dist);
  if (instant) { camera.position.copy(p); controls.target.copy(c); controls.update(); requestRender(); return; }
  camTween = { t0: performance.now(), ms: 900, p0: camera.position.clone(), p1: p, q0: controls.target.clone(), q1: c };
  requestRender();
}
function groupObjs(g) { const out = []; scene.traverse(o => { if (o.isMesh && o.userData.group === g && !o.userData.groups) out.push(o); }); return out; }   // includes the floor plate
function groupInfo(g) {
  const T = A.table, P = A.params, c = A.core;
  return {
    enc: { t: "Encoder (v1 design, d 256)", cat: "enc", fig: "v2enc", params: T.encoder_non_embedding, kv: [["params", `${fi(T.encoder_non_embedding)} + ${fi(P.engram_table)} lookup`], ["layers", "8 (6 S, 2 F) + Engram"], ["Engram", `8 × ${fi(A.ecfg.engram_rows)} × ${A.ecfg.engram_dim}`], ["output", "384-d concepts + tok"]], d: "Same two-stream encoder as v1, one third narrower; the local n-gram knowledge moves to the 80M Engram table." },
    core: { t: "Concept module v2 = D6", cat: "loop", fig: "v2core", params: T.concept_module, kv: [["params", fi(T.concept_module)], ["layers", "4 distinct, full attention"], ["experts", `${c.ffn.E} shared, top-2 (active ${c.ffn.active_width})`], ["passes", "K 8–12 train (windows of 4, detached), R 6 eval"], ["share", `${(100 * T.concept_module / T.compute_params).toFixed(0)} % of compute params`]], d: "Reads top → bottom and loops around the ring: each pass runs the 4 layers, picks experts from the state, and is graded on the next chunk's code." },
    dec: { t: "Decoder (confined) + NextLat", cat: "dec", fig: "v2dec", params: T.decoder, kv: [["params", fi(T.decoder)], ["layers", "plan builder + 4 confined layers"], ["reach", `${A.reach_dec} tokens at any depth`], ["NextLat", `ψ ${fi(P.nextlat_psi)} (training only)`]], d: "Unchanged confinement; NextLat on the final hidden state pushes the plan to carry the history." },
  }[g];
}
function frameView(v, instant) {
  if (v === "all") { setFocus(null); fitTo(["enc", "core", "dec"].flatMap(groupObjs), DIR.all, 0.64, instant); if (!selected && tourIdx < 0 && built) showInfo(null); }
  else { setFocus(v); fitTo(groupObjs(v), DIR[v], 0.8, instant); if (!selected && tourIdx < 0) showInfo(groupInfo(v)); }
}

/* ------------------------------------------------------------------ labels: level of detail */
function updateLabelLOD() {
  const box = labelRenderer.domElement, d = camera.position.distanceTo(controls.target);
  box.classList.toggle("far", labelMode === "auto" && d > 40);
  box.classList.toggle("off", labelMode === "off");
}

/* ------------------------------------------------------------------ info panel */
function catChip(c) { const [n, col] = CAT[c] || ["Part", "#888"]; return `<span class="v3-cat"><i style="background:${col}"></i>${n}</span>`; }
function showInfo(info, obj) {
  const el = document.getElementById("v3b-info"); if (tourIdx >= 0) return;
  const T = A.table;
  if (!info) {
    el.innerHTML = `<div class="v3p-kicker">Explore · v2</div><h5>HRM Thinker v2 (19M) in 3D</h5>
      <p>Same three plates as v1. New: four distinct core layers, the ring of passes with detach marks, the shelf of ${A.core.ffn.E} experts, and NextLat beside the decoder's output.</p>
      <div class="v3-zones">
        <button data-zone="enc"><span class="zn" style="background:#3b78c2">1</span><b>Encoder</b><small>v1 design at d 256 · Engram 80M</small></button>
        <button data-zone="core"><span class="zn" style="background:#2e9c8c">2</span><b>Concept module (D6)</b><small>4 layers × K passes · 16 state-routed experts</small></button>
        <button data-zone="dec"><span class="zn" style="background:#e07b39">3</span><b>Decoder + NextLat</b><small>70 tokens + its own plan · belief-state pressure</small></button>
      </div>
      <div class="v3p-sub">Colour key</div>
      <div class="v3-key"><span><i style="background:#86b4ea"></i>sliding-window layer</span><span><i style="background:#f2a56f"></i>full attention / experts</span>
        <span><i style="background:#f4cd5c"></i>memory / codebook</span><span><i style="background:#4fbcad"></i>concepts &amp; plan</span>
        <span><i style="background:#ef8fa1"></i>loss</span><span><i style="background:#b9a3ec"></i>NextLat (training only)</span>
        <span><i class="ln" style="background:#3b78c2"></i>L / tok stream</span><span><i class="ln" style="background:#d0667a"></i>detach mark</span></div>
      <div class="v3-kv" style="margin-top:16px"><span>compute</span><b>${fi(T.compute_params)}</b><span>stored</span><b>${fi(T.stored_total)}</b><span>width</span><b>${A.ecfg.d_model}</b><span>experts</span><b>${A.core.ffn.E} (top-2)</b></div>
      <div class="v3p-tip">Press “Play passes” to watch the lit experts change from pass to pass. Click any block for details.</div>`;
    el.querySelectorAll("[data-zone]").forEach(b => b.onclick = () => frameView(b.dataset.zone));
    return;
  }
  const share = info.params ? info.params / T.compute_params : null;
  el.innerHTML = `${catChip(info.cat)}<h5>${info.t}</h5><p>${info.d || ""}</p>
    ${share != null ? `<div class="v3-share"><div class="v3-share-l"><span>share of compute parameters</span><b>${(100 * share).toFixed(1)} %</b></div><div class="v3-share-t"><div style="width:${Math.max(1.5, Math.min(100, 100 * share))}%"></div></div></div>` : ""}
    <div class="v3-kv">${(info.kv || []).map(([a, b]) => `<span>${a}</span><b>${b}</b>`).join("")}</div>
    <div class="v3p-actions">${obj ? `<button class="btn sm" id="v3b-fly">Zoom to it</button>` : ""}${info.fig ? `<a class="btn sm" href="#v2arch2d" data-fig2="${info.fig}">Open 2D figure →</a>` : ""}<button class="btn sm ghost" id="v3b-clear">Clear</button></div>`;
  const fly = document.getElementById("v3b-fly"); if (fly) fly.onclick = () => fitTo([obj], null, 1.4);
  document.getElementById("v3b-clear").onclick = () => select(null);
}

/* ------------------------------------------------------------------ guided tour */
const TOUR = [
  { n: "Tokens in", g: "enc", pick: "emb", objs: ["tokens", "emb"], dir: [-0.5, 0.45, 1], x: "Token ids enter at the bottom left; each looks up a 256-number vector (v1: 384). The same table is the output head at the end." },
  { n: "Same encoder, narrower", g: "enc", pick: "Lstream", objs: ["enc0", "enc7", "Lstream", "Gstream"], dir: [-0.6, 0.3, 1], x: "The v1 encoder design at d 256: 6 sliding-window and 2 full-attention layers, local stream L (reach 63) and global stream G. 7.39M parameters instead of 15.5M." },
  { n: "Engram 80M", g: "enc", pick: "engram", objs: ["engram", "engramTables"], dir: [-0.7, 0.35, 1], x: "The n-gram memory grows from 50.3M to 80.0M numbers (208,333 rows per table): more local knowledge by lookup, at the same compute." },
  { n: "384-d concepts", g: "enc", pick: "pool", objs: ["crc", "pool", "concepts"], dir: [-0.45, 0.4, 1], x: "Every 4 tokens become one concept of 384 dims on a sphere. It stays 384 so the working PQ codebook (48 × 128 × 8) is unchanged." },
  { n: "Four distinct layers", g: "core", pick: "core1", objs: ["cin", "core1", "core4"], dir: [0.35, 0.3, 1], x: "The concept module is now 4 distinct full-attention layers (v1: 1 shared layer). Each reads the clean input X̃ and the carry S through gated reads." },
  { n: "The ring of passes", g: "core", pick: "ring", objs: ["ring", "core1", "carry"], dir: [0.2, 0.3, 1], x: "One pass = the 4 layers. Training runs 8–12 passes in windows of 4; the red marks detach the carry, so every window must improve its input. Every pass is graded; the purple final window adds LoopFormer's consistency. Evaluation uses 6 passes." },
  { n: "16 experts", g: "core", pick: "experts", objs: ["experts", "core2"], dir: [0.6, 0.3, 1], x: "Each layer's router picks 2 of 16 shared experts from the state, so later passes can use different experts (the looped-MoE effect). Active width stays 1,024." },
  { n: "Grounded every pass", g: "core", pick: "ncp", objs: ["ncp", "codebook", "pred"], dir: [0.35, 0.55, 1], x: "Every pass predicts the next chunk's 48 codes (NCP). The final pass hands u_K and ĉ_pred to the decoder." },
  { n: "Confined decoder", g: "dec", pick: "dec1", objs: ["plan", "dec1", "dec4", "tok"], dir: [0.6, 0.3, 1], x: "Unchanged: each position sees the last 70 tokens (keys and values from tok) plus its own plan." },
  { n: "NextLat", g: "dec", pick: "nextlat", objs: ["onorm", "nextlat", "lnl", "lntp"], dir: [0.7, 0.3, 1], x: "Training only: ψ predicts the next hidden state from the current one and the next token. Since the plan is the decoder's only channel to older history, this pushes the concept stream to carry a belief state." },
];
function goTour(i) {
  tourIdx = i; const S = TOUR[i];
  setFocus(S.g);
  const obj = parts[S.pick]; selected = obj || null; setOutline();
  fitTo(S.objs.map(id => parts[id]).filter(Boolean), S.dir, 1.05);
  document.querySelectorAll("#tourb [data-i]").forEach(b => { b.classList.toggle("on", +b.dataset.i === i); b.classList.toggle("done", +b.dataset.i < i); });
  const info = obj ? obj.userData.info : null;
  document.getElementById("v3b-info").innerHTML = `<div class="v3p-kicker">Guided tour · ${i + 1} of ${TOUR.length}</div>
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
  document.querySelectorAll("#tourb [data-i]").forEach(b => b.classList.remove("on", "done"));
  select(null); frameView("all");
}
function setupTour() {
  const el = document.getElementById("tourb");
  el.innerHTML = `<button class="tb-tour-start" id="tb-tour-start">▶ Start guided tour</button><div class="tour-steps">` +
    TOUR.map((t, i) => `<button data-i="${i}" title="${t.n}"><span>${i + 1}</span><em>${t.n}</em></button>`).join("") + `</div>`;
  el.querySelectorAll("[data-i]").forEach(b => b.onclick = () => { stopAuto(); goTour(+b.dataset.i); });
  document.getElementById("tb-tour-start").onclick = () => { stopAuto(); goTour(0); };
}

/* ------------------------------------------------------------------ input */
function wireInput(stage, cv) {
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2(), tipEl = document.getElementById("v3b-hover");
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
    if (tourIdx >= 0) { stopAuto(); tourIdx = -1; document.querySelectorAll("#tourb [data-i]").forEach(b => b.classList.remove("on", "done")); }
    select(pick(ev));
  });
  cv.addEventListener("dblclick", ev => { const h = pick(ev); if (h) select(h, true); });
  labelRenderer.domElement.addEventListener("click", ev => {
    const z = ev.target.closest("[data-go]"); if (z && z.dataset.go.startsWith("zone:")) frameView(z.dataset.go.slice(5));
  });
  document.addEventListener("keydown", ev => {
    if (!document.getElementById("tab-v2arch3d").classList.contains("on") || /INPUT|TEXTAREA/.test(document.activeElement?.tagName)) return;
    if (ev.key === "ArrowRight") { stopAuto(); goTour(Math.min(TOUR.length - 1, tourIdx + 1)); }
    else if (ev.key === "ArrowLeft" && tourIdx > 0) { stopAuto(); goTour(tourIdx - 1); }
    else if (ev.key === "Escape") { if (tourIdx >= 0) endTour(); else { select(null); frameView("all"); } }
  });
}

/* ------------------------------------------------------------------ toolbar */
function setupBar() {
  document.querySelectorAll("#v3b-bar button").forEach(b => b.onclick = () => {
    const act = b.dataset.act;
    if (b.dataset.view) { if (tourIdx >= 0) { stopAuto(); tourIdx = -1; document.querySelectorAll("#tourb [data-i]").forEach(x => x.classList.remove("on", "done")); select(null); } frameView(b.dataset.view); }
    else if (act === "explode") { b.classList.toggle("on"); explodeTarget = b.classList.contains("on") ? 1 : 0; requestRender(); }
    else if (act === "labels") { labelMode = { auto: "all", all: "off", off: "auto" }[labelMode]; b.textContent = { auto: "Labels: auto", all: "Labels: all", off: "Labels: off" }[labelMode]; b.classList.toggle("on", labelMode !== "off"); requestRender(); }
    else if (act === "flow") { b.classList.toggle("on"); flowOn = b.classList.contains("on"); requestRender(); }
    else if (act === "passes") { b.classList.toggle("on"); b.classList.contains("on") ? startPasses() : stopPasses(); }
    else if (act === "reset") { if (tourIdx >= 0) endTour(); else { select(null); frameView("all"); } }
    else if (act === "png") { composer.render(); const a = document.createElement("a"); a.download = "hrm-thinker-v2-3d.png"; a.href = renderer.domElement.toDataURL("image/png"); a.click(); }
  });
}

/* ------------------------------------------------------------------ pass animation (illustrative routing) */
function setPass(p) {
  if (!passAnim) return;
  passAnim.markers.forEach((m, i) => { m.material.emissiveIntensity = i === p ? 1.1 : 0.15; m.scale.setScalar(i === p ? 1.45 : 1); });
  const lit = new Set(); let s = 17 + p * 131;
  for (let l = 0; l < 4; l++) for (let k = 0; k < 2; k++) { s = (s * 16807) % 2147483647; lit.add(s % passAnim.experts.length); }
  passAnim.experts.forEach((m, i) => { const on = p >= 0 && lit.has(i); m.material.color.setHex(on ? 0xe0662a : 0xf2a56f); m.material.emissive.setHex(on ? 0xe0662a : 0x000000); m.material.emissiveIntensity = on ? 0.55 : 0; });
  requestRender();
}
function startPasses() { if (passTimer) clearInterval(passTimer); passIdx = -1; passTimer = setInterval(() => { passIdx = (passIdx + 1) % 12; setPass(passIdx); }, 750); }
function stopPasses() { if (passTimer) clearInterval(passTimer); passTimer = null; passIdx = -1; setPass(-1); const b = document.querySelector('#v3b-bar [data-act="passes"]'); if (b) b.classList.remove("on"); }

function resize() {
  const st = document.getElementById("v3b-stage"), w = st.clientWidth, h = st.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); composer.setPixelRatio(renderer.getPixelRatio()); composer.setSize(w, h); labelRenderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix(); requestRender();
}
function stopFlow() {
  stopAuto(); if (passTimer) stopPasses();
  if (!flowOn) return; flowOn = false;
  const b = document.querySelector('#v3b-bar [data-act="flow"]'); if (b) b.classList.remove("on");
}
async function ensureArch() { if (!window.ARCH2) window.ARCH2 = await (await fetch("data/arch_v2.json")).json(); }

window.addEventListener("tabshown", async e => {
  if (e.detail !== "v2arch3d") { if (built) stopFlow(); return; }
  await ensureArch();
  if (!built) build(); else { resize(); requestRender(); }
});
document.addEventListener("visibilitychange", () => { if (document.hidden && built) stopFlow(); });
if (document.getElementById("tab-v2arch3d").classList.contains("on")) ensureArch().then(() => { if (!built) build(); });
