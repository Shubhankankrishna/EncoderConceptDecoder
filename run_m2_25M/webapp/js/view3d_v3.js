/* Proposed v3 3D view, adapted from the v1 scene. Three.js r169 + add-ons (vendor/jsm): physically based materials with image-based lighting (RoomEnvironment),
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
let tourIdx = -1, tourTimer = null;
let passDepth = 2, passIndex = 0;
const passNodes = [];

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
  A = window.ARCH3;
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
  setPassDepth(2);
  applyTheme();
  wireInput(stage, cv);
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(() => { applyTheme(); requestRender(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { applyTheme(); requestRender(); });
  setupBar(); setupTour(); showInfo(null);
  if (innerWidth < 800) {
    labelMode = "off";
    const labels = document.querySelector('#v3c-bar [data-act="labels"]');
    if (labels) { labels.textContent = "Labels: off"; labels.classList.remove("on"); }
  }
  resize(); frameView("all", true);
  built = true;
}

function buildModel() {
  const e = A.ecfg, P = A.params, T = A.table;
  const XE = -10, XC = 0, XD = 10, W = 3.2, D = 8.4, zs = i => -3.75 + i * 0.5;
  zonePlate(XE, 12.5, 12, 0x3b78c2, "enc", "Encoder", `${fp(T.encoder_non_embedding)} + 50.3M memory`, 1);
  zonePlate(XC, 8.2, 12, 0x2e9c8c, "core", "Concept module", fp(T.concept_module), 2);
  zonePlate(XD, 8.2, 12, 0xe07b39, "dec", "Decoder (confined)", fp(T.decoder), 3);

  /* ---------- tokens + embedding */
  for (let i = 0; i < 16; i++) block({ g: "enc", x: XE, y: -0.3, z: zs(i), w: 0.38, h: 0.38, d: 0.38, color: Math.floor(i / 4) % 2 ? C.tokB : C.tok, r: 0.07, id: i ? null : "tokens",
    info: i ? null : { t: "Tokens", cat: "input", fig: "v3whole", kv: [["drawn", "16 of 1,024 positions"], ["vocabulary", fi(e.vocab_size)], ["chunks", "4 tokens → 1 concept"]], d: "Token ids arrive at the bottom of the encoder. Alternating shades mark the 4-token chunks that later become one concept each." } });
  label(anchor([XE + 2.2, -0.3, 3.9], "enc"), "tokens · 16 of 1,024", { level: "minor", cx: 0 });
  block({ g: "enc", id: "emb", x: XE, y: 0.5, w: W, h: 0.26, d: D, color: C.emb, label: "Token embedding", sub: "8,192 × 384 · tied",
    info: { t: "Token embedding E", cat: "input", fig: "v3whole", params: T.token_embedding, kv: [["shape", `${fi(e.vocab_size)} × ${e.d_model}`], ["params", fi(T.token_embedding)], ["use", "e0 = rms(E[x]); tied with the output head"]], d: "Byte-level BPE with 8,192 tokens; digits split. The same matrix E produces the logits at the top of the decoder." } });

  /* ---------- encoder streams + layers */
  const layerY = i => 1.5 + i * 0.95, streamTop = layerY(8) + 0.9;
  glass({ g: "enc", id: "Lstream", x: XE - 1.95, y: (1.0 + streamTop) / 2, w: 0.12, h: streamTop - 1.0, d: D, color: C.L, label: "L", sub: "local · reach 63",
    info: { t: "L stream (local)", cat: "stream", fig: "v3enc", kv: [["reach", `${A.reach_enc} tokens`], ["written by", "S layers, Engram"], ["read by", "S and F layers, CRC, decoder (tok)"]], d: "The only encoder stream the decoder may read. Its reach is bounded because only sliding-window layers write to it." } });
  glass({ g: "enc", id: "Gstream", x: XE + 1.95, y: (1.0 + streamTop) / 2, w: 0.12, h: streamTop - 1.0, d: D, color: C.G, label: "G", sub: "global",
    info: { t: "G stream (global)", cat: "stream", fig: "v3enc", kv: [["reach", "unbounded (FULL layers)"], ["written by", "S layers, F layers, Engram"], ["read by", "F layers, CRC → concepts"]], d: "Carries information from the whole document, but reaches the decoder only through the concept pooling and the concept module." } });
  const rows = [["S", 0], ["E"], ["S", 1], ["S", 2], ["F", 3], ["S", 4], ["S", 5], ["S", 6], ["F", 7]];
  rows.forEach((r, i) => {
    let info, color, lab, sub;
    if (r[0] === "S") {
      color = C.swa; lab = `S${r[1]}`; sub = "sliding window";
      info = { t: `Encoder layer ${r[1]} · S (sliding window)`, cat: "enc", fig: "v3enc", params: P.enc_layer_S, kv: [["params", fi(P.enc_layer_S)], ["reads", "{L}"], ["writes", "{L, G}"], ["attention", `SWA w = ${e.window}, partial RoPE ${e.rope_dims}/${e.head_dim}, GQA ${e.n_heads}:${e.n_kv_heads}`], ["q/k/v conv", `kernel ${e.conv_kernel}, doc-aware`], ["FFN", `SwiGLU ${e.d_model} → ${e.d_ff} → ${e.d_model}`], ["reach added", "(w−1)+(k−1) = 10"]], d: "Gated-residual read of L → gated attention → per-stream write scales; then the same for the FFN." };
    } else if (r[0] === "F") {
      color = C.full; lab = `F${r[1]}`; sub = "full attention";
      info = { t: `Encoder layer ${r[1]} · F (full attention)`, cat: "full", fig: "v3enc", params: P.enc_layer_F, kv: [["params", fi(P.enc_layer_F)], ["reads", "{L, G}"], ["writes", "{G} only"], ["attention", "causal within the document, NoPE, per-head τ"]], d: "Global layer. Because it never writes L, the decoder's token path keeps its 63-token reach." };
    } else {
      color = C.mem; lab = "Engram"; sub = "n-gram memory";
      info = { t: "Engram (hashed n-gram memory)", cat: "mem", fig: "v3engram", params: P.engram_proj, kv: [["table", `${fi(P.engram_table)} params (lookup)`], ["compute", fi(P.engram_proj)], ["orders", e.engram_orders.join(", ")], ["heads", e.engram_heads + " per order"], ["tables", `8 × ${fi(e.engram_rows)} × ${e.engram_dim}`], ["gate", "α = σ(rms(h)·rms(k)/√d)"], ["writes", "{L, G}"]], d: "Canonical-id n-grams are hashed into 8 tables; the looked-up vector is gated by agreement with the hidden state and added to both streams." };
    }
    block({ g: "enc", id: r[0] === "E" ? "engram" : `enc${r[1]}`, x: XE, y: layerY(i), w: W, h: 0.46, d: D, color, label: lab, sub, explode: i * 0.55, info });
  });
  for (let t = 0; t < 8; t++) block({ g: "enc", id: t ? null : "engramTables", x: XE - 4.3 - (t % 4) * 0.36, y: layerY(1) + 0.25, z: -2.2 + Math.floor(t / 4) * 4.4, w: 0.24, h: 2.4, d: 1.6, color: C.mem, r: 0.06, explode: 0.55,
    info: t ? null : { t: "Engram tables", cat: "mem", fig: "v3engram", params: P.engram_table, kv: [["count", "8 (orders 2, 3 × 4 heads)"], ["rows × dim", `${fi(e.engram_rows)} × ${e.engram_dim}`], ["params", fi(P.engram_table)]], d: "One row is read from each table per token; the table is a lookup, not compute, so its 50.3M numbers cost almost nothing to use." } });
  const engA = anchor([XE - 4.85, layerY(1) + 1.85, 0], "enc"); explodables.push({ obj: engA, base: engA.position.y, k: 0.55 });
  label(engA, "8 Engram tables", { level: "minor", sub: "50.3M lookups" });
  flow([[XE - 3.95, layerY(1), 0], [XE - 1.7, layerY(1), 0]], C.mem, { g: "enc", r: 0.06 });
  const yCRC = layerY(8) + 1.25, yPool = yCRC + 0.8, yC = yPool + 1.0;
  block({ g: "enc", id: "crc", x: XE, y: yCRC, w: W, h: 0.34, d: D, color: C.gate, label: "CRC depth-mix", sub: "e0 · L · G", explode: 9 * 0.55,
    info: { t: "CRC depth-mix", cat: "concept", fig: "v3concept", params: P.crc_router, kv: [["params", fi(P.crc_router)], ["mix", "softmax(W rms(G)) over {e0, rms L, rms G}"], ["init", "0.107 / 0.107 / 0.787"]], d: "A per-token router decides how much of the embedding, local and global state goes into the concept." } });
  block({ g: "enc", id: "pool", x: XE, y: yPool, w: W, h: 0.34, d: D, color: C.conc, label: "Concat-pool 4 → 1", sub: "1,536 → 384", explode: 9.3 * 0.55,
    info: { t: "Concept pooling", cat: "concept", fig: "v3concept", params: P.pool, kv: [["W_pool", `${e.chunk * e.d_model} → ${e.d_model}`], ["params", fi(P.pool)], ["init", "mean pooling"], ["output", "ĉ = √384 · c/‖c‖"]], d: "Four consecutive mixed token states are concatenated and projected, then put on the sphere of radius √384." } });
  const cz = [-3, -1, 1, 3];
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 3), pmat(C.conc, { clearcoat: 0.8, rough: 0.25, emissive: C.conc, ei: 0.12 }));
    c.position.set(XE, yC, z); c.castShadow = true; tag(c, "enc"); scene.add(c); explodables.push({ obj: c, base: yC, k: 9.6 * 0.55 });
    if (!j) { parts.concepts = c; c.userData.id = "concepts"; c.userData.info = { t: "Concepts ĉ", cat: "concept", fig: "v3concept", kv: [["count", "1 per 4 tokens (256 per row)"], ["dim", e.d_model], ["norm", `√${e.d_model} = ${Math.sqrt(e.d_model).toFixed(2)}`]], d: "Concepts live on a sphere (drawn as spheres): only their direction carries information. This sequence X is handed to the concept module." }; pickables.push(c); }
    for (let k = 0; k < 4; k++) flow([[XE, yPool + 0.2, zs(4 * j + k)], [XE, yPool + 0.55, zs(4 * j + k) * 0.6 + z * 0.4], [XE, yC - 0.4, z]], C.P, { g: "enc", r: 0.018, opacity: 0.6, arrow: false });
  });
  const cA = anchor([XE, yC + 0.85, 0], "enc"); explodables.push({ obj: cA, base: cA.position.y, k: 9.6 * 0.55 });
  label(cA, "ĉ · concepts", { level: "major", sub: "on a sphere of radius √384" });

  /* ---------- concept module (reads top → bottom) */
  const yX = yC, yLoop = yX - 2.4, yU = yLoop - 1.7, yHead = yU - 1.4, yCB = yHead - 1.5, yPred = yCB - 1.4;
  flow([[XE + 0.6, yC, 0], [-5, yC + 1.0, 0], [XC - 2.1, yX, 0]], C.P, { g: "core", groups: ["enc", "core"], r: 0.075 });
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.4, 3), pmat(C.conc, { clearcoat: 0.8, rough: 0.25, emissive: C.conc, ei: 0.12 }));
    c.position.set(XC, yX, z); c.castShadow = true; tag(c, "core"); scene.add(c);
    if (!j) { parts.X = c; c.userData.id = "X"; c.userData.info = { t: "X = ĉ", cat: "loop", fig: "v3scratch", kv: [["role", "input of every loop pass"]], d: "Re-read by every pass; gradients flow back into the encoder." }; pickables.push(c); }
  });
  label(anchor([XC, yX + 0.8, 0], "core"), "C = ĉ · evidence", { level: "minor" });
  block({ g: "core", id: "zero", x: XC - 3.0, y: yLoop + 0.85, w: 1.45, h: 0.44, d: 1.65, color: C.tok, label: "S₀ = 0", lpos: [0, 0.52, 0.8],
    info: { t: "Zero scratchpad S₀", cat: "loop", fig: "v3scratch", kv: [["shape", "B × 256 × 384"], ["initial values", "all zero"], ["first read", "0.5 σ(2) n(C), nonzero"]], d: "S starts empty. The separate concept stream C supplies evidence on the first pass and remains available on later passes." } });
  block({ g: "core", id: "loop", x: XC, y: yLoop, w: W, h: 0.72, d: D, color: C.full, label: "Shared FULL layer", sub: "applied R times", level: "major",
    info: { t: "Shared dense scratchpad block", cat: "loop", fig: "v3scratch", params: P.core_layer, kv: [["params", fi(P.core_layer)], ["reads", "{C, S}"], ["writes", "S with identity residuals"], ["depth", "K2 / K4 train; fixed R2 / R4 inference"], ["gradient", "full through all passes"], ["attention", "causal, same document, NoPE"]], d: "The same dense layer rereads C and updates S at each pass. This reuses weights but does not detach earlier states. The visualization is a proposed computation graph, not learned v3 behavior." } });
  const ringMat = pmat(0x2e9c8c, { emissive: 0x2e9c8c, ei: 0.25, clearcoat: 0.7 });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.85, 0.07, 16, 64, Math.PI * 1.65), ringMat);
  ring.position.set(XC + 2.55, yLoop, 0); ring.rotation.z = Math.PI * 0.2; ring.castShadow = true; tag(ring, "core"); scene.add(ring);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.17, 0.4, 16), ringMat);
  const a = Math.PI * 0.2 + Math.PI * 1.65; tip.position.set(XC + 2.55 + 0.85 * Math.cos(a), yLoop + 0.85 * Math.sin(a), 0); tip.rotation.z = a; tag(tip, "core"); scene.add(tip);
  label(anchor([XC + 2.55, yLoop + 1.35, 0], "core"), "× R", { level: "major", sub: "R2 / R4 · shared" });
  for (let r = 0; r < 4; r++) {
    const node = block({ g: "core", id: `pass${r + 1}`, x: XC + 4.15, y: yLoop + 0.75 - 0.46 * r, w: 0.44, h: 0.32, d: 0.72, color: C.conc,
      info: { t: `Illustrative pass ${r + 1}`, cat: "loop", fig: "v3depth", kv: [["reads", "C and previous S"], ["writes", `S${r + 1}`], ["weights", "same dense block every pass"], ["gradient", "all executed passes connected"]], d: "The highlight marks an operation in the proposed graph. It is not a measured activation or a prediction that this pass improves an answer." } });
    passNodes.push(node);
  }
  block({ g: "core", id: "u", x: XC, y: yU, w: W, h: 0.3, d: D, color: C.norm, label: "u = zcRMSNorm(Sᴿ)",
    info: { t: "u (post-loop state)", cat: "loop", fig: "v3scratch", kv: [["shape", "(B, 256, 384)"]], d: "Post-loop state per chunk; sent to the decoder's plan builder." } });
  block({ g: "core", id: "ncp", x: XC, y: yHead, w: W, h: 0.34, d: D, color: C.conc, label: "NCP head", sub: "q = W_q u",
    info: { t: "NCP head", cat: "loop", fig: "v3scratch", params: P.ncp_Wq, kv: [["W_q", `384 × 384 = ${fi(P.ncp_Wq)}`], ["τ", `${P.ncp_tau} (one per segment, init 16)`], ["logits", "−‖q_s − e_s,n‖² / τ_s"]], d: "A distance head over the product-quantised vocabulary: it predicts which code each of the 48 dictionaries will pick for the next concept." } });
  // codebook: all 6,144 codes as instanced gems
  const S = A.S, N = A.N, gem = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.03, 0), pmat(0xe0b73a, { rough: 0.3, clearcoat: 0.6 }), S * N);
  const m4 = new THREE.Matrix4(), col = new THREE.Color();
  let k = 0;
  for (let s = 0; s < S; s++) for (let n = 0; n < N; n++) {
    m4.makeTranslation(XC - W / 2 + (n + 0.5) * W / N, yCB + 0.06, -D / 2 + (s + 0.5) * D / S); gem.setMatrixAt(k, m4);
    const v = 0.6 + 0.4 * Math.abs(Math.sin(s * 12.9898 + n * 78.233)); col.setRGB(0.88 * v + 0.1, 0.7 * v + 0.08, 0.2 * v); gem.setColorAt(k++, col);
  }
  tag(gem, "core"); scene.add(gem);
  block({ g: "core", id: "codebook", x: XC, y: yCB, w: W + 0.2, h: 0.06, d: D + 0.2, color: C.mem, opacity: 0.35, cast: false, label: "PQ codebook", sub: `${S} × ${N} codes · ${e.d_model / S} dims`, lpos: [0, 0.1, (D + 0.2) / 2 + 0.02],
    info: { t: "PQ vocabulary (codebook)", cat: "code", fig: "v3scratch", params: P.codebook, kv: [["segments", S], ["codes / segment", N], ["dims / code", e.d_model / S], ["params", fi(P.codebook)], ["init", "k-means++ on complete real training chunks"], ["maintenance", "unused codes checked every 100 optimizer steps"], ["trained by", "L_VQ only"]], d: "All 6,144 code points are drawn. NCP predicts next-chunk codes from the final scratchpad state; only complete real chunks fit the codebook." } });
  block({ g: "core", id: "lvq", x: XC + 3.05, y: yCB, w: 1.3, h: 0.5, d: 1.6, color: C.loss, label: "L_VQ", lpos: [0, 0.55, 0.8],
    info: { t: "VQ loss", cat: "loss", fig: "v3loss", kv: [["form", "‖sg(ĉ) − e_{n*}‖²"], ["chunks", "complete real only"], ["updates", "codebook only"]], d: "Keeps codes near real, complete concepts. Its optimizer-batch denominator is shared across microbatches." } });
  block({ g: "core", id: "lncp", x: XC - 3.05, y: yHead, w: 1.3, h: 0.5, d: 1.6, color: C.loss, label: "L_NCP × 0.1", lpos: [0, 0.55, 0.8],
    info: { t: "Terminal NCP loss", cat: "loss", fig: "v3loss", kv: [["form", "CE(logits_j, codes(ĉ_{j+1}))"], ["weight", "0.1"], ["depth", "final only"], ["mask", "same document; complete real target chunk"]], d: "The final scratchpad predicts the next concept's codes. Intermediate passes are trained through final task gradients." } });
  cz.forEach((z, j) => {
    const c = new THREE.Mesh(new THREE.IcosahedronGeometry(0.36, 3), pmat(C.plan, { clearcoat: 0.8, rough: 0.25, emissive: C.plan, ei: 0.15 }));
    c.position.set(XC, yPred, z); c.castShadow = true; tag(c, "core"); scene.add(c);
    if (!j) { parts.pred = c; c.userData.id = "pred"; c.userData.info = { t: "ĉ_pred", cat: "plan", fig: "v3scratch", kv: [["form", "Σ_n π_n · sg(e_n) per segment"]], d: "The prediction of the next chunk's concept, used by the decoder's plan." }; pickables.push(c); }
  });
  label(anchor([XC, yPred - 0.75, 0], "core"), "ĉ_pred · predicted next concept", { level: "minor" });
  flow([[XC, yX - 0.45, 0], [XC, yLoop + 0.4, 0]], C.P, { g: "core", r: 0.05 });
  flow([[XC, yLoop - 0.4, 0], [XC, yU + 0.18, 0]], C.P, { g: "core", r: 0.05 });
  flow([[XC, yU - 0.18, 0], [XC, yHead + 0.2, 0]], C.P, { g: "core", r: 0.05 });
  flow([[XC, yCB - 0.1, 0.9], [XC, yPred + 0.4, 0.9]], C.P, { g: "core", r: 0.04 });

  /* ---------- decoder (reads bottom → top) */
  const yPlan = 1.2, yDL = i => 2.6 + i * 1.0, yON = yDL(3) + 1.1, yH = yON + 0.9, yLog = yH + 1.1, yNTP = yLog + 1.3;
  flow([[XE - 1.95, streamTop + 0.1, 0], [XE - 1.95, streamTop + 3.4, 0], [0, streamTop + 4.6, 0], [XD - 1.95, streamTop + 3.4, 0], [XD - 1.95, yDL(3) + 0.7, 0]], C.L, { g: "dec", groups: ["enc", "dec"], r: 0.075 });
  label(anchor([0, streamTop + 5.1, 0], "dec"), "tok = zcRMSNorm(L)", { level: "major", sub: `the decoder's only view of tokens · reach ${A.reach_enc}`, group: "dec" });
  glass({ g: "dec", id: "tok", x: XD - 1.95, y: (yPlan + yDL(3) + 0.6) / 2, w: 0.12, h: yDL(3) + 0.6 - yPlan, d: D, color: C.L, label: "tok", sub: "keys & values",
    info: { t: "tok (keys and values)", cat: "stream", fig: "v3decoder", kv: [["shape", "(B, 1024, 384)"], ["reach", A.reach_enc]], d: "Every decoder layer takes its keys and values from tok only (window 8)." } });
  block({ g: "dec", id: "plan", x: XD, y: yPlan, w: W, h: 0.42, d: D, color: C.plan, label: "Plan builder", sub: "P = W_u u + W_c ĉ_pred + o_r", level: "major",
    info: { t: "Plan builder", cat: "plan", fig: "v3plan", params: P.dec_plan, kv: [["params", fi(P.dec_plan)], ["index", "j*(t) = ⌊(t+1)/4⌋ − 1"], ["offset", "o_r(t), r = (t+1) mod 4"], ["null plan", "first 3 tokens of each document"], ["D⁰", "tok + P (ungated add)"]], d: "Each position gets the plan of the newest complete chunk; the target token is never inside it." } });
  flow([[XC + 1.7, yU, 0], [4.6, yU, 0], [6.3, yPlan + 0.1, 0], [XD - 1.75, yPlan, 0]], C.P, { g: "dec", groups: ["core", "dec"], r: 0.065 });
  flow([[XC + 0.5, yPred, 0], [4.6, yPred - 0.2, 1.4], [XD - 1.75, yPlan - 0.05, 1.4]], C.P, { g: "dec", groups: ["core", "dec"], r: 0.065 });
  label(anchor([4.9, yU + 0.45, 0], "dec"), "u", { level: "minor", group: "dec" });
  label(anchor([4.5, yPred - 0.65, 1.4], "dec"), "ĉ_pred", { level: "minor", group: "dec" });
  for (let i = 0; i < 4; i++) {
    block({ g: "dec", id: `dec${i + 1}`, x: XD, y: yDL(i), w: W, h: 0.52, d: D, color: C.dec, label: `Confined layer ${i + 1}`, sub: "keys/values from tok", explode: (i + 1) * 0.55,
      info: { t: `Decoder layer ${i + 1} · confined`, cat: "dec", fig: "v3decoder", params: P.dec_layer, kv: [["params", fi(P.dec_layer)], ["queries", "from GR read of D (plan-aware)"], ["keys, values", "from tok only, SWA w = 8"], ["positions", "partial RoPE 16/64"], ["FFN", "SwiGLU 384 → 1056 → 384"], ["reach", `${A.reach_dec} at any depth`]], d: "D is private to its position: nothing reads D at another position, so position t sees only its own plan." } });
    for (let q = -1; q <= 1; q += 2) flow([[XD - 1.9, yDL(i), q * 2.5], [XD - 1.62, yDL(i), q * 2.5]], C.L, { g: "dec", r: 0.035, arrow: false });
  }
  block({ g: "dec", id: "onorm", x: XD, y: yON, w: W, h: 0.26, d: D, color: C.norm, label: "zcRMSNorm(D)", explode: 5 * 0.55 });
  block({ g: "dec", id: "head", x: XD, y: yH, w: W, h: 0.34, d: D, color: C.head, label: "Tied output head", sub: "Eᵀ · 0 extra params", explode: 5.5 * 0.55,
    info: { t: "Output head (tied)", cat: "out", fig: "v3decoder", kv: [["matrix", "Eᵀ, 384 × 8,192"], ["extra params", "0 (shares the embedding)"]], d: "logits = zcRMSNorm(D) · Eᵀ" } });
  for (let i = 0; i < 16; i++) {
    const h = 0.15 + 0.95 * Math.pow(Math.abs(Math.sin(i * 1.7 + 0.3)), 3);
    block({ g: "dec", x: XD, y: yLog + h / 2 - 0.3, z: zs(i), w: 0.3, h, d: 0.3, color: i === 9 ? 0xe07b39 : C.tok, r: 0.05, explode: 6 * 0.55 });
  }
  label(anchor([XD + 2.0, yLog + 0.2, 4.2], "dec"), "logits · 8,192 per position", { level: "minor", cx: 0 });
  block({ g: "dec", id: "lntp", x: XD, y: yNTP, w: 2.2, h: 0.5, d: 2.2, color: C.loss, label: "L_token", sub: "next-token CE", explode: 6.5 * 0.55, lpos: [0, 0.6, 1.1],
    info: { t: "Token loss", cat: "loss", fig: "v3loss", kv: [["targets", "x_{t+1}"], ["masked", "cross-document and PAD targets"], ["normalization", "valid tokens in full optimizer batch"]], d: "The main language loss. EOS stays a valid target. Its denominator covers the entire accumulated batch." } });
  block({ g: "dec", id: "lanswer", x: XD + 3.0, y: yNTP, w: 1.8, h: 0.5, d: 1.9, color: C.loss, label: "L_answer", sub: "λ ≤ 0.1", explode: 6.5 * 0.55, lpos: [0, 0.6, 0.95],
    info: { t: "Annotated answer-span loss", cat: "loss", fig: "v3loss", kv: [["targets", "correct answer tokens"], ["averaging", "tokens → queries → documents"], ["weight", "0.1 min(1, step/300)"], ["metadata", "verified sidecars required"]], d: "This reweights correct token labels in synthetic answers. It adds no decoder head. The sidecars still need to be constructed and tested." } });
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
  c.y += sph.radius * 0.06;                                     // leave room for the toolbar at the top of the stage
  const p = c.clone().addScaledVector(new THREE.Vector3(...(dir || DIR.core)).normalize(), dist);
  if (instant) { camera.position.copy(p); controls.target.copy(c); controls.update(); requestRender(); return; }
  camTween = { t0: performance.now(), ms: 900, p0: camera.position.clone(), p1: p, q0: controls.target.clone(), q1: c };
  requestRender();
}
function groupObjs(g) { const out = []; scene.traverse(o => { if (o.isMesh && o.userData.group === g && !o.userData.groups) out.push(o); }); return out; }   // includes the floor plate
function groupInfo(g) {
  const T = A.table, P = A.params;
  return {
    enc: { t: "Encoder v1.0", cat: "enc", fig: "v3enc", params: T.encoder_non_embedding, kv: [["params", `${fi(T.encoder_non_embedding)} + ${fi(P.engram_table)} lookup`], ["layers", "8 (6 sliding-window S, 2 full F) + Engram"], ["streams", `L (local, reach ${A.reach_enc}) and G (global)`], ["output", "1 concept per 4 tokens + tok for the decoder"]], d: "Reads the tokens bottom → top with two streams, consults the n-gram memory, then mixes and pools every 4 tokens into one concept." },
    core: { t: "Proposed v3 scratchpad", cat: "loop", fig: "v3scratch", params: T.concept_module - T.codebook, kv: [["compute params", fi(T.concept_module - T.codebook)], ["start", "S₀ = 0"], ["block", "one shared dense layer"], ["depth", "R2/R4; full backprop"], ["output", "u and ĉ_pred → plan"]], d: "The same block rereads C and updates S each pass. A terminal readout and predicted PQ codes form the plan. The highlighted passes are illustrative." },
    dec: { t: "Proposed v3 decoder", cat: "dec", fig: "v3decoder", params: T.decoder, kv: [["params", fi(T.decoder)], ["layers", "plan builder + 4 confined layers"], ["reach", `${A.reach_dec} tokens`], ["loss", "token + answer-span CE"]], d: "Each position combines its private plan with local token features. Long-range evidence must arrive through the plan." },
  }[g];
}
function frameView(v, instant) {
  if (v === "all") { setFocus(null); fitTo(["enc", "core", "dec"].flatMap(groupObjs), DIR.all, innerWidth < 800 ? 1.15 : 0.64, instant); if (!selected && tourIdx < 0 && built) showInfo(null); }
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
  const el = document.getElementById("v3c-info"); if (tourIdx >= 0) return;
  const T = A.table;
  if (!info) {
    el.innerHTML = `<div class="v3p-kicker">Proposed · untrained</div><h5>HRM Thinker v3 in 3D</h5>
      <p>Three modules on three floor plates. Select R2 or R4 and step through the shared scratchpad updates. Highlights illustrate the graph, not learned activations.</p>
      <div class="v3-zones">
        <button data-zone="enc"><span class="zn" style="background:#3b78c2">1</span><b>Encoder</b><small>reads tokens with a local and a global stream; pools every 4 tokens into a concept</small></button>
        <button data-zone="core"><span class="zn" style="background:#2e9c8c">2</span><b>Scratchpad S</b><small>zero start, C reread each pass, full gradient through R2/R4</small></button>
        <button data-zone="dec"><span class="zn" style="background:#e07b39">3</span><b>Decoder</b><small>sees only the last 70 tokens plus its plan; writes the next token</small></button>
      </div>
      <div class="v3p-sub">Colour key</div>
      <div class="v3-key"><span><i style="background:#86b4ea"></i>sliding-window layer</span><span><i style="background:#f2a56f"></i>full attention</span>
        <span><i style="background:#f4cd5c"></i>memory / codebook</span><span><i style="background:#4fbcad"></i>concepts &amp; plan</span>
        <span><i style="background:#ef8fa1"></i>loss</span><span><i style="background:#a996de"></i>embedding / head</span>
        <span><i class="ln" style="background:#3b78c2"></i>L / tok stream</span><span><i class="ln" style="background:#e07b39"></i>G stream</span></div>
      <div class="v3-kv" style="margin-top:16px"><span>compute</span><b>${fi(T.compute_params)}</b><span>stored</span><b>${fi(T.stored_total)}</b><span>width</span><b>${A.ecfg.d_model}</b><span>decoder reach</span><b>${A.reach_dec} tokens</b></div>
      <div class="v3p-tip">Click any block for details, double-click to zoom, or start the guided tour. The scene renders on interaction; Animate flow is optional.</div>`;
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
  { n: "Tokens in", g: "enc", pick: "emb", objs: ["tokens", "emb"], dir: [-0.5, 0.45, 1], x: "Text arrives as token ids. 16 of the 1,024 positions are drawn (4 chunks of 4, alternating shades). Each id looks up a 384-number vector in the embedding table (purple), the same table that produces the output at the very end." },
  { n: "Two streams", g: "enc", pick: "Lstream", objs: ["enc0", "enc7", "Lstream", "Gstream"], dir: [-0.6, 0.3, 1], x: "8 layers stack upward. Blue slabs = sliding-window attention (sees 8 tokens); orange = full attention (whole document). The glass walls are the two streams: blue L (local, reach 63) and orange G (global). Sliding layers write both; full layers write only G." },
  { n: "Engram memory", g: "enc", pick: "engram", objs: ["engram", "engramTables"], dir: [-0.7, 0.35, 1], x: "After layer 0, a hashed memory of 2- and 3-token combinations is consulted: 8 tables, 50.3M numbers. A gate decides how much of each looked-up vector to add. It is a lookup, so it costs almost no compute." },
  { n: "Make concepts", g: "enc", pick: "pool", objs: ["crc", "pool", "concepts"], dir: [-0.45, 0.4, 1], x: "Each token mixes its embedding, local and global states (lavender), then every 4 tokens are pooled into one concept (teal spheres): 256 concepts per 1,024-token row, all on a sphere of radius √384." },
  { n: "Zero scratchpad", g: "core", pick: "zero", objs: ["X", "zero", "loop"], dir: [0.4, 0.3, 1], x: "The writable S starts as exact zero. C is a separate, nonzero read-only evidence stream. The first shared pass reads C and writes a useful state into S." },
  { n: "Repeat the shared block", g: "core", pick: "loop", objs: ["loop", "pass1", "pass4", "u"], dir: [0.4, 0.3, 1], x: "Ordinary training uses two passes 75% of the time and four passes 25%; designated multistep batches use four. The same dense weights are reused. All executed passes receive final task gradients. Use R2 and R4 controls above to illustrate the graph." },
  { n: "Predict the next concept", g: "core", pick: "codebook", objs: ["ncp", "codebook", "pred"], dir: [0.35, 0.55, 1], x: "Each gold point is one code: 48 dictionaries × 128 codes. The NCP head predicts which code each dictionary picks for the NEXT concept; the prediction ĉ_pred (teal spheres) goes to the decoder." },
  { n: "Build the plan", g: "dec", pick: "plan", objs: ["plan", "dec1"], dir: [0.5, 0.35, 1], x: "u (after the loop) and ĉ_pred flow to the decoder's plan builder. Position t gets the plan of the newest complete chunk, so the plan never contains the token being predicted." },
  { n: "Confined decoder", g: "dec", pick: "dec1", objs: ["dec1", "dec4", "tok"], dir: [0.6, 0.3, 1], x: "4 decoder layers. Queries come from each position's own plan-aware state; keys and values come only from tok (the blue wall, last 70 tokens). No position sees another's plan, so long-range knowledge must come through the concepts." },
  { n: "Direct answer credit", g: "dec", pick: "lanswer", objs: ["head", "lntp", "lanswer"], dir: [0.55, 0.35, 1], x: "The tied head scores 8,192 tokens. The proposed loss adds a small ramped weight to annotated correct answer spans, alongside ordinary token loss, terminal next-concept prediction and codebook fitting. No v3 learning result is claimed." },
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
    <div class="v3p-actions"><button class="btn sm" id="tour-prev" ${i ? "" : "disabled"}>← Back</button>
      <button class="btn sm primary" id="tour-next">${i === TOUR.length - 1 ? "Finish" : "Next →"}</button>
      <button class="btn sm ghost" id="tour-play">${tourTimer ? "❚❚ Pause" : "▶ Autoplay"}</button><button class="btn sm ghost" id="tour-end">Exit</button></div>
    <div class="v3p-tip">Keyboard: ← → to move, Esc to exit.</div>`;
  document.getElementById("tour-prev").onclick = () => { stopAuto(); i > 0 && goTour(i - 1); };
  document.getElementById("tour-next").onclick = () => { stopAuto(); i < TOUR.length - 1 ? goTour(i + 1) : endTour(); };
  document.getElementById("tour-play").onclick = () => { tourTimer ? stopAuto() : startAuto(); goTour(tourIdx); };
  document.getElementById("tour-end").onclick = endTour;
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
  el.innerHTML = `<button class="tour-start" id="tourc-start">▶ Start guided tour</button><div class="tour-steps">` +
    TOUR.map((t, i) => `<button data-i="${i}" title="${t.n}"><span>${i + 1}</span><em>${t.n}</em></button>`).join("") + `</div>`;
  el.querySelectorAll("[data-i]").forEach(b => b.onclick = () => { stopAuto(); goTour(+b.dataset.i); });
  document.getElementById("tourc-start").onclick = () => { stopAuto(); goTour(0); };
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
function paintPasses() {
  passNodes.forEach((node,i) => {
    node.visible = i < passDepth;
    node.material.color.setHex(i === passIndex - 1 ? 0x168b76 : 0x9dd6ca);
    node.material.emissive.setHex(i === passIndex - 1 ? 0x168b76 : 0x000000);
    node.material.emissiveIntensity = i === passIndex - 1 ? 0.65 : 0;
  });
  const step=document.querySelector('#v3c-bar [data-act="passes"]');
  if(step)step.textContent=passIndex ? `Step passes · S${passIndex}/${passDepth}` : `Step passes · S₀/${passDepth}`;
  requestRender();
}
function setPassDepth(n) {
  passDepth=n;passIndex=0;
  document.querySelectorAll('#v3c-bar [data-depth]').forEach(b=>b.classList.toggle('on',+b.dataset.depth===n));
  paintPasses();
}
function setupBar() {
  document.querySelectorAll("#v3c-bar button").forEach(b => b.onclick = () => {
    const act = b.dataset.act;
    if (b.dataset.depth) setPassDepth(+b.dataset.depth);
    else if (b.dataset.view) { if (tourIdx >= 0) { stopAuto(); tourIdx = -1; document.querySelectorAll("#tourc [data-i]").forEach(x => x.classList.remove("on", "done")); select(null); } frameView(b.dataset.view); }
    else if (act === "passes") { passIndex=(passIndex+1)%(passDepth+1);paintPasses(); }
    else if (act === "explode") { b.classList.toggle("on"); explodeTarget = b.classList.contains("on") ? 1 : 0; requestRender(); }
    else if (act === "labels") { labelMode = { auto: "all", all: "off", off: "auto" }[labelMode]; b.textContent = { auto: "Labels: auto", all: "Labels: all", off: "Labels: off" }[labelMode]; b.classList.toggle("on", labelMode !== "off"); requestRender(); }
    else if (act === "flow") { b.classList.toggle("on"); flowOn = b.classList.contains("on"); requestRender(); }
    else if (act === "reset") { if (tourIdx >= 0) endTour(); else { select(null); frameView("all"); } }
    else if (act === "png") { composer.render(); const a = document.createElement("a"); a.download = "hrm-thinker-v3-proposed-3d.png"; a.href = renderer.domElement.toDataURL("image/png"); a.click(); }
  });
}

function resize() {
  const st = document.getElementById("v3c-stage"), w = st.clientWidth, h = st.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); composer.setPixelRatio(renderer.getPixelRatio()); composer.setSize(w, h); labelRenderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix(); requestRender();
}
function stopFlow() {
  stopAuto();
  if (!flowOn) return; flowOn = false;
  const b = document.querySelector('#v3c-bar [data-act="flow"]'); if (b) b.classList.remove("on");
}
async function ensureArch() { if (!window.ARCH3) window.ARCH3 = await (await fetch("data/arch_v3.json")).json(); }

window.addEventListener("tabshown", async e => {
  if (e.detail !== "v3arch3d") { if (built) stopFlow(); return; }
  await ensureArch();
  if (!built) build(); else { resize(); requestRender(); }
});
document.addEventListener("visibilitychange", () => { if (document.hidden && built) stopFlow(); });
if (document.getElementById("tab-v3arch3d").classList.contains("on")) ensureArch().then(() => { if (!built) build(); });
