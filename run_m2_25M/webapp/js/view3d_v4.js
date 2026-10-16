/* 3D model of the v4 DRAFT (draft_v4.md r2), derived from js/view3d_v3d.js (same engine, same colours and controls).
   Reads data/arch_v4.json (exact counts from model_v4.py). Ids: #v4c, #v4c-stage, #v4c-bar, #v4c-info, #v4c-hover, #tour4. */
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
  window.__v4cframes = (window.__v4cframes || 0) + 1;          // frame counter (verifies on-demand rendering)
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
  A = window.ARCH4;
  const stage = document.getElementById("v4c-stage"), cv = document.getElementById("v4c");
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
  const e = A.ecfg, P = A.params, T = A.table, c4 = A.core;
  const XE = -10, XC = 0, XD = 10, W = 3.2, D = 8.4, NT = 18, zs = i => -3.825 + i * 0.45;
  zonePlate(XE, 12.5, 12, 0x3b78c2, "enc", "Encoder", `6 tokens → 1 concept · ${fp(T.encoder_non_embedding)} + ${fp(T.engram_table)} memory`, 1);
  zonePlate(XC, 8.2, 12, 0x2e9c8c, "core", "Concept module", `thinks in 2 passes · ${fp(T.concept_module)}`, 2);
  zonePlate(XD, 8.6, 12, 0xe07b39, "dec", "Decoder", `writes the next 6 tokens · ${fp(T.decoder)}`, 3);

  /* ---------- tokens + embedding (18 drawn = 3 chunks of 6) */
  for (let i = 0; i < NT; i++) block({ g: "enc", x: XE, y: -0.3, z: zs(i), w: 0.34, h: 0.34, d: 0.34, color: Math.floor(i / 6) % 2 ? C.tokB : C.tok, r: 0.06, id: i ? null : "tokens",
    info: i ? null : { t: "Tokens", cat: "input", fig: "v4whole", kv: [["drawn", `18 of ${fi(e.T)} positions`], ["vocabulary", fi(e.vocab_size)], ["chunks", "6 tokens → 1 concept"], ["row", `T = ${fi(e.T)} = 170 × 6`]], d: "Token ids arrive at the bottom of the encoder. Alternating shades mark the 6-token chunks that become one concept each." } });
  label(anchor([XE + 2.2, -0.3, 4.1], "enc"), "Tokens in", { level: "major", sub: "18 of 1,020 drawn · 3 chunks of 6", cx: 0 });
  block({ g: "enc", id: "emb", x: XE, y: 0.5, w: W, h: 0.26, d: D, color: C.emb, label: "Token embedding", sub: "8,192 × 256 · tied",
    info: { t: "Token embedding E", cat: "input", fig: "v4whole", params: T.token_embedding, kv: [["shape", `${fi(e.vocab_size)} × ${e.d_model}`], ["params", fi(T.token_embedding) + " (not counted)"], ["use", "e0 = rms(E[x]); tied with the output head"]], d: "The same matrix E produces the logits at the top of the decoder." } });

  /* ---------- encoder streams + layers */
  const rows = [["S", 0], ["E", 1], ["S", 1], ["S", 2], ["F", 3], ["S", 4], ["E", 2], ["S", 5], ["S", 6], ["F", 7]];
  const layerY = i => 1.4 + i * 0.86, streamTop = layerY(rows.length) + 0.7;
  glass({ g: "enc", id: "Lstream", x: XE - 1.95, y: (1.0 + streamTop) / 2, w: 0.12, h: streamTop - 1.0, d: D, color: C.L, label: "L", sub: "local · reach 63",
    info: { t: "L stream (local)", cat: "stream", fig: "v4enc", kv: [["reach", `${A.reach_enc} tokens (unchanged by the loop)`], ["written by", "S layers, both Engram reads"], ["read by", "S and F layers, router, decoder (tok)"]], d: "The only encoder stream the decoder may read. Only S layers and the Engram reads write it, so looping the F layers leaves its reach at 63 (tested)." } });
  glass({ g: "enc", id: "Gstream", x: XE + 1.95, y: (1.0 + streamTop) / 2, w: 0.12, h: streamTop - 1.0, d: D, color: C.G, label: "G", sub: "global",
    info: { t: "G stream (global)", cat: "stream", fig: "v4enc", kv: [["reach", "unbounded (full-attention layers)"], ["written by", "S layers, F layers (twice), Engram"], ["read by", "F layers, router → pooler → concepts"]], d: "Carries document-wide information; concepts are mostly G. It reaches the decoder only through the concepts and the concept module." } });
  const loopRings = [];
  rows.forEach((r, i) => {
    let info, color, lab, sub;
    if (r[0] === "S") {
      color = C.swa; lab = `S${r[1]}`; sub = "sliding window · + shared FFN";
      info = { t: `Encoder layer ${r[1]} · S (sliding window)`, cat: "enc", fig: "v4enc", params: P.enc_layer_S, kv: [["params", fi(P.enc_layer_S)], ["reads", "{L}"], ["writes", "{L, G}"], ["attention", `SWA w = ${e.window}, partial RoPE, 4 heads, GQA kv 2`], ["FFN", `own SwiGLU 1024 + g_ℓ · shared SwiGLU ${e.enc_shared_ff}`]], d: "v3's S layer; the FFN sublayer now adds the encoder's shared FFN as a parallel branch on the same read and write gate." };
    } else if (r[0] === "F") {
      color = C.full; lab = `F${r[1]} × 2`; sub = "full attention · applied twice";
      info = { t: `Encoder layer ${r[1]} · F, applied twice`, cat: "full", fig: "v4enc", params: P.enc_layer_F, kv: [["params", fi(P.enc_layer_F) + " (no new ones)"], ["reads", "{L, G}"], ["writes", "{G} only"], ["loop", "(L, G) → G′ → G″ with the same weights"], ["bypass", "enc_loop = 1 (the 'F loop off' switch-off)"], ["tiny scale", "loop off: +0.022 nats at k 4, +0.002 at k 6"]], d: "Layer-wise loop of the global layers only: they never write L, so the decoder's reach stays exactly 63." };
    } else {
      color = C.mem; lab = `Engram read ${r[1]}`; sub = r[1] === 1 ? "with conv" : "no conv · W_v = 0";
      const p = r[1] === 1 ? P.engram_read1 : P.engram_read2;
      info = { t: `Engram read ${r[1]} (encoder)`, cat: "mem", fig: "v4engram", params: p, kv: [["compute", fi(p)], ["position", r[1] === 1 ? "after layer 0" : "after layer 4"], ["output", r[1] === 1 ? "SiLU(conv₄(α v)) → L, G" : "SiLU(α v) → L, G"], ["tiny scale", "encoder reads off: +0.14 to +0.17 nats (they generalise)"]], d: "Both encoder reads use the rows gathered once from the 80M table; the decoder reads the same rows." };
    }
    block({ g: "enc", id: r[0] === "E" ? `engram${r[1]}` : `enc${r[1]}`, x: XE, y: layerY(i), w: W, h: 0.42, d: D, color, label: lab, sub, explode: i * 0.5, info });
    if (r[0] === "F") {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.05, 12, 48), pmat(0xb5476b, { emissive: 0xb5476b, ei: 0.25, clearcoat: 0.7 }));
      ring.position.set(XE, layerY(i), D / 2 + 0.5); tag(ring, "enc"); scene.add(ring); explodables.push({ obj: ring, base: layerY(i), k: i * 0.5 }); loopRings.push(ring);
      if (r[1] === 3) { parts.loop = ring; ring.userData.id = "loop"; ring.userData.info = info; pickables.push(ring); }
    }
  });
  label(anchor([XE + 0.9, layerY(4) + 0.55, D / 2 + 0.5], "enc"), "F layers × 2", { level: "major", sub: "same weights, writes G only" });
  // encoder shared FFN (one block beside the layers)
  const eshY = (layerY(0) + layerY(9)) / 2, eshH = layerY(9) - layerY(0) + 0.5;
  block({ g: "enc", id: "eshared", x: XE + 2.7, y: eshY, z: 2.4, w: 0.55, h: eshH, d: 2.2, color: 0xf2a56f, r: 0.08, explode: 2.2,
    label: "Shared FFN 384", sub: "all 8 layers", level: "major", lpos: [0, eshH / 2 + 0.45, 1.12],
    info: { t: `Encoder shared SwiGLU ${e.enc_shared_ff}`, cat: "enc", fig: "v4enc", params: P.enc_shared, kv: [["params", `${fi(P.enc_shared)} + ${P.enc_shared_gains} gains`], ["used by", "all 8 layers (10 applications with the loop)"], ["placement", "parallel to each layer's own FFN, same read and write gate"], ["init", "W₂ = 0, g_ℓ = 1 → exactly v3's encoder at step 0"], ["tiny scale", "off: +0.014 / +0.024 nats (used)"]], d: "A central, layer-shared memory in the encoder too, after the core's shared FFN proved its worth for multi-fact work in v3." } });
  rows.forEach((r, i) => { if (r[0] !== "E") flow([[XE + 2.42, layerY(i), 2.4], [XE + 2.05, layerY(i), 2.4]], 0xdc8b56, { g: "enc", r: 0.025, arrow: false }); });
  // one Engram table (8 slabs)
  const tY = (layerY(1) + layerY(6)) / 2, tH = layerY(6) - layerY(1) + 0.9;
  for (let t = 0; t < 8; t++) block({ g: "enc", id: t ? null : "engramTables", x: XE - 4.3 - (t % 4) * 0.36, y: tY, z: -2.2 + Math.floor(t / 4) * 4.4, w: 0.24, h: tH, d: 1.6, color: C.mem, r: 0.06, explode: 1.75,
    info: t ? null : { t: "Engram table (one, shared by four reads)", cat: "mem", fig: "v4engram", params: P.engram_table, kv: [["count", "8 (orders 2, 3 × 4 heads)"], ["rows × dim", `${fi(e.engram_rows)} × ${e.engram_dim}`], ["params", fi(P.engram_table) + " (lookup, not counted)"], ["reads", "2 in the encoder + 2 in the decoder; rows gathered once"]], d: "One row per table per token. The decoder's reads reuse the same gathered rows: no second gather, no second table." } });
  const engA = anchor([XE - 4.85, tY + tH / 2 + 0.5, 0], "enc"); explodables.push({ obj: engA, base: engA.position.y, k: 1.75 });
  label(engA, "Engram memory", { level: "major", sub: "80M n-gram table · 4 reads" });
  flow([[XE - 3.95, layerY(1), 0], [XE - 1.7, layerY(1), 0]], C.mem, { g: "enc", r: 0.06 });
  flow([[XE - 3.95, layerY(6), 0], [XE - 1.7, layerY(6), 0]], C.mem, { g: "enc", r: 0.06 });
  // router + pooler + concepts
  const yCRC = streamTop + 0.5, yAtt = yCRC + 0.75, yPF = yAtt + 0.62, yC = yPF + 1.0;
  block({ g: "enc", id: "crc", x: XE, y: yCRC, w: W, h: 0.3, d: D, color: C.gate, label: "Router mix", sub: "e0 · L · G per token", explode: 10 * 0.5,
    info: { t: "Router mix", cat: "concept", fig: "v4pool", params: P.crc_router, kv: [["params", fi(P.crc_router)], ["mix", "softmax(W rms G) over {e0, rms L, rms G}"]], d: "Chooses which view of each token enters the pooler. It cannot remove a token: that is the pooler's job." } });
  block({ g: "enc", id: "pool", x: XE, y: yAtt, w: W, h: 0.36, d: D, color: C.conc, label: "Attention pooler · 4 heads", sub: "6 tokens → 1 · weights shared across slots", explode: 10.3 * 0.5,
    info: { t: "Attention pooler (6 tokens → 1)", cat: "concept", fig: "v4pool", params: P.pool_Wk + P.pool_Wv + P.pool_Wo + P.pool_q + P.pool_p + P.pool_b, kv: [["heads", "4 × 64, learned queries q_h (zero-init)"], ["keys / values", "W_K(rms m + p_r), W_V(m + p_r), shared across slots"], ["pads", "masked; an all-pad chunk uses uniform weights"], ["init", "W_V = W_O = I → exact masked mean (tested)"], ["tiny scale", "weights tokens by content (within-slot ratios ≠ 1); selects structure until retrieval pays"]], d: "Each head scores the chunk's tokens by content, so a name or value can take most of the weight at any slot; v3's pool weighted by position only." } });
  for (let h = 0; h < 4; h++) block({ g: "enc", x: XE - 1.05 + h * 0.7, y: yAtt + 0.3, z: D / 2 - 0.4, w: 0.4, h: 0.16, d: 0.4, color: 0x2e9c8c, r: 0.05, explode: 10.3 * 0.5 });
  block({ g: "enc", id: "poolffn", x: XE, y: yPF, w: W, h: 0.3, d: D, color: 0xf2a56f, opacity: 0.8, label: "Pooler FFN 384", sub: "composes the heads' picks · arm", explode: 10.6 * 0.5,
    info: { t: "Pooler FFN (SwiGLU 384)", cat: "concept", fig: "v4pool", params: P.pool_ffn, kv: [["form", "c₁ = c₀ + g · SwiGLU(rms c₀)"], ["init", "W₂ = 0, g = 1"], ["tiny scale", "writes (g 0.56–0.71) but switching it off is neutral"], ["recommendation", "off in the main run; priced arm"]], d: "Attention can select but not compose ('not' + 'good'); the FFN composes across the heads' selections. At tiny scale it did not earn its 0.29M parameters." } });
  const cz = [-2.7, 0, 2.7];
  cz.forEach((z, j) => {
    const cm = new THREE.Mesh(new THREE.IcosahedronGeometry(0.44, 3), pmat(C.conc, { clearcoat: 0.8, rough: 0.25, emissive: C.conc, ei: 0.12 }));
    cm.position.set(XE, yC, z); cm.castShadow = true; tag(cm, "enc"); scene.add(cm); explodables.push({ obj: cm, base: yC, k: 10.9 * 0.5 });
    if (!j) { parts.concepts = cm; cm.userData.id = "concepts"; cm.userData.info = { t: "Concepts C = ĉ", cat: "concept", fig: "v4pool", kv: [["count", "1 per 6 tokens (170 per row)"], ["dim", e.d_concept], ["norm", "√256 = 16"]], d: "One concept per 6 tokens: 6 : 1 compression (v3: 4 : 1). The concept module re-reads this sequence C." }; pickables.push(cm); }
    for (let k = 0; k < 6; k++) flow([[XE, yPF + 0.18, zs(6 * j + k)], [XE, yPF + 0.5, zs(6 * j + k) * 0.6 + z * 0.4], [XE, yC - 0.42, z]], C.P, { g: "enc", r: 0.016, opacity: 0.55, arrow: false });
  });
  const cA = anchor([XE, yC + 0.9, 0], "enc"); explodables.push({ obj: cA, base: cA.position.y, k: 10.9 * 0.5 });
  label(cA, "Concepts C", { level: "major", sub: "1 per 6 tokens · 256-d" });

  /* ---------- concept module (reads top → bottom) */
  const yX = yC, yIn = yX - 1.25, yLay = l => yIn - 1.2 - l * 0.95, yU = yLay(3) - 1.0, yHead = yU - 1.0, yCo = yHead - 1.1, yCB = yCo - 0.75, yPred = yCB - 1.2;
  flow([[XE + 0.6, yC, 0], [-5, yC + 1.0, 0], [XC - 2.1, yX, 0]], C.P, { g: "core", groups: ["enc", "core"], r: 0.075 });
  cz.forEach((z, j) => {
    const cm = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 3), pmat(C.conc, { clearcoat: 0.8, rough: 0.25, emissive: C.conc, ei: 0.12 }));
    cm.position.set(XC, yX, z); cm.castShadow = true; tag(cm, "core"); scene.add(cm);
    if (!j) { parts.X = cm; cm.userData.id = "X"; cm.userData.info = { t: "C (evidence)", cat: "loop", fig: "v4core", kv: [["shape", "(B, 170, 256)"], ["role", "re-read by every sublayer of both passes"]], d: "170 concepts per row instead of 256: the core's attention costs ≈ 44 % of v3's per pass." }; pickables.push(cm); }
  });
  block({ g: "core", id: "cin", x: XC, y: yIn, w: W, h: 0.3, d: D, color: C.tok, label: "S₀ = 0 · working memory", sub: "C is re-read, S is written",
    info: { t: "Working memory S", cat: "loop", fig: "v4core", kv: [["S₀", "exact zeros"], ["pass 2", "reads S₁, which holds what pass 1 retrieved"], ["readout", "u = zcRMSNorm(S₂)"]], d: "The second pass builds its queries from a state that already contains pass 1's retrieval: the structure a two-step retrieval needs." } });
  const layerMeshes = [];
  for (let l = 0; l < 4; l++) {
    const m = block({ g: "core", id: `core${l + 1}`, x: XC, y: yLay(l), w: W, h: 0.5, d: D, color: C.full, label: `Core layer ${l + 1}`, sub: `6-head full attention · own FFN ${c4.own_ff}`, level: "minor", explode: -l * 0.35,
      info: { t: `Core layer ${l + 1} (distinct weights)`, cat: "loop", fig: "v4core", params: P.core_layer, kv: [["params", fi(P.core_layer)], ["attention", `${fi(P.core_attn)} · full causal over 170 concepts, 6 × 64`], ["own FFN", `SwiGLU 256 → ${c4.own_ff} → 256`], ["shared FFN", `parallel branch, SwiGLU ${c4.shared_ff}`], ["conditioning", "none (the (t, Δ) table is removed)"]], d: "One of four distinct layers; one pass = all four, two passes in all." } });
    m.material = m.material.clone(); m.material.userData.baseOpacity = 1; layerMeshes.push(m);
    if (l < 3) flow([[XC, yLay(l) - 0.27, 0], [XC, yLay(l + 1) + 0.27, 0]], C.P, { g: "core", r: 0.045 });
  }
  flow([[XC, yIn - 0.17, 0], [XC, yLay(0) + 0.27, 0]], C.P, { g: "core", r: 0.05 });
  const shared = block({ g: "core", id: "shared", x: XC + 2.45, y: (yLay(0) + yLay(3)) / 2, w: 0.75, h: yLay(0) - yLay(3) + 0.6, d: 2.6, color: 0xf2a56f, r: 0.08, explode: -1.2,
    label: "Shared FFN", sub: `SwiGLU ${c4.shared_ff} · all layers`, level: "major", lpos: [0, (yLay(0) - yLay(3)) / 2 + 0.6, 1.32],
    info: { t: `Core shared SwiGLU ${c4.shared_ff}`, cat: "loop", fig: "v4core", params: P.core_shared, kv: [["params", fi(P.core_shared)], ["used by", "all 4 layers in both passes"], ["evidence", "v3: removing it cost multi4 −0.129 → kept full size"], ["tiny scale", "off: +0.008 / +0.012 nats (used)"]], d: "Kept at full size on v3's evidence." } });
  shared.material = shared.material.clone(); shared.material.userData.baseOpacity = 1;
  for (let l = 0; l < 4; l++) flow([[XC + 2.05, yLay(l), 1.0], [XC + 1.65, yLay(l), 1.0]], 0xdc8b56, { g: "core", r: 0.03 });
  // pass ring: 2 markers
  const ringC = [XC - 2.55, (yLay(0) + yLay(3)) / 2, D / 2 + 0.9], ringR = 1.05;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(ringR, 0.045, 12, 96), pmat(0x2e9c8c, { emissive: 0x2e9c8c, ei: 0.2, clearcoat: 0.7, opacity: 0.7 }));
  ring.position.set(...ringC); tag(ring, "core"); scene.add(ring);
  const markers = [];
  for (let p = 0; p < 2; p++) {
    const a = Math.PI / 2 - (p + 0.5) * Math.PI;
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.18, 20, 14), pmat(0x2e9c8c, { emissive: 0x2e9c8c, ei: 0.15, clearcoat: 0.8 }));
    s.position.set(ringC[0] + ringR * Math.cos(a), ringC[1] + ringR * Math.sin(a), ringC[2]); s.castShadow = true; tag(s, "core"); scene.add(s); markers.push(s);
    if (p === 0) { parts.ring = s; s.userData.id = "ring"; s.userData.info = { t: "Two plain passes (M = 2)", cat: "loop", fig: "v4core", kv: [["trajectory", "S₁ = Φ(C, 0), S₂ = Φ(C, S₁)"], ["supervision", "terminal only (u from S₂)"], ["shortcut", "removed (was +42 % core compute in v3)"], ["evidence", "v3: pass 2 multi4 +0.09, pass 3 +0.01"], ["tiny scale", "1 pass instead of 2: +0.008 / +0.014 nats"]], d: "Each marker is one pass through the 4 layers. Press “▶ Play passes”." }; pickables.push(s); }
  }
  label(anchor([ringC[0], ringC[1] + ringR + 0.55, ringC[2]], "core"), "2 passes", { level: "major", sub: "4 core layers per pass · no table" });
  flow([[XC - 1.0, yU - 0.1, D / 2 + 0.05], [ringC[0] + 0.6, ringC[1] - ringR - 0.25, ringC[2]], [ringC[0] - ringR - 0.3, ringC[1], ringC[2]], [ringC[0] + 0.6, ringC[1] + ringR + 0.25, ringC[2]], [XC - 1.0, yLay(0) + 0.1, D / 2 + 0.05]], C.P, { g: "core", r: 0.03, opacity: 0.55 });
  block({ g: "core", id: "readout", x: XC, y: yU, w: W, h: 0.3, d: D, color: C.norm, label: "u = zcRMSNorm(S₂)", sub: "terminal-only supervision",
    info: { t: "Readout u", cat: "loop", fig: "v4core", kv: [["u", "zcRMSNorm(S₂)"], ["to decoder", "u and ĉ_pred"], ["pass-1 NCP", "an arm with a kill criterion, not the default"]], d: "Only the endpoint is graded; intermediate states are never pulled toward the answer." } });
  flow([[XC, yLay(3) - 0.27, 0], [XC, yU + 0.17, 0]], C.P, { g: "core", r: 0.05 });
  block({ g: "core", id: "ncp", x: XC, y: yHead, w: W, h: 0.34, d: D, color: C.conc, label: "Head · q = W_q u", sub: "coarse + fine next-concept codes",
    info: { t: "Next-concept head", cat: "loop", fig: "v4code", params: P.ncp_head, kv: [["coarse", "−‖q − E⁰_k‖² / τ₀ over 1,024"], ["fine", "PQ distances, frame q − E⁰[k_true] in the loss"], ["plan", "ĉ_pred = c̄ + PQexp(q − c̄)"]], d: "Predicts the next concept's coarse 'action' code and its fine detail; nothing from the future reaches the plan." } });
  flow([[XC, yU - 0.17, 0], [XC, yHead + 0.2, 0]], C.P, { g: "core", r: 0.05 });
  // coarse codebook (32 × 32 = 1,024 gems) and PQ (32 × 256 gems)
  const gemPlate = (y, rowsN, colsN, col0, id, lab, sub, info, sz) => {
    const gem = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(sz, 0), pmat(col0, { rough: 0.3, clearcoat: 0.6 }), rowsN * colsN);
    const m4 = new THREE.Matrix4(), col = new THREE.Color(); let gk = 0;
    for (let s = 0; s < rowsN; s++) for (let n = 0; n < colsN; n++) {
      m4.makeTranslation(XC - W / 2 + (n + 0.5) * W / colsN, y + 0.06, -D / 2 + (s + 0.5) * D / rowsN); gem.setMatrixAt(gk, m4);
      const v = 0.6 + 0.4 * Math.abs(Math.sin(s * 12.9898 + n * 78.233)); col.setHex(col0).multiplyScalar(v); gem.setColorAt(gk++, col);
    }
    tag(gem, "core"); scene.add(gem);
    block({ g: "core", id, x: XC, y, w: W + 0.2, h: 0.06, d: D + 0.2, color: C.mem, opacity: 0.35, cast: false, label: lab, sub, lpos: [0, 0.1, (D + 0.2) / 2 + 0.02], info });
  };
  gemPlate(yCo, 32, 32, 0xb5476b, "coarse", "Coarse codebook", "1,024 codes · 10-bit action", { t: "Coarse codebook E⁰", cat: "code", fig: "v4code", params: P.coarse_codebook, kv: [["codes", "1,024 × 256"], ["bits", "10 per concept"], ["role", "NCP target and abstraction measure; the RL action under option B"], ["tiny scale", "perplexity 36–62 % of K, very predictable (CE 1.07–1.53 vs ln K 4.16), five arms"]], d: "A small 'abstract action' per 6 tokens: K^{T/6} = 2^{1.7T} sequences vs V^T = 2^{13T} for tokens." }, 0.045);
  gemPlate(yCB, 32, 256, 0xe0b73a, "codebook", "PQ codebook (residual)", "32 × 256 codes · 8 dims", { t: "Fine PQ codebook", cat: "code", fig: "v4code", params: P.pq_codebook, kv: [["segments × codes × dims", "32 × 256 × 8"], ["on", "the residual ĉ − E⁰_k"], ["bits", "256 per concept (1 per dim)"], ["tiny scale", "fine-level VE on the residual 0.72 (bound 0.75)"]], d: "The detail of each concept; trained only by VQ, restarted from residuals of the updated E⁰ (tested)." }, 0.018);
  const lossB = (id, x, y, lab, info, col2) => block({ g: "core", id, x, y, w: 1.35, h: 0.45, d: 1.6, color: col2 || C.loss, label: lab, lpos: [0, 0.5, 0.8], info });
  lossB("lcc", XC - 3.05, yHead, "0.05 · CE_coarse", { t: "Coarse next-concept loss", cat: "loss", fig: "v4loss", kv: [["form", "CE of k_{j+1} under −‖q − E⁰‖²/τ₀"], ["weight", "0.05"], ["pairs", "pair_ok: complete, same-document targets"]], d: "Teaches the core to predict the next abstract action." });
  lossB("lcf", XC + 3.05, yHead, "0.1 · CE_fine", { t: "Fine next-concept loss", cat: "loss", fig: "v4loss", kv: [["form", "segment-mean CE in the frame q − E⁰[k_true]"], ["weight", "0.1 (v3's NCP weight)"]], d: "Teacher-forced on the true next coarse code inside the loss only." });
  lossB("lvqc", XC - 3.05, yCo, "VQ_coarse", { t: "Coarse VQ", cat: "loss", fig: "v4loss", kv: [["form", "‖sg ĉ − E⁰_k‖² / 32"], ["trains", "E⁰ only (tested)"]], d: "Dominates the printed loss (≈ 7); the dashboard headlines NTP." });
  lossB("lvqf", XC + 3.05, yCB, "VQ_fine", { t: "Fine VQ", cat: "loss", fig: "v4loss", kv: [["form", "segment-mean ‖sg r − e_n‖²"], ["trains", "PQ only (tested)"]], d: "Keeps the fine codes on the residuals." });
  cz.forEach((z, j) => {
    const cm = new THREE.Mesh(new THREE.IcosahedronGeometry(0.38, 3), pmat(C.plan, { clearcoat: 0.8, rough: 0.25, emissive: C.plan, ei: 0.15 }));
    cm.position.set(XC, yPred, z); cm.castShadow = true; tag(cm, "core"); scene.add(cm);
    if (!j) { parts.pred = cm; cm.userData.id = "pred"; cm.userData.info = { t: "ĉ_pred and u", cat: "plan", fig: "v4dec", kv: [["ĉ_pred", "c̄ + PQexp(q − c̄), 256-d"], ["u", "readout of S₂, 256-d"]], d: "Per chunk, the two inputs of the decoder's plan builder." }; pickables.push(cm); }
  });
  label(anchor([XC, yPred - 0.75, 0], "core"), "Plan inputs → decoder", { level: "major", sub: "u and the predicted next concept" });
  flow([[XC, yCB - 0.1, 0.9], [XC, yPred + 0.42, 0.9]], C.P, { g: "core", r: 0.04 });
  passAnim = { markers, layers: layerMeshes, shared };

  /* ---------- decoder (reads bottom → top) */
  const yPlan = 1.2, yDL = i => 2.6 + i * 1.15, yON = yDL(3) + 1.1, yH = yON + 0.9, yLog = yH + 1.1, yNTP = yLog + 1.3;
  flow([[XE - 1.95, streamTop + 0.1, 0], [XE - 1.95, streamTop + 3.6, 0], [0, streamTop + 4.8, 0], [XD - 1.95, streamTop + 3.6, 0], [XD - 1.95, yDL(3) + 0.7, 0]], C.L, { g: "dec", groups: ["enc", "dec"], r: 0.075 });
  label(anchor([0, streamTop + 5.3, 0], "dec"), "Local token stream → decoder", { level: "major", sub: `tok = zcRMSNorm(L) · reach ${A.reach_enc} tokens`, group: "dec" });
  glass({ g: "dec", id: "tok", x: XD - 1.95, y: (yPlan + yDL(3) + 0.6) / 2, w: 0.12, h: yDL(3) + 0.6 - yPlan, d: D, color: C.L, label: "tok", sub: "keys & values", level: "minor",
    info: { t: "tok (keys and values)", cat: "stream", fig: "v4dec", kv: [["shape", `(B, ${fi(e.T)}, 256)`], ["reach", A.reach_enc]], d: "Every decoder layer takes its keys and values from tok only (window 8)." } });
  block({ g: "dec", id: "plan", x: XD, y: yPlan, w: W, h: 0.42, d: D, color: C.plan, label: "Plan builder", sub: "shift 1 chunk · repeat × 6", level: "major",
    info: { t: "Plan builder (shift and repeat)", cat: "plan", fig: "v4dec", params: P.dec_plan, kv: [["params", fi(P.dec_plan)], ["index", "j*(t) = ⌊(t+1)/6⌋ − 1"], ["repeat", "the plan of chunk j at all 6 positions writing chunk j + 1, + offset o_r"], ["null plan", "first 5 tokens of each document"], ["tested", "plan j changes exactly positions 6j+5 … 6j+10"]], d: "NCP's mechanism with k = 6: the predicted next concept is used exactly while its tokens are being written." } });
  flow([[XC + 1.7, yU, 0], [4.6, yU, 0], [6.3, yPlan + 0.1, 0], [XD - 1.75, yPlan, 0]], C.P, { g: "dec", groups: ["core", "dec"], r: 0.065 });
  flow([[XC + 0.5, yPred, 0], [4.6, yPred - 0.2, 1.4], [XD - 1.75, yPlan - 0.05, 1.4]], C.P, { g: "dec", groups: ["core", "dec"], r: 0.065 });
  label(anchor([4.9, yU + 0.45, 0], "dec"), "u", { level: "minor", group: "dec" });
  label(anchor([4.5, yPred - 0.65, 1.4], "dec"), "ĉ_pred", { level: "minor", group: "dec" });
  for (let i = 0; i < 4; i++) {
    block({ g: "dec", id: `dec${i + 1}`, x: XD, y: yDL(i), w: W, h: 0.52, d: D, color: C.dec, label: `Confined layer ${i + 1}`, sub: "keys/values from tok", explode: (i + 1) * 0.55,
      info: { t: `Decoder layer ${i + 1} · confined`, cat: "dec", fig: "v4dec", params: P.dec_layer, kv: [["params", fi(P.dec_layer)], ["queries", "from GR read of D (plan-aware)"], ["keys, values", "from tok only, SWA w = 8, 4 heads"], ["reach", `${A.reach_dec} at any depth`]], d: "D is private to its position: position t sees 70 tokens plus its own plan." } });
    for (let q = -1; q <= 1; q += 2) flow([[XD - 1.9, yDL(i), q * 2.5], [XD - 1.62, yDL(i), q * 2.5]], C.L, { g: "dec", r: 0.035, arrow: false });
    if (i === 0 || i === 2) {
      const yR = yDL(i) + 0.58;
      block({ g: "dec", id: `dread${i}`, x: XD + 0.4, y: yR, z: -2.6, w: 2.0, h: 0.22, d: 2.4, color: C.mem, opacity: 0.85, explode: (i + 1.5) * 0.55, label: `Engram read (after layer ${i})`, sub: "arm · memorises", level: "minor",
        info: { t: `Decoder Engram read after layer ${i}`, cat: "mem", fig: "v4engram", params: P.dec_read, kv: [["params", fi(P.dec_read)], ["rows", "the encoder's gathered e_t (no second gather)"], ["form", "D += write(h) · SiLU(α · W_V e), W_V zero-init"], ["reach", "position-wise: stays 70 (tested)"], ["tiny scale", "+0.47 to +0.70 on memorised text · −0.14 to −0.17 unseen · ≈ 0 to −0.02 on text"], ["recommendation", "off in the main run; mandatory arm"]], d: "The owner's 'spit out common tokens' path. At tiny scale it acted as a memorisation channel, while the encoder's reads generalised." } });
    }
  }
  flow([[XE - 4.85, tY + tH / 2, -4.4], [XE - 4.85, streamTop + 2.4, -4.4], [0, streamTop + 2.8, -4.4], [XD + 0.4, yDL(2) + 1.3, -4.4], [XD + 0.4, yDL(2) + 0.72, -3.4]], C.mem, { g: "dec", groups: ["enc", "dec"], r: 0.05, opacity: 0.6 });
  flow([[XD + 0.4, yDL(2) + 0.45, -3.6], [XD + 0.4, yDL(0) + 0.72, -3.4]], C.mem, { g: "dec", r: 0.035, opacity: 0.6 });
  label(anchor([0, streamTop + 3.3, -4.4], "dec"), "Same Engram rows → decoder (arm)", { level: "minor", group: "dec" });
  block({ g: "dec", id: "onorm", x: XD, y: yON, w: W, h: 0.26, d: D, color: C.norm, label: "zcRMSNorm(D)", explode: 5 * 0.55 });
  block({ g: "dec", id: "head", x: XD, y: yH, w: W, h: 0.34, d: D, color: C.head, label: "Tied output head", sub: "Eᵀ · 0 extra params", explode: 5.5 * 0.55,
    info: { t: "Output head (tied)", cat: "out", fig: "v4dec", kv: [["matrix", "Eᵀ, 256 × 8,192"], ["extra params", "0"]], d: "logits = zcRMSNorm(D) · Eᵀ" } });
  for (let i = 0; i < NT; i++) {
    const h = 0.15 + 0.95 * Math.pow(Math.abs(Math.sin(i * 1.7 + 0.3)), 3);
    block({ g: "dec", x: XD, y: yLog + h / 2 - 0.3, z: zs(i), w: 0.28, h, d: 0.28, color: i === 9 ? 0xe07b39 : C.tok, r: 0.05, explode: 6 * 0.55 });
  }
  label(anchor([XD + 2.0, yLog + 0.2, 4.3], "dec"), "Next-token scores", { level: "minor", sub: "8,192 per position", cx: 0 });
  block({ g: "dec", id: "lntp", x: XD, y: yNTP, w: 2.2, h: 0.5, d: 2.2, color: C.loss, label: "Next-token loss", sub: "NTP · main training signal", level: "major", explode: 6.5 * 0.55, lpos: [0, 0.6, 1.1],
    info: { t: "Next-token loss", cat: "loss", fig: "v4loss", kv: [["targets", "x_{t+1}, same document, not PAD"], ["weight", "1"], ["reaches core", "through u and ĉ_pred"]], d: "The main training signal; the dashboard headlines it because VQ_coarse dominates the printed total." } });

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
  document.querySelectorAll("#v4c-bar [data-view]").forEach(b => b.classList.toggle("on", (b.dataset.view === "all" && !focus) || b.dataset.view === focus));
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
    enc: { t: "Encoder (v3's, refined)", cat: "enc", fig: "v4enc", params: T.encoder_non_embedding, kv: [["params", `${fi(T.encoder_non_embedding)} + ${fi(P.engram_table)} lookup`], ["layers", "8 (6 S · 4 heads, 2 F · 6 heads, F applied twice)"], ["shared FFN", `SwiGLU ${A.ecfg.enc_shared_ff} beside every layer`], ["Engram", "one 80M table, two encoder reads (+ two decoder reads)"], ["pooler", `attention + FFN, 6 tokens → 1 (${fi(P.pooler)})`], ["output", "170 concepts C + tok (reach 63)"]], d: "v3's two-stream encoder with the global layers looped and a content-aware pooler at the top." },
    core: { t: "Concept module v4", cat: "loop", fig: "v4core", params: T.concept_module, kv: [["params", fi(T.concept_module)], ["layers", "4 distinct, 6-head full attention, own FFN 736"], ["shared FFN", `SwiGLU ${c.shared_ff}, all layers`], ["passes", "M = 2 plain, terminal-only"], ["codebooks", "coarse 1,024 + PQ 32 × 256 × 8 on the residual"], ["share", `${(100 * T.concept_module / T.compute_params).toFixed(1)} % of compute params`]], d: "Reads top → bottom and loops around the ring twice. No (t, Δ) table, no shortcut." },
    dec: { t: "Decoder (confined)", cat: "dec", fig: "v4dec", params: T.decoder, kv: [["params", fi(T.decoder)], ["layers", "plan builder + 4 confined layers"], ["plan", "shift 1 chunk, repeat over 6"], ["Engram", "2 reads of the encoder's rows (recommended as an arm)"], ["reach", `${A.reach_dec} tokens at any depth`]], d: "v3's confinement: all long-range information arrives through the plan." },
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
  const el = document.getElementById("v4c-info"); if (tourIdx >= 0) return;
  const T = A.table;
  if (!info) {
    el.innerHTML = `<div class="v3p-kicker">How to read this model</div><h5>HRM Thinker v4 · 19.9M · proposed</h5>
      <p>Data moves left to right across three plates. Click a plate to zoom in, click any block for details.</p>
      <div class="v3-zones">
        <button data-zone="enc"><span class="zn" style="background:#3b78c2">1</span><b>Encoder</b><small>reads the tokens (global layers twice, a shared FFN beside every layer) and an attention + FFN pooler turns every 6 into one concept</small></button>
        <button data-zone="core"><span class="zn" style="background:#2e9c8c">2</span><b>Concept module</b><small>thinks over the concepts in 2 passes of 4 layers and predicts the next concept's coarse and fine codes</small></button>
        <button data-zone="dec"><span class="zn" style="background:#e07b39">3</span><b>Decoder</b><small>sees the last 70 tokens plus the plan, shifted one chunk and repeated over 6 tokens; may read the Engram rows directly (arm)</small></button>
      </div>
      <div class="v3p-sub">Colour key</div>
      <div class="v3-key"><span><i style="background:#86b4ea"></i>sliding-window layer</span><span><i style="background:#f2a56f"></i>full attention / FFN</span>
        <span><i style="background:#f4cd5c"></i>memory / codebook</span><span><i style="background:#4fbcad"></i>concepts, pooler &amp; plan</span>
        <span><i style="background:#b5476b"></i>F loop · coarse codes</span><span><i style="background:#ef8fa1"></i>loss</span>
        <span><i class="ln" style="background:#3b78c2"></i>L / tok stream</span><span><i class="ln" style="background:#f4cd5c"></i>Engram rows</span></div>
      <div class="v3-kv" style="margin-top:16px"><span>compute</span><b>${fi(T.compute_params)}</b><span>recommended main run</span><b>${fi(A.variants.recommended_main_run)}</b><span>stored</span><b>${fi(T.stored_total)}</b><span>tokens per concept</span><b>6</b><span>passes</span><b>M = 2</b></div>
      <div class="v3p-tip">Press “▶ Play passes” to watch the two passes run through the 4 core layers and the shared FFN. Tiny-scale results for every part: the <a href="#v4tiny" data-tab-link="v4tiny">Tiny experiment &amp; verdict</a> tab.</div>`;
    el.querySelectorAll("[data-zone]").forEach(b => b.onclick = () => frameView(b.dataset.zone));
    return;
  }
  const share = info.params ? info.params / T.compute_params : null;
  el.innerHTML = `${catChip(info.cat)}<h5>${info.t}</h5><p>${info.d || ""}</p>
    ${share != null ? `<div class="v3-share"><div class="v3-share-l"><span>share of compute parameters</span><b>${(100 * share).toFixed(1)} %</b></div><div class="v3-share-t"><div style="width:${Math.max(1.5, Math.min(100, 100 * share))}%"></div></div></div>` : ""}
    <div class="v3-kv">${(info.kv || []).map(([a, b]) => `<span>${a}</span><b>${b}</b>`).join("")}</div>
    <div class="v3p-actions">${obj ? `<button class="btn sm" id="v4c-fly">Zoom to it</button>` : ""}${info.fig ? `<a class="btn sm" href="#v4arch2d" data-fig4="${info.fig}">Open 2D figure →</a>` : ""}<button class="btn sm ghost" id="v4c-clear">Clear</button></div>`;
  const fly = document.getElementById("v4c-fly"); if (fly) fly.onclick = () => fitTo([obj], null, 1.4);
  document.getElementById("v4c-clear").onclick = () => select(null);
}

/* ------------------------------------------------------------------ guided tour */
const TOUR = [
  { n: "Tokens in, 6 per chunk", g: "enc", pick: "emb", objs: ["tokens", "emb"], dir: [-0.5, 0.45, 1], x: "Token ids enter at the bottom left; alternating shades mark the 6-token chunks that will each become one concept (v3: 4)." },
  { n: "v3's encoder, kept", g: "enc", pick: "Lstream", objs: ["enc0", "enc7", "Lstream", "Gstream"], dir: [-0.6, 0.3, 1], x: "SSSFSSSF with the local stream L (reach 63) and the global stream G, gated residuals, one 80M Engram table read after layers 0 and 4." },
  { n: "Global layers applied twice", g: "enc", pick: "loop", objs: ["enc3", "enc7", "loop"], dir: [-0.3, 0.3, 1], x: "F layers 3 and 7 run twice with the same weights. They write only G, so the decoder's reach stays 63 (looping an S layer would push it to 123: tested). Tiny scale: worth +0.022 nats at k 4, +0.002 at k 6." },
  { n: "A shared FFN in the encoder", g: "enc", pick: "eshared", objs: ["eshared", "enc1", "enc5"], dir: [0.4, 0.3, 1], x: "One SwiGLU 384 beside every layer's own FFN, zero-initialised so step 0 equals v3. Tiny scale: switching it off costs +0.014 to +0.024 nats." },
  { n: "6 tokens → 1 concept, by content", g: "enc", pick: "pool", objs: ["crc", "pool", "poolffn", "concepts"], dir: [-0.45, 0.4, 1], x: "Four attention heads with weights shared across slots score the chunk's tokens by content, then a small FFN composes their picks. v3's pool weighted by position only. Tiny scale: content-selective; the FFN did not earn its parameters (arm)." },
  { n: "Two plain passes", g: "core", pick: "ring", objs: ["ring", "core1", "core4", "readout"], dir: [0.2, 0.3, 1], x: "Four distinct 6-head layers read C and write S, twice from S₀ = 0. No (t, Δ) table, no shortcut, terminal-only: v3's 24 h switch-offs showed the second pass is what pays (+0.09 multi4)." },
  { n: "The shared FFN", g: "core", pick: "shared", objs: ["shared", "core2"], dir: [0.7, 0.3, 1], x: "One SwiGLU 1024 used by all four layers in both passes, kept full size because removing it cost v3 multi4 −0.129." },
  { n: "Two-level codebook", g: "core", pick: "coarse", objs: ["ncp", "coarse", "codebook", "pred"], dir: [0.35, 0.55, 1], x: "The head predicts the next concept's coarse code (1,024 — a 10-bit 'abstract action') and its fine PQ codes on the residual. The plan gets the soft expectation; the true code is used only inside the loss." },
  { n: "Shift and repeat", g: "dec", pick: "plan", objs: ["plan", "dec1", "tok"], dir: [0.6, 0.3, 1], x: "The plan made at chunk j is added at all 6 positions that write chunk j + 1. The decoder still sees only the last 70 tokens plus that plan." },
  { n: "Decoder Engram reads (arm)", g: "dec", pick: "dread0", objs: ["dread0", "dread2", "dec1", "dec3"], dir: [0.3, 0.35, -1], x: "Two reads of the encoder's Engram rows, after decoder layers 0 and 2. Tiny scale: they memorised repeated text (+0.47 to +0.70) and hurt on new text (−0.14 to −0.17), so they are recommended off in the main run and tested as an arm." },
];
function goTour(i) {
  tourIdx = i; const S = TOUR[i];
  setFocus(S.g);
  const obj = parts[S.pick]; selected = obj || null; setOutline();
  fitTo(S.objs.map(id => parts[id]).filter(Boolean), S.dir, 1.05);
  document.querySelectorAll("#tour4 [data-i]").forEach(b => { b.classList.toggle("on", +b.dataset.i === i); b.classList.toggle("done", +b.dataset.i < i); });
  const info = obj ? obj.userData.info : null;
  document.getElementById("v4c-info").innerHTML = `<div class="v3p-kicker">Guided tour · ${i + 1} of ${TOUR.length}</div>
    <div class="v3-prog"><div style="width:${100 * (i + 1) / TOUR.length}%"></div></div>
    <h5>${S.n}</h5><p>${S.x}</p>
    ${info ? `${catChip(info.cat)}<div class="v3-kv">${(info.kv || []).slice(0, 5).map(([a, b]) => `<span>${a}</span><b>${b}</b>`).join("")}</div>` : ""}
    <div class="v3p-actions"><button class="btn sm" id="t4-tour-prev" ${i ? "" : "disabled"}>← Back</button>
      <button class="btn sm primary" id="t4-tour-next">${i === TOUR.length - 1 ? "Finish" : "Next →"}</button>
      <button class="btn sm ghost" id="t4-tour-play">${tourTimer ? "❚❚ Pause" : "▶ Autoplay"}</button><button class="btn sm ghost" id="t4-tour-end">Exit</button></div>
    <div class="v3p-tip">Keyboard: ← → to move, Esc to exit.</div>`;
  document.getElementById("t4-tour-prev").onclick = () => { stopAuto(); i > 0 && goTour(i - 1); };
  document.getElementById("t4-tour-next").onclick = () => { stopAuto(); i < TOUR.length - 1 ? goTour(i + 1) : endTour(); };
  document.getElementById("t4-tour-play").onclick = () => { tourTimer ? stopAuto() : startAuto(); goTour(tourIdx); };
  document.getElementById("t4-tour-end").onclick = endTour;
}
function startAuto() { tourTimer = setInterval(() => { if (tourIdx < TOUR.length - 1) goTour(tourIdx + 1); else endTour(); }, 7000); }
function stopAuto() { if (tourTimer) clearInterval(tourTimer); tourTimer = null; }
function endTour() {
  stopAuto(); tourIdx = -1;
  document.querySelectorAll("#tour4 [data-i]").forEach(b => b.classList.remove("on", "done"));
  select(null); frameView("all");
}
function setupTour() {
  const el = document.getElementById("tour4");
  el.innerHTML = `<button class="tour-start" id="t4-tour-start">▶ Start guided tour</button><div class="tour-steps">` +
    TOUR.map((t, i) => `<button data-i="${i}" title="${t.n}"><span>${i + 1}</span><em>${t.n}</em></button>`).join("") + `</div>`;
  el.querySelectorAll("[data-i]").forEach(b => b.onclick = () => { stopAuto(); goTour(+b.dataset.i); });
  document.getElementById("t4-tour-start").onclick = () => { stopAuto(); goTour(0); };
}

/* ------------------------------------------------------------------ input */
function wireInput(stage, cv) {
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2(), tipEl = document.getElementById("v4c-hover");
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
    if (tourIdx >= 0) { stopAuto(); tourIdx = -1; document.querySelectorAll("#tour4 [data-i]").forEach(b => b.classList.remove("on", "done")); }
    select(pick(ev));
  });
  cv.addEventListener("dblclick", ev => { const h = pick(ev); if (h) select(h, true); });
  labelRenderer.domElement.addEventListener("click", ev => {
    const z = ev.target.closest("[data-go]"); if (z && z.dataset.go.startsWith("zone:")) frameView(z.dataset.go.slice(5));
  });
  document.addEventListener("keydown", ev => {
    if (!document.getElementById("tab-v4arch3d").classList.contains("on") || /INPUT|TEXTAREA/.test(document.activeElement?.tagName)) return;
    if (ev.key === "ArrowRight") { stopAuto(); goTour(Math.min(TOUR.length - 1, tourIdx + 1)); }
    else if (ev.key === "ArrowLeft" && tourIdx > 0) { stopAuto(); goTour(tourIdx - 1); }
    else if (ev.key === "Escape") { if (tourIdx >= 0) endTour(); else { select(null); frameView("all"); } }
  });
}

/* ------------------------------------------------------------------ toolbar */
let traj = "2";
function setupBar() {
  document.querySelectorAll("#v4c-bar button").forEach(b => b.onclick = () => {
    const act = b.dataset.act;
    if (b.dataset.view) { if (tourIdx >= 0) { stopAuto(); tourIdx = -1; document.querySelectorAll("#tour4 [data-i]").forEach(x => x.classList.remove("on", "done")); select(null); } frameView(b.dataset.view); }
    else if (b.dataset.traj) { traj = b.dataset.traj; document.querySelectorAll("#v4c-bar [data-traj]").forEach(x => x.classList.toggle("on", x === b)); showTraj(); if (passTimer) startPasses(); }
    else if (act === "explode") { b.classList.toggle("on"); explodeTarget = b.classList.contains("on") ? 1 : 0; requestRender(); }
    else if (act === "labels") { labelMode = { auto: "all", all: "off", off: "auto" }[labelMode]; b.textContent = { auto: "Labels: auto", all: "Labels: all", off: "Labels: off" }[labelMode]; b.classList.toggle("on", labelMode !== "off"); requestRender(); }
    else if (act === "flow") { b.classList.toggle("on"); flowOn = b.classList.contains("on"); requestRender(); }
    else if (act === "passes") { b.classList.toggle("on"); b.classList.contains("on") ? startPasses() : stopPasses(); }
    else if (act === "reset") { if (tourIdx >= 0) endTour(); else { select(null); frameView("all"); } }
    else if (act === "png") { composer.render(); const a = document.createElement("a"); a.download = "hrm-thinker-v4-draft-3d.png"; a.href = renderer.domElement.toDataURL("image/png"); a.click(); }
  });
  showTraj();
}

/* ------------------------------------------------------------------ pass animation (illustrative: pass p, core layer l) */
function showTraj() {
  if (!passAnim) return;
  const n = +traj;
  passAnim.markers.forEach((m, i) => { m.visible = i < n; });
  setPass(-1);
}
function setPass(k) {                                            // k = pass * 4 + layer, or -1
  if (!passAnim) return;
  const p = k < 0 ? -1 : Math.floor(k / 4), l = k < 0 ? -1 : k % 4;
  passAnim.markers.forEach((m, i) => { m.material.emissiveIntensity = i === p ? 1.1 : 0.15; m.scale.setScalar(i === p ? 1.45 : 1); });
  passAnim.layers.forEach((m, li) => { m.material.emissive.setHex(li === l ? 0xe0662a : 0x000000); m.material.emissiveIntensity = li === l ? 0.45 : 0; });
  passAnim.shared.material.emissive.setHex(l >= 0 ? 0xe0662a : 0x000000); passAnim.shared.material.emissiveIntensity = l >= 0 ? 0.35 + 0.1 * l : 0;
  const tip = document.getElementById("v4c-hover");
  if (p >= 0 && tourIdx < 0) { tip.innerHTML = `<b>${traj === "2" ? "Main path" : "1-pass switch-off"} · pass ${p + 1} of ${traj} · core layer ${l + 1}</b><span>${p === 0 ? "reads C and S₀ = 0" : "reads C and S₁ (what pass 1 retrieved)"}</span><em>illustrative, not trained activations</em>`; tip.style.transform = "translate(14px, 14px)"; tip.classList.add("on"); }
  else if (p < 0) tip.classList.remove("on");
  requestRender();
}
function startPasses() { if (passTimer) clearInterval(passTimer); passIdx = -1; const n = (+traj) * 4 + 2; passTimer = setInterval(() => { passIdx = (passIdx + 1) % n; setPass(passIdx < n - 2 ? passIdx : -1); }, 650); }
function stopPasses() { if (passTimer) clearInterval(passTimer); passTimer = null; passIdx = -1; setPass(-1); const b = document.querySelector('#v4c-bar [data-act="passes"]'); if (b) b.classList.remove("on"); }
function resize() {
  const st = document.getElementById("v4c-stage"), w = st.clientWidth, h = st.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); composer.setPixelRatio(renderer.getPixelRatio()); composer.setSize(w, h); labelRenderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix(); requestRender();
}
function stopFlow() {
  stopAuto(); if (passTimer) stopPasses();
  if (!flowOn) return; flowOn = false;
  const b = document.querySelector('#v4c-bar [data-act="flow"]'); if (b) b.classList.remove("on");
}
async function ensureArch() { if (!window.ARCH4) window.ARCH4 = await (await fetch("data/arch_v4.json")).json(); }

window.addEventListener("tabshown", async e => {
  if (e.detail !== "v4arch3d") { if (built) stopFlow(); return; }
  await ensureArch();
  if (!built) build(); else { resize(); requestRender(); }
});
document.addEventListener("visibilitychange", () => { if (document.hidden && built) stopFlow(); });
if (document.getElementById("tab-v4arch3d").classList.contains("on")) ensureArch().then(() => { if (!built) build(); });
