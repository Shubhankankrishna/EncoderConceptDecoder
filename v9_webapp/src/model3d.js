/* V9 Atlas · 3D scenes (three.js 0.160 from jsdelivr).
   Model scene: four chunks along x. Encoder on the back plane (z < 0), decoder on the front plane (z > 0),
   concept core, hypothesis block and K codes rising above each chunk; plans arc from chunk j's chosen code to chunk j+1's decoder.
   Tree scene: radial PUCT tree over codes, re-grounded every R hops. Built lazily when the 3D tab opens; renders only while visible. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const COL = { tok: 0x9aa3b2, enc: 0x3b78c2, con: 0x2e9c8c, bh: 0x1f7f72, code: 0xd9a227, sel: 0xe07b39, dec: 0x7357c9, frz: 0xaab2bf, out: 0xb9c0cc, gold: 0xf0b429 };
const ZE = -2.4, ZD = 2.4, NT = 16;
const CHUNK = ["edge", "last prompt chunk", "hop 1", "hop 2"];
const CHOSEN = [1, 3, 0, 2];
const GOAL = [0, 2, 1, 2];
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let built = false, visible = false, renderer, labelRenderer, scene, camera, controls, raycaster, pointer;
let modelG, treeG, mode = "model", K = 8, R = 2, explode = 0, flowOn = true, labelsOn = true, treeOpts = { kind: "tree", token: false, label: "" };
let pickables = [], flows = [], explodables = [], hovered = null, selectedId = null, camTween = null, lastT = 0;

const cur = () => (window.V9 && window.V9.getVariant && window.V9.getVariant()) || { id: "ALL", train: null, K: 8, body: "hier", core: "dense", label: "All parts", sub: "", params: {} };
const stt = id => { const V = cur(); return V.train ? (V.train[id] || "N") : "T"; };
const Kv = () => { const V = cur(); if (V.codeFree) return 0; return V.id === "ALL" ? K : (V.K || 0); };
function pbox(G, part, o) {          // a box whose look follows the variant's train state: T normal, F frozen grey, N omitted
  const st = stt(part); if (st === "N") return null;
  if (st === "F") o = Object.assign({}, o, { color: COL.frz, opacity: 0.55, ei: 0.02 });
  return box(G, Object.assign({ id: part }, o));
}
const tokX = t => (t - 7.5) * 0.62 + (Math.floor(t / 4) - 1.5) * 0.5;
const chunkX = j => (tokX(4 * j) + tokX(4 * j + 3)) / 2;

/* ------------------------------------------------------------------ helpers */
function mat(color, o = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: o.rough ?? 0.45, metalness: 0.05, transparent: (o.opacity ?? 1) < 1, opacity: o.opacity ?? 1,
    depthWrite: (o.opacity ?? 1) >= 1, emissive: o.emissive ?? color, emissiveIntensity: o.ei ?? 0.04, side: o.side ?? THREE.FrontSide });
}
function add(parent, mesh, id, level, base) {
  parent.add(mesh);
  mesh.castShadow = true; mesh.receiveShadow = true;
  if (id) { mesh.userData.id = id; mesh.userData.ei0 = mesh.material.emissiveIntensity; pickables.push(mesh); }
  if (level != null) { mesh.userData.level = level; mesh.userData.baseY = base ?? mesh.position.y; explodables.push(mesh); }
  return mesh;
}
function box(parent, o) {
  const g = new RoundedBoxGeometry(o.w, o.h, o.d, 2, Math.min(0.07, o.h / 2.5, o.w / 2.5));
  const m = new THREE.Mesh(g, mat(o.color, o)); m.position.set(o.x, o.y, o.z);
  return add(parent, m, o.id, o.level, o.y);
}
function label(parent, html, pos, cls = "") {
  const el = document.createElement("div"); el.className = "l3 " + cls; el.innerHTML = html;
  const obj = new CSS2DObject(el); obj.position.set(...pos); parent.add(obj); return obj;
}
function tube(parent, pts, color, r, id, o = {}) {
  const curve = new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(...p)), false, "centripetal");
  const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, r, 8, false), mat(color, { opacity: o.opacity ?? 0.85, ei: 0.15 }));
  add(parent, m, id);
  if (o.flow !== false) addFlow(parent, curve, color, o.dots ?? 3, o.speed ?? 0.35);
  return curve;
}
function addFlow(parent, curve, color, n, speed) {
  const dots = [];
  for (let i = 0; i < n; i++) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.075, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    s.material.color.setHex(color).lerp(new THREE.Color(0xffffff), 0.45); parent.add(s); dots.push(s);
  }
  flows.push({ curve, dots, speed, phase: Math.random() });
}
function disposeGroup(g) {
  g.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); if (o.element) o.element.remove(); });
  g.parent && g.parent.remove(g);
}

/* ------------------------------------------------------------------ model scene */
function plate(G, z, color, w, d) { const p = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08, depthWrite: false })); p.rotation.x = -Math.PI / 2; p.position.set(0, -0.32, z); G.add(p); }
function buildModel() {
  if (modelG) disposeGroup(modelG);
  pickables = pickables.filter(p => p.userData.scene === "tree"); flows = []; explodables = [];
  modelG = new THREE.Group(); scene.add(modelG);
  const V = cur();
  if (V.body === "flat") buildFlat(modelG, V); else buildHier(modelG, V);
  applyExplode(); applyLabels();
}
function buildFlat(G, V) {
  const Z = 0, bandCol = i => (i < 3 ? COL.enc : i < 7 ? COL.con : COL.dec);
  plate(G, Z, COL.enc, 13, 2.6);
  label(G, `<b>${V.label}</b><span>${V.sub}</span>`, [-7.4, 8.4, Z], "zone major");
  label(G, "<b>Flat Transformer</b><span>10 blocks · token stream only</span>", [-7.4, 5.6, Z], "zone major");
  for (let t = 0; t < NT; t++) {
    const x = tokX(t), j = Math.floor(t / 4);
    box(G, { x, y: 0, z: Z, w: 0.42, h: 0.42, d: 0.42, color: j < 2 ? COL.tok : 0xc3c9d3, id: "tokens", level: 0 });
    for (let i = 0; i < 10; i++) box(G, { x, y: 0.62 + 0.46 * i, z: Z, w: 0.5, h: 0.3, d: 0.95, color: bandCol(i), id: "flatblk", level: 1 + i });
    box(G, { x, y: 0.62 + 0.46 * 10 + 0.08, z: Z, w: 0.4, h: 0.4, d: 0.4, color: COL.out, id: "head", level: 11 });
  }
  for (let j = 0; j < 4; j++) {
    const cx = chunkX(j);
    label(G, `<b>${CHUNK[j]}</b><span>chunk ${j} · no meaning to this model</span>`, [cx, -0.55, 0], "minor");
    tube(G, [[cx, 0.1, Z + 0.62], [cx, 2.6, Z + 0.64], [cx, 5.2, Z + 0.62]], COL.enc, 0.03, "flatblk", { opacity: 0.4, dots: 2 });
  }
  [["blocks 1–3 ≙ E3", 1.1, COL.enc], ["blocks 4–7 ≙ C4", 2.9, COL.con], ["blocks 8–10 ≙ D3", 4.5, COL.dec]].forEach(b => label(G, `<b>${b[0]}</b>`, [tokX(15) + 1.6, b[1], Z], "minor"));
  // ghost: where the hierarchy's concept stack would sit
  const ghost = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(13, 2.6, 1.6)), new THREE.LineDashedMaterial({ color: 0x8a93a3, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.55 }));
  ghost.computeLineDistances(); ghost.position.set(0, 7.2, Z); G.add(ghost);
  label(G, "<b>no concept stream, no codes, no plan</b><span>where the hierarchy's C4 core, B_H and codes would be</span>", [0, 7.2, Z], "minor");
}
function addAdapters(G, V) {
  const ad = V.adapter; if (!ad) return;
  const col = 0x7357c9, gated = ad.kind === "lsl" && ad.gate, rec = V.core === "recurrent";
  const bar = (x, y, z, w) => box(G, { x, y, z, w, h: 0.07, d: 0.16, color: col, ei: 0.4, id: "lora" });
  const gate = (x, y, z) => { const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.17), mat(0xd9a227, { ei: 0.5 })); m.position.set(x, y, z); G.add(m); m.userData.id = "gate"; m.userData.ei0 = 0.5; m.castShadow = true; pickables.push(m); };
  const span = tokX(15) - tokX(0) + 0.9, cx0 = (tokX(0) + tokX(15)) / 2, xl = tokX(0) - 0.7;
  if (ad.where.includes("enc")) for (let k = 0; k < 3; k++) { bar(cx0, 0.7 + 0.55 * k + 0.28, ZE + 0.57, span); if (gated) gate(xl, 0.7 + 0.55 * k + 0.28, ZE + 0.57); }
  if (ad.where.includes("dec")) [0.75, 1.3, 1.85].forEach(y => { bar(cx0, y + 0.28, ZD + 0.57, span); if (gated) gate(xl, y + 0.28, ZD + 0.57); });
  if (ad.where.includes("core")) for (let j = 0; j < 4; j++) { const cx = chunkX(j); for (let l = 0; l < 4; l++) { const y = (l < 3 ? 3.85 + 0.5 * l : 5.45) + 0.2; bar(cx, y, ZE + 0.6, 2.0); if (gated && j === 0) gate(cx - 1.25, y, ZE + 0.6); } }
  label(G, `<b>${gated ? "LSL adapters + gates" : "LoRA adapters"}</b><span>rank ${ad.rank} · ${ad.targets}${gated ? " · gold diamonds are the gates" : ""}</span>`, [xl - 0.2, 0.2, ZE + 1.1], "minor");
}
function buildHier(G, V) {
  const Kc = Kv(), rec = V.core === "recurrent", hasT = stt("teacher") !== "N";
  plate(G, ZE, COL.enc, 13, 2.6); plate(G, ZD, COL.dec, 13, 2.6);
  label(G, `<b>${V.label}</b><span>${V.sub}</span>`, [-7.4, 8.6, ZE], "zone major");
  label(G, `<b>Encoder E3</b><span>${stt("enc") === "F" ? "frozen" : "trains"} · back plane</span>`, [-7.4, 2.2, ZE], "zone major");
  label(G, `<b>Decoder D3</b><span>${stt("dec") === "F" ? "frozen" : "trains"} · front plane · writes chunk j+1</span>`, [-7.4, 2.2, ZD], "zone major");
  label(G, rec ? "<b>Recurrent HRM core</b><span>L, L, H, L, L then B_H = visit 6</span>" : "<b>Concept core C4</b><span>dense</span>", [-7.4, 4.4, ZE], "zone major");
  label(G, Kc > 1 ? `<b>B_H + ${Kc} codes</b><span>selector π · best of m = 2</span>` : Kc === 1 ? "<b>B_H · one concept (K = 1 = V8)</b><span>no codes, no selector</span>" : "<b>one concept prediction ĉ</b><span>no codes, no selector</span>", [-7.4, 6.5, ZE], "zone major");
  for (let t = 0; t < NT; t++) {
    const x = tokX(t), j = Math.floor(t / 4), prompt = j < 2;
    box(G, { x, y: 0, z: ZE, w: 0.42, h: 0.42, d: 0.42, color: prompt ? COL.tok : 0xc3c9d3, id: "tokens", level: 0 });
    ["E1", "E2", "E3"].forEach((n, k) => pbox(G, "enc", { x, y: 0.7 + 0.55 * k, z: ZE, w: 0.5, h: 0.32, d: 0.95, color: COL.enc, level: 1 + k }));
    [0.75, 1.3, 1.85].forEach((y, k) => pbox(G, "dec", { x, y, z: ZD, w: 0.5, h: 0.32, d: 0.95, color: COL.dec, level: 1 + k }));
    pbox(G, "head", { x, y: 2.5, z: ZD, w: 0.4, h: 0.4, d: 0.4, color: COL.out, level: 4 });
    pbox(G, "dsum", { x, y: 0.2, z: ZD, w: 0.5, h: 0.14, d: 0.95, color: COL.dec, opacity: 0.55, level: 0 });
  }
  for (let j = 0; j < 4; j++) {
    const cx = chunkX(j);
    label(G, `<b>${CHUNK[j]}</b><span>chunk ${j}</span>`, [cx, -0.55, 0], "minor");
    if (stt("pool") !== "N") { const fun = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 1.15, 0.6, 28, 1, true), mat(stt("pool") === "F" ? COL.frz : COL.con, { opacity: 0.35, side: THREE.DoubleSide, ei: 0.2 })); fun.position.set(cx, 2.45, ZE); add(G, fun, "pool", 4); }
    if (stt("cin") !== "N") { const sph = new THREE.Mesh(new THREE.SphereGeometry(0.3, 28, 18), mat(stt("cin") === "F" ? COL.frz : COL.con, { ei: 0.15, opacity: stt("cin") === "F" ? 0.55 : 1 })); sph.position.set(cx, 3.15, ZE); add(G, sph, "cin", 5); }
    if (!rec) for (let l = 0; l < 3; l++) pbox(G, "core", { x: cx, y: 3.85 + 0.5 * l, z: ZE, w: 2.1, h: 0.3, d: 1.1, color: COL.con, level: 6 + l });
    else {
      const ys = [0, 1, 2, 3, 4].map(i => 3.72 + 0.33 * i);
      ys.forEach((y, i) => pbox(G, "core", { x: cx, y, z: ZE, w: 2.1, h: 0.22, d: 1.1, color: i === 2 ? 0xe07b39 : COL.con, level: 6 + i * 0.6 }));
      tube(G, [[cx - 1.2, ys[4], ZE + 0.3], [cx - 1.5, (ys[0] + ys[4]) / 2, ZE + 0.3], [cx - 1.2, ys[0], ZE + 0.3]], COL.con, 0.025, "core", { opacity: 0.55, dots: 1, speed: 0.3 });
      tube(G, [[cx + 1.1, ys[2], ZE - 0.3], [cx + 1.45, 4.6, ZE - 0.3], [cx + 1.1, 5.45, ZE - 0.3]], 0xe07b39, 0.02, "bh0", { opacity: 0.5, dots: 1, speed: 0.2 });
    }
    pbox(G, Kc > 1 ? "bhk" : "bh0", { x: cx, y: 5.45, z: ZE, w: 2.1, h: 0.34, d: 1.1, color: Kc > 1 ? COL.bh : COL.con, level: 9 });
    // hypotheses + selector bars
    let chosenPos = null;
    if (Kc === 0) {
      const sp = new THREE.Mesh(new THREE.SphereGeometry(0.2, 22, 14), mat(stt("pq") === "F" ? COL.frz : COL.con, { ei: 0.3 })); sp.position.set(cx, 6.3, ZE); add(G, sp, "pq", 10); chosenPos = sp.position.clone();
    } else {
      const step = Math.min(0.27, 2.0 / Kc), ch = CHOSEN[j] % Kc;
      const pis = Array.from({ length: Kc }, (_, i) => (i === ch ? 3 : 1) * (0.6 + 0.4 * Math.abs(Math.sin(7 * i + j)))), ps = pis.reduce((a, b) => a + b, 0);
      for (let i = 0; i < Kc; i++) {
        const x = cx + (i - (Kc - 1) / 2) * step, z = ZE + 0.25 * Math.cos((i - (Kc - 1) / 2) * 0.5), on = i === ch, part = on && stt("bestm") !== "N" ? "bestm" : "hyp";
        const c = pbox(G, part, { x, y: 6.3, z, w: on ? 0.24 : 0.18, h: on ? 0.24 : 0.18, d: on ? 0.24 : 0.18, color: COL.code, ei: on ? 0.6 : 0.05, opacity: on ? 1 : 0.75, level: 10 });
        if (on) chosenPos = c ? c.position.clone() : new THREE.Vector3(x, 6.3, z);
        if (Kc > 1) { const h = Math.max(0.05, 1.4 * pis[i] / ps); pbox(G, "sel", { x, y: 6.6 + h / 2, z: ZE - 0.45, w: Math.min(0.16, step * 0.7), h, d: 0.08, color: COL.sel, level: 11 }); }
      }
    }
    // flows
    tube(G, [[cx, 0.1, ZE + 0.6], [cx, 1.5, ZE + 0.62], [cx, 3.0, ZE + 0.6]], COL.enc, 0.025, "enc", { opacity: 0.35, dots: 2 });
    tube(G, [[cx + 1.15, 3.15, ZE], [cx + 1.2, 4.7, ZE], [cx + 1.15, 6.2, ZE]], COL.con, 0.025, "core", { opacity: 0.35, dots: 2 });
    tube(G, [[cx - 0.6, 2.0, ZE + 0.5], [cx - 0.6, 1.4, 0], [cx - 0.6, 0.3, ZD - 0.5]], COL.enc, 0.035, "H", { opacity: 0.45, dots: 2 });
    tube(G, [[cx + 0.9, 0.2, ZD + 0.6], [cx + 0.9, 1.4, ZD + 0.62], [cx + 0.9, 2.5, ZD + 0.6]], COL.dec, 0.025, "dec", { opacity: 0.35, dots: 2 });
    const nx = j < 3 ? chunkX(j + 1) : chunkX(3) + 2.3;
    tube(G, [[chosenPos.x, 6.45, chosenPos.z], [chosenPos.x + 0.4, 7.5, ZE + 1.4], [(chosenPos.x + nx) / 2 + 0.3, 5.4, ZD + 1.2], [nx, 1.0, ZD + 0.9], [nx, 0.32, ZD + 0.55]], COL.con, 0.05, "plan", { dots: 3, speed: 0.25 });
  }
  label(G, V.codeFree ? "<b>code-free plan (ĉ⁽⁰⁾, A⁽⁰⁾) steers chunk j+1</b>" : "<b>plan from chunk j steers chunk j+1</b>", [chunkX(1) + 1.3, 6.9, 0.6], "minor");
  label(G, "<b>H_t</b><span>decoder still reads true tokens</span>", [chunkX(0) - 0.6, 2.9, 0.4], "minor");
  label(G, "<b>chunk 4 →</b>", [chunkX(3) + 2.3, 0.0, ZD + 0.6], "minor");
  addAdapters(G, V);
  if (hasT) {
    const tx = chunkX(3) + 3.6;
    for (let l = 0; l < 4; l++) box(G, { x: tx, y: 3.85 + 0.5 * l, z: ZE, w: 1.4, h: 0.3, d: 1.1, color: COL.frz, opacity: 0.45, id: "teacher", level: 6 + l });
    label(G, "<b>Frozen S1 teacher</b><span>targets C_j+1" + (stt("ldist") !== "N" ? " · distills u⁽⁰⁾" : "") + "</span>", [tx, 6.0, ZE], "major");
    if (stt("ldist") !== "N") tube(G, [[tx - 0.7, 4.6, ZE], [tx - 1.1, 5.0, ZE + 0.3], [chunkX(3) + 1.1, 5.45, ZE + 0.3]], COL.frz, 0.025, "ldist", { opacity: 0.5, dots: 1, speed: 0.2 });
  }
  if (stt("vtau") !== "N") { const vx = chunkX(3) + 3.6; box(G, { x: vx, y: 6.6, z: ZE, w: 0.9, h: 0.5, d: 0.6, color: COL.sel, id: "vtau" }); label(G, "<b>V_ψ · τ_ψ</b><span>two 128→1 heads on u⁽⁰⁾</span>", [vx, 7.4, ZE], "minor"); }
}

/* ------------------------------------------------------------------ tree scene */
function buildTree() {
  if (treeG) disposeGroup(treeG);
  if (treeOpts.kind !== "tree") return buildFan();
  pickables = pickables.filter(p => p.userData.scene !== "tree");
  treeG = new THREE.Group(); scene.add(treeG); treeG.visible = mode === "tree";
  const G = treeG, DEPTH = 4, B = 3, tk = !!treeOpts.token, Rr = treeOpts.R != null ? treeOpts.R : R;
  const include = p => p.length <= 2 || (p.length === 3 && (p[0] === GOAL[0] || p[1] === GOAL[1])) || (p.length === 4 && p[0] === GOAL[0] && (p[1] === GOAL[1]) && (p[2] === GOAL[2] || p[3] === 0));
  let nodes = 0, grounds = 0;
  const lineMat = new THREE.LineBasicMaterial({ color: COL.frz, transparent: true, opacity: 0.6 });
  function node(path, a0, a1, parentPos) {
    const d = path.length, ang = (a0 + a1) / 2, rad = d * 2.5;
    const pos = new THREE.Vector3(rad * Math.cos(ang), 8 - d * 1.9, rad * Math.sin(ang));
    let m = 0; while (m < d && path[m] === GOAL[m]) m++;
    const onGoal = m === d, visits = Math.round(2 + 40 * (m / Math.max(1, d)) * (onGoal ? 1 : 0.35) / (1 + 0.3 * d));
    const grounded = tk ? d > 0 : Rr > 0 && d > 0 && d % Rr === 0;
    const size = d === 0 ? 0.55 : 0.16 + 0.05 * Math.sqrt(visits);
    let mesh;
    if (d === 0 || grounded) mesh = new THREE.Mesh(new RoundedBoxGeometry(size * 1.6, size * 1.6, size * 1.6, 2, 0.05), mat(d === 0 ? 0x2b3240 : tk ? COL.dec : COL.enc, { ei: onGoal ? 0.35 : 0.05, emissive: onGoal ? COL.gold : undefined }));
    else mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 20, 14), mat(COL.con, { ei: onGoal ? 0.45 : 0.05, emissive: onGoal ? COL.gold : undefined }));
    mesh.position.copy(pos); G.add(mesh);
    mesh.userData = { scene: "tree", id: "treenode", path, visits, grounded, onGoal, ei0: mesh.material.emissiveIntensity }; pickables.push(mesh);
    if (d > 0) { nodes++; if (grounded) grounds++; }
    if (onGoal && d > 0 && Rr === 0) { const ring = new THREE.Mesh(new THREE.TorusGeometry(size + 0.18, 0.035, 8, 32), mat(COL.dec, { ei: 0.4 })); ring.position.copy(pos); ring.rotation.x = Math.PI / 2; G.add(ring); }
    if (parentPos) {
      if (onGoal) {
        const c = new THREE.CatmullRomCurve3([parentPos, parentPos.clone().lerp(pos, 0.5).add(new THREE.Vector3(0, -0.2, 0)), pos]);
        G.add(new THREE.Mesh(new THREE.TubeGeometry(c, 20, 0.045, 8, false), mat(COL.gold, { ei: 0.4 })));
        flows.push({ curve: c, dots: [], speed: 0, phase: 0 });
      } else G.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([parentPos, pos]), lineMat));
    }
    if (d < DEPTH) for (let a = 0; a < B; a++) { const p = path.concat([a]); if (include(p)) { const w = (a1 - a0) / B; node(p, a0 + a * w, a0 + (a + 1) * w, pos); } }
  }
  node([], 0, Math.PI * 2, null);
  for (let d = 1; d <= DEPTH; d++) {
    const grounded = tk ? true : Rr > 0 && d % Rr === 0;
    label(G, `<b>hop ${d}</b><span>${tk ? "child = forced colour, then decode 4 tokens (28)" : grounded ? "re-grounded: decode 4 tokens, re-encode" : "decoder-free: code → core step"}</span>`, [d * 2.5 + 1.2, 8 - d * 1.9, 0], grounded ? "major" : "minor");
    if (grounded) { const ring = new THREE.Mesh(new THREE.RingGeometry(d * 2.5 - 0.15, d * 2.5 + 0.15, 96), new THREE.MeshBasicMaterial({ color: COL.enc, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false })); ring.rotation.x = -Math.PI / 2; ring.position.y = 8 - d * 1.9; G.add(ring); }
  }
  label(G, `<b>${treeOpts.label || "root"}</b><span>${tk ? "token tree: three colours + STOP per node" : "prompt encoded once"}</span>`, [0, 8.7, 0], "major");
  const cost = tk ? 28 * nodes : 5 * nodes + 24 * grounds + (Rr === 0 ? 24 * DEPTH : 0);
  G.userData.summary = { nodes, grounds, cost, token: Math.floor(cost / 28), tk, Rr };
  applyLabels();
}


/* ------------------------------------------------------------------ fan scenes: best-of-N and the answer loop */
function buildFan() {
  treeG = new THREE.Group(); scene.add(treeG); treeG.visible = mode === "tree";
  pickables = pickables.filter(p => p.userData.scene !== "tree");
  const G = treeG, loop = treeOpts.kind === "loop", n = loop ? 5 : 8, win = loop ? 4 : 5, D = 4; let nodes = 0;
  const node = (x, y, c, info, ring) => {
    const m = new THREE.Mesh(new RoundedBoxGeometry(0.5, 0.5, 0.5, 2, 0.06), mat(c, { ei: ring ? 0.5 : 0.08 })); m.position.set(x, y, 0); G.add(m);
    m.userData = Object.assign({ scene: "tree", id: "fannode", ei0: ring ? 0.5 : 0.08 }, info); pickables.push(m); nodes++; return m;
  };
  const root = new THREE.Mesh(new RoundedBoxGeometry(0.9, 0.9, 0.9, 2, 0.06), mat(0x2b3240)); root.position.set(0, 9.2, 0); G.add(root);
  root.userData = { scene: "tree", id: "fannode", kind: "prompt" }; pickables.push(root);
  label(G, `<b>${treeOpts.label}</b><span>prompt encoded once per sample</span>`, [0, 10.1, 0], "major");
  const line = (a, b, c) => { G.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), new THREE.LineBasicMaterial({ color: c || COL.frz, transparent: true, opacity: 0.6 }))); };
  for (let i = 0; i < n; i++) {
    const x = (i - (n - 1) / 2) * 1.8, ok = i === win, gen = loop && i < 3, cont = loop && i >= 3;
    let prev = new THREE.Vector3(0, 9.2, 0);
    for (let d = 0; d < D; d++) {
      const y = 7.6 - d * 1.5, m = node(x, y, cont ? 0x9b7be0 : COL.dec, { kind: "chunk", chain: i + 1, depth: d + 1, cont, ok, loop }, ok);
      line(prev, m.position, cont ? 0x7357c9 : COL.frz); prev = m.position;
    }
    const v = new THREE.Mesh(new THREE.SphereGeometry(0.28, 20, 14), mat(ok ? COL.gold : COL.frz, { ei: ok ? 0.6 : 0.05 })); v.position.set(x, 7.6 - D * 1.5 - 0.3, 0); G.add(v);
    v.userData = { scene: "tree", id: "fannode", kind: "verifier", chain: i + 1, ok, ei0: ok ? 0.6 : 0.05 }; pickables.push(v); line(prev, v.position);
    label(G, `<b>${loop ? (cont ? "CONT " + (i - 2) : "GEN " + (i + 1)) : "sample " + (i + 1)}</b><span>${ok ? "verified ✓" : "failed"}</span>`, [x, 7.6 - D * 1.5 - 1.0, 0], ok ? "major" : "minor");
    if (cont) { // refine arc from a failed GEN answer to this continuation
      const src = (i - 3) * 1.8 - (n - 1) * 0.9 + 0.0; const a = new THREE.Vector3((i - 3 - (n - 1) / 2) * 1.8, 7.6 - 1.5, 0.4), b = new THREE.Vector3(x, 7.6 - 1.5, 0.4);
      const c = new THREE.CatmullRomCurve3([a, a.clone().lerp(b, 0.5).add(new THREE.Vector3(0, 1.2, 1.2)), b]); G.add(new THREE.Mesh(new THREE.TubeGeometry(c, 24, 0.04, 8, false), mat(0x7357c9, { ei: 0.4 })));
    }
  }
  label(G, loop ? "<b>Thompson sampling</b><span>GEN (dark violet) vs CONT (light violet: refine a failed answer)</span>" : "<b>no search</b><span>independent T = 1 samples until the verifier accepts</span>", [0, 1.9, 0], "major");
  G.userData.summary = { nodes, grounds: 0, cost: nodes * 28, token: nodes, fan: true, loop };
  applyLabels();
}

/* ------------------------------------------------------------------ inspector */
const EXTRA = {
  tokens: { t: "Tokens (4 chunks drawn)", cat: ["Input", "--tok"], d: "Sixteen positions in four chunks: a prompt edge, the last prompt chunk, hop 1 and hop 2. One chunk is one reasoning step.", src: "§4.2" },
  head: { t: "Output tokens", cat: ["Decoder", "--dec"], d: "Logits through the tied head. Each chunk's tokens are steered by the plan arriving from the previous chunk's chosen code.", src: "§5.1" },
};
function inspect(obj) {
  const host = $("#v3-insp"); if (!host) return;
  const V = cur();
  if (!obj) {
    if (mode === "tree") {
      const sm = treeG ? treeG.userData.summary : null;
      const fan = sm && sm.fan, tk = sm && sm.tk, Rr = treeOpts.R != null ? treeOpts.R : R;
      host.innerHTML = `<div class="cat"><i style="background:var(--con)"></i>Search · §9</div><h3>${V.kind === "search" ? V.label : "Decoder-free concept MCTS at R = " + (Rr === 0 ? "∞" : Rr)}</h3>
        <p>${fan ? (sm.loop ? "Dark violet cubes are fresh inner searches (GEN). Light violet cubes continue a failed answer (CONT) with the refine kernel; Thompson sampling splits the shared budget between them. The gold chain is the one the verifier accepts." : "Eight independent samples, each decoded fully and checked by the verifier. There is no tree: the budget is spent on more draws. The gold chain is the first accepted answer.") : tk ? "Violet cubes are token-tree children: each forces one of three colours (or STOP) and decodes the hop, 28 block-positions each. Gold marks the most-visited path." : `Teal spheres are decoder-free children: one code, one core step. Blue cubes sit on re-grounded hops. Gold marks the most-visited path${Rr === 0 ? ", decoded once at the end (violet rings)" : ""}. Node size grows with visits.`}</p>
        ${sm && !fan ? `<dl class="kv"><dt>children drawn</dt><dd>${sm.nodes}</dd><dt>re-groundings</dt><dd>${sm.grounds}</dd><dt>block-positions</dt><dd>${sm.cost}</dd><dt>token children, same spend</dt><dd>${sm.token}</dd></dl>` : ""}
        <p class="muted" style="font-size:12.5px">Illustrative scene with synthetic outcomes. Costs use §9.3 dense-core counts: 5 per decoder-free child, 28 per token child, +24 per re-grounding.</p><div class="src">draft_v9.md §9.2–§9.6</div>`;
    } else {
      const P = V.params || {}, fmt = n => (n == null ? "not stated" : n.toLocaleString("en-US"));
      host.innerHTML = `<div class="cat"><i style="background:var(${({ S1: "--enc", S2: "--con", S3: "--dec", S4: "--sel", S6: "--loss" })[V.stage] || "--code"})"></i>${V.id === "ALL" ? "3D model" : V.stage + " · " + (V.queue || "")}</div><h3>${V.label}</h3>
        <p>${V.sub || ""}</p>
        <dl class="kv"><dt>active parameters</dt><dd>${fmt(P.total)}</dd><dt>trainable</dt><dd>${fmt(P.trainable)}</dd>${V.K ? `<dt>codes K</dt><dd>${V.id === "ALL" ? K : V.K}</dd>` : ""}${V.core ? `<dt>core</dt><dd>${V.core}</dd>` : ""}</dl>
        <p>${V.body === "flat" ? "One plane, ten stacked blocks per token position, and nothing above them: the dashed box is where the hierarchy's concept stack would be." : "The encoder is on the back plane and the decoder on the front. Above each chunk: pooling funnel, concept, core blocks, the hypothesis block B_H, then the codes (amber) with selector bars (orange)."} <b>Grey translucent parts are frozen</b> in this arm; parts it does not have are not drawn. Click any part.</p>
        ${V.note ? `<p class="muted" style="font-size:12.5px">${V.note}</p>` : ""}<div class="src">draft_v9.md ${V.source || "§5–§6"}</div>`;
    }
    return;
  }
  const u = obj.userData;
  if (u.id === "fannode") {
    host.innerHTML = `<div class="cat"><i style="background:var(--dec)"></i>${u.kind === "prompt" ? "Prompt" : u.kind === "verifier" ? "Verifier" : "Decoded chunk"}</div><h3>${u.kind === "prompt" ? "Prompt, encoded once" : u.kind === "verifier" ? `Verifier · chain ${u.chain}` : `Chain ${u.chain} · hop ${u.depth}`}</h3>
      <p>${u.kind === "verifier" ? (u.ok ? "The binary verifier accepts this answer. The first accepted answer ends the search." : "The verifier rejects this answer. Each call is counted against the budget.") : u.kind === "prompt" ? "Every sample re-encodes the prompt and decodes from the start; nothing is shared between draws." : u.cont ? "A continuation: the refine kernel (GradCuit or a random direction of the same norm) edits a code at a uniformly chosen committed chunk of a failed answer, then decoding resumes." : "A fully decoded hop (28 block-positions); the answer is complete only when the terminator chunk appears."}</p><div class="src">draft_v9.md §9.3, §9.6</div>`;
    return;
  }
  if (u.id === "treenode") {
    host.innerHTML = `<div class="cat"><i style="background:var(--con)"></i>Search node</div><h3>${u.path.length ? u.path.map(a => "k" + (a + 1)).join(" → ") : "Root"}</h3>
      <dl class="kv"><dt>depth</dt><dd>hop ${u.path.length}</dd><dt>visits (synthetic)</dt><dd>${u.visits}</dd><dt>node type</dt><dd>${u.path.length === 0 ? "root" : treeOpts.token ? "token child (decoded)" : u.grounded ? "re-grounded" : "decoder-free"}</dd><dt>on chosen path</dt><dd>${u.onGoal ? "yes" : "no"}</dd></dl>
      <p>${treeOpts.token ? "Forced one colour at this hop, then greedy decode of 4 tokens under the code-free plan: 28 block-positions per child." : u.grounded ? "This hop was decoded into 4 tokens under the committed plan, then re-encoded, and the rolled concepts were replaced with true pooled concepts (+24 block-positions)." : u.path.length ? "Expanded without the decoder: code k → ĉ⁽ᵏ⁾ → one incremental core step → u⁽⁰⁾, π, V_ψ, τ_ψ (≈5 block-positions)." : "The prompt is encoded once and the core caches are built. π at the last prompt chunk is the prior for hop 1."}</p><div class="src">draft_v9.md §9.2</div>`;
    return;
  }
  const I = (window.V9 && window.V9.pinfo && window.V9.pinfo(u.id)) || EXTRA[u.id];
  if (!I) { host.innerHTML = ""; return; }
  const ex = EXTRA[u.id] || I, cat = ex.cat || I.cat || ["Part", "--ink3"], stt_ = V.train && V.train[u.id];
  host.innerHTML = `<div class="cat"><i style="background:var(${cat[1]})"></i>${cat[0]}</div><h3>${ex.t}</h3><p>${ex.d}</p>${I.eq ? `<pre>${I.eq}</pre>` : ""}${I.kv ? `<dl class="kv">${I.kv.map(k => `<dt>${k[0]}</dt><dd>${k[1]}</dd>`).join("")}</dl>` : ""}${stt_ && V.id !== "ALL" ? `<p><span class="pill ${stt_ === "T" ? "gate" : "untested"}">${V.label}: ${stt_ === "T" ? "trains / active" : stt_ === "F" ? "frozen" : "not present"}</span></p>` : ""}<div class="src">draft_v9.md ${ex.src || I.src || ""}</div>`;
}
function highlight(id, on) {
  pickables.forEach(p => { if (p.userData.id === id && id !== "treenode" && id !== "fannode") p.material.emissiveIntensity = on ? 0.45 : p.userData.ei0; });
}

/* ------------------------------------------------------------------ controls */
function applyExplode() { explodables.forEach(m => { m.position.y = m.userData.baseY + m.userData.level * explode * 0.42; }); flows.forEach(f => f.dots.forEach(d => d.visible = explode < 0.02 && flowOn)); }
function applyLabels() { if (labelRenderer) labelRenderer.domElement.style.display = labelsOn ? "" : "none"; }
function setMode(m) {
  mode = m; modelG.visible = m === "model"; if (!treeG) buildTree(); treeG.visible = m === "tree";
  const ALLV = cur().id === "ALL"; $("#v3-k").hidden = m !== "model" || !ALLV; $("#v3-view").hidden = m !== "model"; $("#v3-r").hidden = m !== "tree" || !ALLV;
  $("#v3-explode").parentElement.hidden = m !== "model";
  // hide CSS2D labels of the inactive group (CSS2DRenderer ignores parent visibility in some cases)
  [modelG, treeG].forEach(g => g.traverse(o => { if (o.element) o.element.style.visibility = g.visible ? "" : "hidden"; }));
  selectedId = null; inspect(null);
  view(m === "tree" ? "tree" : "all");
}
const VIEWS = { all: [[2.5, 11.5, 22.5], [0.8, 3.4, 0]], flat: [[2.5, 9.5, 20], [0.8, 3.6, 0]], enc: [[7, 4.5, -12], [0, 1.4, ZE]], con: [[2, 10.5, 8.5], [0, 5.5, ZE]], dec: [[-6, 4, 11], [0, 1.3, ZD]], tree: [[0, 17, 31], [0, 2.4, 0]], fan: [[0, 8.5, 17], [0, 5.2, 0]] };
function view(v, instant) {
  if (v === "all" && cur().body === "flat") v = "flat";
  if (v === "tree" && treeOpts.kind !== "tree") v = "fan";
  const [p, t] = VIEWS[v] || VIEWS.all;
  camTween = { p0: camera.position.clone(), p1: new THREE.Vector3(...p), q0: controls.target.clone(), q1: new THREE.Vector3(...t), t0: performance.now(), ms: instant ? 1 : 700 };
}
function segWire(sel, attr, fn) {
  document.querySelectorAll(sel + " button").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(sel + " button").forEach(x => x.classList.toggle("on", x === b)); fn(b.dataset[attr]);
  }));
}
function setHud() {
  const h = $("#hud3d"); if (!h) return;
  const V = cur(); h.textContent = mode === "model" ? `${V.label}${Kv() ? " · K = " + Kv() : ""} · drag to orbit · scroll to zoom · click a part` : `${treeOpts.label || "tree"}${treeOpts.kind === "tree" ? " · R = " + ((treeOpts.R != null ? treeOpts.R : R) === 0 ? "∞" : (treeOpts.R != null ? treeOpts.R : R)) : ""} · gold = accepted / most-visited · click a node`;
}

/* ------------------------------------------------------------------ init + loop */
function applyTheme() { if (!scene) return; const bg = css("--panel2") || "#f8f9fb"; scene.background = new THREE.Color(bg); }
function init() {
  const stage = $("#stage3d"), cv = $("#cv3d");
  try { renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true }); }
  catch (e) { stage.insertAdjacentHTML("beforeend", `<div class="no3d">WebGL is not available in this browser, so the 3D view can't render. The 2D tabs cover the same content.</div>`); return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  scene = new THREE.Scene(); applyTheme();
  camera = new THREE.PerspectiveCamera(36, 1, 0.1, 300);
  camera.position.set(2.5, 11.5, 22.5);
  labelRenderer = new CSS2DRenderer({ element: $("#labels3d") });
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8c0cc, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.7); sun.position.set(8, 16, 12); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); Object.assign(sun.shadow.camera, { left: -14, right: 14, top: 14, bottom: -6, near: 1, far: 50 }); scene.add(sun);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.ShadowMaterial({ opacity: 0.12 })); floor.rotation.x = -Math.PI / 2; floor.position.y = -0.35; floor.receiveShadow = true; scene.add(floor);
  const grid = new THREE.GridHelper(40, 40, 0x8a93a3, 0x8a93a3); grid.position.y = -0.34; grid.material.transparent = true; grid.material.opacity = 0.12; scene.add(grid);
  controls = new OrbitControls(camera, cv);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.08, minDistance: 4, maxDistance: 45, maxPolarAngle: Math.PI * 0.495 });
  controls.target.set(0, 3.2, 0);
  raycaster = new THREE.Raycaster(); pointer = new THREE.Vector2();

  buildModel();
  // input
  const pick = e => {
    const r = cv.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(pickables.filter(p => (p.userData.scene === "tree") === (mode === "tree")), false);
    return hits.length ? hits[0].object : null;
  };
  cv.addEventListener("pointermove", e => {
    const o = pick(e);
    if (hovered && hovered !== o) { if (hovered.userData.id === "treenode" || hovered.userData.id === "fannode") hovered.material.emissiveIntensity = hovered.userData.ei0; else if (hovered.userData.id !== selectedId) highlight(hovered.userData.id, false); }
    hovered = o;
    if (o) { if (o.userData.id === "treenode" || o.userData.id === "fannode") o.material.emissiveIntensity = 0.6; else highlight(o.userData.id, true); }
    cv.style.cursor = o ? "pointer" : "grab";
  });
  let down = null;
  cv.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
  cv.addEventListener("pointerup", e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const o = pick(e);
    if (selectedId) highlight(selectedId, false);
    selectedId = o && o.userData.id !== "treenode" && o.userData.id !== "fannode" ? o.userData.id : null;
    if (selectedId) highlight(selectedId, true);
    inspect(o);
  });
  segWire("#v3-scene", "s", setMode);
  function syncVariant() {
    const V = cur();
    if (V.kind === "search") {
      treeOpts = { kind: V.method === "best_of_n" ? "fan" : V.method === "answer_loop" ? "loop" : "tree", token: !!V.token && V.method !== "best_of_n", R: V.R, label: V.label };
      buildTree(); [...document.querySelectorAll("#v3-scene button")].forEach(b => b.classList.toggle("on", b.dataset.s === "tree")); setMode("tree");
    } else {
      treeOpts = { kind: "tree", token: false, label: "" }; buildTree(); buildModel();
      [...document.querySelectorAll("#v3-scene button")].forEach(b => b.classList.toggle("on", b.dataset.s === "model")); setMode("model");
    }
    setHud();
  }
  window.addEventListener("v9variant", syncVariant); syncVariant();
  segWire("#v3-k", "k", v => { K = +v; buildModel(); selectedId = null; inspect(null); setHud(); });
  segWire("#v3-r", "r", v => { R = +v; buildTree(); inspect(null); setHud(); });
  segWire("#v3-view", "v", v => view(v));
  $("#v3-flow").addEventListener("change", e => { flowOn = e.target.checked; applyExplode(); });
  $("#v3-labels").addEventListener("change", e => { labelsOn = e.target.checked; applyLabels(); });
  $("#v3-explode").addEventListener("input", e => { explode = +e.target.value; applyExplode(); });
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { flowOn = false; $("#v3-flow").checked = false; applyExplode(); }
  window.addEventListener("v9theme", applyTheme);
  new ResizeObserver(resize).observe(stage); resize();
  inspect(null); setHud(); view(mode === "tree" ? "tree" : "all", true);
  built = true;
}
function resize() {
  const stage = $("#stage3d"); const w = stage.clientWidth, h = stage.clientHeight; if (!w || !h) return;
  renderer.setSize(w, h, false); labelRenderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
function frame(t) {
  if (!visible) return;
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (t - (lastT || t)) / 1000); lastT = t;
  if (camTween) {
    const k = Math.min(1, (performance.now() - camTween.t0) / camTween.ms), e = 1 - Math.pow(1 - k, 3);
    camera.position.lerpVectors(camTween.p0, camTween.p1, e); controls.target.lerpVectors(camTween.q0, camTween.q1, e);
    if (k >= 1) camTween = null;
  }
  controls.update();
  if (flowOn && explode < 0.02) flows.forEach(f => { f.phase = (f.phase + dt * f.speed) % 1; f.dots.forEach((d, i) => d.position.copy(f.curve.getPointAt((f.phase + i / f.dots.length) % 1))); });
  renderer.render(scene, camera); labelRenderer.render(scene, camera);
}
function onTab(id) {
  visible = id === "model3d";
  if (visible && !built) init();
  if (visible && built) { resize(); lastT = 0; requestAnimationFrame(frame); }
}
window.addEventListener("v9tab", e => onTab(e.detail));
if (document.querySelector("#tab-model3d.on")) onTab("model3d");
