/* V9 Guide · 3D scenes. three.js 0.160 as an ES module straight from jsDelivr (no import map, so no addons: orbit, labels and rounded boxes are written here).
   scene "model": the architecture of any arm in ../src/variants.js (frozen parts grey, absent parts missing, K codes, adapters + gates).
   scene "tree":  search shapes of S5 (receding horizon, G(R), IMAG, token tree).
   Each scene builds when it is first near the viewport and renders only while visible. */
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const { H, ui } = window.G;
const V9V = window.V9V;
const COL = { tok: 0x9aa3b2, enc: 0x3b78c2, con: 0x2e9c8c, bh: 0x1f7f72, code: 0xd9a227, sel: 0xe07b39, dec: 0x7357c9, frz: 0xaab2bf, out: 0xb9c0cc, gold: 0xf0b429, ink: 0x2b3240, lora: 0x7357c9 };
const ZE = -2.4, ZD = 2.4, NT = 16;
const CHUNK = ["a road", "last road of the puzzle", "answer hop 1", "answer hop 2"];
const CHOSEN = [1, 3, 0, 2];
const tokX = t => (t - 7.5) * 0.62 + (Math.floor(t / 4) - 1.5) * 0.5;
const chunkX = j => (tokX(4 * j) + tokX(4 * j + 3)) / 2;

/* plain-language part descriptions (keys are the part ids used in variants.js) */
const PINFO = {
  tokens: ["Tokens", "--tok", "The text as ID numbers, four to a chunk. Four chunks are drawn: puzzle roads, then answer hops. One chunk is one reasoning step."],
  enc: ["Encoder, 3 blocks", "--enc", "Reads the real tokens. Each block lets every token look back at the earlier ones, then think. This is ordinary language-model work, on the back plane."],
  emb: ["Token embedding", "--enc", "Swaps each token ID for a vector of 128 numbers. The same table is used again at the output end."],
  H: ["Token features", "--enc", "The encoder's output for each token. They go two places: up to be pooled into concepts, and straight to the decoder."],
  pool: ["Pooling funnel", "--con", "Averages the four token vectors of a chunk into one concept vector."],
  cin: ["Concept", "--con", "One summary note per chunk. In S2 closed-loop practice, the model's own guesses are fed in here instead of the real notes."],
  core: ["Concept stack", "--con", "Blocks that read only concepts, four times fewer positions than tokens, so each block is cheap. In S2 this is a fresh copy that gets trained."],
  bh0: ["Last concept block, no code", "--con", "The final block run plainly. Its output is the model's one default guess, and feeds the selector and the teacher check."],
  codes: ["Codes", "--code", "A handful of learnable code vectors, each a different 'what if' that nudges the last block."],
  bhk: ["Last concept block, run once per code", "--code", "The same block and weights, asked K times, each time with a different code added. One pass serves all K hypotheses."],
  pq: ["Guess head", "--con", "Turns a vector into a blend of stickers, the guess of the next concept."],
  Q: ["Codebook of stickers", "--code", "The fixed set of reusable vectors the guesses are blended from."],
  hyp: ["Hypotheses", "--code", "Each code commits to one possible next concept. Winner-take-all training pulls each onto a different real possibility."],
  sel: ["Selector", "--sel", "Bars showing how likely each hypothesis is to be the right one. It becomes the search prior in S5 and the policy trained in S4."],
  bestm: ["Best of m codes", "--sel", "In training, m codes are drawn using the selector's odds and the closest one steers the writer."],
  plan: ["Plan arc", "--con", "The chosen guess travels to the decoder and steers the next chunk. It arrives one chunk later than it was made."],
  teacher: ["Frozen S1 teacher", "--frozen", "The S1 model that S2 learns from. It supplies the true next concepts and is never changed."],
  dsum: ["Decoder input", "--dec", "Token features plus the hint, added together before the decoder reads them."],
  dec: ["Decoder, 3 blocks", "--dec", "Writes the answer tokens, on the front plane. In S5 it only runs when we commit to a step."],
  head: ["Output", "--dec", "Turns the decoder's last vector into probabilities for every token."],
  flatblk: ["Plain transformer block", "--enc", "One of ten identical blocks over tokens. No pooling, no concept stream, no hint."],
  lora: ["Adapter (LoRA)", "--dec", "A thin add-on to a block's weights. The block underneath stays frozen and only the add-on learns."],
  gate: ["Gate", "--loss", "A switch that turns an adapter on only for inputs that look like the ones it was trained on."],
  vtau: ["Value and stop heads", "--sel", "Two tiny judges: how good is this position, and is stopping legal here? They replace the checker inside the search."],
};

/* ---------------------------------------------------------------- three helpers */
const GEO = new Map();
function rbox(w, h, d, r) {
  if (r == null) r = 0.06;
  const key = [w, h, d, r].map(v => v.toFixed(3)).join("|"); if (GEO.has(key)) return GEO.get(key);
  r = Math.min(r || 0.06, w / 2.3, h / 2.3, d / 2.3); const b = Math.min(r, d / 2.3), cr = Math.max(0.001, r - b * 0.999), W = w - 2 * b, Hh = h - 2 * b, x0 = -W / 2, y0 = -Hh / 2;
  const s = new THREE.Shape(); s.moveTo(x0 + cr, y0); s.lineTo(x0 + W - cr, y0); s.quadraticCurveTo(x0 + W, y0, x0 + W, y0 + cr); s.lineTo(x0 + W, y0 + Hh - cr); s.quadraticCurveTo(x0 + W, y0 + Hh, x0 + W - cr, y0 + Hh); s.lineTo(x0 + cr, y0 + Hh); s.quadraticCurveTo(x0, y0 + Hh, x0, y0 + Hh - cr); s.lineTo(x0, y0 + cr); s.quadraticCurveTo(x0, y0, x0 + cr, y0);
  const g = new THREE.ExtrudeGeometry(s, { depth: Math.max(0.001, d - 2 * b), bevelEnabled: true, bevelThickness: b, bevelSize: b, bevelSegments: 2, curveSegments: 4 }); g.translate(0, 0, -(d - 2 * b) / 2);
  GEO.set(key, g); return g;
}
const SPH = new THREE.SphereGeometry(1, 20, 14);
const mat = (color, o) => { o = o || {}; const op = o.opacity == null ? 1 : o.opacity; return new THREE.MeshStandardMaterial({ color, roughness: o.rough == null ? 0.5 : o.rough, metalness: 0.05, transparent: op < 1, opacity: op, depthWrite: op >= 1, emissive: o.emissive == null ? color : o.emissive, emissiveIntensity: o.ei == null ? 0.05 : o.ei, side: o.side || THREE.FrontSide }); };

function mountScene(fig) {
  const kind = fig.dataset.scene, VIDS = (fig.dataset.variants || "").split(",").filter(Boolean);
  const cap = fig.querySelector("figcaption");
  const uiBar = H("div", { class: "fig-ui" }), stage = H("div", { class: "stage3", role: "img", "aria-label": fig.dataset.title || "3D scene" }), insp = H("div", { class: "insp3", "aria-live": "polite" });
  const title = H("div", { class: "fig-title" }, fig.dataset.title || "", fig.dataset.tag ? H("i", {}, fig.dataset.tag) : null);
  fig.insertBefore(title, cap); fig.insertBefore(uiBar, cap); fig.insertBefore(stage, cap); fig.insertBefore(insp, cap);
  const canvas = H("canvas"); stage.append(canvas);
  const layer = H("div", { style: "position:absolute;inset:0;overflow:hidden;pointer-events:none" }); stage.append(layer);
  stage.append(H("div", { class: "hud3" }, "drag to rotate · + and − to zoom · click a part"));
  const hex = n => "#" + n.toString(16).padStart(6, "0");
  if (kind === "model") stage.append(H("div", { class: "leg3", html: [["tokens", COL.tok], ["encoder", COL.enc], ["concepts", COL.con], ["codes", COL.code], ["selector", COL.sel], ["decoder", COL.dec], ["frozen", COL.frz]].map(([n, c]) => `<span><i style="background:${hex(c)}"></i>${n}</span>`).join("") }));
  if (kind === "tree") stage.append(H("div", { class: "leg3", html: [["imagined step", COL.con], ["decoded step", COL.enc], ["token child", COL.dec], ["most-visited path", COL.gold]].map(([n, c]) => `<span><i style="background:${hex(c)}"></i>${n}</span>`).join("") }));
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" }); } catch (e) { stage.replaceChildren(H("div", { class: "no3d" }, "This browser could not start WebGL, so the 3D scene is unavailable. The 2D diagrams on this page show the same architecture.")); return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(38, 1, 0.1, 300);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8892a6, 1.15)); const dl = new THREE.DirectionalLight(0xffffff, 1.5); dl.position.set(6, 14, 10); scene.add(dl);
  const raycaster = new THREE.Raycaster(), ptr = new THREE.Vector2();

  /* ---- state */
  let V = V9V.byId(VIDS[0]) || V9V.ALL, Kcur = V.K || 0, explode = 0, flowOn = true, labelsOn = true, treeMode = "rh1", group = null, pick = [], flows = [], labels = [], explodables = [], sel = null, hover = null, vis = false, dirty = true, summary = null;
  const cam = { th: 0.08, ph: 1.22, r: 24, t: new THREE.Vector3(0.8, 3.4, 0) }, tween = { on: false };
  const VIEWS = { all: [[2.2, 10.8, 20.0], [0.6, 3.1, 0]], flat: [[2.5, 8.0, 16.5], [0.4, 3.2, 0]], enc: [[6, 4.5, -10], [0, 1.4, ZE]], con: [[2, 9.5, 7.0], [0, 5.5, ZE]], dec: [[-5, 4, 9], [0, 1.3, ZD]], top: [[0, 26, 4], [0, 3, 0]], rh: [[12, 8.5, 31], [12, 2.0, 0]], tree: [[0, 17, 31], [0, 2.4, 0]], fan: [[0, 9, 20], [0, 5.2, 0]], side: [[30, 6, 2], [0, 4, 0]] };
  function setCam(p, t) { const tt = new THREE.Vector3(...t), pp = new THREE.Vector3(...p), d = pp.clone().sub(tt); cam.r = d.length(); cam.ph = Math.acos(d.y / cam.r); cam.th = Math.atan2(d.x, d.z); cam.t.copy(tt); }
  function applyCam() { const s = Math.sin(cam.ph); camera.position.set(cam.t.x + cam.r * s * Math.sin(cam.th), cam.t.y + cam.r * Math.cos(cam.ph), cam.t.z + cam.r * s * Math.cos(cam.th)); camera.lookAt(cam.t); }
  function view(v, instant) {
    const [p, t] = VIEWS[v] || VIEWS.all; const tt = new THREE.Vector3(...t), pp = new THREE.Vector3(...p), d = pp.clone().sub(tt), r = d.length();
    const to = { r, ph: Math.acos(d.y / r), th: Math.atan2(d.x, d.z), t: tt }; if (instant) { cam.r = to.r; cam.ph = to.ph; cam.th = to.th; cam.t.copy(to.t); dirty = true; return; }
    let dth = to.th - cam.th; while (dth > Math.PI) dth -= 2 * Math.PI; while (dth < -Math.PI) dth += 2 * Math.PI;
    Object.assign(tween, { on: true, t0: performance.now(), from: { r: cam.r, ph: cam.ph, th: cam.th, t: cam.t.clone() }, to: Object.assign(to, { th: cam.th + dth }) });
  }
  const fit = () => { const w = stage.clientWidth || 600, h = stage.clientHeight || 400; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); const narrow = camera.aspect < 1.2; cam.fov = narrow; dirty = true; };
  new ResizeObserver(fit).observe(stage);

  /* ---- scene building */
  function clearScene() { if (group) { group.traverse(o => { if (o.material) o.material.dispose(); }); scene.remove(group); } layer.replaceChildren(); labels = []; pick = []; flows = []; explodables = []; group = new THREE.Group(); scene.add(group); sel = null; hover = null; }
  function box(o) { const m = new THREE.Mesh(rbox(o.w, o.h, o.d, o.r), mat(o.color, o)); m.position.set(o.x, o.y, o.z); group.add(m); if (o.id) { m.userData.id = o.id; m.userData.ei0 = m.material.emissiveIntensity; pick.push(m); } if (o.level != null) { m.userData.level = o.level; m.userData.baseY = o.y; explodables.push(m); } return m; }
  function ball(x, y, z, r, color, id, o) { o = o || {}; const m = new THREE.Mesh(SPH, mat(color, o)); m.scale.setScalar(r); m.position.set(x, y, z); group.add(m); if (id) { m.userData.id = id; m.userData.ei0 = m.material.emissiveIntensity; Object.assign(m.userData, o.info || {}); pick.push(m); } if (o.level != null) { m.userData.level = o.level; m.userData.baseY = y; explodables.push(m); } return m; }
  function label(html, pos, cls) { const el = H("div", { class: "l3 " + (cls || ""), html }); layer.append(el); labels.push({ el, p: new THREE.Vector3(...pos) }); }
  function tube(pts, color, r, id, o) { o = o || {}; const curve = new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(...p)), false, "centripetal"); const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, r, 8, false), mat(color, { opacity: o.opacity == null ? 0.85 : o.opacity, ei: 0.15 })); group.add(m); if (id) { m.userData.id = id; m.userData.ei0 = 0.15; pick.push(m); } if (o.flow !== false) { const dots = []; for (let i = 0; i < (o.dots || 3); i++) { const d = new THREE.Mesh(SPH, new THREE.MeshBasicMaterial({ color: new THREE.Color(color).lerp(new THREE.Color(0xffffff), 0.5) })); d.scale.setScalar(0.075); group.add(d); dots.push(d); } flows.push({ curve, dots, speed: o.speed || 0.35, phase: Math.random() }); } return curve; }
  function plate(z, color, w, d) { const p = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08, depthWrite: false })); p.rotation.x = -Math.PI / 2; p.position.set(0, -0.32, z); group.add(p); }

  function buildModel() {
    clearScene(); const T = V.train || {}, stt = id => T[id] || "N", Kc = Kcur, rec = V.core === "recurrent";
    const pbox = (part, o) => { const st = stt(part); if (st === "N") return null; if (st === "F") o = Object.assign({}, o, { color: COL.frz, opacity: 0.55, ei: 0.02 }); return box(Object.assign({ id: part }, o)); };
    if (V.body === "flat") {
      const Z = 0, band = i => (i < 3 ? COL.enc : i < 7 ? COL.con : COL.dec); plate(Z, COL.enc, 13, 2.6);
      for (let t = 0; t < NT; t++) {
        const x = tokX(t), j = Math.floor(t / 4); box({ x, y: 0, z: Z, w: 0.42, h: 0.42, d: 0.42, color: j < 2 ? COL.tok : 0xc3c9d3, id: "tokens", level: 0 });
        for (let i = 0; i < 10; i++) box({ x, y: 0.62 + 0.46 * i, z: Z, w: 0.5, h: 0.3, d: 0.95, color: band(i), id: "flatblk", level: 1 + i });
        box({ x, y: 0.62 + 4.6 + 0.08, z: Z, w: 0.4, h: 0.4, d: 0.4, color: COL.out, id: "head", level: 11 });
      }
      for (let j = 0; j < 4; j++) { const cx = chunkX(j); label(`<b>${CHUNK[j]}</b><span>chunk ${j}</span>`, [cx, -0.6, 0], "minor"); tube([[cx, 0.1, Z + 0.62], [cx, 2.6, Z + 0.64], [cx, 5.2, Z + 0.62]], COL.enc, 0.03, "flatblk", { opacity: 0.4, dots: 2 }); }
      [["blocks 1 to 3", 1.1], ["blocks 4 to 7", 2.9], ["blocks 8 to 10", 4.5]].forEach(b => label(`<b>${b[0]}</b>`, [tokX(15) + 1.7, b[1], Z], "minor"));
      const ghost = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(13, 2.6, 1.6)), new THREE.LineDashedMaterial({ color: 0x8a93a3, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.6 })); ghost.computeLineDistances(); ghost.position.set(0, 7.2, Z); group.add(ghost);
      label("<b>no concept stack, no guesses</b><span>this is where the hierarchy's concept stack would be</span>", [0, 7.2, Z], "minor");
      return;
    }
    plate(ZE, COL.enc, 13, 2.6); plate(ZD, COL.dec, 13, 2.6);
    label(`<b>Encoder</b><span>${stt("enc") === "F" ? "frozen" : "learns"} · back plane</span>`, [-6.0, 2.2, ZE], "major right");
    label(`<b>Decoder</b><span>${stt("dec") === "F" ? "frozen" : "learns"} · front plane</span>`, [-6.0, 2.2, ZD], "major right");
    label(rec ? "<b>Recurrent concept core</b><span>blocks revisited several times</span>" : "<b>Concept stack</b><span>reads concepts only</span>", [-6.0, 4.6, ZE], "major right");
    label(Kc > 1 ? `<b>Last block × ${Kc} codes</b><span>selector bars · best of 2</span>` : Kc === 1 ? "<b>Last block, one guess</b><span>no codes, no selector</span>" : "<b>One guess per chunk</b><span>no codes, no selector</span>", [-6.0, 6.5, ZE], "major right");
    for (let t = 0; t < NT; t++) {
      const x = tokX(t), j = Math.floor(t / 4), prompt = j < 2;
      box({ x, y: 0, z: ZE, w: 0.42, h: 0.42, d: 0.42, color: prompt ? COL.tok : 0xc3c9d3, id: "tokens", level: 0 });
      [0, 1, 2].forEach(k => pbox("enc", { x, y: 0.7 + 0.55 * k, z: ZE, w: 0.5, h: 0.32, d: 0.95, color: COL.enc, level: 1 + k }));
      [0.75, 1.3, 1.85].forEach((y, k) => pbox("dec", { x, y, z: ZD, w: 0.5, h: 0.32, d: 0.95, color: COL.dec, level: 1 + k }));
      pbox("head", { x, y: 2.5, z: ZD, w: 0.4, h: 0.4, d: 0.4, color: COL.out, level: 4 }); pbox("dsum", { x, y: 0.2, z: ZD, w: 0.5, h: 0.14, d: 0.95, color: COL.dec, opacity: 0.55, level: 0 });
    }
    for (let j = 0; j < 4; j++) {
      const cx = chunkX(j); label(`<b>${CHUNK[j]}</b><span>chunk ${j}</span>`, [cx, -0.6, 0], "minor");
      if (stt("pool") !== "N") { const fun = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 1.15, 0.6, 28, 1, true), mat(stt("pool") === "F" ? COL.frz : COL.con, { opacity: 0.35, side: THREE.DoubleSide, ei: 0.2 })); fun.position.set(cx, 2.45, ZE); group.add(fun); fun.userData.id = "pool"; fun.userData.ei0 = 0.2; pick.push(fun); fun.userData.level = 4; fun.userData.baseY = 2.45; explodables.push(fun); }
      if (stt("cin") !== "N") ball(cx, 3.15, ZE, 0.3, stt("cin") === "F" ? COL.frz : COL.con, "cin", { ei: 0.15, opacity: stt("cin") === "F" ? 0.55 : 1, level: 5 });
      if (!rec) for (let l = 0; l < 3; l++) pbox("core", { x: cx, y: 3.85 + 0.5 * l, z: ZE, w: 2.1, h: 0.3, d: 1.1, color: COL.con, level: 6 + l });
      else { const ys = [0, 1, 2, 3, 4].map(i => 3.72 + 0.33 * i); ys.forEach((y, i) => pbox("core", { x: cx, y, z: ZE, w: 2.1, h: 0.22, d: 1.1, color: i === 2 ? COL.sel : COL.con, level: 6 + i * 0.6 })); tube([[cx - 1.2, ys[4], ZE + 0.3], [cx - 1.5, (ys[0] + ys[4]) / 2, ZE + 0.3], [cx - 1.2, ys[0], ZE + 0.3]], COL.con, 0.025, "core", { opacity: 0.55, dots: 1, speed: 0.3 }); }
      pbox(Kc > 1 ? "bhk" : "bh0", { x: cx, y: 5.45, z: ZE, w: 2.1, h: 0.34, d: 1.1, color: Kc > 1 ? COL.bh : COL.con, level: 9 });
      let chosen = null;
      if (Kc === 0) { chosen = ball(cx, 6.3, ZE, 0.2, stt("pq") === "F" ? COL.frz : COL.con, "pq", { ei: 0.3, level: 10 }).position.clone(); }
      else {
        const step = Math.min(0.27, 2.0 / Kc), ch = CHOSEN[j] % Kc, pis = Array.from({ length: Kc }, (_, i) => (i === ch ? 3 : 1) * (0.6 + 0.4 * Math.abs(Math.sin(7 * i + j)))), ps = pis.reduce((a, b) => a + b, 0);
        for (let i = 0; i < Kc; i++) {
          const x = cx + (i - (Kc - 1) / 2) * step, z = ZE + 0.25 * Math.cos((i - (Kc - 1) / 2) * 0.5), on = i === ch, part = on && stt("bestm") !== "N" ? "bestm" : "hyp";
          const c = stt(part) === "N" ? null : box({ id: part, x, y: 6.3, z, w: on ? 0.24 : 0.18, h: on ? 0.24 : 0.18, d: on ? 0.24 : 0.18, color: COL.code, ei: on ? 0.6 : 0.05, opacity: on ? 1 : 0.75, level: 10 });
          if (on) chosen = c ? c.position.clone() : new THREE.Vector3(x, 6.3, z);
          if (Kc > 1 && stt("sel") !== "N") { const hh = Math.max(0.05, 1.4 * pis[i] / ps); box({ id: "sel", x, y: 6.6 + hh / 2, z: ZE - 0.45, w: Math.min(0.16, step * 0.7), h: hh, d: 0.08, color: COL.sel, level: 11 }); }
        }
      }
      tube([[cx, 0.1, ZE + 0.6], [cx, 1.5, ZE + 0.62], [cx, 3.0, ZE + 0.6]], COL.enc, 0.025, "enc", { opacity: 0.35, dots: 2 });
      tube([[cx + 1.15, 3.15, ZE], [cx + 1.2, 4.7, ZE], [cx + 1.15, 6.2, ZE]], COL.con, 0.025, "core", { opacity: 0.35, dots: 2 });
      tube([[cx - 0.6, 2.0, ZE + 0.5], [cx - 0.6, 1.4, 0], [cx - 0.6, 0.3, ZD - 0.5]], COL.enc, 0.035, "H", { opacity: 0.45, dots: 2 });
      tube([[cx + 0.9, 0.2, ZD + 0.6], [cx + 0.9, 1.4, ZD + 0.62], [cx + 0.9, 2.5, ZD + 0.6]], COL.dec, 0.025, "dec", { opacity: 0.35, dots: 2 });
      if (stt("plan") !== "N") { const nx = j < 3 ? chunkX(j + 1) : chunkX(3) + 2.3; tube([[chosen.x, 6.45, chosen.z], [chosen.x + 0.4, 7.5, ZE + 1.4], [(chosen.x + nx) / 2 + 0.3, 5.4, ZD + 1.2], [nx, 1.0, ZD + 0.9], [nx, 0.32, ZD + 0.55]], COL.con, 0.035, "plan", { dots: 3, speed: 0.25, opacity: 0.7 }); }
    }
    label("<b>the guess from chunk j steers chunk j + 1</b>", [chunkX(1) + 1.3, 6.95, 0.6], "minor");
    label("<b>token features H</b><span>the decoder still reads real tokens</span>", [chunkX(3) + 2.6, 1.6, 0.0], "minor");
    /* adapters and gates (S6 arms) */
    const ad = V.adapter; if (ad) {
      const gated = ad.kind === "lsl" && ad.gate, bar = (x, y, z, w) => box({ x, y, z, w, h: 0.07, d: 0.16, color: COL.lora, ei: 0.4, id: "lora" }), gate = (x, y, z) => { const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.17), mat(COL.code, { ei: 0.5 })); m.position.set(x, y, z); group.add(m); m.userData.id = "gate"; m.userData.ei0 = 0.5; pick.push(m); };
      const span = tokX(15) - tokX(0) + 0.9, cx0 = (tokX(0) + tokX(15)) / 2, xl = tokX(0) - 0.7;
      if (ad.where.includes("enc")) for (let k = 0; k < 3; k++) { bar(cx0, 0.7 + 0.55 * k + 0.28, ZE + 0.57, span); if (gated) gate(xl, 0.7 + 0.55 * k + 0.28, ZE + 0.57); }
      if (ad.where.includes("dec")) [0.75, 1.3, 1.85].forEach(y => { bar(cx0, y + 0.28, ZD + 0.57, span); if (gated) gate(xl, y + 0.28, ZD + 0.57); });
      if (ad.where.includes("core")) for (let j = 0; j < 4; j++) { const cx = chunkX(j); for (let l = 0; l < 3; l++) { const y = 3.85 + 0.5 * l + 0.2; bar(cx, y, ZE + 0.6, 2.0); if (gated && j === 0) gate(cx - 1.25, y, ZE + 0.6); } }
      label(`<b>${gated ? "adapters + gates" : "adapters"}</b><span>rank ${ad.rank}${gated ? " · gold diamonds are the gates" : ""}</span>`, [xl - 0.2, 0.2, ZE + 1.1], "minor");
    }
    if (stt("teacher") !== "N") { const tx = chunkX(3) + 3.6; for (let l = 0; l < 4; l++) box({ x: tx, y: 3.85 + 0.5 * l, z: ZE, w: 1.4, h: 0.3, d: 1.1, color: COL.frz, opacity: 0.45, id: "teacher", level: 6 + l }); label("<b>Frozen S1 teacher</b><span>supplies the true next concepts</span>", [tx, 6.0, ZE], "major"); }
    if (stt("vtau") !== "N") { const vx = chunkX(3) + 3.6; box({ x: vx, y: 6.6, z: ZE, w: 0.9, h: 0.5, d: 0.6, color: COL.sel, id: "vtau" }); label("<b>value + stop heads</b><span>two tiny linear judges</span>", [vx, 7.4, ZE], "minor"); }
    applyExplode();
  }

  /* ---- search shapes (S5) */
  function buildTree() {
    clearScene(); const mode = treeMode, tk = mode === "token", R = mode === "g2" ? 2 : mode === "g4" ? 4 : mode === "g1" ? 1 : mode === "imag" ? 0 : 1;
    let free = 0, grounded = 0, exec = 0;
    const lineMat = new THREE.LineBasicMaterial({ color: COL.frz, transparent: true, opacity: 0.65 });
    const line = (a, b, c) => group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), c ? new THREE.LineBasicMaterial({ color: c }) : lineMat));
    const cube = (p, s, color, info) => { const m = new THREE.Mesh(rbox(s, s, s, s * 0.18), mat(color, { ei: info && info.gold ? 0.5 : 0.06, emissive: info && info.gold ? COL.gold : color })); m.position.copy(p); group.add(m); m.userData = Object.assign({ id: "treenode", ei0: m.material.emissiveIntensity }, info); pick.push(m); return m; };
    const orb = (p, s, color, info) => { const m = new THREE.Mesh(SPH, mat(color, { ei: info && info.gold ? 0.55 : 0.06, emissive: info && info.gold ? COL.gold : color })); m.scale.setScalar(s); m.position.copy(p); group.add(m); m.userData = Object.assign({ id: "treenode", ei0: m.material.emissiveIntensity }, info); pick.push(m); return m; };
    if (mode.startsWith("rh")) {
      const Hh = mode === "rh2" ? 2 : 1, STEPS = 4, DX = 6.4;
      for (let s = 0; s <= STEPS; s++) {
        const x = s * DX, root = new THREE.Vector3(x, 0, 0); cube(root, 0.9, s === 0 ? COL.ink : COL.enc, { kind: "root", note: s === 0 ? "The prompt, read once." : "A real, decoded and re-read position. Search restarts from here.", gold: false });
        if (s === STEPS) { label("<b>DONE</b><span>answer checked by the verifier</span>", [x, -1.4, 0], "minor"); break; }
        const kids = [-2.0, 0, 2.0], ch = 1;
        kids.forEach((z, k) => {
          const c1 = new THREE.Vector3(x + 1.7, 2.1, z), best = k === ch; orb(c1, 0.3, COL.con, { kind: "free", note: "One code tried: a guess and one cheap step in concept space. The decoder does not run.", gold: best }); free++; line(root, c1, best ? COL.gold : null);
          if (Hh === 2) [-0.8, 0.8].forEach(dz => { const c2 = new THREE.Vector3(x + 3.2, 4.2, z + dz); orb(c2, 0.24, COL.con, { kind: "free", note: "A second imagined step. Still no decoder.", gold: best && dz > 0 }); free++; line(c1, c2, best && dz > 0 ? COL.gold : null); });
        });
        const next = new THREE.Vector3((s + 1) * DX, 0, 0); tube([[x + 0.5, 0.1, 0], [x + 2.8, -0.9, 0.2], [next.x - 0.5, 0.1, 0]], COL.dec, 0.06, "treenode", { dots: 2, speed: 0.3 }); exec++;
        label("<b>search</b><span>decoder-free</span>", [x + 1.7, Hh === 2 ? 5.1 : 3.2, 0], "minor"); if (s < 3) label("<b>execute</b><span>decode 4 tokens</span>", [x + 3.2, -1.8, 0.2], "minor");
      }
      label(`<b>${mode === "rh2" ? "RH(2): look two steps ahead, then commit one" : "RH(1): look one step ahead, then commit one"}</b><span>only three of the K codes are drawn per step</span>`, [0, 7.0, 0], "major left");
      summary = { free, grounded: 0, exec, cost: 5 * free + 28 * exec, token: 0, mode };
      return;
    }
    if (mode === "bon" || mode === "loop") {
      const lp = mode === "loop", n = lp ? 5 : 8, win = lp ? 4 : 5, D = 4; let nodes = 0;
      cube(new THREE.Vector3(0, 9.2, 0), 0.9, COL.ink, { kind: "root", note: "The prompt. Every attempt starts from here." });
      label(`<b>${lp ? "Answer loop: GEN = fresh attempt, CONT = refine a failed one" : "Best of N: independent samples until one is accepted"}</b><span>${lp ? "a bandit rule chooses between them" : "no tree, the budget buys more draws"}</span>`, [0, 10.3, 0], "major");
      for (let i = 0; i < n; i++) {
        const x = (i - (n - 1) / 2) * 1.9, cont = lp && i >= 3, ok = i === win; let prev = new THREE.Vector3(0, 9.2, 0);
        for (let d = 0; d < D; d++) { const p = new THREE.Vector3(x, 7.6 - d * 1.5, 0); cube(p, 0.5, cont ? 0x9b7be0 : COL.dec, { kind: "token", note: "A decoded step of a full answer (28 units of work).", gold: false }); line(prev, p, cont ? COL.lora : null); prev = p; nodes++; }
        const v = new THREE.Vector3(x, 7.6 - D * 1.5 - 0.3, 0); orb(v, 0.28, ok ? COL.gold : COL.frz, { kind: "verifier", note: ok ? "The checker accepts this answer. The first accepted answer ends the search." : "The checker rejects this answer. Each call costs budget.", gold: ok }); line(prev, v);
        label(`<b>${lp ? (cont ? "CONT " + (i - 2) : "GEN " + (i + 1)) : "sample " + (i + 1)}</b><span>${ok ? "accepted" : "rejected"}</span>`, [x, 7.6 - D * 1.5 - 1.1, 0], ok ? "major" : "minor");
        if (cont) { const a = new THREE.Vector3((i - 3 - (n - 1) / 2) * 1.9, 7.6 - 1.5, 0.4), b = new THREE.Vector3(x, 7.6 - 1.5, 0.4), c = new THREE.CatmullRomCurve3([a, a.clone().lerp(b, 0.5).add(new THREE.Vector3(0, 1.2, 1.2)), b]); group.add(new THREE.Mesh(new THREE.TubeGeometry(c, 24, 0.04, 8, false), mat(COL.lora, { ei: 0.4 }))); }
      }
      summary = { free: 0, grounded: nodes, exec: 0, cost: nodes * 28, token: nodes, mode }; return;
    }
    const DEPTH = 4, B = 3, GOAL = [0, 2, 1, 2], include = p => p.length <= 2 || (p.length === 3 && (p[0] === GOAL[0] || p[1] === GOAL[1])) || (p.length === 4 && p[0] === GOAL[0] && p[1] === GOAL[1] && (p[2] === GOAL[2] || p[3] === 0));
    (function node(path, a0, a1, parent) {
      const d = path.length, ang = (a0 + a1) / 2, rad = d * 2.5, pos = new THREE.Vector3(rad * Math.cos(ang), 8 - d * 1.9, rad * Math.sin(ang));
      let m = 0; while (m < d && path[m] === GOAL[m]) m++; const onGoal = m === d, visits = Math.round(2 + 40 * (m / Math.max(1, d)) * (onGoal ? 1 : 0.35) / (1 + 0.3 * d)), g = tk ? d > 0 : R > 0 && d > 0 && d % R === 0, size = d === 0 ? 0.55 : 0.16 + 0.05 * Math.sqrt(visits);
      const info = { kind: d === 0 ? "root" : tk ? "token" : g ? "ground" : "free", visits, gold: onGoal && d > 0, path };
      if (d === 0 || g) cube(pos, size * 1.6, d === 0 ? COL.ink : tk ? COL.dec : COL.enc, info); else orb(pos, size, COL.con, info);
      if (d > 0) { if (g) grounded++; else free++; }
      if (parent) { if (onGoal) { const c = new THREE.CatmullRomCurve3([parent, parent.clone().lerp(pos, 0.5).add(new THREE.Vector3(0, -0.2, 0)), pos]); group.add(new THREE.Mesh(new THREE.TubeGeometry(c, 20, 0.045, 8, false), mat(COL.gold, { ei: 0.4 }))); } else line(parent, pos); }
      if (d < DEPTH) for (let a = 0; a < B; a++) { const p = path.concat([a]); if (include(p)) { const w = (a1 - a0) / B; node(p, a0 + a * w, a0 + (a + 1) * w, pos); } }
    })([], 0, Math.PI * 2, null);
    for (let d = 1; d <= DEPTH; d++) { const g = tk || (R > 0 && d % R === 0); label(`<b>hop ${d}</b><span>${tk ? "forced colour, then decode (28)" : g ? "decoded and re-read" : "decoder-free (5)"}</span>`, [d * 2.5 + 1.3, 8 - d * 1.9, 0], g ? "major" : "minor"); if (g) { const ring = new THREE.Mesh(new THREE.RingGeometry(d * 2.5 - 0.12, d * 2.5 + 0.12, 96), new THREE.MeshBasicMaterial({ color: tk ? COL.dec : COL.enc, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false })); ring.rotation.x = -Math.PI / 2; ring.position.y = 8 - d * 1.9; group.add(ring); } }
    const lab = { g1: "G(1): every step decoded", g2: "G(2): decoded every 2nd step", g4: "G(4): decoded every 4th step", imag: "IMAG: imagine it all, decode only the final path", token: "Token tree: every child is decoded" }[mode];
    label(`<b>${lab}</b><span>${tk ? "three colours + STOP per node" : "prompt read once"}</span>`, [0, 8.9, 0], "major");
    const finalDecode = mode === "imag" ? DEPTH * 24 : 0;   // the chosen path is decoded once at the end
    summary = { free, grounded, exec: 0, cost: tk ? 28 * grounded : 5 * free + 29 * grounded + finalDecode, token: tk ? grounded : Math.floor((5 * free + 29 * grounded + finalDecode) / 28), mode, tk };
  }

  /* ---- inspector */
  function inspDefault() {
    if (kind === "tree") {
      const s = summary || {}, txt = { rh1: "Receding horizon with one step of lookahead. At every real step the model tries each code in concept space (decoder-free), picks the most visited, decodes only that one, re-reads it, and searches again.", rh2: "Receding horizon, two steps of lookahead. Imagined for two steps, then one real step. Deeper looking costs a little and risks drifting from reality.", g1: "G(1): one tree for the whole puzzle where every child is decoded and re-read. Safe and expensive. A control for the decoder-free idea.", g2: "G(2): imagine two steps, then decode and re-read, repeating. A dial between G(1) and IMAG.", g4: "G(4): imagine four steps between each decode. Cheap, but errors can pile up.", imag: "IMAG: imagine the whole route in concept space and decode only the winning path at the very end. The cheapest, and the most dependent on the imagined steps staying true.", bon: "The strongest simple baseline: draw whole answers at random, check each one, and stop at the first accepted. There is no tree, so the budget buys more draws.", loop: "The outer loop wrapped around the best search. Fresh attempts (GEN) and refinements of a failed answer (CONT, by nudging a code) share one budget, and a bandit rule decides which to try next.", token: "The competitor: a tree whose children are tokens. Each child forces one colour, then decodes the rest of the hop with the real decoder." }[s.mode || treeMode];
      insp.innerHTML = `<div class="cat"><i style="background:var(--code)"></i>S5 · search shape</div><h5>${fig.querySelector(".seg button.on") ? fig.querySelector(".seg button.on").textContent : ""}</h5><p>${txt}</p><p class="muted" style="font-size:13px">Teal balls are imagined steps (about 5 units of work, no decoder). Blue cubes are decoded and re-read positions. Violet cubes are token-tree children (28 units each). Gold marks the most-visited path. Drawn: ${s.free || 0} imagined, ${s.grounded || 0} decoded${s.cost ? `, about <b>${s.cost}</b> units of work, against ${s.token ? `<b>${s.token * 28}</b> for a token tree of the same size` : "a token tree"}` : ""}. Synthetic illustration.</p>`;
      return;
    }
    const P = V.params || {}, f = n => (n == null ? "not stated" : n.toLocaleString("en-US")), tint = { S1: "--enc", S2: "--con", S3: "--dec", S4: "--sel", S5: "--code", S6: "--loss" }[V.stage] || "--ink3";
    insp.innerHTML = `<div class="cat"><i style="background:var(${tint})"></i>${V.stage} · ${V.tier === "full" ? "full column (optional)" : "mandatory"}</div><h5>${V.label}${Kcur > 1 && V.K > 1 ? ` · K = ${Kcur}` : ""}</h5><p>${V.sub || ""}. Parameters in use: <b>${f(P.total)}</b>${P.trainable != null ? `, learning: <b>${f(P.trainable)}</b>` : ""}.</p><p class="muted" style="font-size:13px">${V.note ? V.note + " " : ""}Click any part to read what it does. Grey translucent parts are frozen in this arm, and parts the arm does not have are not drawn.</p>`;
  }
  function inspPart(o) {
    const u = o.userData;
    if (u.id === "treenode") { const kindTxt = { verifier: "The checker.", root: "The prompt, read once.", free: "An imagined step: one code tried, one cheap step in concept space. The decoder did not run.", ground: "A decoded and re-read position: 4 tokens written, read back, concepts replaced by the real ones.", token: "A token-tree child: one colour forced, then the hop decoded by the real decoder." }[u.kind] || u.note || ""; insp.innerHTML = `<div class="cat"><i style="background:var(--code)"></i>Search node</div><h5>${u.kind === "root" ? "Root" : u.kind === "verifier" ? "Checker" : u.kind === "free" ? "Imagined step" : u.kind === "token" ? "Token child" : u.kind === "ground" ? "Decoded step" : "Node"}${u.visits ? ` · ${u.visits} visits (synthetic)` : ""}</h5><p>${u.note || kindTxt}</p>${u.gold ? "<p>This node is on the most-visited path.</p>" : ""}`; return; }
    const I = PINFO[u.id]; if (!I) { inspDefault(); return; }
    const st = V.train && V.train[u.id], state = st === "T" ? "learns in this arm" : st === "F" ? "frozen in this arm" : st === "N" ? "not present in this arm" : "";
    insp.innerHTML = `<div class="cat"><i style="background:var(${I[1]})"></i>${V.label || ""}</div><h5>${I[0]}</h5><p>${I[2]}</p>${state ? `<p><span class="chip ${st === "T" ? "mand" : "hist"}">${state}</span></p>` : ""}`;
  }

  /* ---- controls */
  function applyExplode() { explodables.forEach(m => { m.position.y = m.userData.baseY + m.userData.level * explode * 0.42; }); flows.forEach(f => f.dots.forEach(d => { d.visible = explode < 0.02 && flowOn; })); dirty = true; }
  function rebuild() { if (kind === "tree") buildTree(); else buildModel(); applyExplode(); labels.forEach(l => { l.el.style.display = labelsOn ? "" : "none"; }); inspDefault(); dirty = true; }
  let kslider = null;
  const pickVariant = id => { V = V9V.byId(id) || V; Kcur = V.K || 0; if (kslider) { kslider.hidden = !(V.K > 1); if (V.K > 1) Kcur = kslider.get(); } rebuild(); view(V.body === "flat" ? "flat" : "all"); };
  if (kind === "model") {
    const opts = VIDS.map(id => { const v = V9V.byId(id); return [id, v ? v.label : id]; });
    if (opts.length > 1) uiBar.append(ui.seg(opts, VIDS[0], pickVariant, "arm"));
    if (fig.dataset.k) { kslider = ui.range("codes K", 2, 16, 1, 8, v => { Kcur = v; rebuild(); }, v => String(v)); kslider.hidden = !(V.K > 1); uiBar.append(kslider); if (V.K > 1) Kcur = 8; }
  }
  if (kind === "tree") {
    const MODES = [["rh1", "RH(1)"], ["rh2", "RH(2)"], ["g1", "G(1)"], ["g2", "G(2)"], ["g4", "G(4)"], ["imag", "IMAG"], ["token", "Token tree"], ["bon", "Best of N"], ["loop", "Answer loop"]];
    uiBar.append(ui.seg(MODES.filter(m => !fig.dataset.modes || fig.dataset.modes.split(",").includes(m[0])), treeMode, m => { treeMode = m; rebuild(); view(m.startsWith("rh") ? "rh" : (m === "bon" || m === "loop") ? "fan" : "tree"); }, "search shape"));
  }
  if (kind === "model") uiBar.append(ui.range("explode layers", 0, 1, 0.02, 0, v => { explode = v; applyExplode(); }, v => Math.round(v * 100) + "%"));
  uiBar.append(ui.check("data flow", true, v => { flowOn = v; applyExplode(); }), ui.check("labels", true, v => { labelsOn = v; labels.forEach(l => { l.el.style.display = v ? "" : "none"; }); }));
  const vb = kind === "model" ? [["all", "Whole"], ["enc", "Encoder"], ["con", "Concepts"], ["dec", "Decoder"], ["side", "Side"]] : [["tree", "Overview"], ["top", "From above"]];
  const vseg = H("div", { class: "seg", role: "group", "aria-label": "camera" }); vb.forEach(([k, l]) => { const b = H("button", { type: "button" }, l); b.addEventListener("click", () => view(k === "all" && V.body === "flat" ? "flat" : (kind === "tree" && k === "tree" && treeMode.startsWith("rh") ? "rh" : k))); vseg.append(b); });
  uiBar.append(vseg, ui.btn("−", () => { cam.r = Math.min(70, cam.r * 1.18); dirty = true; }), ui.btn("+", () => { cam.r = Math.max(5, cam.r / 1.18); dirty = true; }));

  /* ---- pointer: drag rotates, shift/right-drag pans, pinch zooms, click picks */
  const pts = new Map(); let dragMoved = 0, last = null, pinch = 0;
  stage.addEventListener("pointerdown", e => { stage.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); dragMoved = 0; last = [e.clientX, e.clientY]; tween.on = false; if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = Math.hypot(a[0] - b[0], a[1] - b[1]); } });
  stage.addEventListener("pointermove", e => {
    if (!pts.has(e.pointerId)) { hoverAt(e); return; }
    const prev = pts.get(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2) { const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (pinch) { cam.r = Math.min(70, Math.max(5, cam.r * pinch / d)); } pinch = d; dirty = true; dragMoved += 5; return; }
    const dx = e.clientX - prev[0], dy = e.clientY - prev[1]; dragMoved += Math.abs(dx) + Math.abs(dy);
    if (e.shiftKey || e.buttons === 2) { const s = cam.r * 0.0016, right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1); cam.t.addScaledVector(right, -dx * s).addScaledVector(up, dy * s); }
    else { cam.th -= dx * 0.006; cam.ph = Math.min(Math.PI - 0.12, Math.max(0.12, cam.ph - dy * 0.006)); }
    dirty = true;
  });
  const up = e => { const was = pts.has(e.pointerId); pts.delete(e.pointerId); pinch = 0; if (was && dragMoved < 6 && pts.size === 0) clickAt(e); };
  stage.addEventListener("pointerup", up); stage.addEventListener("pointercancel", e => { pts.delete(e.pointerId); });
  stage.addEventListener("wheel", e => { if (e.ctrlKey || e.metaKey) { e.preventDefault(); cam.r = Math.min(70, Math.max(5, cam.r * Math.exp(e.deltaY * 0.01))); dirty = true; } }, { passive: false });
  stage.addEventListener("dblclick", () => view(kind === "tree" ? "tree" : V.body === "flat" ? "flat" : "all"));
  stage.addEventListener("contextmenu", e => e.preventDefault());
  function rayHit(e) { const b = canvas.getBoundingClientRect(); ptr.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1); raycaster.setFromCamera(ptr, camera); const h = raycaster.intersectObjects(pick, false); return h.length ? h[0].object : null; }
  function setEI(o, on) { if (o && o.material && o.userData.ei0 != null && !(sel && sel.userData.id === o.userData.id && o.userData.id !== "treenode" && !on)) o.material.emissiveIntensity = on ? 0.5 : o.userData.ei0; }
  function hoverAt(e) { if (e.pointerType === "touch") return; const o = rayHit(e); if (o !== hover) { if (hover && !(sel && sel.userData.id === hover.userData.id)) pick.forEach(p => { if (p.userData.id === hover.userData.id && p.userData.id !== "treenode") p.material.emissiveIntensity = p.userData.ei0; }); hover = o; if (o && o.userData.id !== "treenode") pick.forEach(p => { if (p.userData.id === o.userData.id) p.material.emissiveIntensity = 0.5; }); stage.style.cursor = o ? "pointer" : ""; dirty = true; } }
  function clickAt(e) { const o = rayHit(e); pick.forEach(p => { if (p.userData.id !== "treenode") p.material.emissiveIntensity = p.userData.ei0; }); sel = o; if (o) { if (o.userData.id !== "treenode") pick.forEach(p => { if (p.userData.id === o.userData.id) p.material.emissiveIntensity = 0.5; }); inspPart(o); } else inspDefault(); dirty = true; }

  /* ---- render */
  const v3 = new THREE.Vector3(); let time = 0;
  function render(dt) {
    if (tween.on) { const k = Math.min(1, (performance.now() - tween.t0) / 650), e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2; cam.r = THREE.MathUtils.lerp(tween.from.r, tween.to.r, e); cam.ph = THREE.MathUtils.lerp(tween.from.ph, tween.to.ph, e); cam.th = THREE.MathUtils.lerp(tween.from.th, tween.to.th, e); cam.t.lerpVectors(tween.from.t, tween.to.t, e); if (k >= 1) tween.on = false; dirty = true; }
    if (flowOn && explode < 0.02) { time += dt || 0; flows.forEach(f => f.dots.forEach((d, i) => { const u = (time * f.speed + f.phase + i / f.dots.length) % 1; d.position.copy(f.curve.getPointAt(u)); })); dirty = true; }
    if (!dirty) return; dirty = false;
    applyCam(); renderer.render(scene, camera);
    const w = stage.clientWidth, h = stage.clientHeight;
    labels.forEach(l => { v3.copy(l.p).project(camera); const ok = v3.z < 1 && Math.abs(v3.x) < 1.15 && Math.abs(v3.y) < 1.15; l.el.style.visibility = ok ? "" : "hidden"; if (ok) l.el.style.transform = `translate(${((v3.x + 1) / 2 * w).toFixed(1)}px, ${((1 - v3.y) / 2 * h).toFixed(1)}px) translate(${l.el.classList.contains("left") ? "0" : l.el.classList.contains("right") ? "-100%" : "-50%"}, -50%)`; });
  }
  let raf = 0, lastT = 0;
  function frame(now) { raf = 0; if (!vis) return; render(Math.min(0.05, (now - lastT) / 1000)); lastT = now; raf = requestAnimationFrame(frame); }
  new IntersectionObserver(es => es.forEach(e => { vis = e.isIntersecting; if (vis) dirty = true; if (vis && !raf) { lastT = performance.now(); raf = requestAnimationFrame(frame); } }), { rootMargin: "100px" }).observe(stage);

  setCam(...(kind === "tree" ? VIEWS.rh : V.body === "flat" ? VIEWS.flat : VIEWS.all));
  fit(); rebuild(); render(0.016);
  const api = { fig, render: n => { for (let i = 0; i < (n || 1); i++) { dirty = true; render(0.05); } }, view, setVariant: id => { V = V9V.byId(id); rebuild(); }, get state() { return { V: V.id, K: Kcur, picks: pick.length, labels: labels.length, flows: flows.length, summary }; } };
  (window.G3 = window.G3 || { scenes: [], render: n => window.G3.scenes.forEach(s => s.render(n)) }).scenes.push(api);
}

function boot() { document.querySelectorAll("figure.fig3[data-scene]").forEach(fig => { try { mountScene(fig); } catch (e) { console.error("3D scene", e); fig.append(H("div", { class: "no3d" }, "This 3D scene could not be built.")); } }); }
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
