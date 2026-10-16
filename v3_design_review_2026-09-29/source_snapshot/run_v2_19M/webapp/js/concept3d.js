/* Concept v2 — 3D view: each design's EXPECTED trajectory of the plan state across passes (illustrative, not measured).
   x = pass (depth), y/z = a 2D picture of the plan state. Gold spheres = valid next-chunk plans (several: the target is multimodal).
   Renders on demand only. */
import * as THREE from "three";
import { OrbitControls } from "../vendor/jsm/controls/OrbitControls.js";

let built = false, renderer, scene, camera, controls, pending = false;
const groups = {};
const COL = { ROW0: 0x8a93a3, D1: 0x3b78c2, D2: 0xe07b39, D3: 0x2e9c8c, D4: 0x8f63c9, D5: 0xd14f7a, D6: 0x3a9a4a };
const TARGETS = [[12, 2.2, 1.2], [12, -1.8, 2.0], [12, 0.6, -2.4], [12, -2.4, -1.2]];

function req() { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; controls.update(); renderer.render(scene, camera); }); }

function tube(pts, color, r = 0.07, opacity = 1) {
  const curve = new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(...p)));
  const m = new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05, transparent: opacity < 1, opacity });
  return new THREE.Mesh(new THREE.TubeGeometry(curve, 160, r, 10, false), m);
}
function dots(pts, color, r = 0.13) {
  const g = new THREE.Group(), geo = new THREE.SphereGeometry(r, 16, 12), m = new THREE.MeshStandardMaterial({ color });
  pts.forEach(p => { const s = new THREE.Mesh(geo, m); s.position.set(...p); g.add(s); });
  return g;
}
const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

function trajectories() {
  const start = [0, 0.3, -0.2], T = TARGETS[0];
  const P = {};
  // today: collapses to a fixed point by pass ~2, flat afterwards (measured behaviour, drawn schematically)
  P.ROW0 = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(i => i < 2 ? lerp(start, [i * 1.5, 0.9, 0.3], i / 2) : [i * 1.5, 0.9, 0.3]);
  // D1: moves longer, may still flatten (the open question E4/E5 decide)
  P.D1 = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(i => { const t = 1 - Math.exp(-i / 3); return [i * 1.5, ...lerp(start.slice(1), [1.8, 1.0], t)]; });
  // D2: long trajectory spirals toward the endpoint; shortcut chord drawn separately
  P.D2 = [0, 1, 2, 3, 4, 5, 6].map(i => { const t = i / 6, a = t * Math.PI * 1.5; return [i * 2, lerp(start.slice(1), T.slice(1), t)[0] + Math.sin(a) * (1 - t) * 1.4, lerp(start.slice(1), T.slice(1), t)[1] + Math.cos(a) * (1 - t) * 1.4]; });
  // D3: windows of 4 passes, each window improves the previous (detached) state
  P.D3 = []; for (let i = 0; i <= 12; i++) { const t = 1 - Math.pow(0.78, i); P.D3.push([i, ...lerp(start.slice(1), TARGETS[1].slice(1), t).map((v, j) => v + (j ? 0.25 : -0.25) * Math.sin(i * 1.3) * (1 - t))]); }
  // D4: each pass takes a different (expert-routed) turn
  P.D4 = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(i => { const t = i / 8; const z = lerp(start.slice(1), TARGETS[2].slice(1), t); return [i * 1.5, z[0] + (i % 2 ? 0.9 : -0.9) * (1 - t), z[1] + (i % 3 ? 0.5 : -0.7) * (1 - t)]; });
  // D5: shared passes, then 4 candidate plans fork towards 4 different valid targets
  P.D5 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => lerp(start, [9, 0, 0], i / 9));
  // D6: D3's stepwise improvement per window + D4's expert-routed turns, converging to a valid plan
  P.D6 = []; for (let i = 0; i <= 12; i++) { const t = 1 - Math.pow(0.74, i); const z = lerp(start.slice(1), TARGETS[3].slice(1), t); P.D6.push([i, z[0] + (i % 2 ? 0.7 : -0.7) * (1 - t), z[1] + (i % 3 ? 0.4 : -0.6) * (1 - t)]); }
  return P;
}

function build() {
  const cv = document.getElementById("cv2-3d");
  renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
  camera.position.set(6, 9, 20);
  controls = new OrbitControls(camera, cv); controls.target.set(6, 0, 0); controls.addEventListener("change", req);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.6));
  const dl = new THREE.DirectionalLight(0xffffff, 1.4); dl.position.set(5, 10, 8); scene.add(dl);
  const grid = new THREE.GridHelper(28, 28, 0x9aa3b2, 0xd5dae2); grid.position.set(6, -3.2, 0); scene.add(grid);
  const ax = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-1, -3.1, 3.5), 15, 0x5a6270, 0.5, 0.3); scene.add(ax);
  TARGETS.forEach(t => { const s = new THREE.Mesh(new THREE.SphereGeometry(0.34, 24, 16), new THREE.MeshStandardMaterial({ color: 0xe0b73a, emissive: 0x6a4a00, emissiveIntensity: 0.25 })); s.position.set(...t); scene.add(s); });
  const P = trajectories();
  for (const k of Object.keys(P)) {
    const g = new THREE.Group(); g.add(tube(P[k], COL[k])); g.add(dots(P[k], COL[k]));
    if (k === "D2") g.add(tube([P.D2[0], [5, 1.8, 1.8], P.D2[6]], COL.D2, 0.045, 0.7));
    if (k === "D3" || k === "D6") [4, 8].forEach(i => { const r = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.05, 8, 24), new THREE.MeshStandardMaterial({ color: 0x1b6b60 })); r.position.set(...P[k][i]); r.rotation.y = Math.PI / 2; g.add(r); });
    if (k === "D5") TARGETS.forEach(t => g.add(tube([P.D5[9], lerp(P.D5[9], t, 0.5), t], COL.D5, 0.05)));
    groups[k] = g; scene.add(g);
  }
  const ro = new ResizeObserver(() => { const w = cv.clientWidth, h = cv.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); req(); });
  ro.observe(cv);
  built = true;
}

function highlight(k) {
  for (const [key, g] of Object.entries(groups)) {
    const on = k === "all" || key === k;
    g.traverse(o => { if (o.material) { o.material.transparent = true; o.material.opacity = on ? 1 : 0.12; } });
  }
  req();
}

function show(k) { if (!document.getElementById("cv2-3d").clientWidth) return; if (!built) build(); highlight(k); }
window.addEventListener("tabshown", e => { if (e.detail === "concept") show(window.__cv2sel || "D3"); });
window.addEventListener("cv2select", e => { window.__cv2sel = e.detail; show(e.detail); });
document.addEventListener("click", e => { const b = e.target.closest("[data-cv2show]"); if (b && built) highlight(b.dataset.cv2show); });
// the tab may already be open when this module finishes loading (three.js loads after app.js fired "tabshown")
if (document.getElementById("tab-concept")?.classList.contains("on")) requestAnimationFrame(() => show(window.__cv2sel || "D3"));
