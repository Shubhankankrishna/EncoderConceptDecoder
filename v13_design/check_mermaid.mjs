// Syntax check for every ```mermaid block in a Markdown file (v13_NOTES.md §14).
//
//   node v13_design/check_mermaid.mjs v13_NOTES.md <dir>
//
// <dir> holds a node_modules with mermaid@11.4.1 and jsdom@25.0.1 (npm install there; kept out of the repository).
// Each block is parsed by mermaid.parse under a jsdom window. Exit code 1 if any block fails.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [, , mdPath, depDir] = process.argv;
if (!mdPath || !depDir) {
  console.error('usage: node check_mermaid.mjs <file.md> <dir with node_modules>');
  process.exit(2);
}
const load = (p) => import(pathToFileURL(join(depDir, 'node_modules', p)).href);

const { JSDOM } = await load('jsdom/lib/api.js');
const dom = new JSDOM('<!doctype html><html><body></body></html>', { pretendToBeVisual: true });
for (const k of ['window', 'document', 'DOMParser', 'Element', 'HTMLElement', 'SVGElement', 'Node']) {
  globalThis[k] = k === 'window' ? dom.window : dom.window[k];
}
const mermaid = (await load('mermaid/dist/mermaid.core.mjs')).default;
mermaid.initialize({ startOnLoad: false });

const lines = readFileSync(mdPath, 'utf8').split('\n');
const blocks = [];
for (let i = 0; i < lines.length; i++) {
  if (lines[i].trim() === '```mermaid') {
    const start = i + 1;
    let j = start;
    while (j < lines.length && lines[j].trim() !== '```') j++;
    blocks.push({ line: start + 1, text: lines.slice(start, j).join('\n') });
    i = j;
  }
}
let failed = 0;
for (const [n, b] of blocks.entries()) {
  try {
    await mermaid.parse(b.text);
    console.log(`block ${n + 1} (line ${b.line}): ok`);
  } catch (e) {
    failed++;
    console.log(`block ${n + 1} (line ${b.line}): FAILED\n  ${String(e.message || e).split('\n').slice(0, 6).join('\n  ')}`);
  }
}
console.log(`${blocks.length} blocks, ${failed} failed`);
process.exit(failed ? 1 : 0);
