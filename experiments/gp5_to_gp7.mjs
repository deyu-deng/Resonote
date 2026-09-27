// Convert Resonote's .gp5 (Guitar Pro 5) to the modern .gp (GP7/8) format
// using alphaTab's Gp7Exporter — Guitar Pro 8 opens both, but .gp carries
// the newer format features. Usage: node gp5_to_gp7.mjs in.gp5 out.gp
//
// The exporters are only shipped in dist/alphaTab.core.mjs (the main ESM
// entry tree-shakes them out), and the package "exports" map blocks
// subpaths — so the core file is imported by absolute file URL.
import * as fs from 'fs';
import * as path from 'path';
import { pathToFileURL } from 'url';

const [src, dst] = process.argv.slice(2);
if (!src || !dst) {
  console.error('usage: node gp5_to_gp7.mjs <in.gp5> <out.gp>');
  process.exit(2);
}

// walk up from cwd to find the installed core bundle
function findCore(start) {
  let dir = start;
  while (true) {
    const cand = path.join(dir, 'node_modules', '@coderline', 'alphatab',
                           'dist', 'alphaTab.core.mjs');
    if (fs.existsSync(cand)) return cand;
    const parent = path.dirname(dir);
    if (parent === dir) {
      throw new Error('node_modules/@coderline/alphatab not found — ' +
                      'run: npm install @coderline/alphatab');
    }
    dir = parent;
  }
}

const core = await import(pathToFileURL(findCore(process.cwd())).href);
const score = core.importer.ScoreLoader.loadScoreFromBytes(
    new Uint8Array(fs.readFileSync(src)), new core.Settings());
console.log('loaded:', score.title, '| bars:', score.masterBars.length);

const exporter = new core.exporter.Gp7Exporter();
const bytes = exporter.export(score, new core.Settings());
fs.writeFileSync(dst, Buffer.from(bytes));
console.log('wrote', dst, bytes.length, 'bytes (GP7 format)');

// round-trip: read our own .gp back with alphaTab's GP7 reader
const back = core.importer.ScoreLoader.loadScoreFromBytes(
    new Uint8Array(fs.readFileSync(dst)), new core.Settings());
let notes = 0;
for (const t of back.tracks)
  for (const st of t.staves)
    for (const bar of st.bars)
      for (const v of bar.voices)
        if (v && v.beats) for (const b of v.beats) notes += b.notes?.length || 0;
console.log('round-trip OK: bars', back.masterBars.length, '| notes', notes);
