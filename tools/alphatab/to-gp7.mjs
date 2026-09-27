// Convert a .gp5 to the modern .gp (GP7/8) container using alphaTab's
// Gp7Exporter. Guitar Pro 8 opens .gp5 natively, so this is for the features
// the old container cannot carry.
//
//   node tools/alphatab/to-gp7.mjs <in.gp5> <out.gp>
//
// The exporters only ship in dist/alphaTab.core.mjs (the main ESM entry
// tree-shakes them out) and the package "exports" map blocks that subpath,
// so it is imported by absolute file URL.
import * as fs from 'fs';
import * as path from 'path';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';

const [src, dst] = process.argv.slice(2);
if (!src || !dst) {
  console.error('usage: node to-gp7.mjs <in.gp5> <out.gp>');
  process.exit(2);
}

const require = createRequire(import.meta.url);
let core;
try {
  core = path.join(require.resolve('@coderline/alphatab'), '..', '..',
                   'dist', 'alphaTab.core.mjs');
  if (!fs.existsSync(core)) throw new Error(core);
} catch {
  console.error('alphaTab not installed — run: npm install  (in tools/alphatab/)');
  process.exit(2);
}
const alphaTab = await import(pathToFileURL(core).href);

const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(
    new Uint8Array(fs.readFileSync(src)), new alphaTab.Settings());
console.log('loaded:', score.title, '| bars:', score.masterBars.length);

const bytes = new alphaTab.exporter.Gp7Exporter()
    .export(score, new alphaTab.Settings());
fs.writeFileSync(dst, Buffer.from(bytes));
console.log('wrote', dst, bytes.length, 'bytes (GP7 format)');

// round-trip: read our own .gp back with alphaTab's GP7 reader
const back = alphaTab.importer.ScoreLoader.loadScoreFromBytes(
    new Uint8Array(fs.readFileSync(dst)), new alphaTab.Settings());
let notes = 0;
for (const t of back.tracks)
  for (const st of t.staves)
    for (const bar of st.bars)
      for (const v of bar.voices)
        if (v && v.beats) for (const b of v.beats) notes += b.notes?.length || 0;
console.log('round-trip OK: bars', back.masterBars.length, '| notes', notes);
