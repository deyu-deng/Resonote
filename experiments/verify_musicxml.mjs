// Independent validation: load Resonote's MusicXML with alphaTab (MPL-2.0),
// a completely different implementation from the exporter.
import * as fs from 'fs';
import * as alphaTab from '@coderline/alphatab';

const file = process.argv[2];
const data = new Uint8Array(fs.readFileSync(file));
let score;
try {
  score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(data,
      new alphaTab.Settings());
} catch (e) {
  console.log('PARSE FAILED:', e && e.message ? e.message : e);
  process.exit(1);
}

console.log('=== alphaTab parsed the MusicXML OK ===');
console.log('title      :', score.title);
console.log('tempo      :', score.tempo);
console.log('masterBars :', score.masterBars.length);

let notes = 0, empty = 0, dup = 0, tracks = 0;
let tuning = [];
for (const t of score.tracks) {
  tracks++;
  if (t.staves.length && t.staves[0].tuning) {
    tuning = t.staves[0].tuning.values || t.staves[0].tuning;
  }
  for (const st of t.staves) {
    for (const bar of st.bars) {
      for (const v of bar.voices) {
        if (!v || !v.beats) continue;
        for (const b of v.beats) {
          if (!b.notes || b.notes.length === 0) { empty++; continue; }
          const ss = b.notes.map(n => n.string);
          if (new Set(ss).size !== ss.length) dup++;
          notes += b.notes.length;
        }
      }
    }
  }
}
console.log('tracks     :', tracks);
console.log('notes      :', notes);
console.log('empty beats:', empty);
console.log('collisions :', dup);
console.log('tuning     :', JSON.stringify(tuning));
