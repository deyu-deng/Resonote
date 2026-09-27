// Independent verification of a Resonote export, using alphaTab (MPL-2.0) —
// a completely different implementation from the writers that produced the
// file (pyguitarpro for .gp5, our own exporter for MusicXML).
//
//   node tools/alphatab/verify-score.mjs <file.gp5|file.gp|file.musicxml|file.mid>
//
// "A file was written" is not the same as "Guitar Pro opens it and a human can
// play it", so this re-reads the artifact and checks the rules that break
// either claim:
//   * two notes on one string at one instant  -> unplayable, and corrupts GP5
//   * more notes attacking at once than one hand can pluck
//   * a fretboard span no left hand reaches
//
// Two things make the counting non-trivial, and both are easy to get wrong:
//   * Sustained notes carry a *tie chain* in GP5, so the raw note count is
//     higher than the number of attacks. Only notes without tieOrigin are
//     plucked; count those.
//   * Our MusicXML uses one voice per string (a string cannot sound twice),
//     so simultaneity lives ACROSS voices, not within a beat. Voices are
//     therefore walked on a shared time axis, in quarters from the bar start.
import * as fs from 'fs';
import * as path from 'path';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';

const require = createRequire(import.meta.url);
// alphaTab's package "exports" map hides dist/alphaTab.core.mjs, which is the
// bundle carrying the format sniffers, so resolve it by absolute path.
let alphaTab;
try {
  const core = path.join(require.resolve('@coderline/alphatab'), '..', '..',
                         'dist', 'alphaTab.core.mjs');
  alphaTab = await import(pathToFileURL(core).href);
} catch {
  console.error('alphaTab not installed — run: npm install  (in tools/alphatab/)');
  process.exit(2);
}

const file = process.argv[2];
if (!file) {
  console.error('usage: node verify-score.mjs <score file>');
  process.exit(2);
}

let score;
try {
  score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(
      new Uint8Array(fs.readFileSync(file)), new alphaTab.Settings());
} catch (e) {
  console.log('PARSE FAILED:', e && e.message ? e.message : e);
  process.exit(1);
}

/** alphaTab stores a beat's value as a note divisor: 4 = quarter, 8 = eighth,
 *  and it also uses 256 for the zero-length filler beats it inserts where a
 *  voice starts on a tie. Quarters = 4 / divisor, so no lookup table (a table
 *  silently rounding an unknown divisor up to a whole eighth shifted a whole
 *  bar's notes onto the next barline). */
function beatQuarters(beat, warnings) {
  const base = 4 / (beat.duration || 8);
  let q = base;
  for (let d = 1; d <= (beat.dots || 0); d++) q += base / 2 ** d;
  if (beat.tupletNumerator && beat.tupletDenominator) {
    q *= beat.tupletDenominator / beat.tupletNumerator;
  } else if (beat.tupletNumerator) {
    warnings.tuplet = true;                  // duration then needs its group
  }
  return q;
}

/** quarters from score start to the beginning of each bar */
function barStarts(score) {
  const starts = [];
  let t = 0;
  for (const mb of score.masterBars) {
    starts.push(t);
    t += (mb.timeSignatureNumerator || 4) * 4 / (mb.timeSignatureDenominator || 4);
  }
  return starts;
}

/** attacks: one {t, string, fret} per plucked note, across every voice */
function collect_attacks(track, starts, warnings) {
  const attacks = [];
  let rawNotes = 0, tied = 0;
  for (const staff of track.staves) {
    for (const [voiceIndex, firstVoice] of staff.bars[0].voices.entries()) {
      if (!firstVoice) continue;
      for (const [barIndex, bar] of staff.bars.entries()) {
        let cursor = starts[barIndex];
        for (const beat of bar.voices[voiceIndex]?.beats || []) {
          for (const note of beat.notes || []) {
            rawNotes++;
            // A sustained note is written once as an attack and then again in
            // every value it is tied through. Both readers flag the
            // continuation, but not the same way: the GP5 reader fills
            // tieOrigin, the MusicXML reader only sets isTieDestination.
            if ((note.tieOrigin && note.tieOrigin.length) ||
                note.isTieDestination) { tied++; continue; }
            attacks.push({ t: cursor, string: note.string, fret: note.fret });
          }
          cursor += beatQuarters(beat, warnings);
        }
      }
    }
  }
  return { attacks, rawNotes, tied };
}

const warnings = {};
const starts = barStarts(score);
const attacks = [];
let rawNotes = 0, tiedNotes = 0;
for (const track of score.tracks) {
  const r = collect_attacks(track, starts, warnings);
  attacks.push(...r.attacks);
  rawNotes += r.rawNotes;
  tiedNotes += r.tied;
}

// bucket attacks onto the finest grid we emit (1/64 of a quarter is plenty)
const buckets = new Map();
for (const a of attacks) {
  const key = Math.round(a.t * 256);
  if (!buckets.has(key)) buckets.set(key, []);
  buckets.get(key).push(a);
}
const atOnce = [...buckets.values()].map(v => v.length);
const maxAtOnce = atOnce.length ? Math.max(...atOnce) : 0;
const collisions = [...buckets.values()]
    .filter(v => new Set(v.map(a => a.string)).size !== v.length);
const spans = [...buckets.values()].filter(v => v.length > 1)
    .map(v => Math.max(...v.map(a => a.fret)) - Math.min(...v.map(a => a.fret)));
const avgSpan = spans.length ? spans.reduce((a, x) => a + x, 0) / spans.length : 0;
const over5 = spans.filter(x => x > 5).length;
const high = attacks.filter(a => a.fret >= 12).length;
const bars = score.tracks.reduce((a, t) => a + t.staves[0].bars.length, 0);
const tunings = score.tracks[0].staves[0].stringTuning.tunings;

console.log(`=== alphaTab read ${path.basename(file)} ===`);
console.log('title      :', score.title);
console.log('tempo      :', score.tempo);
console.log('tracks     :', score.tracks.length, '| bars:', bars);
console.log('tuning     :', tunings.join(','));
console.log(`notes      : ${attacks.length} attacks` +
            (tiedNotes ? ` (+${tiedNotes} tie continuations, ${rawNotes} written)` : ''));
console.log('max at once:', maxAtOnce);
console.log('fret span  : avg %s, >5 frets on %s%% of simultaneous attacks',
            avgSpan.toFixed(1),
            spans.length ? (100 * over5 / spans.length).toFixed(0) : '0');
console.log('>=12 fret  : %s%% of attacks',
            attacks.length ? (100 * high / attacks.length).toFixed(0) : '0');
if (warnings.tuplet) console.log('! tuplets present: durations may be mis-added');

let failed = 0;
if (collisions.length) {
  failed = 1;
  console.log(`FAIL: ${collisions.length} instant(s) put two notes on one string` +
              ` — Guitar Pro cannot open such a .gp5`);
}
if (maxAtOnce > 5) {
  failed = 1;
  console.log(`FAIL: ${maxAtOnce} notes attack at once, more than one hand plucks`);
}
console.log(failed ? 'RESULT: NOT PLAYABLE' : 'RESULT: OK (parsed + playable by these checks)');
process.exit(failed);
