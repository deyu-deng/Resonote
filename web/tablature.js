/* ============================================================
   Resonote · Tablature SVG renderer (prototype)
   Renders a 6-string guitar tab from a note array.
   note = { string:1..6, fret:0..24, finger:0..4, onset:beat, role }
   string 6 = low E (top line), string 1 = high e (bottom line)
   ============================================================ */

const TAB = (() => {
  const STRINGS = 6;
  const STEP = 56;        // px per beat
  const LEFT = 54;        // left padding (string labels)
  const TOP = 26;
  const LINE_GAP = 26;
  const DOT_R = 12;

  const ROLE_COLOR = {
    melody:  "#7c5cff",
    bass:    "#19d3da",
    harmony: "#f5a623",
    optional:"#8a93a8",
  };
  const LABELS = ["E", "A", "D", "G", "B", "e"]; // top→bottom (string 6..1)

  function render(container, notes, opts = {}) {
    const beats = Math.max(...notes.map((n) => n.onset), 0) + 4;
    const w = LEFT + beats * STEP + 24;
    const h = TOP * 2 + (STRINGS - 1) * LINE_GAP;
    const yOf = (s) => TOP + (STRINGS - s) * LINE_GAP; // s:1..6

    let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="吉他六线谱">`;

    // string lines
    for (let s = 1; s <= STRINGS; s++) {
      const y = yOf(s);
      svg += `<line x1="${LEFT}" y1="${y}" x2="${w - 16}" y2="${y}" stroke="var(--glass-brd)" stroke-width="1.4"/>`;
      svg += `<text x="14" y="${y + 4}" fill="var(--fg-dim)" font-size="13" font-family="var(--font)" font-weight="700">${LABELS[s - 1]}</text>`;
    }

    // measure separators every 4 beats
    for (let b = 0; b <= beats; b += 4) {
      const x = LEFT + b * STEP;
      svg += `<line x1="${x}" y1="${TOP}" x2="${x}" y2="${yOf(1)}" stroke="var(--glass-brd)" stroke-width="1" opacity="0.5"/>`;
    }

    // notes
    for (const n of notes) {
      if (n.onset < 0) continue;
      const x = LEFT + n.onset * STEP + STEP / 2;
      const y = yOf(n.string);
      const c = ROLE_COLOR[n.role] || "#7c5cff";
      svg += `<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="${c}" opacity="0.92"/>`;
      svg += `<text x="${x}" y="${y + 4.5}" fill="#fff" font-size="12.5" font-weight="800" text-anchor="middle" font-family="var(--font)">${n.fret}</text>`;
    }

    svg += `</svg>`;
    container.innerHTML = svg;
  }

  return { render };
})();
