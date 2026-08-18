/* ============================================================
   Resonote front-end logic — zero frameworks.
   Drag/drop or pick a file -> POST /api/arrange (raw bytes).
   "Try the sample" -> GET /api/demo. Renders the result.
   ============================================================ */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const drop    = $("#drop");
  const fileInp = $("#file");
  const styleSel = $("#style");
  const instInp  = $("#instruction");
  const sampleBtn = $("#sampleBtn");
  const statusEl  = $("#status");
  const resultEl  = $("#result");
  const statsEl   = $("#stats");
  const tabEl     = $("#tab");
  const audioEl   = $("#audio");
  const dlGp5  = $("#dlGp5");
  const dlMidi = $("#dlMidi");
  const dlHtml = $("#dlHtml");

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- theme ---------------- */
  const themeBtns = document.querySelectorAll("[data-theme-set]");
  const stored = localStorage.getItem("resonote-theme") || "system";

  function applyTheme(t) {
    let resolved = t;
    if (t === "system") {
      resolved = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }
    document.documentElement.setAttribute("data-theme", resolved);
    themeBtns.forEach((b) =>
      b.classList.toggle("active", b.dataset.themeSet === t));
  }
  applyTheme(stored);
  themeBtns.forEach((b) =>
    b.addEventListener("click", () => {
      const t = b.dataset.themeSet;
      localStorage.setItem("resonote-theme", t);
      applyTheme(t);
    }));
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
    if ((localStorage.getItem("resonote-theme") || "system") === "system") applyTheme("system");
  });

  /* ---------------- magnetic micro-interaction ---------------- */
  function magnetize(el, strength = 0.16) {
    if (reduce) return;
    el.addEventListener("mousemove", (e) => {
      const r = el.getBoundingClientRect();
      const x = (e.clientX - (r.left + r.width / 2)) * strength;
      const y = (e.clientY - (r.top + r.height / 2)) * strength;
      el.style.transform = `translate(${x}px, ${y}px)`;
    });
    el.addEventListener("mouseleave", () => { el.style.transform = ""; });
  }
  magnetize(drop);
  magnetize(sampleBtn);

  /* ---------------- status helpers ---------------- */
  function setLoading(msg) {
    statusEl.hidden = false;
    statusEl.className = "status loading";
    statusEl.innerHTML = `<span class="spinner"></span>${msg}`;
    resultEl.hidden = true;
    drop.classList.add("busy");
  }
  function showError(msg) {
    statusEl.hidden = false;
    statusEl.className = "status error";
    statusEl.textContent = "⚠ " + msg;
    drop.classList.remove("busy");
  }
  function clearStatus() { statusEl.hidden = true; statusEl.className = "status"; }

  function qs() {
    const p = new URLSearchParams();
    p.set("style", styleSel.value);
    if (instInp.value.trim()) p.set("instruction", instInp.value.trim());
    return p.toString();
  }

  /* ---------------- result rendering ---------------- */
  function chip(label, value) {
    return `<span class="chip">${label} <b>${value}</b></span>`;
  }
  function renderResult(d) {
    const rc = d.role_counts || {};
    statsEl.innerHTML =
      chip("Key", d.key || "—") +
      chip("Tempo", Math.round(d.tempo) + " BPM") +
      chip("Notes", d.note_count) +
      chip("Melody", rc.melody || 0) +
      chip("Bass", rc.bass || 0) +
      chip("Harmony", rc.harmony || 0);

    audioEl.src = d.wav_url;
    dlGp5.href = d.gp5_url;
    dlMidi.href = d.midi_url;
    dlHtml.href = d.html_url;
    tabEl.textContent = d.ascii_tab;

    clearStatus();
    resultEl.hidden = false;
    drop.classList.remove("busy");
    resultEl.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }

  /* ---------------- requests ---------------- */
  async function postFile(file) {
    setLoading("Transcribing & arranging…");
    try {
      const res = await fetch("/api/arrange?" + qs(), {
        method: "POST",
        headers: { "X-Filename": file.name },
        body: file,
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Arrangement failed.");
      renderResult(data);
    } catch (e) {
      showError(e.message || String(e));
    }
  }

  async function runDemo() {
    setLoading("Arranging the sample melody…");
    try {
      const res = await fetch("/api/demo?" + qs());
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Demo failed.");
      renderResult(data);
    } catch (e) {
      showError(e.message || String(e));
    }
  }

  /* ---------------- file handling ---------------- */
  const OK = /\.(mid|midi|mp3|wav)$/i;
  function handleFile(file) {
    if (!file) return;
    if (!OK.test(file.name)) {
      showError("Unsupported file. Use .mid / .midi / .mp3 / .wav.");
      return;
    }
    postFile(file);
  }

  drop.addEventListener("click", () => fileInp.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInp.click(); }
  });
  fileInp.addEventListener("change", () => handleFile(fileInp.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault(); drop.classList.add("drag");
    }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault(); drop.classList.remove("drag");
    }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    handleFile(f);
  });

  // allow dropping anywhere on the page (nicer UX)
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  sampleBtn.addEventListener("click", runDemo);
})();
