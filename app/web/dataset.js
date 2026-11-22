"use strict";

/* ── dataset.js — data-driven Dataset page ────────────────────────────────
   Fetches /static/data/results.json (shared cache via results.js) and
   renders the Dataset page. Does NOT modify app.js.
   ─────────────────────────────────────────────────────────────────────── */

/* ── XSS-safe text escaping (local copy, results.js may not be loaded yet) */
function _dsEsc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ── Loading / error helpers ────────────────────────────────────────────── */

function _dsShowLoading(container) {
  container.innerHTML = `<div class="ds-loading" role="status" aria-live="polite">
    <span class="spinner" aria-hidden="true"></span>
    <span>Loading dataset…</span>
  </div>`;
}

function _dsShowError(container, err) {
  container.innerHTML = `<div class="ds-error" role="alert">
    <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
      <circle cx="8" cy="8" r="6.5"/>
      <path d="M8 5v3.5M8 10.5v.5"/>
    </svg>
    <div>
      <strong>Could not load dataset.</strong>
      <span class="muted">${_dsEsc(String(err.message || err))}</span>
    </div>
    <button class="btn ghost sm" id="ds-retry-btn" type="button">Retry</button>
  </div>`;
  const btn = container.querySelector("#ds-retry-btn");
  if (btn) {
    btn.addEventListener("click", () => {
      /* Bust the shared cache so a re-fetch actually fires */
      if (typeof _resultsCache !== "undefined") window._resultsCache = null;
      _dsShowLoading(container);
      fetchResultsData()
        .then((data) => _renderDatasetPage(container, data))
        .catch((e) => _dsShowError(container, e));
    });
  }
}

/* ── Render ─────────────────────────────────────────────────────────────── */

function _renderDatasetPage(container, data) {
  container.dataset.loaded = "1";

  const dataset    = data.dataset    || {};
  const split      = data.split      || {};
  const rejections = data.rejections || [];
  const leakage    = data.leakage    || {};

  const total    = dataset.total    || 0;
  const verified = dataset.verified || 0;
  const rejected = total - verified;

  const train    = split.train    || 0;
  const val      = split.val      || 0;
  const heldOut  = split.held_out || 0;

  /* ── Rejection rows ── */
  const rejRows = rejections
    .map(
      (r) => `<tr>
        <td><code>${_dsEsc(r.category)}</code></td>
        <td class="num">${r.count}</td>
        <td class="ds-rej-reason">${_dsEsc(r.reason)}</td>
      </tr>`
    )
    .join("");

  /* ── Leakage numbers ── */
  const earlyOverlap = leakage.early_overlap || 0;
  const earlyTotal   = leakage.early_total   || 0;
  const leakageFix   = leakage.fix           || "";

  container.innerHTML = `
    <!-- ── Dataset overview ── -->
    <section class="ds-section" aria-labelledby="ds-overview-h">
      <div class="ds-section-head">
        <p class="eyebrow">Overview</p>
        <h2 id="ds-overview-h" class="ds-h2">Dataset overview</h2>
        <p class="ds-lede">${total} candidates from 15 templates × 20 shape configurations. ${verified} verified on a Tesla T4; ${rejected} rejected.</p>
      </div>
      <div class="ds-overview-cards">
        <div class="ds-ov-card">
          <span class="ds-ov-label">Templates</span>
          <span class="ds-ov-value">15</span>
          <span class="ds-ov-sub">5 LLM-serving categories</span>
        </div>
        <div class="ds-ov-card">
          <span class="ds-ov-label">Candidates</span>
          <span class="ds-ov-value">${total}</span>
          <span class="ds-ov-sub">15 templates × 20 shapes</span>
        </div>
        <div class="ds-ov-card ds-ov-card-ok">
          <span class="ds-ov-label">Verified</span>
          <span class="ds-ov-value">${verified}</span>
          <span class="ds-ov-sub">on Tesla T4 GPU</span>
        </div>
        <div class="ds-ov-card ds-ov-card-warn">
          <span class="ds-ov-label">Rejected</span>
          <span class="ds-ov-value">${rejected}</span>
          <span class="ds-ov-sub">tolerance / timeout / OOM</span>
        </div>
      </div>
    </section>

    <!-- ── Rejections ── -->
    <section class="ds-section" aria-labelledby="ds-rej-h">
      <div class="ds-section-head">
        <p class="eyebrow">Quality gate</p>
        <h2 id="ds-rej-h" class="ds-h2">Rejections</h2>
        <p class="ds-lede">${rejected} of ${total} candidates failed the verification harness and were excluded from the dataset.</p>
      </div>
      <div class="ds-table-wrap">
        <table class="ds-table" aria-label="Rejection breakdown by category and reason">
          <thead>
            <tr>
              <th scope="col">Category</th>
              <th scope="col" class="num">Count</th>
              <th scope="col">Reason</th>
            </tr>
          </thead>
          <tbody>${rejRows || '<tr><td colspan="3" class="ds-empty-cell">No rejection data.</td></tr>'}</tbody>
        </table>
      </div>
    </section>

    <!-- ── Train / val / held-out split ── -->
    <section class="ds-section" aria-labelledby="ds-split-h">
      <div class="ds-section-head">
        <p class="eyebrow">Data split</p>
        <h2 id="ds-split-h" class="ds-h2">Train / val / held-out split</h2>
      </div>
      <div class="rp-cards" style="max-width:860px">
        <div class="rp-stat-card">
          <span class="rp-stat-label">Train</span>
          <span class="rp-stat-value">${train}</span>
          <span class="rp-stat-sub">examples</span>
        </div>
        <div class="rp-stat-card">
          <span class="rp-stat-label">Validation</span>
          <span class="rp-stat-value">${val}</span>
          <span class="rp-stat-sub">examples</span>
        </div>
        <div class="rp-stat-card rp-stat-card-accent">
          <span class="rp-stat-label">Held-out test</span>
          <span class="rp-stat-value">${heldOut}</span>
          <span class="rp-stat-sub">examples (leakage-checked)</span>
        </div>
        <div class="rp-stat-card">
          <span class="rp-stat-label">Verified total</span>
          <span class="rp-stat-value">${verified}</span>
          <span class="rp-stat-sub">of ${total} built</span>
        </div>
      </div>
      <p class="rp-split-note muted" style="margin-top:14px">Template-level holdout — the held-out set shares no template with training. Integrity tests block accidental overlap.</p>
    </section>

    <!-- ── Leakage story ── -->
    <section class="ds-section" aria-labelledby="ds-leak-h">
      <div class="ds-section-head">
        <p class="eyebrow">Data integrity</p>
        <h2 id="ds-leak-h" class="ds-h2">Leakage story</h2>
      </div>
      <div class="ds-leakage-box">
        <div class="ds-leak-row">
          <div class="ds-leak-card ds-leak-card-warn" aria-label="Earlier evaluation: ${earlyOverlap} of ${earlyTotal} overlapping with training">
            <span class="ds-leak-label">Earlier eval subset</span>
            <span class="ds-leak-num">${earlyOverlap} / ${earlyTotal}</span>
            <span class="ds-leak-sub">entries overlapped with training split</span>
          </div>
          <div class="ds-leak-arrow" aria-hidden="true">
            <svg viewBox="0 0 32 16" fill="none">
              <path d="M0 8h26M22 4l6 4-6 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <div class="ds-leak-card ds-leak-card-ok" aria-label="Fixed: template-level holdout, zero overlap">
            <span class="ds-leak-label">Current split</span>
            <span class="ds-leak-num">0</span>
            <span class="ds-leak-sub">train/test overlap — template-level holdout</span>
          </div>
        </div>
        <p class="ds-leak-fix">${_dsEsc(leakageFix)}</p>
      </div>
    </section>
  `;
}

/* ── Main init ──────────────────────────────────────────────────────────── */

function initDatasetPage() {
  const container = document.getElementById("dataset-page-content");
  if (!container) return;
  if (container.dataset.loaded === "1") return; /* idempotent */

  _dsShowLoading(container);

  /* Reuse the shared fetch from results.js if available, else fetch directly */
  const fetchFn = typeof fetchResultsData === "function"
    ? fetchResultsData
    : () => fetch("/static/data/results.json").then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      });

  fetchFn()
    .then((data) => _renderDatasetPage(container, data))
    .catch((err) => _dsShowError(container, err));
}

/* ── Bootstrap ──────────────────────────────────────────────────────────── */

function _bootstrapDataset() {
  /* Dataset page: populate if already visible */
  if (!document.getElementById("page-dataset")?.hidden) {
    initDatasetPage();
  }

  /* Hook for router.js to call when navigating to dataset page */
  window.datasetPageReady = function () {
    initDatasetPage();
  };
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _bootstrapDataset);
} else {
  _bootstrapDataset();
}
