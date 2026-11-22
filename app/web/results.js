"use strict";

/* ── results.js — shared results-loading logic ────────────────────────────
   Used by both the Home page results snapshot and the full Results page.
   Does NOT modify app.js.
   ────────────────────────────────────────────────────────────────────────── */

const RESULTS_JSON_URL = "/static/data/results.json";

/* Cache the fetch so both consumers share one network request per page load. */
let _resultsCache = null;
let _resultsPending = null;

/**
 * Fetch results.json once; subsequent callers get the cached promise.
 * @returns {Promise<Object>} The parsed results JSON.
 */
function fetchResultsData() {
  if (_resultsCache) return Promise.resolve(_resultsCache);
  if (_resultsPending) return _resultsPending;
  _resultsPending = fetch(RESULTS_JSON_URL)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((data) => {
      _resultsCache = data;
      _resultsPending = null;
      return data;
    })
    .catch((err) => {
      _resultsPending = null;
      throw err;
    });
  return _resultsPending;
}

/* ── Status badge helper ──────────────────────────────────────────────────── */

function _statusBadge(status) {
  if (status === "verified") {
    return `<span class="res-status-badge res-status-ok">verified</span>`;
  }
  if (status === "re-run in progress") {
    return `<span class="res-status-badge res-status-pending">re-run in progress</span>`;
  }
  return `<span class="res-status-badge">${_esc(status)}</span>`;
}

/* ── pass@1 cell helper ───────────────────────────────────────────────────── */

function _pass1Cell(val) {
  if (val === "pending") {
    return `<span class="home-res-pending">pending</span>`;
  }
  /* "0 / 88" → styled numerator + denominator */
  const m = /^(\d+)\s*\/\s*(\d+)$/.exec(val.trim());
  if (m) {
    return `<span class="res-num-inline">${m[1]}</span><span class="res-den-inline"> / ${m[2]}</span>`;
  }
  return _esc(val);
}

/* ── XSS-safe text escaping ───────────────────────────────────────────────── */

function _esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ═══════════════════════════════════════════════════════════════════════════
   HOME PAGE — populate #home-results table
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Fetch results.json and populate the home-page #home-results table.
 * Runs only when #home-results exists in the DOM.
 */
function initHomeResults() {
  const container = document.getElementById("home-results");
  if (!container) return;

  const tbody = container.querySelector("tbody");
  if (!tbody) return;

  /* Show spinner while loading */
  tbody.innerHTML = `<tr><td colspan="3" class="home-results-loading-row">
    <span class="spinner" aria-hidden="true"></span> Loading…
  </td></tr>`;

  fetchResultsData()
    .then((data) => {
      const rows = (data.results || [])
        .map(
          (r) => `<tr>
          <td>${_esc(r.model)}</td>
          <td class="num">${_pass1Cell(r.pass_at_1)}</td>
          <td class="muted">${_esc(r.notes || "")}</td>
        </tr>`
        )
        .join("");
      tbody.innerHTML = rows || `<tr><td colspan="3" class="home-results-loading-row muted">No data.</td></tr>`;
    })
    .catch((err) => {
      tbody.innerHTML = `<tr><td colspan="3" class="home-results-loading-row muted">
        Could not load results. <button class="btn ghost sm" onclick="initHomeResults()">Retry</button>
      </td></tr>`;
      console.warn("[results.js] home results fetch failed:", err);
    });
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESULTS PAGE — full data-driven results page
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Render the full #page-results content from results.json.
 * Called by router.js hook (window.resultsPageReady) on first visit.
 */
function initResultsPage() {
  const container = document.getElementById("results-page-content");
  if (!container) return;
  if (container.dataset.loaded === "1") return; /* idempotent */

  _showResultsLoading(container);

  fetchResultsData()
    .then((data) => _renderResultsPage(container, data))
    .catch((err) => _showResultsError(container, err));
}

function _showResultsLoading(container) {
  container.innerHTML = `<div class="rp-loading" role="status" aria-live="polite">
    <span class="spinner" aria-hidden="true"></span>
    <span>Loading results…</span>
  </div>`;
}

function _showResultsError(container, err) {
  container.innerHTML = `<div class="rp-error" role="alert">
    <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
      <circle cx="8" cy="8" r="6.5"/>
      <path d="M8 5v3.5M8 10.5v.5"/>
    </svg>
    <div>
      <strong>Could not load results data.</strong>
      <span class="muted">${_esc(String(err.message || err))}</span>
    </div>
    <button class="btn ghost sm" id="results-retry-btn" type="button">Retry</button>
  </div>`;
  const btn = container.querySelector("#results-retry-btn");
  if (btn) {
    btn.addEventListener("click", () => {
      _resultsCache = null;
      _showResultsLoading(container);
      fetchResultsData()
        .then((data) => _renderResultsPage(container, data))
        .catch((e) => _showResultsError(container, e));
    });
  }
}

function _renderResultsPage(container, data) {
  container.dataset.loaded = "1";

  const results = data.results || [];
  const split = data.split || {};
  const dataset = data.dataset || {};
  const errors = data.error_breakdown || [];
  const disclaimer = data.disclaimer || "";

  const totalErrors = errors.reduce((s, e) => s + e.count, 0);
  const maxErrorCount = errors.reduce((m, e) => Math.max(m, e.count), 0);

  /* ── Results table ── */
  const tableRows = results
    .map(
      (r) => `<tr>
      <td>${_esc(r.model)}</td>
      <td>${_esc(r.quantization)}</td>
      <td>${_esc(r.eval_set)}</td>
      <td class="num">${_pass1Cell(r.pass_at_1)}</td>
      <td>${_statusBadge(r.status)}</td>
    </tr>`
    )
    .join("");

  /* ── Split card ── */
  const splitTotal = (split.train || 0) + (split.val || 0) + (split.held_out || 0);

  /* ── Error bars ── */
  const errorRows = errors
    .map((e) => {
      const pct = totalErrors > 0 ? Math.round((e.count / totalErrors) * 100) : 0;
      const barW = maxErrorCount > 0 ? Math.round((e.count / maxErrorCount) * 100) : 0;
      return `<div class="rp-err-row">
        <span class="rp-err-label">${_esc(e.category)}</span>
        <div class="rp-err-bar-wrap" role="img" aria-label="${_esc(e.category)}: ${e.count} of ${totalErrors}">
          <div class="rp-err-bar" style="width:${barW}%"></div>
        </div>
        <span class="rp-err-count">${e.count}</span>
        <span class="rp-err-pct muted">${pct}%</span>
      </div>`;
    })
    .join("");

  container.innerHTML = `
    <!-- ── Results table ── -->
    <section class="rp-section" aria-label="Evaluation results table">
      <div class="rp-section-head">
        <p class="eyebrow">Evaluation</p>
        <h2 class="rp-h2">Model results</h2>
        <p class="rp-lede">Base model zero-shot evaluation on the held-out test set. Fine-tuned adapter numbers are pending training completion.</p>
      </div>
      <div class="rp-table-wrap">
        <table class="rp-table" aria-label="Evaluation results">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Quantization</th>
              <th scope="col">Eval set</th>
              <th scope="col" class="num">pass@1</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    </section>

    <!-- ── Split + dataset cards ── -->
    <section class="rp-section rp-cards-section" aria-label="Dataset split">
      <div class="rp-section-head">
        <p class="eyebrow">Dataset</p>
        <h2 class="rp-h2">Train / val / held-out split</h2>
      </div>
      <div class="rp-cards">
        <div class="rp-stat-card">
          <span class="rp-stat-label">Train</span>
          <span class="rp-stat-value">${split.train || "—"}</span>
          <span class="rp-stat-sub">examples</span>
        </div>
        <div class="rp-stat-card">
          <span class="rp-stat-label">Validation</span>
          <span class="rp-stat-value">${split.val || "—"}</span>
          <span class="rp-stat-sub">examples</span>
        </div>
        <div class="rp-stat-card rp-stat-card-accent">
          <span class="rp-stat-label">Held-out test</span>
          <span class="rp-stat-value">${split.held_out || "—"}</span>
          <span class="rp-stat-sub">examples (leakage-checked)</span>
        </div>
        <div class="rp-stat-card">
          <span class="rp-stat-label">Verified total</span>
          <span class="rp-stat-value">${dataset.verified || "—"}</span>
          <span class="rp-stat-sub">of ${dataset.total || "—"} built</span>
        </div>
      </div>
      <p class="rp-split-note muted">Template-level holdout — the held-out set shares no template with training. Integrity tests block accidental overlap.</p>
    </section>

    <!-- ── Error breakdown ── -->
    <section class="rp-section" aria-label="Base model error breakdown">
      <div class="rp-section-head">
        <p class="eyebrow">Error analysis</p>
        <h2 class="rp-h2">Base model failure breakdown</h2>
        <p class="rp-lede">All ${totalErrors} failures on the 88-example held-out set, manually categorized.</p>
      </div>
      <div class="rp-err-chart" aria-label="Error category bar chart">
        ${errorRows}
      </div>
    </section>

    <!-- ── Disclaimer ── -->
    ${
      disclaimer
        ? `<div class="rp-disclaimer" role="note">
        <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
          <circle cx="8" cy="8" r="6.5"/>
          <path d="M8 7v4M8 4.5v.5"/>
        </svg>
        <span>${_esc(disclaimer)}</span>
      </div>`
        : ""
    }
  `;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Bootstrap — run after DOM is ready
   ═══════════════════════════════════════════════════════════════════════════ */

function _bootstrapResults() {
  /* Home page: always populate the results table if it exists */
  initHomeResults();

  /* Results page: populate if already visible */
  if (!document.getElementById("page-results")?.hidden) {
    initResultsPage();
  }

  /* Hook for router.js to call when navigating to results page */
  window.resultsPageReady = function () {
    initResultsPage();
  };
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _bootstrapResults);
} else {
  _bootstrapResults();
}
