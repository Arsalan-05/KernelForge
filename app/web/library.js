"use strict";

/* ── library.js — template library page logic ─────────────────────────────
   Handles #/library page: fetches /templates, groups by category, expandable
   detail with code blocks, category filter chips, copy buttons, deep-links.
   Does NOT modify app.js or router.js.
   ────────────────────────────────────────────────────────────────────────── */

/* ── Category metadata ───────────────────────────────────────────────────── */

const CATEGORY_META = {
  quantized_matmul: { label: "quantized_matmul", color: "#f59e0b" },
  attention:        { label: "attention",         color: "#60a5fa" },
  kv_cache:         { label: "kv_cache",          color: "#a78bfa" },
  norm:             { label: "norm",              color: "#34d399" },
  rope:             { label: "rope",              color: "#f87171" },
};

/* Fallback for unknown categories */
function _catColor(cat) {
  return (CATEGORY_META[cat] || {}).color || "#7b8697";
}
function _catLabel(cat) {
  return (CATEGORY_META[cat] || {}).label || cat;
}

/* ── XSS-safe text ───────────────────────────────────────────────────────── */

function _esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ── State ───────────────────────────────────────────────────────────────── */

let _libTemplates = null;      /* cached list from /templates */
let _libFilter    = null;      /* active category filter */
let _libInitted   = false;     /* guard against double-init */

/* ── Fetch /templates ────────────────────────────────────────────────────── */

function _fetchTemplates() {
  if (_libTemplates) return Promise.resolve(_libTemplates);
  return fetch("/templates")
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((data) => {
      _libTemplates = Array.isArray(data) ? data : [];
      return _libTemplates;
    });
}

/* ── Fetch single template detail ────────────────────────────────────────── */

function _fetchTemplateDetail(id) {
  return fetch(`/templates/${encodeURIComponent(id)}`)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
}

/* ── Copy to clipboard ───────────────────────────────────────────────────── */

function _copyCode(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = orig; }, 1500);
  }).catch(() => {
    btn.textContent = "Copy failed";
    setTimeout(() => { btn.textContent = "Copy"; }, 2000);
  });
}

/* ── Highlight code if hljs is available ─────────────────────────────────── */

function _highlightEl(el) {
  if (window.hljs && el && !el.dataset.highlighted) {
    window.hljs.highlightElement(el);
  }
}

/* ── Render filter chips ─────────────────────────────────────────────────── */

function _renderFilters(container, categories) {
  const bar = container.querySelector(".lib-filters");
  if (!bar) return;

  const all = document.createElement("button");
  all.className = "lib-filter-btn" + (_libFilter === null ? " active" : "");
  all.textContent = "All";
  all.setAttribute("type", "button");
  all.setAttribute("aria-pressed", String(_libFilter === null));
  all.addEventListener("click", () => {
    _libFilter = null;
    _applyFilter(container, categories);
  });
  bar.appendChild(all);

  for (const cat of categories) {
    const btn = document.createElement("button");
    btn.className = "lib-filter-btn" + (_libFilter === cat ? " active" : "");
    btn.setAttribute("type", "button");
    btn.setAttribute("data-cat", cat);
    btn.setAttribute("aria-pressed", String(_libFilter === cat));
    btn.innerHTML = `<span class="lib-filter-dot" aria-hidden="true" style="background:${_esc(_catColor(cat))}"></span>${_esc(_catLabel(cat))}`;
    btn.addEventListener("click", () => {
      _libFilter = _libFilter === cat ? null : cat;
      _applyFilter(container, categories);
    });
    bar.appendChild(btn);
  }
}

function _applyFilter(container, categories) {
  /* Update button states */
  container.querySelectorAll(".lib-filter-btn").forEach((btn) => {
    const cat = btn.dataset.cat || null;
    const active = cat === _libFilter;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", String(active));
  });

  /* Show/hide category groups */
  container.querySelectorAll(".lib-category").forEach((group) => {
    const cat = group.dataset.category;
    group.hidden = _libFilter !== null && cat !== _libFilter;
  });
}

/* ── Expand a template: fetch detail and inject code panes ───────────────── */

function _expandTemplate(templateEl, templateId) {
  const detail = templateEl.querySelector(".lib-detail");
  if (!detail) return;

  if (detail.dataset.loaded === "1") return; /* already loaded */

  detail.innerHTML = `<div class="lib-detail-loading">
    <span class="spinner" aria-hidden="true"></span> Loading template…
  </div>`;

  _fetchTemplateDetail(templateId)
    .then((data) => {
      const refCode = data.pytorch_reference || "";
      /* The catalog's detail() returns `reference_explanation` as the
         optimization_explanation from the gpu entry; there is no model-generated
         Triton kernel in the catalog — only hand-written reference templates. */
      const kernelCode = data.triton_kernel || data.reference_kernel || "";

      detail.innerHTML = `
        <div class="lib-code-panes">
          <div class="lib-code-pane">
            <div class="lib-pane-head">
              <span class="lib-pane-label">PyTorch reference · <code>Model</code></span>
              <button class="btn ghost sm lib-copy-btn" type="button" aria-label="Copy PyTorch reference">Copy</button>
            </div>
            <div class="lib-code-view">
              <pre><code class="language-python lib-ref-code"></code></pre>
            </div>
          </div>
          <div class="lib-code-pane">
            <div class="lib-pane-head">
              <span class="lib-pane-label">
                Triton kernel
                <span class="lib-ref-tag" title="This is a hand-written reference template, not model output">Hand-written reference template (not model output)</span>
              </span>
              <button class="btn ghost sm lib-copy-btn" type="button" aria-label="Copy Triton kernel">Copy</button>
            </div>
            <div class="lib-code-view">
              <pre><code class="language-python lib-kernel-code"></code></pre>
            </div>
          </div>
        </div>
      `;
      detail.dataset.loaded = "1";

      const refEl    = detail.querySelector(".lib-ref-code");
      const kernelEl = detail.querySelector(".lib-kernel-code");
      if (refEl)    { refEl.textContent    = refCode;    _highlightEl(refEl); }
      if (kernelEl) { kernelEl.textContent = kernelCode; _highlightEl(kernelEl); }

      const [copyRef, copyKernel] = detail.querySelectorAll(".lib-copy-btn");
      if (copyRef)    copyRef.addEventListener("click", () => _copyCode(refCode, copyRef));
      if (copyKernel) copyKernel.addEventListener("click", () => _copyCode(kernelCode, copyKernel));
    })
    .catch((err) => {
      detail.innerHTML = `<div class="lib-detail-error">
        <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
          <circle cx="8" cy="8" r="6.5"/><path d="M8 5v3.5M8 10.5v.5"/>
        </svg>
        <span>Failed to load template: ${_esc(String(err.message || err))}</span>
        <button class="btn ghost sm lib-retry-detail" type="button">Retry</button>
      </div>`;
      const retryBtn = detail.querySelector(".lib-retry-detail");
      if (retryBtn) retryBtn.addEventListener("click", () => {
        delete detail.dataset.loaded;
        _expandTemplate(templateEl, templateId);
      });
    });
}

/* ── Toggle template open/closed ─────────────────────────────────────────── */

function _toggleTemplate(templateEl, templateId) {
  const isOpen = templateEl.classList.contains("open");
  /* Close all others */
  templateEl.closest(".lib-templates").querySelectorAll(".lib-template.open").forEach((el) => {
    if (el !== templateEl) el.classList.remove("open");
  });
  templateEl.classList.toggle("open", !isOpen);
  if (!isOpen) {
    _expandTemplate(templateEl, templateId);
    templateEl.querySelector(".lib-template-btn").setAttribute("aria-expanded", "true");
  } else {
    templateEl.querySelector(".lib-template-btn").setAttribute("aria-expanded", "false");
  }
}

/* ── Render the library ───────────────────────────────────────────────────── */

function _renderLibrary(container, templates) {
  const categories = [...new Set(templates.map((t) => t.category))];

  /* Group by category */
  const grouped = {};
  for (const cat of categories) grouped[cat] = [];
  for (const t of templates) {
    if (!grouped[t.category]) grouped[t.category] = [];
    grouped[t.category].push(t);
  }

  const content = container.querySelector(".lib-content");
  if (!content) return;

  /* Filters */
  _renderFilters(container, categories);

  /* Check for deep-link ?op= */
  const deepOp = new URLSearchParams(location.hash.split("?")[1] || "").get("op");

  /* Inject category groups */
  content.innerHTML = "";
  for (const cat of categories) {
    const group = grouped[cat];
    const catEl = document.createElement("div");
    catEl.className = "lib-category";
    catEl.dataset.category = cat;

    catEl.innerHTML = `
      <div class="lib-cat-head">
        <span class="lib-cat-dot" aria-hidden="true" style="background:${_esc(_catColor(cat))}"></span>
        <span class="lib-cat-label">${_esc(_catLabel(cat))}</span>
        <span class="lib-cat-count">${group.length} template${group.length !== 1 ? "s" : ""}</span>
      </div>
      <div class="lib-templates"></div>
    `;

    const tplList = catEl.querySelector(".lib-templates");
    for (const t of group) {
      const tplEl = document.createElement("div");
      tplEl.className = "lib-template";
      tplEl.dataset.id = t.id;

      const badges = [];
      if (t.interpreter_checked) badges.push(`<span class="badge interp" title="Passed Triton CPU interpreter">interp ✓</span>`);
      if (t.gpu_verified > 0)     badges.push(`<span class="badge gpu"   title="${t.gpu_verified} GPU-verified entries">GPU</span>`);

      tplEl.innerHTML = `
        <button class="lib-template-btn" type="button" aria-expanded="false" aria-controls="lib-detail-${_esc(t.id)}">
          <svg class="lib-tpl-caret" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M6 4l4 4-4 4"/>
          </svg>
          <span>
            <span class="lib-tpl-name">${_esc(t.op_name)}</span>
            <span class="lib-tpl-id">${_esc(t.id)}</span>
            <span class="lib-tpl-desc">${_esc(t.description || "")}</span>
          </span>
          <span class="lib-tpl-meta">${badges.join("")}</span>
        </button>
        <div class="lib-detail" id="lib-detail-${_esc(t.id)}" role="region" aria-label="${_esc(t.op_name)} code"></div>
      `;

      const btn = tplEl.querySelector(".lib-template-btn");
      btn.addEventListener("click", () => _toggleTemplate(tplEl, t.id));

      tplList.appendChild(tplEl);

      /* Auto-expand if this is the deep-linked op */
      if (deepOp && t.id === deepOp) {
        setTimeout(() => {
          tplEl.scrollIntoView({ behavior: "smooth", block: "start" });
          _toggleTemplate(tplEl, t.id);
        }, 80);
      }
    }

    content.appendChild(catEl);
  }

  /* Apply current filter */
  _applyFilter(container, categories);
}

/* ── Page init ───────────────────────────────────────────────────────────── */

function initLibraryPage() {
  const page = document.getElementById("page-library");
  if (!page) return;

  /* Allow re-init if hash changes (deep-link) but not on plain revisit */
  const deepOp = new URLSearchParams(location.hash.split("?")[1] || "").get("op");
  if (_libInitted && !deepOp) return;
  _libInitted = true;

  const container = page.querySelector(".lib-wrap");
  if (!container) return;

  /* Show loading state */
  const content = container.querySelector(".lib-content");
  if (content) {
    content.innerHTML = `<div class="lib-loading">
      <span class="spinner" aria-hidden="true"></span> Loading templates…
    </div>`;
  }

  _fetchTemplates()
    .then((templates) => {
      if (content) content.innerHTML = "";
      /* Clear and rebuild filters bar */
      const bar = container.querySelector(".lib-filters");
      if (bar) bar.innerHTML = "";
      _renderLibrary(container, templates);
    })
    .catch((err) => {
      if (content) {
        content.innerHTML = `<div class="lib-error">
          <svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
            <circle cx="8" cy="8" r="6.5"/><path d="M8 5v3.5M8 10.5v.5"/>
          </svg>
          <span>Could not load templates: ${_esc(String(err.message || err))}</span>
          <button class="btn ghost sm" id="lib-retry-btn" type="button">Retry</button>
        </div>`;
        const retryBtn = content.querySelector("#lib-retry-btn");
        if (retryBtn) retryBtn.addEventListener("click", () => {
          _libTemplates = null;
          _libInitted = false;
          initLibraryPage();
        });
      }
      console.warn("[library.js] templates fetch failed:", err);
    });
}

/* Expose for router.js hook */
window.libraryPageReady = initLibraryPage;

/* Bootstrap on DOMContentLoaded if already on the library page */
function _bootstrapLibrary() {
  if (!document.getElementById("page-library")?.hidden) {
    initLibraryPage();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _bootstrapLibrary);
} else {
  _bootstrapLibrary();
}
