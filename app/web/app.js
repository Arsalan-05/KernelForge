"use strict";

const CATEGORY_META = {
  quantized_matmul: { label: "Quantized matmul", color: "#f472b6" },
  attention: { label: "Attention (prefill)", color: "#60a5fa" },
  kv_cache: { label: "KV-cache decode", color: "#a78bfa" },
  norm: { label: "Normalization", color: "#34d399" },
  rope: { label: "RoPE", color: "#fbbf24" },
};
const CATEGORY_ORDER = Object.keys(CATEGORY_META);
const VARIANT_LABEL = { base: "Base model", "fine-tuned": "Fine-tuned", remote: "Hosted model", mock: "Mock mode" };
const DEVICE_LABEL = { cuda: "GPU", cpu: "Interpreter" };
const HISTORY_KEY = "kernelforge.history.v2";
const HISTORY_MAX = 20;

const CUSTOM_TEMPLATE = `import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replace with the op you want a Triton kernel for.
        return torch.nn.functional.silu(x) * x


def get_inputs():
    return [torch.randn(4096, 4096, dtype=torch.float16)]


def get_init_inputs():
    return []
`;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  health: null,
  templates: [],
  selected: null, // template detail | {custom: true}
  device: "cuda",
  deviceTouched: false,
  samples: 1,
  repair: 0,
  running: false,
  abort: null,
  run: null,
  history: loadHistory(),
};

/* ---------------- utilities ---------------- */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

let toastTimer;
function toast(message, isError = false) {
  const t = $("#toast");
  t.textContent = message;
  t.classList.toggle("err", isError);
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), isError ? 5000 : 2200);
}

function fmtMs(seconds) {
  if (seconds == null) return "—";
  const ms = seconds * 1000;
  return ms >= 100 ? `${ms.toFixed(0)} ms` : ms >= 10 ? `${ms.toFixed(1)} ms` : `${ms.toFixed(3)} ms`;
}

function fmtSecs(s) {
  return s == null ? "—" : s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
}

function fmtSpeedup(x) {
  return x == null ? "—" : `${x.toFixed(2)}×`;
}

function fmtUptime(s) {
  if (s == null) return "—";
  if (s < 3600) return `up ${Math.floor(s / 60)}m`;
  return `up ${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function highlightInto(codeEl, text) {
  if (window.hljs && text) {
    codeEl.innerHTML = window.hljs.highlight(text, { language: "python", ignoreIllegals: true }).value;
  } else {
    codeEl.textContent = text || "";
  }
}

function lineNumbers(text) {
  const n = Math.max(1, (text || "").split("\n").length);
  let out = "";
  for (let i = 1; i <= n; i++) out += i + "\n";
  return out;
}

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${label} copied`);
  } catch {
    toast("Clipboard unavailable in this context", true);
  }
}

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) throw new Error(await errorDetail(resp));
  return resp.json();
}

async function errorDetail(resp) {
  try {
    const body = await resp.json();
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
    return body.detail || `HTTP ${resp.status}`;
  } catch {
    return `HTTP ${resp.status}`;
  }
}

/* ---------------- health / status ---------------- */

async function refreshHealth() {
  try {
    const h = await api("/health");
    state.health = h;
    const pill = $("#model-pill");
    pill.dataset.variant = h.model_variant;
    const shortName = h.model_name.split("/").pop();
    $("#model-pill-text").textContent =
      h.model_variant === "mock" ? "Mock mode · no model" : `${VARIANT_LABEL[h.model_variant]} · ${shortName}`;
    pill.title = h.banner;

    const gpu = h.gpu_name ? h.gpu_name.replace(/^NVIDIA\s+/, "") : "No CUDA GPU";
    const triton = h.triton_version ? `Triton ${h.triton_version}` : "no Triton";
    $("#gpu-pill-text").textContent = `${gpu} · ${triton}`;

    $("#banner").dataset.variant = h.model_variant;
    $("#banner-text").textContent = h.banner;

    $("#flow-model-name").textContent = shortName;
    if (h.model_variant === "fine-tuned") {
      $("#flow-model-sub").textContent = "+ fine-tuned QLoRA adapter";
      $("#limit-model").replaceChildren(
        el("strong", {}, "Fine-tuned adapter, small dataset. "),
        `This demo runs ${shortName} with a QLoRA adapter trained on a small (~300-example target) curated serving-kernel dataset.`
      );
    } else if (h.model_variant === "remote") {
      $("#flow-model-sub").textContent = "hosted API · general-purpose";
      $("#limit-model").replaceChildren(
        el("strong", {}, "Hosted general-purpose model. "),
        `This deployment has no GPU, so text comes from ${shortName} over an API. It isn't KernelForge's fine-tuned model; the search loop, harness and repair feedback are the same code the GPU demo runs.`
      );
    } else if (h.model_variant === "mock") {
      $("#flow-model-name").textContent = "Mock";
      $("#flow-model-sub").textContent = "template echo · no model";
      $("#limit-model").replaceChildren(
        el("strong", {}, "No model on this deployment. "),
        "Mock mode replays the hand-written template kernels so the pipeline and harness can be exercised. Nothing here is model-generated."
      );
    }
    if (!h.cuda_available) {
      $("#footer-note").textContent = h.interpreter_available
        ? "This server has no GPU: kernels are checked with Triton's CPU interpreter (correctness only, no speedups)."
        : "This server has no GPU and no Triton, so kernels can't be verified here.";
    }
    if (h.public) {
      const limit = h.limits?.rate_limit_per_min;
      $("#limit-sandbox").replaceChildren(
        el("strong", {}, "Public demo limits. "),
        `Code runs in a subprocess with CPU/file-size limits and a scrubbed environment, after a static policy check (torch/triton/math imports only; no file, process or dunder access).${limit ? ` Each visitor gets ${limit} runs per minute.` : ""}`
      );
    }
    // Without a GPU the interpreter is the only harness that can actually run.
    if (!state.deviceTouched && !h.cuda_available && h.interpreter_available && state.device !== "cpu") {
      setDevice("cpu", false);
    }
  } catch {
    const pill = $("#model-pill");
    pill.dataset.variant = "offline";
    $("#model-pill-text").textContent = "Backend offline";
    $("#gpu-pill-text").textContent = "—";
    // Banner intentionally keeps the base-model text: that's the truthful default.
  }
}

async function refreshStats() {
  let s;
  try {
    s = await api("/stats");
  } catch {
    return;
  }
  $("#stats-uptime").textContent = fmtUptime(s.uptime_s);
  const cells = [
    ["requests", s.requests],
    ["pass rate", s.pass_rate == null ? "—" : `${Math.round(s.pass_rate * 100)}%`],
    ["candidates", s.candidates_generated],
    ["tok/s", s.tokens_per_s ?? "—"],
    ["harness runs", s.verifications_run],
    ["cache hits", s.verification_cache_hits],
    ["fixed by repair", s.repairs_that_passed],
    ["best speedup", fmtSpeedup(s.best_speedup)],
  ];
  $("#stats").replaceChildren(
    ...cells.map(([k, v]) => el("div", { class: "stat" }, el("span", { class: "stat-v" }, String(v)), el("span", { class: "stat-k" }, k)))
  );
}

/* ---------------- op library ---------------- */

async function loadTemplates() {
  try {
    state.templates = await api("/templates");
  } catch (err) {
    state.templates = [];
    toast(`Couldn't load examples: ${err.message}`, true);
  }
  $("#op-count").textContent = `${state.templates.length} ops`;
  renderOpGroups();

  const wanted = new URLSearchParams(location.search).get("op");
  const initial = state.templates.find((t) => t.id === wanted) || state.templates[0];
  if (initial) selectTemplate(initial.id);
  else selectCustom();
}

function opBadges(t) {
  const badges = [];
  if (t.interpreter_checked) badges.push(el("span", { class: "badge interp", title: "Passes Triton's CPU interpreter (correctness only)" }, "interp ✓"));
  if (t.gpu_verified) {
    const speed = t.median_speedup != null ? ` · ${t.median_speedup}×` : "";
    badges.push(el("span", { class: "badge gpu", title: "Harness-verified dataset entries on a GPU (median speedup)" }, `GPU ${t.gpu_verified}${speed}`));
  }
  return badges.length ? el("span", { class: "op-badges" }, badges) : null;
}

function renderOpGroups() {
  const q = $("#op-search").value.trim().toLowerCase();
  const root = $("#op-groups");
  root.replaceChildren();
  for (const cat of CATEGORY_ORDER) {
    const meta = CATEGORY_META[cat];
    const all = state.templates.filter((t) => t.category === cat);
    const items = q
      ? all.filter((t) => `${t.op_name} ${t.id} ${t.description} ${meta.label}`.toLowerCase().includes(q))
      : all;
    if (q && !items.length) continue;

    const list = el("div", { class: "op-list" });
    if (!items.length) list.append(el("div", { class: "op-empty" }, "No templates"));
    for (const t of items) {
      list.append(
        el(
          "button",
          {
            class: "op-item" + (state.selected && state.selected.id === t.id ? " on" : ""),
            type: "button",
            dataset: { id: t.id },
            title: t.description,
            onclick: () => selectTemplate(t.id),
          },
          el("span", {}, el("span", { class: "op-name" }, t.op_name), el("span", { class: "op-sub" }, t.id), opBadges(t))
        )
      );
    }
    root.append(
      el(
        "details",
        { class: "op-group", open: true },
        el(
          "summary",
          {},
          caretIcon(),
          el("span", { class: "cat-swatch", style: `background:${meta.color}` }),
          meta.label,
          el("span", { class: "n" }, String(all.length))
        ),
        list
      )
    );
  }
}

function caretIcon() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "caret");
  svg.setAttribute("viewBox", "0 0 10 10");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "M3 1.5L7 5 3 8.5");
  svg.append(path);
  return svg;
}

function markSelected() {
  $$(".op-item[data-id]").forEach((b) => b.classList.toggle("on", !!state.selected && b.dataset.id === state.selected.id));
  $("#custom-op").classList.toggle("on", !!state.selected && !!state.selected.custom);
}

// The interpreter runs every program on the CPU, so it gets the small smoke shape of the same op.
function referenceFor(t, device = state.device) {
  return device === "cpu" && t.smoke_reference ? t.smoke_reference : t.pytorch_reference;
}

function descriptionFor(t, device = state.device) {
  return device === "cpu" && t.smoke_description ? t.smoke_description : t.description;
}

async function selectTemplate(id) {
  try {
    const t = await api(`/templates/${encodeURIComponent(id)}`);
    state.selected = t;
    $("#op-title").textContent = t.op_name;
    const chip = $("#op-category");
    chip.hidden = false;
    chip.textContent = CATEGORY_META[t.category]?.label || t.category;
    chip.style.color = CATEGORY_META[t.category]?.color || "";
    $("#description").value = descriptionFor(t);
    setReference(referenceFor(t));
    history.replaceState(null, "", `?op=${encodeURIComponent(t.id)}${location.hash}`);
    markSelected();
  } catch (err) {
    toast(`Couldn't load ${id}: ${err.message}`, true);
  }
}

function selectCustom() {
  state.selected = { custom: true };
  $("#op-title").textContent = "Custom op";
  const chip = $("#op-category");
  chip.hidden = false;
  chip.textContent = "custom";
  chip.style.color = "";
  $("#description").value = "Fused SiLU-gate: silu(x) * x over a large fp16 activation tensor.";
  setReference(CUSTOM_TEMPLATE);
  history.replaceState(null, "", location.pathname + location.hash);
  markSelected();
  $("#reference").focus();
}

/* ---------------- editor + controls ---------------- */

function setReference(text) {
  const ta = $("#reference");
  ta.value = text;
  syncGutter();
  ta.scrollTop = 0;
}

function syncGutter() {
  const ta = $("#reference");
  const gutter = $("#ref-gutter");
  gutter.textContent = lineNumbers(ta.value);
  gutter.scrollTop = ta.scrollTop;
}

function setupEditor() {
  const ta = $("#reference");
  ta.addEventListener("input", syncGutter);
  ta.addEventListener("scroll", () => ($("#ref-gutter").scrollTop = ta.scrollTop));
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      const { selectionStart: s, selectionEnd: end, value } = ta;
      ta.value = value.slice(0, s) + "    " + value.slice(end);
      ta.selectionStart = ta.selectionEnd = s + 4;
      syncGutter();
    }
  });
}

function setSegment(id, value) {
  $$(`#${id} button`).forEach((b) => {
    const on = b.dataset.value === String(value);
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
  });
}

function setDevice(value, byUser = true) {
  const previous = state.device;
  state.device = value;
  if (byUser) state.deviceTouched = true;
  setSegment("device-seg", value);
  const t = state.selected;
  // Swap to the matching shape only if the user hasn't edited the reference.
  if (t && !t.custom && previous !== value && $("#reference").value === referenceFor(t, previous)) {
    setReference(referenceFor(t, value));
    if ($("#description").value === descriptionFor(t, previous)) $("#description").value = descriptionFor(t, value);
    if (byUser) toast(value === "cpu" ? "Using the small interpreter shape of this op" : "Using the full serving shape of this op");
  }
}

/* ---------------- pipeline ---------------- */

function setStep(name, stepState, meta) {
  const step = $(`.step[data-step="${name}"]`);
  step.dataset.state = stepState;
  if (meta !== undefined) {
    const metaEl = $(".step-meta", step);
    metaEl.textContent = meta;
    metaEl.title = meta === "—" ? "" : meta;
  }
}

function resetPipeline() {
  for (const s of ["queue", "generate", "parse", "verify", "select"]) setStep(s, "idle", "—");
}

let tickTimer;
function startTicker(fn) {
  stopTicker();
  tickTimer = setInterval(fn, 200);
}
function stopTicker() {
  clearInterval(tickTimer);
}

/* ---------------- output rendering ---------------- */

function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
  $$(".tab-panel").forEach((p) => p.classList.toggle("on", p.dataset.panel === name));
}

function renderKernel(kernel, reference) {
  const has = !!kernel;
  $("#kernel-empty").hidden = has;
  $("#kernel-view").hidden = !has;
  $("#copy-kernel").disabled = !has;
  $("#download-kernel").disabled = !has;
  highlightInto($("#kernel-code"), kernel || "");
  $("#kernel-gutter").textContent = has ? lineNumbers(kernel) : "";
  highlightInto($("#cmp-ref"), reference || "");
  highlightInto($("#cmp-kernel"), kernel || "# No kernel parsed from the model output.");
}

function renderExplanation(text, hasKernel, ran = true) {
  const box = $("#explanation");
  box.replaceChildren();
  if (!ran) {
    box.append(el("p", { class: "muted" }, "The optimization rationale appears here. The model writes it alongside the kernel; it isn't a caption added afterwards."));
    return;
  }
  if (!hasKernel) {
    box.append(el("p", { class: "muted" }, "No kernel was parsed, so there's no rationale to show. The full model output is under Raw output."));
    return;
  }
  if (!text) {
    box.append(el("p", { class: "muted" }, "The model produced a kernel but no explanation section."));
    return;
  }
  for (const para of text.split(/\n{2,}/)) box.append(el("p", {}, para.trim()));
}

function renderRaw(text, live) {
  $("#raw-code").textContent = text || "";
  $("#raw-cursor").hidden = !live;
  $("#live-dot").hidden = !live;
  if (live) {
    const view = $("#raw-view");
    view.scrollTop = view.scrollHeight;
  }
}

function verifyHead(stateChip, extra) {
  return el("div", { class: "card-head" }, el("h3", {}, "Verification"), extra, stateChip);
}

function renderVerification(v, phase, label) {
  const card = $("#verify-card");
  card.replaceChildren();
  const who = label ? el("span", { class: "chip subtle mono" }, label) : null;

  if (phase === "idle") {
    card.append(
      verifyHead(el("span", { class: "chip subtle state-chip" }, "idle")),
      el("div", { class: "verify-placeholder" }, "Correctness and speedup against the PyTorch reference appear here after a run.")
    );
    return;
  }

  if (phase === "running") {
    const interp = state.device === "cpu";
    card.append(
      verifyHead(el("span", { class: "chip accent state-chip" }, "running"), who),
      el(
        "div",
        { class: "verify-body" },
        verdict(
          "run", "", "Running in sandbox subprocess",
          interp ? "compile → correctness in Triton's CPU interpreter (no timing)" : "compile → correctness → benchmark on the GPU"
        ),
        facts(interp
          ? ["interpreter", `atol ${$("#atol").value}`, `rtol ${$("#rtol").value}`]
          : ["warmup 10", "iters 50", `atol ${$("#atol").value}`, `rtol ${$("#rtol").value}`])
      )
    );
    return;
  }

  if (!v || !v.ran) {
    card.append(
      verifyHead(el("span", { class: "chip subtle state-chip" }, "not run"), who),
      el("div", { class: "verify-body" }, verdict("skip", "–", "Not run", v?.skipped_reason || "Skipped."))
    );
    return;
  }

  const runFacts = facts([
    v.interpreted ? "Triton interpreter" : DEVICE_LABEL[v.device] || v.device,
    `atol ${v.atol}`,
    `rtol ${v.rtol}`,
    v.passed && v.iters ? `median of ${v.iters} after ${v.warmup} warmup` : null,
    v.wall_time_s != null ? `harness ${fmtSecs(v.wall_time_s)}` : null,
    v.cached ? "cached result" : null,
    v.duplicate_of ? `same kernel as ${v.duplicate_of}` : null,
  ]);

  if (v.passed && v.speedup == null) {
    card.append(
      verifyHead(el("span", { class: "chip accent state-chip" }, "correct"), who),
      el(
        "div",
        { class: "verify-body" },
        verdict("ok", "✓", "Correct in the interpreter", "Matches the PyTorch reference within tolerance on a small shape"),
        el("div", { class: "verify-placeholder", style: "padding:0" },
          "The interpreter checks correctness only. Speed needs the GPU harness, so this run isn't GPU-verified."),
        runFacts
      )
    );
    return;
  }

  if (v.passed) {
    const faster = v.speedup >= 1;
    const max = Math.max(v.reference_time_s || 0, v.candidate_time_s || 0) || 1;
    card.append(
      verifyHead(el("span", { class: "chip accent state-chip" }, "passed"), who),
      el(
        "div",
        { class: "verify-body" },
        verdict("ok", "✓", "Correct", "Matches the PyTorch reference within tolerance"),
        el(
          "div",
          { class: "speedup" },
          el("span", { class: `speedup-value ${faster ? "faster" : "slower"}` }, fmtSpeedup(v.speedup)),
          el("span", { class: "speedup-label" }, faster ? "faster than PyTorch eager" : "slower than PyTorch eager")
        ),
        el(
          "div",
          { class: "bars" },
          barRow("PyTorch eager", "ref", v.reference_time_s, max),
          barRow("Generated", "cand", v.candidate_time_s, max)
        ),
        runFacts
      )
    );
    return;
  }

  const incorrect = v.status === "ok";
  const title = incorrect ? "Incorrect" : { timeout: "Timed out", crash: "Crashed", error: "Failed to run" }[v.status] || v.status;
  const sub = incorrect
    ? "Ran, but the output doesn't match the reference within tolerance"
    : v.status === "timeout"
      ? "Killed at the sandbox timeout"
      : "The kernel raised or failed to compile in the sandbox";
  const body = el("div", { class: "verify-body" }, verdict("bad", "✕", title, sub));
  if (v.error) body.append(el("pre", { class: "errlog" }, v.error.length > 4000 ? "…" + v.error.slice(-4000) : v.error));
  body.append(runFacts);
  card.append(verifyHead(el("span", { class: "chip state-chip", style: "color:var(--err)" }, incorrect ? "incorrect" : v.status), who), body);
}

function verdict(kind, icon, title, sub) {
  return el(
    "div",
    { class: `verdict ${kind}` },
    el("div", { class: "verdict-icon" }, icon),
    el("div", {}, el("div", { class: "verdict-title" }, title), el("div", { class: "verdict-sub" }, sub))
  );
}

function facts(items) {
  return el("div", { class: "facts" }, items.filter(Boolean).map((f) => el("span", { class: "chip" }, f)));
}

function barRow(label, cls, seconds, max) {
  const bar = el("div", { class: `bar ${cls}` }, el("span", { style: "width:0%" }));
  requestAnimationFrame(() => requestAnimationFrame(() => (bar.firstChild.style.width = `${((seconds || 0) / max) * 100}%`)));
  return el("div", { class: "bar-row" }, el("span", { class: "label" }, label), bar, el("span", { class: "val" }, fmtMs(seconds)));
}

/* ---------------- candidates ---------------- */

function verdictOf(c) {
  const v = c.verification;
  if (c.phase === "streaming") return { text: "generating", cls: "info" };
  if (c.phase === "verifying") return { text: "verifying", cls: "info" };
  if (c.parsed && c.parsed.status !== "ok") return { text: "unparseable", cls: "warn" };
  if (!v) return { text: c.parsed ? "parsed" : "waiting", cls: "skip" };
  if (!v.ran) return { text: "not run", cls: "skip" };
  if (v.passed) return { text: v.speedup == null ? "correct" : "passed", cls: "ok" };
  return { text: v.status === "ok" ? "incorrect" : v.status, cls: "bad" };
}

function renderCandidate(r, c) {
  const best = r.selected?.candidate === c.id;
  const v = verdictOf(c);
  const meta = [
    `${c.raw.length.toLocaleString()} chars`,
    c.verification?.duplicate_of ? `= ${c.verification.duplicate_of}` : null,
    c.verification?.cached && !c.verification?.duplicate_of ? "cached" : null,
    c.verification?.wall_time_s != null && !c.verification?.cached ? `harness ${fmtSecs(c.verification.wall_time_s)}` : null,
  ].filter(Boolean).join(" · ");
  c.node.replaceChildren(
    el("div", { class: "cand-top" }, el("span", { class: "cand-id" }, c.id), best ? el("span", { class: "cand-best" }, "SELECTED") : null),
    el(
      "div",
      { class: "cand-verdict" },
      el("span", { class: `tag ${v.cls}` }, v.text),
      c.verification?.speedup != null ? el("span", { class: "cand-speed" }, fmtSpeedup(c.verification.speedup)) : null
    ),
    el("div", { class: "cand-meta" }, meta)
  );
  c.node.dataset.phase = c.phase;
  c.node.classList.toggle("best", best);
  c.node.classList.toggle("focus", r.focus === c.id);
}

function addRound(r, ev) {
  const round = el("div", { class: "round" });
  const title = ev.kind === "sample"
    ? `${ev.num_candidates} ${ev.num_candidates === 1 ? "sample" : "samples"}`
    : `Repair of ${ev.repairing}`;
  round.append(
    el("div", { class: "round-head" }, el("span", { class: "round-k" }, `round ${ev.round}`), el("span", { class: "round-title" }, title))
  );
  if (ev.example) {
    r.example = ev.example;
    $("#search-card").hidden = false;
    round.append(el("p", { class: "round-note" },
      "Prompt included a solved example for a different op: ",
      el("strong", {}, ev.example.op_name || ev.example.id),
      " (from the verified template library), plus Triton API notes."));
  }
  if (ev.feedback) {
    round.append(el("details", { class: "feedback" }, el("summary", {}, "Harness feedback sent to the model"), el("pre", {}, ev.feedback)));
  }
  const grid = el("div", { class: "cand-grid" });
  round.append(grid);
  $("#rounds").append(round);
  for (const id of ev.candidates) {
    const c = { id, round: ev.round, raw: "", parsed: null, verification: null, phase: "streaming", node: null };
    c.node = el("button", { class: "cand", type: "button", onclick: () => focusCandidate(r, id, true) });
    grid.append(c.node);
    r.cands.set(id, c);
    renderCandidate(r, c);
  }
}

function focusCandidate(r, id, pinned = false) {
  const c = r.cands.get(id);
  if (!c) return;
  if (pinned) r.pinned = true;
  const prev = r.cands.get(r.focus);
  r.focus = id;
  if (prev) renderCandidate(r, prev);
  renderCandidate(r, c);

  const chip = $("#focus-chip");
  chip.hidden = r.cands.size < 2;
  chip.textContent = r.selected?.candidate === id ? `${id} · selected` : id;

  const p = c.parsed;
  renderKernel(p?.triton_kernel, r.reference);
  renderExplanation(p?.optimization_explanation, p?.status === "ok", !!p);
  renderRaw(c.raw, c.phase === "streaming");
  $("#unparseable-callout").hidden = !p || p.status === "ok";
  if (c.phase === "verifying") renderVerification(null, "running", id);
  else if (c.verification) renderVerification(c.verification, "result", id);
  else renderVerification(null, "idle");
  if (pinned) showTab(p?.status === "ok" ? "kernel" : "raw");
}

function summarizeSearch(r) {
  const all = [...r.cands.values()];
  const passed = all.filter((c) => c.verification?.passed).length;
  const rounds = new Set(all.map((c) => c.round)).size;
  $("#search-summary").textContent =
    `${all.length} candidate${all.length === 1 ? "" : "s"} · ${rounds} round${rounds === 1 ? "" : "s"} · ${passed} passed`;
}

/* ---------------- run ---------------- */

function setRunning(running) {
  state.running = running;
  const btn = $("#run-btn");
  btn.classList.toggle("stop", running);
  $("#run-label").textContent = running ? "Stop" : state.samples > 1 || state.repair > 0 ? "Run kernel search" : "Generate kernel";
  $("#run-kbd").textContent = running ? "esc" : navigator.platform.includes("Mac") ? "⌘ ↵" : "Ctrl ↵";
}

async function run() {
  if (state.running) return;
  const description = $("#description").value.trim();
  const reference = $("#reference").value;
  if (!description || !reference.trim()) {
    toast("Add an op description and a PyTorch reference first", true);
    return;
  }

  const atol = parseFloat($("#atol").value);
  const rtol = parseFloat($("#rtol").value);
  const payload = {
    description,
    pytorch_reference: reference,
    verify: $("#verify").checked,
    device: state.device,
    tolerance_atol: Number.isFinite(atol) ? atol : 0.01,
    tolerance_rtol: Number.isFinite(rtol) ? rtol : 0.01,
    num_candidates: state.samples,
    repair_rounds: state.repair,
  };

  const r = (state.run = {
    started: performance.now(),
    genStarted: null,
    meta: null,
    reference,
    description,
    opName: state.selected?.custom ? "Custom op" : state.selected?.op_name || "Custom op",
    category: state.selected?.custom ? "custom" : state.selected?.category,
    cands: new Map(),
    focus: null,
    pinned: false,
    selected: null,
    tokens: 0,
    stopped: false,
    error: null,
    done: null,
    rafPending: false,
  });

  resetPipeline();
  renderKernel(null, reference);
  renderExplanation(null, false, false);
  renderRaw("", true);
  renderVerification(null, "idle");
  $("#unparseable-callout").hidden = true;
  $("#focus-chip").hidden = true;
  $("#rounds").replaceChildren();
  $("#search-summary").textContent = "";
  $("#search-card").hidden = state.samples === 1 && state.repair === 0;
  showTab("raw");
  setRunning(true);
  setStep("queue", "active", "requesting…");

  state.abort = new AbortController();
  try {
    const resp = await fetch("/generate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: state.abort.signal,
    });
    if (!resp.ok) throw new Error(await errorDetail(resp));

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (line) handleEvent(r, JSON.parse(line));
      }
    }
    if (!r.selected && !r.error) throw new Error("Stream ended before the search finished.");
  } catch (err) {
    stopTicker();
    if (err.name === "AbortError") {
      r.stopped = true;
      markActiveSteps("warn", "stopped");
      toast("Stopped");
    } else {
      r.error = err.message;
      markActiveSteps("failed", "error");
      toast(err.message, true);
    }
    const c = r.cands.get(r.focus);
    if (c) renderRaw(c.raw, false);
  } finally {
    stopTicker();
    setRunning(false);
    state.abort = null;
    if (r.selected || r.error) pushHistory(r);
    refreshHealth();
    refreshStats();
  }
}

function markActiveSteps(stepState, meta) {
  $$('.step[data-state="active"]').forEach((s) => setStep(s.dataset.step, stepState, meta));
}

function scheduleLiveRender(r) {
  if (r.rafPending) return;
  r.rafPending = true;
  requestAnimationFrame(() => {
    r.rafPending = false;
    const c = r.cands.get(r.focus);
    if (c && c.phase === "streaming") renderRaw(c.raw, true);
    for (const cand of r.cands.values()) if (cand.phase === "streaming") renderCandidate(r, cand);
  });
}

function countsByPhase(r, round) {
  const cands = [...r.cands.values()].filter((c) => c.round === round);
  return {
    total: cands.length,
    parsed: cands.filter((c) => c.parsed).length,
    kernels: cands.filter((c) => c.parsed?.status === "ok").length,
    verified: cands.filter((c) => c.verification).length,
    passed: cands.filter((c) => c.verification?.passed).length,
  };
}

function handleEvent(r, ev) {
  switch (ev.event) {
    case "meta":
      r.meta = ev;
      break;
    case "queued":
      setStep("queue", "active", "waiting for GPU…");
      startTicker(() => setStep("queue", "active", `waiting ${fmtSecs((performance.now() - r.started) / 1000)}`));
      break;
    case "started": {
      const waited = (performance.now() - r.started) / 1000;
      setStep("queue", "done", waited > 0.5 ? `waited ${fmtSecs(waited)}` : "acquired");
      r.genStarted = performance.now();
      setStep("generate", "active", "loading…");
      break;
    }
    case "round": {
      r.round = ev.round;
      addRound(r, ev);
      if (!r.pinned || ev.round > 0) focusCandidate(r, ev.candidates[0]);
      showTab("raw");
      const label = ev.kind === "sample" ? `${ev.num_candidates}× sample` : `repair ${ev.round}`;
      const t0 = performance.now();
      startTicker(() => {
        const chars = [...r.cands.values()].filter((c) => c.round === ev.round).reduce((n, c) => n + c.raw.length, 0);
        setStep("generate", "active", `${label} · ${fmtSecs((performance.now() - t0) / 1000)} · ${chars.toLocaleString()} chars`);
      });
      if (ev.round > 0) {
        setStep("parse", "active", "—");
        setStep("verify", "active", "—");
      }
      break;
    }
    case "token": {
      const c = r.cands.get(ev.candidate);
      if (c) {
        c.raw += ev.text;
        scheduleLiveRender(r);
      }
      break;
    }
    case "generated": {
      stopTicker();
      r.tokens += ev.num_tokens;
      for (const c of r.cands.values()) {
        if (c.round === ev.round && c.phase === "streaming") {
          c.phase = "generated";
          renderCandidate(r, c);
        }
      }
      const tokLabel = { mock: "words", remote: "≈tok" }[r.meta?.model_variant] || "tok";
      setStep("generate", "done", `${ev.num_tokens} ${tokLabel} · ${ev.tokens_per_s ?? "—"}/s · ${fmtSecs(ev.generation_time_s)}`);
      const focused = r.cands.get(r.focus);
      if (focused) renderRaw(focused.raw, false);
      break;
    }
    case "parsed": {
      const c = r.cands.get(ev.candidate);
      if (!c) break;
      c.parsed = ev;
      c.raw = ev.raw_output;
      c.phase = "parsed";
      renderCandidate(r, c);
      if (r.focus === c.id) focusCandidate(r, c.id);
      if (r.focus === c.id && ev.status === "ok" && !r.pinned) showTab("kernel");
      const n = countsByPhase(r, c.round);
      const allDone = n.parsed === n.total;
      setStep("parse", allDone ? (n.kernels ? "done" : "failed") : "active",
        n.total > 1 ? `${n.kernels}/${n.total} kernels` : ev.status === "ok" ? (ev.optimization_explanation ? "kernel + explanation" : "kernel, no explanation") : "no kernel found");
      break;
    }
    case "verifying": {
      const c = r.cands.get(ev.candidate);
      if (!c) break;
      c.phase = "verifying";
      renderCandidate(r, c);
      if (!r.pinned && r.focus !== c.id) focusCandidate(r, c.id);
      else if (r.focus === c.id) renderVerification(null, "running", c.id);
      const t0 = performance.now();
      startTicker(() => setStep("verify", "active", `${c.id} · ${state.device === "cpu" ? "interpreter" : "sandbox"} · ${fmtSecs((performance.now() - t0) / 1000)}`));
      break;
    }
    case "verification": {
      stopTicker();
      const c = r.cands.get(ev.candidate);
      if (!c) break;
      c.verification = ev;
      c.phase = "done";
      renderCandidate(r, c);
      if (r.focus === c.id) renderVerification(ev, "result", c.id);
      summarizeSearch(r);
      const n = countsByPhase(r, c.round);
      if (!ev.ran) setStep("verify", "skipped", shortSkip(ev.skipped_reason));
      else if (n.total > 1) setStep("verify", n.verified < n.total ? "active" : n.passed ? "done" : "failed", `${n.passed}/${n.total} passed`);
      else if (ev.passed) setStep("verify", "done", ev.speedup == null ? "correct · interpreter" : `correct · ${fmtSpeedup(ev.speedup)}`);
      else setStep("verify", "failed", ev.status === "ok" ? "incorrect" : ev.status);
      break;
    }
    case "selected": {
      r.selected = ev;
      summarizeSearch(r);
      for (const c of r.cands.values()) renderCandidate(r, c);
      focusCandidate(r, ev.candidate);
      showTab(ev.status === "ok" ? "kernel" : "raw");
      const v = ev.verification || {};
      const state_ = v.passed ? "done" : v.ran ? "failed" : "skipped";
      const label = r.cands.size > 1 ? `${ev.candidate} · ` : "";
      setStep("select", state_, v.passed ? `${label}${v.speedup == null ? "correct" : fmtSpeedup(v.speedup)}` : `${label}${ev.num_passed}/${ev.num_candidates} passed`);
      $(".step[data-step='select'] .step-meta").title = ev.reason;
      break;
    }
    case "error":
      stopTicker();
      r.error = ev.detail;
      markActiveSteps("failed", "error");
      renderRaw((r.cands.get(r.focus)?.raw || "") + `\n\n[${ev.detail}]`, false);
      toast(ev.detail, true);
      break;
    case "done":
      r.done = ev;
      break;
  }
}

function shortSkip(reason) {
  if (!reason) return "skipped";
  if (reason.includes("CUDA")) return "skipped · no GPU";
  if (reason.includes("Triton")) return "skipped · no Triton";
  if (reason.includes("disabled")) return "skipped · disabled";
  if (reason.includes("parsed")) return "skipped · no kernel";
  return "skipped";
}

/* ---------------- history ---------------- */

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(0, HISTORY_MAX)));
  } catch {
    /* storage full or disabled: history stays in memory */
  }
}

function pushHistory(r) {
  const s = r.selected;
  state.history.unshift({
    id: `${Date.now()}`,
    time: new Date().toISOString(),
    opName: r.opName,
    category: r.category,
    variant: r.meta?.model_variant || state.health?.model_variant || "unknown",
    device: r.meta?.device || state.device,
    description: r.description,
    reference: r.reference,
    selected: s
      ? {
          candidate: s.candidate, status: s.status, triton_kernel: s.triton_kernel,
          optimization_explanation: s.optimization_explanation, raw_output: s.raw_output,
          verification: s.verification, reason: s.reason, candidates: s.candidates,
        }
      : null,
    totalTime: r.done?.total_time_s ?? null,
    error: r.error,
  });
  state.history = state.history.slice(0, HISTORY_MAX);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  const body = $("#history-body");
  body.replaceChildren();
  $("#history-count").textContent = state.history.length ? `· ${state.history.length}` : "";
  if (!state.history.length) {
    body.append(el("tr", { class: "empty-row" }, el("td", { colspan: 8 }, "No runs yet. Results stay in this browser only.")));
    return;
  }
  for (const h of state.history) {
    const s = h.selected;
    const v = s?.verification;
    const output = h.error
      ? el("span", { class: "tag bad" }, "error")
      : s?.status === "ok"
        ? el("span", { class: "tag ok" }, "kernel")
        : el("span", { class: "tag warn" }, "unparseable");
    const verification = !v
      ? el("span", { class: "tag skip" }, "—")
      : !v.ran
        ? el("span", { class: "tag skip" }, "not run")
        : v.passed
          ? el("span", { class: "tag ok" }, v.interpreted ? "correct · interp" : "correct")
          : el("span", { class: "tag bad" }, v.status === "ok" ? "incorrect" : v.status);
    const n = s?.candidates?.length || 0;
    const rounds = new Set((s?.candidates || []).map((c) => c.round)).size;
    body.append(
      el(
        "tr",
        { dataset: { id: h.id }, onclick: () => restoreRun(h), title: s?.reason || "Restore this run" },
        el("td", {}, new Date(h.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })),
        el("td", {}, h.opName),
        el("td", {}, VARIANT_LABEL[h.variant] || h.variant),
        el("td", {}, n ? `${n} cand · ${rounds} rnd` : "—"),
        el("td", {}, output),
        el("td", {}, verification),
        el("td", { class: "num" }, v?.passed ? fmtSpeedup(v.speedup) : "—"),
        el("td", { class: "num" }, fmtSecs(h.totalTime))
      )
    );
  }
}

function restoreRun(h) {
  if (state.running) return;
  $("#description").value = h.description;
  setReference(h.reference);
  const p = h.selected;
  renderKernel(p?.triton_kernel, h.reference);
  renderExplanation(p?.optimization_explanation, p?.status === "ok", !!p);
  renderRaw(p?.raw_output || "", false);
  $("#unparseable-callout").hidden = !p || p.status === "ok";
  renderVerification(p?.verification, p?.verification ? "result" : "idle", p?.candidate);
  resetPipeline();
  $("#search-card").hidden = true;
  $("#focus-chip").hidden = true;
  state.run = null;
  showTab(p?.status === "ok" ? "kernel" : "raw");
  $("#studio").scrollIntoView({ behavior: "smooth" });
  toast(`Restored run from ${new Date(h.time).toLocaleTimeString()}`);
}

/* ---------------- wiring ---------------- */

function currentKernel() {
  const r = state.run;
  return r?.cands.get(r.focus)?.parsed?.triton_kernel || $("#kernel-code").textContent;
}

function wire() {
  setupEditor();
  $("#op-search").addEventListener("input", renderOpGroups);
  $("#custom-op").addEventListener("click", selectCustom);
  $("#reset-btn").addEventListener("click", () => {
    if (state.selected?.custom) selectCustom();
    else if (state.selected?.id) selectTemplate(state.selected.id);
  });
  $("#copy-ref").addEventListener("click", () => copyText($("#reference").value, "Reference"));
  $("#copy-kernel").addEventListener("click", () => copyText(currentKernel(), "Kernel"));
  $("#download-kernel").addEventListener("click", () => {
    const name = (state.selected?.id || "custom_op").replace(/[^a-z0-9_]+/gi, "_");
    const blob = new Blob([currentKernel()], { type: "text/x-python" });
    const a = el("a", { href: URL.createObjectURL(blob), download: `${name}_triton.py` });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $$("#device-seg button").forEach((b) => b.addEventListener("click", () => setDevice(b.dataset.value)));
  $$("#samples-seg button").forEach((b) =>
    b.addEventListener("click", () => {
      state.samples = Number(b.dataset.value);
      setSegment("samples-seg", state.samples);
      setRunning(state.running);
    })
  );
  $$("#repair-seg button").forEach((b) =>
    b.addEventListener("click", () => {
      state.repair = Number(b.dataset.value);
      setSegment("repair-seg", state.repair);
      setRunning(state.running);
    })
  );
  $$(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  $("#run-btn").addEventListener("click", () => (state.running ? state.abort?.abort() : run()));
  $("#clear-history").addEventListener("click", () => {
    state.history = [];
    saveHistory();
    renderHistory();
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      run();
    } else if (e.key === "Escape" && state.running) {
      state.abort?.abort();
    }
  });
  document.addEventListener("click", (e) => {
    const adv = $(".advanced");
    if (adv.open && !adv.contains(e.target)) adv.open = false;
  });
}

function init() {
  wire();
  setRunning(false);
  renderVerification(null, "idle");
  renderHistory();
  refreshHealth();
  refreshStats();
  loadTemplates();
  setInterval(refreshHealth, 20000);
  setInterval(refreshStats, 20000);
}

document.addEventListener("DOMContentLoaded", init);
