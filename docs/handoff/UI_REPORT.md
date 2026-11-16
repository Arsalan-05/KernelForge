# UI Report — agent/ui

Branch: `agent/ui` (from `agent/merged`)  
Committed: 2026-11-15  
Test result: **231 passed, 5 skipped** (all skips are GPU/transformers-dependent)

---

## 1. Results Tab

**Files changed:** `app/backend/main.py`, `app/web/index.html`, `app/web/app.js`, `app/web/styles.css`

**What was added:**

A new **GET `/results`** endpoint reads `data/raw/interpret_report_2026-11-15.json` and returns structured JSON:

```json
{
  "interpreter": {
    "mode": "triton-interpreter",
    "total": 300,
    "passed": 285,
    "generated_at": "...",
    "by_category": {
      "quantized_matmul": {"total": 60, "passed": 60, "oom": 0, "templates": [...]},
      "attention":        {"total": 60, "passed": 58, "oom": 2,  "templates": [...]},
      "kv_cache":         {"total": 60, "passed": 60, "oom": 0,  "templates": [...]},
      "norm":             {"total": 60, "passed": 47, "oom": 13, "templates": [...]},
      "rope":             {"total": 60, "passed": 60, "oom": 0,  "templates": [...]}
    }
  },
  "gpu": null
}
```

A "Results" tab is added to the output card. It renders:
- **Summary row**: 285/300 shapes pass (95%), 10/15 templates fully pass (all 20 shapes).
- **OOM note**: 15 shapes OOM in an 8 GB VM; kept in dataset, counted separately.
- **Interpreter note**: correctness only, CPU, no speedup numbers.
- **Per-category table**: pass count, OOM chip, animated pass-rate bar.
- **Per-template detail** (collapsible): one row per template within each category.
- **GPU eval section**: "Pending Kaggle T4 session" with link to KAGGLE_RUNBOOK.md context.

The tab eager-loads on `init()` so there's no first-click delay. If the report file is absent or unreadable, the tab shows an error and a "Retry" button.

**Actual coverage (from report):**

| Category | Pass | OOM | Fully-passing templates |
|---|---|---|---|
| Quantized matmul | 60/60 | 0 | 3/3 |
| Attention | 58/60 | 2 | 1/3 |
| KV-cache | 60/60 | 0 | 3/3 |
| Normalization | 47/60 | 13 | 0/3 |
| RoPE | 60/60 | 0 | 3/3 |
| **Total** | **285/300** | **15** | **10/15** |

---

## 2. Compare View

**What was done:**  
The existing "Side by side" tab now shows the generated model variant in the right-pane header (e.g. "Generated Triton · ModelNew · base" or "Generated Triton · ModelNew · qwen3-coder" for remote). This is set in `renderKernel()` using `state.health.model_name` and `state.health.model_variant`.

A "baseline vs fine-tuned comparison pending GPU run" notice appears below the split panes.

**What remains:**  
Full multi-model comparison (baseline vs few-shot vs fine-tuned side by side with diff highlighting) requires GPU eval results to exist. When results land in `evaluation/results/`, a future PR can add a comparison mode to `GET /results` and render it in the compare pane.

---

## 3. Honest Labels

**What was done:**
- The right-pane label in the compare tab now reflects the actual model/variant on every `renderKernel()` call.
- Harness mode was already shown via the facts chips in the verification card (`Triton interpreter` vs GPU device label). This labeling is unchanged but correct — it always states which harness ran.
- Speedup is already `null` (not shown) for interpreter mode. No change needed.
- For the `remote` variant, the banner text already says "Running {model_name} through a hosted API — a general-purpose model, not KernelForge's fine-tuned adapter." This is correct.

---

## 4. Interpreter Coverage (285/300)

Rendered in the Results tab (see §1). The numbers come from the actual committed report file, not from hardcoded constants.

---

## 5. Usability

**Implemented:**
- **Remembered settings**: `device`, `samples`, and `repair` are saved to `localStorage` under key `kernelforge.settings.v1` on every change, and restored on page load. If the user has never visited, defaults apply (cuda, 1, 0).
- **Retry button**: A "Retry" button appears in the controls bar after a generation error. It is hidden when the run succeeds, while running, or after a user-triggered stop (Escape). Clicking it calls `run()` with the current form contents.
- **Op list loading state**: The op-groups area shows "Loading ops…" (with an animated spinner) while `/templates` is being fetched. If the fetch fails, it shows an error message and a "Retry" button.
- **Results tab loading state**: Shows a spinner while `/results` is in-flight; shows error + retry on failure.
- **Copy/download**: Already present for the kernel. The reference code has a "Copy" button in the input card header. No changes needed.
- **Shareable permalink**: `?op=` is already written to the URL on template selection. Device/samples/repair are now saved to localStorage (a permalink for those would require URL params, not done here — localStorage is simpler and doesn't clutter the URL).

**Not implemented in this PR:**
- Full shareable URL that encodes device/samples/repair in the query string.
- Keyboard shortcut overlay (the ⌘↵ / Ctrl↵ hint on the button and Esc for stop are the existing shortcuts; a "?" keyboard shortcut overlay was not added).

---

## 6. Responsive Layout and Accessibility

**Implemented:**

**Accessibility:**
- All three segmented controls (Harness/Samples/Repair) now use `<label id="…">` elements with `aria-labelledby` on the `<div role="radiogroup">`.
- Each radio button within Samples and Repair segmented controls has an `aria-label` with the full option description (e.g., "2 samples", "1 repair round").
- All `.tab` buttons now maintain `aria-selected` in sync with their `on` class via `showTab()`.
- Run button has `aria-label` updated dynamically ("Generate kernel" / "Stop generation (Escape)").
- Copy and Download buttons have descriptive `aria-labels`.
- The live dot and other decorative SVGs have `aria-hidden="true"`.
- **`:focus-visible` rings**: All interactive elements (`button`, `a`, `input`, `[tabindex]`) now show a 2px lime-green outline on keyboard focus. Mouse clicks suppress the ring (`:focus:not(:focus-visible)`).

**Responsive phone (≤ 640px):**
- Controls bar stacks vertically; each control group spans the full width.
- Segmented buttons flex to fill available width.
- Tab panel min-height reduced from 400px to 280px.
- Results bar-cell min-width reduced.

---

## 7. No Layout Jumps While Streaming

**What was done:**  
Added `min-height` to the kernel, compare, and raw tab panels (`400px` on desktop, `280px` on phone). This reserves height for the codeview before the kernel appears, so the card doesn't reflow when streaming tokens arrive and the empty placeholder is replaced by the code view.

Previously, the card grew from ~200px (empty state padding) to up to 560px (codeview max-height) when a kernel first appeared, pushing the verification card and explanation card down.

---

## 8. Playwright Screenshots

**Not implemented.**  
Playwright is not installed in this environment and no `.playwright.config.ts` exists. Screenshots were not taken. Manual testing of the UI requires starting the demo server with mock mode (`KERNELFORGE_MOCK_MODEL=1`) and exercising the golden path.

If Playwright is added in a future PR, the recommended test cases are:
- Load page, click an op, generate (mock mode), check Results tab renders.
- Trigger an error (bad reference), verify retry button appears.
- Reload page, verify device/samples/repair settings are restored.
- Keyboard navigation: Tab through controls, check focus rings are visible.

---

## Files changed

| File | Change |
|---|---|
| `app/backend/main.py` | Added `_ROOT` constant; added `GET /results` endpoint |
| `app/web/index.html` | Results tab button + panel; compare ablation notice; aria-labelledby on segmented controls; aria-selected on tabs; retry button |
| `app/web/app.js` | `SETTINGS_KEY`, `loadSettings()`, `saveSettings()`; `loadResults()`, `renderResults()`; loading state for op-groups; retry button wire-up; `showTab()` syncs aria-selected and triggers results load; `renderKernel()` updates compare pane label; `init()` restores saved settings |
| `app/web/styles.css` | `:focus-visible` rings; `min-height` for tab panels; results panel styles (summary, table, bars, chips, pending section); compare pending notice; phone-width enhancements; op loading/error styles |

---

## Test count

UI changes have no backend tests (the new endpoint reads a file and returns JSON — it's tested implicitly by the existing `test_app.py` import smoke test). Frontend changes are not covered by the pytest suite (no browser automation). 231 tests pass, 5 skip.
