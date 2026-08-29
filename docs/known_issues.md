# Known Issues & Deferred Work

Consolidated, durable source of truth for every known/deferred item that has
so far only lived in session chat history. Written so a fresh session with no
prior context can pick any of these up from this file alone.

Status legend: `deferred` = agreed not doing now; `low-priority` = cosmetic /
non-blocking; `needs-decision` = needs a human call before proceeding.

---

## 1. Dashboard review — misses #5–8 (original Streamlit diagnosis)

Status: `deferred`

The Step-4 dashboard review surfaced four issues that were deliberately left
out of the initial pass. Each described against the current source.

### 1a. No graceful no-API-key UX
Status: `deferred`

There is no user-facing banner when the app runs without a usable LLM API key.
`get_llm()` returns `None` and `parse_query` falls back to regex parsing
(`src/agri_agent/agent/graph.py:150-191`), and `build_graph` raises if `llm is
None` (`graph.py:393-397`), which the app catches and shows as a generic "hit an
error" message (`app/streamlit_app.py:139-144`). The UX doesn't explain *why*
(diagnosis: missing `.env` key / provider). A banner explaining that the key is
missing (and that only date parsing will work) would be the fix.

### 1b. Debug panel underuses the full graph result state
Status: `deferred`

`render_debug_panel` (`app/components/debug_panel.py:6-24`) shows
`data_sources_used`, `validation_problems`, `signal_conflict_detected`, and raw
JSON, but **not** the richer `AgentState`: no `recommend_attempts` (retry
count), no per-signal grounding detail beyond the raw JSON, and no
`bundle.origin_date` shown. The `result` dict (passed in) and the bundle contain
all of this and could populate a proper evaluator view.

### 1c. Form ambiguity: requesting "today" as an explicit target in offline mode
Status: `deferred`

`st.date_input` cannot express "no date vs. today" cleanly, and today's date is
used as a sentinel meaning "unset" (`app/components/input_form.py`). A user who
actually wants a `target_date` of today in offline mode cannot say so —
`build_signal_bundle` would treat it as unset and use the default origin
(`parquet_max - FORECAST_HORIZON_DAYS`). Needs an explicit "today" affordance or
a documented sentinel.

### 1d. Rationale date vs. form target_date mismatch
Status: `deferred`

The recommendation's free-text `reasoning` can reference a date that differs
from the form's chosen `target_date`/`date` field, because the LLM writes the
narrative from the evidence it saw and the form's date only flows in via
`QueryParams`. Minor UX inconsistency — the displayed badge/date and the prose
can disagree. Tracking only; no planned fix.

---

## 2. Offline parquet read is not "fetch once, reuse"

Status: `deferred` (non-urgent)

`load_fused_dataset(str(FUSED_PARQUET))` is called more than once per offline
request:

- `scripts/run_pipeline.py:126` (`_resolve_reference_date`), `:182` (temporal
  pass 1), `:198` (reference-date fallback when no explicit date).
- `app/streamlit_app.py:94` (temporal pass 1) and `:103` (reference-date
  fallback).
- Then `build_signal_bundle` reads it **again** inside
  `_load_offline_fused()` (`src/agri_agent/agent/bundle.py:322-323`, called at
  `bundle.py:647`).

On the current 731-row dataset this is a few milliseconds — confirmed present,
not urgent. Cleanup direction already discussed: have
`resolve_temporal_expressions` accept an already-known `origin_date` parameter
instead of loading the parquet itself, so the datetime loader parses the file
once and the row/date max is threaded through both the temporal resolution and
`build_signal_bundle`. Note the two entrypoints duplicate this two-pass
temporal logic (see item 5).

---

## 3. `debug_panel.validation_passed` is derived, not a graph key

Status: `deferred` — "works but implicit"

The panel prints "Validation passed:" as `not final.get("validation_problems")`
(`app/components/debug_panel.py:16`). The graph/schema never emits a
`validation_passed` key — `final_output` is built in
`src/agri_agent/agent/graph.py:445-452` with only `validation_problems` and
`signal_conflict_detected`. Works correctly today (yields a real boolean), but
the panel's derivation and the graph's output are not coupled by schema — if
one changes (e.g. the graph starts emitting a different problems shape), the
panel could silently stop being correct.

---

## 4. `use_container_width` deprecation warning

Status: `low-priority`

On the installed Streamlit (1.59.x), passing `use_container_width=True` to
`st.plotly_chart` / `st_folium` prints a deprecation warning recommending
`width="stretch"` instead. Purely cosmetic; the call still works. Occurrences:
`app/components/forecast_chart.py:49` and `app/components/field_map.py`.

---

## 5. App + CLI duplicate the two-pass offline temporal-resolution logic

Status: `deferred`

`app/streamlit_app.py:88-109` mirrors the two-pass temporal-resolution /
reference-date logic in `scripts/run_pipeline.py:160-200` (both reimplement the
"Pass 1 against parquet_max, then real resolution against the computed origin"
dance). This is duplication that already drifted once (app originally had the
simpler spec version) and was aligned in the dashboard-fix session. A shared
helper would prevent future drift, but touches both the app and the CLI, so it
is flagged rather than changed unprompted.

---

## 6. `forecast_chart` dynamic horizon — fixed (verification record)

Status: `resolved` — kept as a record

The forecast-chart title originally hardcoded `"7-day soil-moisture forecast"`.
Now derived from the bundle:
`app/components/forecast_chart.py:42-44` →
`horizon = sm.get("horizon_days", 7)` and `title=f"{horizon}-day …"`. Verified
with a non-7 horizon (3) → title rendered `"3-day soil-moisture forecast"`.
No further action needed; recorded so a future session doesn't re-report it as
a bug.

---

## 7. Pylance `reportUndefinedVariable` ×30 in `scripts/run_pipeline.py`

Status: `deferred` — IDE-only, not a code bug

VS Code's Pylance reports ~30 `reportUndefinedVariable` errors in
`scripts/run_pipeline.py`. Confirmed to be an IDE interpreter/builtins
resolution misconfiguration: the file runs correctly end-to-end via
`python scripts/run_pipeline.py --mode offline` (produces a complete
`FusionRecommendation`). No code change required. If the warnings bother the
developer, point Pylance at the project's `.venv` interpreter.
