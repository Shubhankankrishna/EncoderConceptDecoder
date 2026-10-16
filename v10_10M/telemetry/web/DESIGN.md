# V10 live dashboard: reference and verification

The primary visual reference is the user's preferred **http://127.0.0.1:8791/#v3train**, served from `run_m2_25M/webapp`. Its actual V3 training page, design history, styles, telemetry controller and chart implementations were inspected before implementation. The dashboard is available at **http://127.0.0.1:8766/#live**.

## What the older design gets right

`run_m2_25M/webapp/DESIGN.md`, `css/style.css`, `js/v3train.js`, `js/telemetry.js`, `js/charts.js` and `js/chart.js` organize the page around one immediate question: is the run healthy? A run hero carries the current step, duration, progress and a short verdict. A machine strip supplies context. Six compact metrics lead into section chips, health cards, and detailed charts with prominent latest values. Native typography, restrained color, white panels, thin borders and rounded corners keep a dense scientific interface readable. Dark mode and a single-column phone layout preserve the same hierarchy.

The historical design notes favor polishing this arrangement rather than replacing it with a different navigation system. Those notes were used as reference material; the current user's explicit preference selected this version.

`v9_webapp/src/{page.html,style.css,live.js,app.js}` and `guide/NOTES.md` emphasize module colors, provenance, and the difference between measured, planned and illustrative states. Those principles inform the labels, not the visual direction. `run_v2_19M/webapp` and `run_v3_19M/webapp` were also reviewed as older variants.

## What was carried into V10

- Sticky workspace header, a lightly tinted run hero, progress bar, six machine cells, six clickable sparkline metrics and section chips.
- Health cards with a short reading and its practical limits. Operational checks and research evidence remain separate.
- Two-column chart cards, bold latest values, diagnostic labels, question-mark explainers and collapsible records.
- Native system fonts and the reference's blue/teal/orange/purple accents. Light/dark themes and a tested 402-pixel layout.

V10 has different evidence than the historical V3 run. The UI therefore uses actual logged steps rather than interpolating them, labels the separate Engram budget, marks unrecorded GPU utilization as unavailable, and exposes known math/data and concept limitations. Historical V3 numbers and success claims were not reused.

## Data and interaction rules

The frontend is buildless HTML/CSS/ES modules with no external assets or chart dependency. It polls the independent localhost collector every five seconds while visible. A dashboard pause changes only frontend polling. HTTP failures retain the last successful snapshot and its timestamp. Collector freshness and transport freshness are separate checks.

Events are incrementally paginated by event ID and deduplicated by ID. Training paths preserve ingestion order and break on duplicate/rolled-back steps or missing values. Backfilled development files are plotted by checkpoint order, so an initial evaluation ingested late cannot create a false late-run loss spike. EMA resets at segment boundaries. Only SVG geometry is reduced, preserving bucket extrema; hover/keyboard inspection and CSV downloads retain raw points. Charts redraw on new data, control changes or resizing, and defer work while offscreen.

Routine evaluation curves always show their complete recorded history. The evaluation selector separates trainer samples from balanced checkpoint audits. Correct/count, Wilson intervals, per-module scores and unmodified responses remain visible. Missing trials are not rendered as zero accuracy. Model strings are HTML-escaped. Live dense preclip gradients and offline component/layer gradients have different labels and sources.

JSON snapshots and metric CSVs are server-generated HTTP attachments. Snapshot events share a consistent SQLite read-transaction watermark; overview and catalog are explicitly labeled as independent completion-time snapshots. No training controls or arbitrary filesystem paths are exposed.

## Verification on 11 October 2026

**16 automated tests passed:** ten frontend data-rule tests and six backend contract tests. They cover missing/zero values, final status metrics, Wilson intervals, duplicate/restarted steps, backfilled evaluations, EMA gaps, extrema preservation, cursor progress, escaped text, partial JSONL writes, collector restarts, malformed records, source rotation, corrected evaluations, read-only API/CORS, static asset allowlisting and export consistency.

Browser checks used the actual running API and a separate disposable fixture:

- Real run hero, loss curves, smoothing toggle, recorded-history windows and step/hour axes.
- Raw keyboard inspection, explainer disclosure, legend toggling and fixed source labels.
- Routine and balanced evaluations, per-module counts, correct-answer filtering, family selection, search and pagination.
- Component/layer gradient selection by family/objective; searchable catalog and raw metric series.
- Light/dark themes. Live, evaluation and diagnostic views at 402 pixels without horizontal page overflow; viewport override reset afterward.
- Pause/resume and direct section navigation.
- Full JSON and CSV downloads through the browser. The downloaded CSV value exactly matches the API; a 1,284-event snapshot preserved unique IDs and its watermark. The native snapshot-link download also completed successfully.
- Isolated fixtures for zero-trial accuracy, missing measurements, malicious-looking response text, empty runs, a 503 outage preserving the last good snapshot, and a stale collector despite a successful HTTP request. No fixture data enters the real store. The fixture server was stopped after testing.
- No runtime warnings/errors observed on the finished real dashboard. The only fixture HTTP errors were intentional outage responses.

The trainer retained PID 52954 and its original source SHA-256. No training source, dataset, optimizer, checkpoint tensor, or deadline changed. Only the independent telemetry service was restarted to load the UI routes and exports. Its measured RSS was about 38 MiB at the final check; browser CPU/GPU overhead was not benchmarked.

Screenshots are saved in `../data/ui_qa/`. Machine-readable checks and the final run/service state are in `../data/ui_qa/verification.json`.
