# Performance notes

Compute Bazaar runs as a local research platform, not a public deployment.
This page lists the performance checklist used during the Phase 0 cleanup:
what was applied and where, and which items don't apply to this project.

## Applied

| Item | Where | What it does |
|---|---|---|
| Index the database | `backend/app/db/models.py` | Composite indexes `(agent_id, is_winner)`, `(auction_id, is_winner)`, `(status, round_number)` plus `allocations.agent_id` for the dashboard and settlement queries. |
| Database connection pooling | `backend/app/db/models.py::_engine_options` | SQLite keeps a light setup. Postgres gets `pool_size`/`max_overflow`/`pool_recycle`/`pool_pre_ping`, tunable via `DB_POOL_*` env vars. |
| N+1 database queries | `endpoints.py`, `scheduler.py` | Dashboard metrics come from grouped aggregates; the current-round view eager-loads bids/allocations with their agents; settlement reuses already-loaded bids. `tests/test_performance_layer.py` checks the statement count stays constant as agents grow. |
| Cache expensive queries / server-side caching | `backend/app/core/cache.py` | `VersionedTTLCache` reuses dashboard metrics while a cheap fingerprint (max ids, closed-round count) is unchanged, for at most 5 s. Writes invalidate immediately. |
| Cache API responses | `backend/app/core/http.py`, `/resources` | Vendor bundles: `Cache-Control: public, max-age=604800`. Own static files: `no-cache` + ETag revalidation. `/resources`: `private, max-age=5`. |
| Compress API payloads | `backend/app/main.py` | `GZipMiddleware(minimum_size=1024)`. |
| Paginate large lists | `GET /api/v1/resources?limit=&offset=` | Bounded `limit` (1–500), total in `X-Total-Count`. Dashboard history was already windowed. |
| Defer non-critical scripts | `static/index.html` | Chart.js and `app.js` load with `defer`; the CDN fallback is handled in `app.js` instead of `document.write`. |
| Loading skeletons | `static/index.html` | Shimmer rows in the agent, ticker and history tables until the first payload; respects `prefers-reduced-motion`. |
| Debounce input handlers | `static/app.js::debounceAsync` | Simulation buttons ignore repeats within 400 ms and while a request is in flight. |
| Unnecessary re-renders | `static/app.js::fetchIfChanged` | Polls skip DOM/chart work when the payload is byte-identical, pause in hidden tabs, and charts refresh with `update('none')`. |
| Unused dependencies | `backend/requirements.txt` | Dropped `alembic`, `aiosqlite`, `passlib`, `python-multipart`; test tooling moved to `requirements-dev.txt`. |
| Compress images | `docs/screenshots/` | ~590 KB → ~120 KB each (progressive JPEG, q80). |

## Not applicable

| Item | Why it's skipped |
|---|---|
| Load balancer, CDN | Single-process local server; no deployment target. |
| Lighthouse audit | Needs a hosted build; the dashboard is a local dev tool. |
| Split code into chunks, minify JS/CSS | No JS build step. The dashboard is one ~27 KB script, served gzipped. Vendor bundles ship pre-minified. |
| Lazy loading | The dashboard has no images or below-the-fold heavy content. |
