"""
End-to-end check of the running app in a real Chrome browser (Playwright).

Start the server first (python -m uvicorn backend.app.main:app --port 8000), then:

    pip install playwright
    python scripts/e2e_check.py                 # uses the installed Google Chrome
    python scripts/e2e_check.py --screenshots   # also refresh docs/screenshots/live-*.jpg
    python scripts/e2e_check.py --groq          # also exercise the live Groq agent mode

Every check prints PASS/FAIL; the script exits non-zero if anything fails.
Console errors and failed network requests are collected throughout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
SHOTS = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}", flush=True)
    return ok


def api(method: str, path: str, body: dict | None = None, headers: dict | None = None):
    req = urllib.request.Request(
        BASE + path, method=method, headers={"Content-Type": "application/json", **(headers or {})}
    )
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(req, data, timeout=30) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def text(page, sel: str) -> str:
    return page.locator(sel).inner_text().strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", action="store_true")
    ap.add_argument("--groq", action="store_true")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not args.headed)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        console_errors, failed = [], []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("requestfailed", lambda r: failed.append(f"{r.method} {r.url} {r.failure}"))
        page.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 else None)

        api("POST", "/api/v1/simulation/stop")  # start from a known state

        # 1. Dashboard loads and hydrates
        page.goto(BASE + "/", wait_until="domcontentloaded")
        check("dashboard HTML served", "Compute Bazaar" in page.title(), page.title())
        page.wait_for_function("typeof Chart !== 'undefined'", timeout=20000)
        check("Chart.js loaded (deferred script)", True)
        page.wait_for_function("!document.querySelector('#agent-performance-table-body .skeleton-row')", timeout=15000)
        check("skeleton rows replaced by data", True)
        check(
            "KPI: active agents populated",
            text(page, "#metric-active-agents") not in ("", "0", "--"),
            text(page, "#metric-active-agents"),
        )
        canvases = page.evaluate("[...document.querySelectorAll('canvas')].filter(c => c.width > 0).length")
        check("four charts rendered", canvases >= 4, f"{canvases} canvases")
        page.wait_for_function("typeof ws !== 'undefined' && ws && ws.readyState === 1", timeout=15000)
        check("live WebSocket feed connected", True)

        # 2. Classic simulation end to end
        start_round = text(page, "#metric-round-number")
        page.click("#btn-start-classic")
        page.wait_for_function(
            "document.querySelector('#status-text').innerText.toLowerCase().includes('online')", timeout=15000
        )
        check("start classic simulation -> status online", True, text(page, "#status-text"))
        page.wait_for_function(
            "document.querySelectorAll('#bid-ticker-body tr:not(#empty-bids-message):not(.skeleton-row)').length > 0",
            timeout=30000,
        )
        check("live bid ticker receives bids", True)
        deadline = time.time() + 40
        while time.time() < deadline and text(page, "#metric-round-number") == start_round:
            time.sleep(1)
        check(
            "auction rounds advance",
            text(page, "#metric-round-number") != start_round,
            f"{start_round} -> {text(page, '#metric-round-number')}",
        )
        page.wait_for_timeout(4000)
        hist_rows = page.locator("#auction-history-body tr").count()
        check("closed-round history populated", hist_rows > 0, f"{hist_rows} rows")
        welfare = float(text(page, "#metric-social-welfare").replace(",", "") or 0)
        check("social welfare > 0 after settlement", welfare > 0, f"${welfare:,.2f}")
        jain = float(text(page, "#metric-jains-index") or 0)
        check("Jain index in [0, 1]", 0.0 <= jain <= 1.0, f"{jain}")

        # 3. Backend API flow while the dashboard is open
        name = f"E2E-Agent-{int(time.time())}"
        st, agent = api(
            "POST", "/api/v1/agents/register", {"name": name, "description": "end-to-end check", "tier": "gold"}
        )
        check("register agent via API", st == 201 and agent.get("api_key", "").startswith("cb_"), str(st))
        key = agent.get("api_key", "")
        st, tok = api("POST", f"/api/v1/auth/token?api_key={key}")
        check("exchange API key for JWT", st == 200 and "access_token" in tok, str(st))
        auth = {"Authorization": f"Bearer {tok.get('access_token', '')}"}
        st, me = api("GET", "/api/v1/agents/me", headers=auth)
        check("JWT-authenticated profile", st == 200 and me.get("name") == name, str(st))
        st, bid = api(
            "POST",
            "/api/v1/bids/submit",
            {
                "resource_bundle": ["GPU-Slot-1", "CPU-Cores-8"],
                "bid_amount": 120.0,
                "task_description": "e2e bundle",
                "priority": "high",
            },
            headers={"X-API-Key": key},
        )
        check("submit combinatorial bid", st == 201, f"{st} {bid.get('id') if isinstance(bid, dict) else bid}")
        st, bad = api(
            "POST", "/api/v1/bids/submit", {"resource_bundle": ["NOPE"], "bid_amount": 10.0}, headers={"X-API-Key": key}
        )
        check("invalid resource rejected", st == 400, str(st))
        st, _ = api("GET", "/api/v1/agents/me", headers={"Authorization": "Bearer not-a-token"})
        check("bad JWT rejected", st == 401, str(st))
        st, res = api("GET", "/api/v1/resources?limit=2")
        check("paginated resources", st == 200 and len(res) == 2, f"{st}, {len(res)} items")
        st, cmp_ = api("GET", "/api/v1/analytics/comparison")
        check("analytics comparison endpoint", st == 200 and cmp_["total_rounds"] >= 1, str(cmp_.get("total_rounds")))
        page.wait_for_timeout(12000)  # let the next round close and settle the API bid
        st, metrics = api("GET", "/api/v1/dashboard/metrics")
        row = next((a for a in metrics["agents"] if a["name"] == name), {})
        check(
            "API bid settled in a closed round",
            row.get("total_bids", 0) >= 1 and row.get("wins", 0) + row.get("losses", 0) >= 1,
            f"bids={row.get('total_bids')} wins={row.get('wins')} spent=${row.get('total_spent')}",
        )
        agents_text = page.locator("#agent-performance-table-body").inner_text()
        check("new agent appears on dashboard leaderboard", name in agents_text)

        if args.screenshots:
            SHOTS.mkdir(parents=True, exist_ok=True)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(800)
            page.screenshot(path=str(SHOTS / "live-dashboard-overview.jpg"), quality=82)
            for sel, fname in [
                ("#resourceUtilChart", "live-market-charts.jpg"),
                ("#agent-performance-table-body", "live-agents-console.jpg"),
                ("#bid-ticker-body", "live-ticker-history.jpg"),
            ]:
                page.locator(sel).scroll_into_view_if_needed()
                page.wait_for_timeout(800)
                page.screenshot(path=str(SHOTS / fname), quality=82)
            check("screenshots written", True, str(SHOTS))

        # 4. Stop simulation
        with page.expect_response(lambda r: "/simulation/stop" in r.url):
            page.click("#btn-stop-sim")
        page.wait_for_function(
            "!document.querySelector('#status-text').innerText.toLowerCase().includes('online')", timeout=15000
        )
        check("stop simulation", True, text(page, "#status-text"))

        # 5. Optional: live Groq agents
        if args.groq:
            page.click("#btn-start-groq")
            page.wait_for_function(
                "document.querySelector('#status-text').innerText.toLowerCase().includes('online')", timeout=15000
            )
            page.wait_for_timeout(45000)
            calls = int(text(page, "#metric-llm-calls").replace(",", "") or 0)
            check(
                "Groq autonomous mode makes LLM calls",
                calls > 0,
                f"{calls} calls, model: {text(page, '#nav-llm-status')}",
            )
            page.click("#btn-stop-sim")
            page.wait_for_timeout(2000)

        # 6. API docs
        page.goto(BASE + "/docs", wait_until="networkidle")
        page.wait_for_selector(".swagger-ui .opblock", timeout=20000)
        ops = page.locator(".opblock").count()
        check("Swagger UI renders endpoints", ops >= 8, f"{ops} operations")

        real_errors = list(console_errors)
        check("no console errors", not real_errors, "; ".join(real_errors[:3]))
        real_failed = [
            f for f in failed if "favicon" not in f and "/api/v1/agents/me" not in f and "bids/submit" not in f
        ]
        check("no failed network requests", not real_failed, "; ".join(real_failed[:3]))
        browser.close()

    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
