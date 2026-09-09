"""
=============================================================
 COMPUTE BAZAAR — FastAPI Main Server Application (Weeks 3-8)
=============================================================
 Orchestrates:
   - FastAPI REST API + WebSocket routing
   - Database schema initialization
   - Demo resources pre-seeding
   - Background tasks lifecycle: Auction Scheduler Loop
   - Static files serving for Dashboard UI
=============================================================
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from loguru import logger

from backend.app.db.models import init_db, SessionLocal, Resource, Agent, Auction
from backend.app.api.endpoints import router as api_router
from backend.app.api.websocket import manager
from backend.app.auction.scheduler import start_scheduler, stop_scheduler


# ── Download Swagger and Chart.js assets locally for offline use ──────
def download_static_assets_if_missing():
    """Download Swagger UI and Chart.js assets to serve locally in a background thread to prevent blocking server boot."""
    import threading
    
    def _download():
        import urllib.request
        
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(static_dir, exist_ok=True)
        
        assets = {
            "swagger-ui-bundle.js": "https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
            "swagger-ui.css": "https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
            "chart.umd.js": "https://unpkg.com/chart.js@4/dist/chart.umd.js"
        }
        
        for filename, url in assets.items():
            filepath = os.path.join(static_dir, filename)
            if not os.path.exists(filepath):
                logger.info(f"Downloading {filename} from {url} to serve locally...")
                try:
                    req = urllib.request.Request(
                        url, 
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    with urllib.request.urlopen(req, timeout=15) as response:
                        with open(filepath, "wb") as f:
                            f.write(response.read())
                    logger.info(f"Successfully downloaded {filename}")
                except Exception as e:
                    logger.warning(f"Could not download {filename} from CDN ({e}). Will fall back to remote CDN urls.")
                    
    threading.Thread(target=_download, daemon=True).start()


# ── Seed Default Resources & Agents on Startup ───────────────
def seed_resources_if_empty():
    """Seed base resources, simulated agents, and initial round if database is fresh."""
    db = SessionLocal()
    try:
        # Seed Resources
        r_count = db.query(Resource).count()
        if r_count == 0:
            logger.info("Database is empty. Pre-seeding default compute resources...")
            resources = [
                Resource(name="GPU-Slot-1",     resource_type="gpu",        capacity=1.0,  unit="slot",      cost_per_unit=2.50),
                Resource(name="GPU-Slot-2",     resource_type="gpu",        capacity=1.0,  unit="slot",      cost_per_unit=2.50),
                Resource(name="API-Budget-100",  resource_type="api_budget", capacity=100.0,unit="dollars",   cost_per_unit=1.00),
                Resource(name="RAM-Block-32GB",  resource_type="ram",        capacity=32.0, unit="gb",        cost_per_unit=0.10),
                Resource(name="CPU-Cores-8",    resource_type="cpu",        capacity=8.0,  unit="cores",     cost_per_unit=0.05),
            ]
            db.bulk_save_objects(resources)
            db.commit()
            logger.info(f"Seeded {len(resources)} default compute resources.")

        # Seed Agents
        a_count = db.query(Agent).count()
        if a_count == 0:
            logger.info("Pre-seeding default autonomous AI agents...")
            agents = [
                Agent(name="DevBot-Gold",       description="Autonomous Coding Agent",       api_key="cb_gold_01",   budget_total=1000.0, budget_remaining=850.0, tier="gold"),
                Agent(name="ResearchBot-Gold",   description="Autonomous Research Agent",     api_key="cb_gold_02",   budget_total=1000.0, budget_remaining=820.0, tier="gold"),
                Agent(name="ETLBot-Silver",      description="Autonomous Data ETL Agent",     api_key="cb_silver_01", budget_total=600.0,  budget_remaining=480.0, tier="silver"),
                Agent(name="QuantBot-Bronze",    description="Autonomous Quantitative Agent", api_key="cb_bronze_01",budget_total=400.0,  budget_remaining=350.0, tier="bronze"),
            ]
            db.bulk_save_objects(agents)
            db.commit()
            logger.info(f"Seeded {len(agents)} default AI agents.")

        # Seed Open Auction Round if none exists
        auc_count = db.query(Auction).count()
        if auc_count == 0:
            logger.info("Pre-seeding initial auction round...")
            auction = Auction(round_number=1, status="open")
            db.add(auction)
            db.commit()

    except Exception as e:
        logger.error(f"Error seeding default resources and agents: {e}")
    finally:
        db.close()


# ── Context Manager: Startup / Shutdown Events ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup Actions ──
    logger.info("Starting up Compute Bazaar server...")
    
    # 0. Download static assets for offline/local usage
    download_static_assets_if_missing()
    
    # 1. Initialize SQLite Database Schema
    init_db()
    
    # 2. Seed resource items & baseline agents
    seed_resources_if_empty()
    
    # 3. Start Periodic Auction Loop background task
    start_scheduler()
    
    yield
    
    # ── Shutdown Actions ──
    logger.info("Shutting down Compute Bazaar server...")
    
    # 1. Stop Periodic Auction Loop background task
    stop_scheduler()


# ── App Initialization ───────────────────────────────────────
app = FastAPI(
    title="Compute Bazaar",
    description="Real-Time Multi-Agent Combinatorial Auction Market",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints router
app.include_router(api_router, prefix="/api/v1")


# ── WebSockets Router: Real-time broadcast feed ──────────────
@app.websocket("/ws")
async def websocket_ws(websocket: WebSocket):
    """Real-time updates pushed directly to dashboard."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


@app.websocket("/ws/auction-feed")
async def websocket_feed(websocket: WebSocket):
    """Real-time updates pushed directly to dashboard."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


# ── Serve Dashboard Client ───────────────────────────────────

# Mount static files (/static/app.js)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount index HTML at root path
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h3>Compute Bazaar Dashboard UI files not found.</h3>", status_code=404)


# Custom Swagger UI route utilizing local static assets or fallback CDNs
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    swagger_js = "/static/swagger-ui-bundle.js" if os.path.exists(os.path.join(STATIC_DIR, "swagger-ui-bundle.js")) else "https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"
    swagger_css = "/static/swagger-ui.css" if os.path.exists(os.path.join(STATIC_DIR, "swagger-ui.css")) else "https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"
    
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url=swagger_js,
        swagger_css_url=swagger_css,
    )


# Swagger API docs custom redirect
@app.get("/docs-redirect")
def docs_redirect():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Run Main Server block ────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
