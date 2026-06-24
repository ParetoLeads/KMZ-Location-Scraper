from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from server.routes.analyze import router as analyze_router
from server.routes.health import router as health_router

app = FastAPI(title="KMZ Location Scraper")
app.include_router(health_router)
app.include_router(analyze_router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
