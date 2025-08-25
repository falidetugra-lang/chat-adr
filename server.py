from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

PUBLIC = Path(__file__).parent / "public"
app.mount("/static", StaticFiles(directory=PUBLIC), name="static")

@app.get("/")
def home():
    return FileResponse(PUBLIC / "index.html")

@app.get("/api/ping")
def ping():
    return {"ok": True}

@app.get("/api/query")
def query(q: str):
    # Respuesta de prueba: luego metemos tu lógica real
    return {"result": f"Buscaste: {q}"}

@app.get("/{path:path}")
def spa(path: str):
    if path.startswith("api/"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(PUBLIC / "index.html")
