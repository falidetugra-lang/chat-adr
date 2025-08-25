from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Dict, Any

import json
import numpy as np

# Embeddings locales (sin API): all-MiniLM-L6-v2
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

app = FastAPI()

# ---- Rutas de archivos ----
ROOT = Path(__file__).parent
PUBLIC = ROOT / "public"
TEXTS_PATH = ROOT / "texts.json"       # lista de strings (trozos)
META_PATH  = ROOT / "metadata.json"    # lista de objetos con info de cada trozo

# ---- Servir frontend ----
app.mount("/static", StaticFiles(directory=PUBLIC), name="static")

@app.get("/")
def home():
    return FileResponse(PUBLIC / "index.html")

# ---- Carga de datos/embeddings al iniciar ----
MODEL: SentenceTransformer | None = None
DOCS: List[str] = []
META: List[Dict[str, Any]] = []
EMBEDS: np.ndarray | None = None

def _load_corpus():
    global DOCS, META
    if TEXTS_PATH.exists():
        DOCS = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
    else:
        DOCS = []
    if META_PATH.exists():
        META = json.loads(META_PATH.read_text(encoding="utf-8"))
    else:
        META = [{} for _ in DOCS]
    # Si hay desajuste de longitudes, igualamos
    if len(META) != len(DOCS):
        META = (META + [{}] * len(DOCS))[:len(DOCS)]

def _ensure_embeddings():
    """
    Calcula embeddings de los DOCS si no están en memoria.
    (Para datasets pequeños esto es suficiente; Render free aguanta bien.)
    """
    global MODEL, EMBEDS
    if MODEL is None:
        MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if EMBEDS is None:
        if DOCS:
            EMBEDS = np.array(MODEL.encode(DOCS, normalize_embeddings=True))
        else:
            EMBEDS = np.zeros((0, 384), dtype="float32")  # tamaño de MiniLM

def _search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Devuelve los k fragmentos más similares por coseno.
    """
    if not query.strip() or EMBEDS is None or EMBEDS.shape[0] == 0:
        return []
    qv = MODEL.encode([query], normalize_embeddings=True)[0]
    sims = EMBEDS @ qv  # coseno al estar normalizados
    idx = np.argsort(-sims)[:k]
    out: List[Dict[str, Any]] = []
    for rank, i in enumerate(idx, start=1):
        score = float(sims[i])
        item = {
            "rank": rank,
            "score": score,
            "text": DOCS[i],
            "source": META[i].get("source") or META[i].get("file") or "",
            "page_from": META[i].get("page_from") or META[i].get("page") or None,
            "page_to": META[i].get("page_to") or None,
        }
        out.append(item)
    return out

@app.on_event("startup")
def _startup():
    _load_corpus()
    _ensure_embeddings()

@app.get("/api/ping")
def ping():
    return {"ok": True, "docs": len(DOCS)}

@app.get("/api/query")
def query(q: str = Query(..., min_length=2), k: int = 5):
    """
    Endpoint que usa tu frontend. Devuelve un texto 'respuesta' y
    además adjunta las fuentes por si quieres mostrarlas.
    """
    _ensure_embeddings()
    hits = _search(q, k=k)
    if not hits:
        return {"result": "No encontré nada que encaje con la consulta."}

    # Respuesta simple: concatenamos los top-3 trozos como borrador
    top_text = "\n\n".join(h["text"] for h in hits[:3])
    respuesta = f"Resultados más relevantes para «{q}»:\n\n{top_text}"

    return {
        "result": respuesta,
        "sources": hits   # por si luego quieres mostrarlas en la UI
    }

# Fallback SPA
@app.get("/{path:path}")
def spa(path: str):
    if path.startswith("api/"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(PUBLIC / "index.html")
