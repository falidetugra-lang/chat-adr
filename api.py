# -*- coding: utf-8 -*-
# Requisitos (una vez):
#   python -m pip install fastapi "uvicorn[standard]"

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss, json
from pathlib import Path

# ==== Config ====
BASE_DIR  = Path(r"C:\Users\pc\Documents\Personal\Marca personal\chat_ADR")
INDEX_DIR = BASE_DIR / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"

FAISS_PATH = INDEX_DIR / "index.faiss"
META_PATH  = INDEX_DIR / "metadata.json"
TEXTS_PATH = INDEX_DIR / "texts.json"

# ==== App ====
app = FastAPI(title="ADR Search API", version="1.0.0")

# Habilitar CORS (permite que cualquier web haga consultas a la API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # puedes restringir a ["https://tu-dominio.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Carga de índice y modelo ====
index = faiss.read_index(str(FAISS_PATH))
with open(META_PATH, "r", encoding="utf-8") as f:
    metas = json.load(f)["metadatas"]
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    texts = json.load(f)

model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# ==== Endpoints ====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 3, preview_chars: int = 500):
    # Vectorizar consulta
    q_emb = model.encode([q], convert_to_numpy=True, device=DEVICE, normalize_embeddings=True).astype("float32")
    # Buscar en FAISS
    D, I = index.search(q_emb, k)

    results = []
    for rank, idx in enumerate(I[0]):
        if idx < 0:
            continue
        m = metas[idx]
        frag = texts[idx] if idx < len(texts) else ""
        if preview_chars and len(frag) > preview_chars:
            frag_show = frag[:preview_chars] + "..."
        else:
            frag_show = frag
        results.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "source": Path(m.get("source","")).name,
            "page_from": m.get("page_from"),
            "page_to": m.get("page_to"),
            "chunk_index": m.get("chunk"),
            "text": frag_show
        })
    return {"query": q, "k": k, "results": results}
