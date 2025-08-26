# -*- coding: utf-8 -*-
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import faiss

# ==== Rutas relativas al repo (Railway) ====
BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "index.faiss"
META_PATH  = BASE_DIR / "metadata.json"
TEXTS_PATH = BASE_DIR / "texts.json"   # opcional

app = FastAPI(title="ADR Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restringe si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Estado global (carga perezosa) ====
index = None
metas = None
texts = None
model = None

def lazy_init():
    """Carga índice FAISS, metadatos y el modelo sólo la primera vez."""
    global index, metas, texts, model
    if index is None:
        index = faiss.read_index(str(FAISS_PATH))

    if metas is None:
        with open(META_PATH, "r", encoding="utf-8") as f:
            metas = json.load(f)["metadatas"]

    if texts is None:
        if TEXTS_PATH.exists():
            with open(TEXTS_PATH, "r", encoding="utf-8") as f:
                _t = json.load(f)
            texts = _t if isinstance(_t, list) else [""] * index.ntotal
        else:
            texts = [""] * index.ntotal

    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

@app.get("/health")
def health():
    try:
        lazy_init()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 3, preview_chars: int = 500):
    lazy_init()

    # Embedding
    q_emb = model.encode(
        [q],
        convert_to_numpy=True,
        device="cpu",
        normalize_embeddings=True
    ).astype("float32")

    # Búsqueda
    D, I = index.search(q_emb, k)

    out = []
    for rank, idx in enumerate(I[0]):
        if idx < 0:
            continue
        m = metas[idx]
        frag = texts[idx] if idx < len(texts) else ""
        if preview_chars and len(frag) > preview_chars:
            frag = frag[:preview_chars] + "..."
        out.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "source": Path(m.get("source","")).name,
            "page_from": m.get("page_from"),
            "page_to": m.get("page_to"),
            "chunk_index": m.get("chunk"),
            "text": frag
        })

    return {"query": q, "k": k, "results": out}





