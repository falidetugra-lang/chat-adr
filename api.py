# -*- coding: utf-8 -*-
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import os, json, faiss, traceback
from fastapi.staticfiles import StaticFiles

# ===== BASE =====
BASE_DIR   = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "index.faiss"
META_PATH  = BASE_DIR / "metadata.json"
TEXTS_PATH = BASE_DIR / "texts.json"

# ===== APP =====
app = FastAPI(title="ADR Search API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (ej: chat.html dentro de /public)
if (BASE_DIR / "public").exists():
    app.mount("/web", StaticFiles(directory=str(BASE_DIR / "public")), name="web")

# ===== ESTADO GLOBAL =====
index = None
metas  = None
texts  = None
embed_model = None
gen_client  = None


def lazy_init():
    """Carga FAISS, metadatos, textos y cliente OpenAI una sola vez."""
    global index, metas, texts, gen_client

    if index is None:
        if not FAISS_PATH.exists():
            raise RuntimeError("No se encuentra index.faiss. Genera el índice con ingest.py")
        index = faiss.read_index(str(FAISS_PATH))

    if metas is None:
        if not META_PATH.exists():
            raise RuntimeError("No se encuentra metadata.json")
        with open(META_PATH, "r", encoding="utf-8") as f:
            metas = json.load(f)["metadatas"]

    if texts is None:
        if not TEXTS_PATH.exists():
            raise RuntimeError("No se encuentra texts.json")
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            texts = json.load(f)
        if not isinstance(texts, list):
            raise RuntimeError("texts.json debe ser una lista de strings")

    if gen_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta variable OPENAI_API_KEY en Railway")
        gen_client = OpenAI(api_key=api_key)


# ===== ENDPOINTS =====
@app.get("/health")
def health():
    try:
        lazy_init()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = 3, preview_chars: int = 500):
    try:
        lazy_init()
        from sentence_transformers import SentenceTransformer
        global embed_model
        if embed_model is None:
            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        q_emb = embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(q_emb, k)

        results = []
        for rank, idx in enumerate(I[0]):
            if idx < 0: 
                continue
            m = metas[idx]
            frag = texts[idx] if idx < len(texts) else ""
            if preview_chars and len(frag) > preview_chars:
                frag = frag[:preview_chars] + "..."
            results.append({
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "source": Path(m.get("source", "")).name,
                "page_from": m.get("page_from"),
                "page_to": m.get("page_to"),
                "chunk_index": m.get("chunk"),
                "text": frag
            })
        return {"query": q, "k": k, "results": results}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/chat")
def chat(q: str = Query(..., min_length=2), k: int = 5, max_ctx_chars: int = 12000):
    try:
        lazy_init()
        from sentence_transformers import SentenceTransformer
        global embed_model
        if embed_model is None:
            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        q_emb = embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(q_emb, k)

        ctx_parts, sources, total = [], [], 0
        for rank, idx in enumerate(I[0]):
            if idx < 0: 
                continue
            m = metas[idx]
            frag = texts[idx] if idx < len(texts) else ""
            src = {
                "rank": rank + 1,
                "source": Path(m.get("source","")).name,
                "page_from": m.get("page_from"),
                "page_to": m.get("page_to"),
                "chunk": m.get("chunk")
            }
            part = f"[{rank+1}] {frag}"
            if total + len(part) > max_ctx_chars:
                break
            ctx_parts.append(part)
            sources.append(src)
            total += len(part)

        context = "\n\n".join(ctx_parts) if ctx_parts else "No hay fragmentos recuperados."

        system_msg = (
            "Eres un asistente experto en ADR 2025, RD 97/2014 y LOTT 9/2013. "
            "Responde solo con la información del CONTEXTO. "
            "Si no está en el contexto, responde que no consta en los documentos. "
            "Sé breve y añade al final una sección 'Fuentes' citando [número] y archivo/páginas."
        )
        user_msg = f"Pregunta: {q}\n\nCONTEXTO:\n{context}"

        resp = gen_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
        )
        answer = resp.choices[0].message.content
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})




















