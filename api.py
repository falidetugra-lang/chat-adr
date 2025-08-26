# -*- coding: utf-8 -*-
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os, json, zipfile
import faiss

# ===== RUTAS =====
BASE_DIR   = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "index.faiss"
ZIP_PATH   = BASE_DIR / "index.zip"
META_PATH  = BASE_DIR / "metadata.json"
TEXTS_PATH = BASE_DIR / "texts.json"   # debe existir para /chat

# ===== APP =====
app = FastAPI(title="ADR Search API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ESTADO GLOBAL =====
index = None
metas  = None
texts  = None
embed_model = None  # solo para buscar (ya lo tienes creado al generar el índice)
gen_client  = None  # cliente OpenAI para generar respuesta

def ensure_index_file():
    if not FAISS_PATH.exists() and ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(BASE_DIR)

def lazy_init():
    """Carga FAISS, metadatos, textos y cliente OpenAI (una sola vez)."""
    global index, metas, texts, gen_client
    if index is None:
        ensure_index_file()
        index = faiss.read_index(str(FAISS_PATH))

    if metas is None:
        with open(META_PATH, "r", encoding="utf-8") as f:
            metas = json.load(f)["metadatas"]

    if texts is None:
        if not TEXTS_PATH.exists():
            raise RuntimeError("Falta texts.json (necesario para /chat)")
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            t = json.load(f)
        # debe ser lista de strings con los fragmentos
        if not isinstance(t, list):
            raise RuntimeError("texts.json debe ser una lista de strings (fragmentos)")
        texts = t

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
    lazy_init()

    # Recuperación: usamos el índice FAISS ya creado (MiniLM)
    # Para consultar FAISS necesitamos el embedding; como el índice se generó con
    # sentence-transformers MiniLM, aquí calculamos el embedding con ese modelo.
    # Para no descargarlo ahora (y no romper despliegue), pedimos a FAISS que
    # busque con consulta nula (no se puede). Así que mejor cargamos el modelo mínimo.
    # -> Usaremos sentence-transformers solo si hace falta. Si prefieres sin modelo,
    #    deja este endpoint como está y usa solo /chat (que recupera abajo).
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

@app.get("/chat")
def chat(q: str = Query(..., min_length=2), k: int = 5, max_ctx_chars: int = 12000):
    """
    Chat RAG: recupera k fragmentos del índice y genera respuesta con OpenAI,
    citando las fuentes (archivo/páginas) de tus PDFs.
    """
    lazy_init()

    # 1) Recuperación con el mismo modelo del índice
    from sentence_transformers import SentenceTransformer
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    q_emb = embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k)

    ctx_parts, sources = [], []
    total = 0
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

    # 2) Generación con OpenAI (modelo económico y solvente)
    system_msg = (
        "Eres un asistente experto en ADR, RD 97/2014 y LOTT 9/2013. "
        "Responde solo con la información proporcionada en el CONTEXTO. "
        "Si no está en el contexto, responde claramente que no consta en los documentos. "
        "Sé breve, preciso y añade, al final, una sección 'Fuentes' citando [número] y archivo/páginas."
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











