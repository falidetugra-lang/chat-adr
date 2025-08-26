# -*- coding: utf-8 -*-
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import os
import json
import faiss
import traceback

# ====== Rutas de ficheros ======
BASE_DIR   = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "index.faiss"
META_PATH  = BASE_DIR / "metadata.json"
TEXTS_PATH = BASE_DIR / "texts.json"

# ====== App ======
app = FastAPI(title="ADR Search API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restringe a tu dominio si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sirve chat.html si lo tienes en /public
if (BASE_DIR / "public").exists():
    app.mount("/web", StaticFiles(directory=str(BASE_DIR / "public")), name="web")

# ====== Estado global (cache) ======
index = None
metas = None
texts = None
embed_model = None       # SentenceTransformer para recuperar en FAISS
reranker = None          # CrossEncoder para reordenar candidatos
gen_client = None        # OpenAI para generar la respuesta


def lazy_init():
    """
    Carga índice FAISS, metadatos, textos y cliente OpenAI una sola vez.
    """
    global index, metas, texts, gen_client

    # Índice FAISS
    if index is None:
        if not FAISS_PATH.exists():
            raise RuntimeError("No se encuentra index.faiss. Genera el índice con ingest.py")
        index = faiss.read_index(str(FAISS_PATH))

    # Metadatos
    if metas is None:
        if not META_PATH.exists():
            raise RuntimeError("No se encuentra metadata.json")
        with open(META_PATH, "r", encoding="utf-8") as f:
            metas = json.load(f).get("metadatas", [])

    # Textos (lista de strings)
    if texts is None:
        if not TEXTS_PATH.exists():
            raise RuntimeError("No se encuentra texts.json")
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            t = json.load(f)
        if not isinstance(t, list):
            raise RuntimeError("texts.json debe ser una lista de strings")
        texts = t

    # Cliente OpenAI
    if gen_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta variable OPENAI_API_KEY en Railway")
        gen_client = OpenAI(api_key=api_key)


# ====== Endpoints ======
@app.get("/health")
def health():
    try:
        lazy_init()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/search")
def search(q: str = Query(..., min_length=2),
           k: int = 3,
           preview_chars: int = 500):
    """
    Búsqueda simple (sin generación). Devuelve trozos recuperados.
    """
    try:
        lazy_init()
        from sentence_transformers import SentenceTransformer
        global embed_model
        if embed_model is None:
            embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
            )

        q_emb = embed_model.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

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
def chat(q: str = Query(..., min_length=2),
         k_faiss: int = 20,           # pedimos bastantes candidatos a FAISS
         top_n: int = 5,              # nos quedamos con pocos tras reranking
         max_ctx_chars: int = 12000,  # límite de contexto a enviar al modelo
         min_rerank_score: float = 0.25  # umbral mínimo de relevancia
         ):
    """
    Chat RAG: recupera de FAISS, reranquea con CrossEncoder, genera respuesta con OpenAI.
    """
    try:
        lazy_init()

        # 1) Modelo de embeddings (para FAISS)
        from sentence_transformers import SentenceTransformer, CrossEncoder
        global embed_model, reranker

        if embed_model is None:
            embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
            )

        # 2) Reranker (Cross-Encoder) para ordenar por relevancia real
        if reranker is None:
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 3) Recuperación inicial (K grande)
        q_emb = embed_model.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        D, I = index.search(q_emb, k_faiss)

        candidates = []
        for rank, idx in enumerate(I[0]):
            if idx < 0:
                continue
            m = metas[idx]
            frag = texts[idx] if idx < len(texts) else ""
            candidates.append({
                "faiss_rank": rank + 1,
                "text": frag,
                "meta": {
                    "source": Path(m.get("source", "")).name,
                    "page_from": m.get("page_from"),
                    "page_to": m.get("page_to"),
                    "chunk": m.get("chunk"),
                }
            })

        if not candidates:
            return {
                "answer": "No encuentro información en los documentos para esta consulta. "
                          "Intenta con otra redacción o sé más específico.",
                "sources": []
            }

        # 4) Reranking
        pairs = [(q, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)  # mayor = más relevante
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        top = candidates[:top_n]

        # 5) Umbral mínimo de confianza
        if top and top[0]["rerank_score"] < min_rerank_score:
            return {
                "answer": ("Según el índice, no hay fragmentos suficientemente relevantes para "
                           "responder con seguridad. Reformula la pregunta (por ejemplo citando "
                           "capítulo/artículo o palabra clave concreta)."),
                "sources": []
            }

        # 6) Construimos el contexto limitado
        ctx_parts, sources, total = [], [], 0
        for i, c in enumerate(top, start=1):
            part = f"[{i}] {c['text']}"
            if total + len(part) > max_ctx_chars:
                break
            ctx_parts.append(part)
            meta = c["meta"]
            sources.append({
                "rank": i,
                "source": meta["source"],
                "page_from": meta["page_from"],
                "page_to": meta["page_to"],
                "chunk": meta["chunk"]
            })
            total += len(part)

        context = "\n\n".join(ctx_parts) if ctx_parts else "No hay fragmentos recuperados."

        # 7) Prompt estricto y formato de salida (CORREGIDO e indentado)
        system_msg = (
            "Eres un asistente experto en normativa de transporte de mercancías peligrosas. "
            "Tu conocimiento proviene únicamente de tres fuentes documentales: "
            "ADR 2025, RD 97/2014 y LOTT 9/2013. "
            "NO debes usar información externa ni inventar. "
            "Si la respuesta no está en el CONTEXTO proporcionado, responde exactamente: "
            "'No consta en los documentos proporcionados.' "
            "Responde siempre de forma concisa, clara y técnica, en español. "
            "Usa un tono profesional, como un consejero de seguridad ADR. "
            "No hagas interpretaciones ni consejos prácticos fuera del texto normativo."
        )

        user_msg = (
            f"Pregunta del usuario: {q}\n\n"
            f"CONTEXTO (fragmentos numerados del índice legal):\n{context}\n\n"
            "Instrucciones de salida:\n"
            "1) Responde de manera breve (máx. 5 líneas) y solo con información del CONTEXTO.\n"
            "2) Si corresponde, incluye 'Fundamento: Artículo/Capítulo/Sección' según aparezca en el texto.\n"
            "3) Añade una sección final 'Fuentes:' citando [n] archivo y páginas.\n"
            "4) Si la información no se encuentra en el CONTEXTO, responde únicamente: "
            "'No consta en los documentos proporcionados.'\n"
        )

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

























