from flask import Flask, jsonify, request
from openai import OpenAI
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# -------- Config --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY en Render.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Modelo de embeddings LOCAL (debe coincidir con el usado al crear el index)
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# RAG params
TOP_K = 5
SIM_THRESHOLD = 0.30        # sube/baja este valor según tus datos
MAX_CONTEXT_CHARS = 3500    # limita el tamaño de contexto que enviamos al modelo

# Rutas de tus ficheros
FAISS_PATH = "index.faiss"
TEXTS_PATH = "texts.json"

# -------- Carga del índice y textos --------
embedder = SentenceTransformer(EMB_MODEL_NAME)

if not (os.path.exists(FAISS_PATH) and os.path.exists(TEXTS_PATH)):
    raise RuntimeError("No encuentro index.faiss y/o texts.json en la raíz del proyecto.")

index = faiss.read_index(FAISS_PATH)
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)  # debe ser lista/array con los textos alineados con el índice


def embed_text(text: str) -> np.ndarray:
    vec = embedder.encode([text], normalize_embeddings=True)  # shape (1, d)
    return vec.astype("float32")


def retrieve(query: str, k: int = TOP_K, threshold: float = SIM_THRESHOLD):
    q_vec = embed_text(query)  # (1, d)
    D, I = index.search(q_vec, k)  # distancias y posiciones
    D = D[0]
    I = I[0]

    # FAISS con vectores normalizados + distancia L2 ≈ 2 - 2*similitud
    # Si tus vectores están normalizados y el índice es L2, podemos recuperar similitud aproximada:
    # sim = 1 - (dist/2)
    results = []
    for dist, idx in zip(D, I):
        if idx < 0 or idx >= len(docs):
            continue
        sim = 1.0 - (float(dist) / 2.0)
        if sim >= threshold:
            results.append({"text": docs[idx], "score": sim})

    return results


SYSTEM_PROMPT = (
    "Eres un asistente de consultas ADR. Debes responder EXCLUSIVAMENTE con la información "
    "incluida en el CONTEXTO proporcionado. Si la respuesta no está en el CONTEXTO, responde "
    "literalmente: 'Lo siento, no encuentro esa información en la base ADR'. No inventes datos, "
    "no uses conocimientos externos."
)


def build_prompt(context_chunks, question):
    context_text = "\n\n".join([c["text"] for c in context_chunks])[:MAX_CONTEXT_CHARS]
    prompt = (
        f"CONTEXTO:\n{context_text}\n\n"
        f"PREGUNTA: {question}\n\n"
        "Instrucciones: Responde en español de forma breve y clara usando solo el CONTEXTO. "
        "Si no hay información suficiente en el CONTEXTO, responde: "
        "'Lo siento, no encuentro esa información en la base ADR'."
    )
    return prompt


@app.get("/api/ping")
def ping():
    return jsonify({"ok": True})


@app.get("/api/query")
def query():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Falta parámetro q"}), 400

    # 1) Recuperación
    hits = retrieve(q, k=TOP_K, threshold=SIM_THRESHOLD)

    if not hits:
        # No hay contexto relevante. NO llamamos al modelo.
        return jsonify({
            "answer": "Lo siento, no encuentro esa información en la base ADR",
            "sources": [],
            "notes": "sin_contexto"
        })

    # 2) Armamos prompt con CONTEXTO
    prompt = build_prompt(hits, q)

    # 3) Llamada al modelo: solo puede usar el CONTEXTO
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=300
        )
        text = resp.output_text.strip()

        # Guardarraíl adicional: si el modelo intenta salirse
        if "no encuentro esa información" in text.lower():
            text = "Lo siento, no encuentro esa información en la base ADR"

        return jsonify({
            "answer": text,
            "sources": hits,   # aquí podrías devolver solo páginas/ids si las tuvieras
            "notes": "ok"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Render/Heroku
if __name__ == "__main__":
    # Para correr local (opcional):  flask run  /  python server.py
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))


