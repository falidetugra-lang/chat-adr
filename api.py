from fastapi.responses import JSONResponse
import traceback

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
            "Eres un asistente experto en ADR, RD 97/2014 y LOTT 9/2013. "
            "Responde solo con la información del CONTEXTO. Si no está en el contexto, dilo claramente. "
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
















