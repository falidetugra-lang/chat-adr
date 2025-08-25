import os
from flask import Flask, send_from_directory, request, jsonify
from openai import OpenAI

# App estática: servimos /public
app = Flask(__name__, static_folder="public", static_url_path="/")

# Cliente OpenAI (usa tu OPENAI_API_KEY de Render)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Home -> index.html
@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

# Healthcheck
@app.get("/api/ping")
def api_ping():
    return jsonify(ok=True)

# Lógica de respuesta del chat (reutilizable)
def _chat_reply(user_message: str) -> str:
    if not user_message:
        return "Escribe una pregunta."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente sobre normativa ADR. Responde breve y claro."
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error consultando el modelo: {e}"

# Endpoint principal de chat
@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    answer = _chat_reply(message)
    return jsonify(answer=answer)

# Alias compatible con el front antiguo (/api/query)
@app.post("/api/query")
def api_query():
    return chat()

# Servir estáticos / SPA fallback
@app.get("/<path:path>")
def static_proxy(path):
    fp = os.path.join(app.static_folder, path)
    if os.path.exists(fp):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

