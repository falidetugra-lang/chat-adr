from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Cliente OpenAI con API Key desde variable de entorno
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/api/query", methods=["GET"])
def query():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Falta el parámetro 'q'"}), 400

    try:
        # Llamada a la API de OpenAI
        response = client.responses.create(
            model="gpt-5",   # o "gpt-4.1" si no tienes acceso a GPT-5
            input=q
        )

        answer = response.output[0].content[0].text
        return jsonify({"result": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "✅ API de Chat ADR funcionando"


if __name__ == "__main__":
    # Solo para pruebas locales
    app.run(host="0.0.0.0", port=5000, debug=True)

