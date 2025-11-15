from flask import Flask, request
from chatbot import get_response
import os

app = Flask(__name__)

@app.get("/")
def home():
    return "ViloFury API Running!"

@app.post("/chat")
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return {"error": "No message provided"}, 400

    message = data["message"].strip()
    if not message:
        return {"response": "Please send a valid message."}

    try:
        reply = get_response(message)
        return {"response": reply}
    except Exception as e:
        print("Error in /chat:", e)
        return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway dynamic port
    app.run(host="0.0.0.0", port=port)
