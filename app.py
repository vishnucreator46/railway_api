from flask import Flask, request
from chatbot import get_response

app = Flask(__name__)

@app.get("/")
def home():
    return "ViloFury API Running!"

@app.post("/chat")
def chat():
    data = request.get_json()
    msg = data.get("message")
    reply = get_response(msg)
    return {"response": reply}

if __name__ == "__main__":
    app.run()
