from flask import Flask, request
from chatbot import get_response

app = Flask(__name__)

@app.get("/")
def home():
    return "ViloFury API Running!"


@app.post("/chat")
def chat():
    data = request.get_json()

    # Validate incoming request
    if not data or "message" not in data:
        return {"error": "No message provided"}, 400

    message = data["message"].strip()
    if not message:
        return {"response": "Please send a valid message."}

    # Get chatbot response
    try:
        reply = get_response(message)
        return {"response": reply}
    except Exception as e:
        # Print server-side error for debugging
        print("Error in /chat:", e)
        return {"error": "Internal server error"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
