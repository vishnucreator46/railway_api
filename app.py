from flask import Flask, request
import json
import random
import wikipedia
import pickle
import os

app = Flask(__name__)

# ------------------------------
# Load Model, Vectorizer, Intents
# ------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

INTENTS_PATH = os.path.join(script_dir, "intents.json")
MODEL_PATH = os.path.join(script_dir, "model.pkl")
VECTORIZER_PATH = os.path.join(script_dir, "vectorizer.pkl")

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))


# ------------------------------
# API Routes
# ------------------------------
@app.get("/")
def home():
    return "ViloFury API Running!"


@app.post("/chat")
def chat():
    data = request.get_json()
    message = data.get("message")

    # Convert message
    X = vectorizer.transform([message])
    predicted_tag = model.predict(X)[0]

    # Get confidence (LinearSVC)
    try:
        confidence = max(model.decision_function(X))
    except:
        confidence = 1  # always high fallback

    # If confident enough → return custom JSON response
    if confidence > 0.5:
        for intent in intents["intents"]:
            if intent["tag"] == predicted_tag:
                return {"response": random.choice(intent["responses"])}

    # Otherwise → Wikipedia fallback
    try:
        wiki_summary = wikipedia.summary(message, sentences=2)
        return {"response": wiki_summary}
    except:
        return {"response": "I couldn't find an answer."}


# ------------------------------
# Run Local
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
