import json
import random
import pickle
import os
import wikipedia

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(script_dir, "model.pkl")
VECTORIZER_PATH = os.path.join(script_dir, "vectorizer.pkl")
INTENTS_PATH = os.path.join(script_dir, "intents.json")

# --- Load model & vectorizer ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model files not found. Run 'train.py' first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# --- Load intents ---
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)


def get_intent_response(user_msg):
    """Return predicted response and confidence from intents.json"""
    X = vectorizer.transform([user_msg])
    decision_scores = model.decision_function(X)
    
    # Handle multiclass shape
    if len(decision_scores.shape) == 1:
        confidence = max(decision_scores)
    else:
        confidence = max(decision_scores[0])
    
    tag = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response, confidence

    return None, 0


def get_response(user_msg):
    """Main response function with fallback to Wikipedia"""
    CONFIDENCE_THRESHOLD = 0.5

    # 1. Try intent-based response
    intent_response, confidence = get_intent_response(user_msg)
    if intent_response and confidence > CONFIDENCE_THRESHOLD:
        return intent_response

    # 2. Wikipedia fallback
    try:
        wiki_summary = wikipedia.summary(user_msg, sentences=2)
        return wiki_summary
    except Exception:
        pass

    # 3. Final fallback
    return intent_response or "I'm not sure about that, but I can learn if you teach me!"
