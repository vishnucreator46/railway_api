import json
import random
import pickle
import os
import wikipedia

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "model.pkl")
VECTORIZER_PATH = os.path.join(script_dir, "vectorizer.pkl")
INTENTS_PATH = os.path.join(script_dir, "intents.json")

# Load model & vectorizer
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model files not found. Run 'train.py' first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Load intents
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)


def get_intent_response(user_msg):
    """Predict intent and return response with confidence"""
    X = vectorizer.transform([user_msg])
    decision_scores = model.decision_function(X)
    confidence = max(decision_scores[0]) if len(decision_scores.shape) > 1 else max(decision_scores)
    tag = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response, confidence

    return None, 0


def get_response(user_msg):
    """Return the most relevant response with fallback to Wikipedia"""
    CONFIDENCE_THRESHOLD = 0.6  # Only use intent if confident

    # 1. Check intents
    response, confidence = get_intent_response(user_msg)
    if response and confidence >= CONFIDENCE_THRESHOLD:
        return response

    # 2. Wikipedia fallback
    try:
        wiki_summary = wikipedia.summary(user_msg, sentences=2)
        return wiki_summary
    except wikipedia.exceptions.DisambiguationError as e:
        options = ", ".join(e.options[:5])
        return f"Your query is ambiguous. Did you mean: {options}?"
    except wikipedia.exceptions.PageError:
        return "I couldn't find information on that. Could you rephrase?"
    except Exception:
        return "I'm not sure about that, but I can learn if you teach me!"
