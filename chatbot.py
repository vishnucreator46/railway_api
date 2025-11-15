import json
import random
import pickle
import sys
import os

# Add the parent directory to the Python path to find 'wikipedia_api'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wikipedia_api import get_wikipedia_summary # Using your more robust API script

# Get the directory of the current script to build correct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Define file paths ---
MODEL_PATH = os.path.join(script_dir, "model.pkl")
VECTORIZER_PATH = os.path.join(script_dir, "vectorizer.pkl")
INTENTS_PATH = os.path.join(script_dir, "intents.json")

# --- Check if model files exist before loading ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("❌ Error: Model files not found.")
    print(f"Please run 'train.py' in the '{os.path.basename(script_dir)}' directory first to generate 'model.pkl' and 'vectorizer.pkl'.")
    sys.exit(1)

# Load model + vectorizer
model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# Load intents
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

def get_intent_response(user_msg):
    """
    Returns a response and confidence score from intents.json if matched.
    """
    X = vectorizer.transform([user_msg])
    
    # Get confidence score and predicted tag
    confidence = model.decision_function(X).max()
    tag = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response, confidence

    return None, 0


def get_response(user_msg):
    # A confidence threshold for the intent model. Adjust if needed.
    CONFIDENCE_THRESHOLD = 0.5 

    # Step 1: Try intents.json
    intent_response, confidence = get_intent_response(user_msg)

    # Step 2: If confidence is high, use the intent response
    if intent_response and confidence > CONFIDENCE_THRESHOLD:
        return intent_response

    # Step 3: If confidence is low, try Wikipedia as a fallback
    if intent_response is None or confidence <= CONFIDENCE_THRESHOLD:
        wiki_answer = get_wikipedia_summary(user_msg)
        if wiki_answer:
            return wiki_answer

    # Step 4: If Wikipedia also fails, use the low-confidence intent response or a final fallback message
    return intent_response or "I'm not sure about that, but I can learn if you teach me!"
