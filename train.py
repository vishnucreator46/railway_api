import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Get the directory of the current script to build correct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(script_dir, "intents.json")
MODEL_PATH = os.path.join(script_dir, "model.pkl")
VECTORIZER_PATH = os.path.join(script_dir, "vectorizer.pkl")

# Load intents
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train model
model = LinearSVC()
model.fit(X, tags)

# Save model & vectorizer
pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

print("Training complete! model.pkl and vectorizer.pkl generated.")
