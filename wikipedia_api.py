import wikipedia

def get_wikipedia_summary(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return "I couldn't find an answer on Wikipedia."
