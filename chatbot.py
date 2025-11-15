def get_response(user_msg):
    """Return response from intents first, then fallback to Wikipedia only if no intent matches"""
    
    # 1. Check intents (ignore confidence threshold)
    response, _ = get_intent_response(user_msg)
    if response:
        return response  # Always return intent response if matched

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
