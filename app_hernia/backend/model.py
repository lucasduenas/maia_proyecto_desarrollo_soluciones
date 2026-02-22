import random

def predict_hernia(image_bytes: bytes):
    """
    Placeholder for ML inference.
    """
    has_hernia = random.choice([True, False])

    return {
        "has_hernia": has_hernia,
        "confidence": round(random.uniform(0.75, 0.95), 2),
        "type": "Tiene hernia" if has_hernia else "No tiene hernia"
    }