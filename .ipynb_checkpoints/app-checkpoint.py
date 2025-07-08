from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os

# Paths relative to where app.py is run
MODEL_PATH = os.path.join("notebook", "model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join("notebook", "model", "tokenizer.pkl")
MAX_LENGTH = 100

# Load model and tokenizer
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    """Basic text cleaning to match model training preprocessing"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
    pred = model.predict(padded)[0][0]

    confidence_bar = "🟩" * int(pred * 10) + "⬜" * (10 - int(pred * 10))

    if pred > 0.5:
        sentiment = f"Positive sentiment (confidence: {pred:.3f})"
    else:
        sentiment = f"Negative sentiment (confidence: {1 - pred:.3f})"

    return sentiment, confidence_bar, pred

if __name__ == "__main__":
    text = input("Enter tweet: ")
    sentiment, bar, conf = predict_sentiment(text)
    print("Sentiment:", sentiment)
    print(f"Confidence: {bar} ({conf:.2f})")
