from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessing import clean_text
import pickle

model = load_model("model/sentiment_model.keras")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    return ['Negative', 'Neutral', 'Positive'][pred.argmax()]

if __name__ == "__main__":
    text = input("Enter tweet: ")
    print("Sentiment:", predict_sentiment(text))
