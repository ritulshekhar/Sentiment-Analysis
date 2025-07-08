from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import numpy as np
import re

# Text preprocessing function (matching your training)
def clean_text(text):
    """
    Basic text cleaning to match model training preprocessing:
    - Lowercasing
    - Remove punctuation and non-alphabetic characters
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Load model and tokenizer
try:
    model = load_model("model/sentiment_model.keras")
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

def predict_sentiment_debug(text):
    """
    Debug version to see what's happening
    """
    print(f"\n--- DEBUG INFO ---")
    print(f"Original text: '{text}'")
    
    # Preprocess text
    cleaned = clean_text(text)
    print(f"Cleaned text: '{cleaned}'")
    
    # Tokenize
    seq = tokenizer.texts_to_sequences([cleaned])
    print(f"Tokenized sequence: {seq}")
    
    # Pad
    padded = pad_sequences(seq, maxlen=100)
    print(f"Padded sequence shape: {padded.shape}")
    print(f"Padded sequence (first 20): {padded[0][:20]}")
    
    # Predict
    pred = model.predict(padded, verbose=1)
    print(f"Raw predictions: {pred}")
    print(f"Prediction shape: {pred.shape}")
    
    # Check if binary or multi-class
    if pred.shape[1] == 1:
        # Binary classification with sigmoid
        probability = pred[0][0]
        predicted_class = 1 if probability > 0.5 else 0
        confidence = probability if predicted_class == 1 else (1 - probability)
        labels = ['Negative', 'Positive']
    else:
        # Multi-class classification
        predicted_class = pred.argmax()
        confidence = pred[0][predicted_class]
        if pred.shape[1] == 2:
            labels = ['Negative', 'Positive']
        else:
            labels = ['Negative', 'Neutral', 'Positive']
    
    print(f"Predicted class index: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Final prediction: {labels[predicted_class]}")
    print(f"--- END DEBUG ---\n")
    
    return labels[predicted_class], confidence

def predict_sentiment(text):
    """
    Regular prediction function for binary sentiment classification
    """
    try:
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)
        pred = model.predict(padded, verbose=0)
        
        # Handle both binary and multi-class outputs
        if pred.shape[1] == 1:
            # Binary classification with sigmoid (0-1 output)
            probability = pred[0][0]
            predicted_class = 1 if probability > 0.5 else 0
            confidence = probability if predicted_class == 1 else (1 - probability)
            labels = ['Negative', 'Positive']
        else:
            # Multi-class classification with softmax
            predicted_class = pred.argmax()
            confidence = pred[0][predicted_class]
            if pred.shape[1] == 2:
                labels = ['Negative', 'Positive']
            else:
                labels = ['Negative', 'Neutral', 'Positive']
        
        return labels[predicted_class], confidence
        
    except Exception as e:
        return f"Error: {e}", 0.0

def create_confidence_bar(confidence, length=10):
    """Create a visual confidence bar"""
    filled = int(confidence * length)
    bar = '🟩' * filled + '⬜' * (length - filled)
    return f"{bar} ({confidence:.2f})"

def main():
    """Main application loop"""
    print("=== Binary Sentiment Analysis Tool ===")
    print("Predicts: Positive or Negative sentiment")
    print("Type 'quit' to exit, 'debug' for debug mode")
    
    debug_mode = False
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() == 'quit':
            print("Goodbye!")
            break
        elif text.lower() == 'debug':
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            continue
            
        if not text:
            print("Please enter some text.")
            continue
        
        if debug_mode:
            sentiment, confidence = predict_sentiment_debug(text)
        else:
            sentiment, confidence = predict_sentiment(text)
            
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {create_confidence_bar(confidence)}")

if __name__ == "__main__":
    main()