# utils/preprocessing.py

import re

def clean_text(text):
    """
    Basic text cleaning to match model training preprocessing:
    - Lowercasing
    - Remove punctuation and non-alphabetic characters
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()
