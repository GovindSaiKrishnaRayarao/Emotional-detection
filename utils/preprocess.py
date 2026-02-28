# utils/preprocess.py

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Lowercase, remove non-alphanumeric, and stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(words)
