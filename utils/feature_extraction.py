# utils/feature_extraction.py

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def extract_features(texts, ngram_range=(1, 4)):
    """
    Convert list of text to n-gram feature vectors.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
