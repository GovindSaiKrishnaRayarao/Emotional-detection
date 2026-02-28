# utils/visualization.py

import emoji

EMOJI_MAP = {
    "joy": "😂",
    "fear": "😨",
    "anger": "😡",
    "sadness": "😢",
    "disgust": "🤢",
    "shame": "😳",
    "guilt": "😔"
}

def show_emotion_emoji(emotion_label):
    """
    Returns emoji representation for a given emotion.
    """
    return EMOJI_MAP.get(emotion_label, "❓")
