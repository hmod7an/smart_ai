"""
Chatbot module — NLP-powered intent detection using TF-IDF + Logistic Regression.
Falls back to rule-based matching if the model is not trained yet.
"""
import re
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
EXAMPLES_PATH = os.path.join(MODELS_DIR, "intent_examples.json")

_pipeline = None   # cached trained pipeline
_labels   = []


def _load_examples():
    with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts  = [item["text"]  for item in data["intents"]]
    labels = [item["label"] for item in data["intents"]]
    return texts, labels


def train_intent_model():
    """Train TF-IDF + Logistic Regression pipeline and return evaluation metrics."""
    global _pipeline, _labels
    texts, labels = _load_examples()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=None
    )

    _pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
        ("clf",   LogisticRegression(max_iter=500, C=1.0)),
    ])
    _pipeline.fit(X_train, y_train)
    _labels = list(set(labels))

    y_pred   = _pipeline.predict(X_test) if X_test else []
    accuracy = accuracy_score(y_test, y_pred) if X_test else 1.0
    report   = classification_report(y_test, y_pred, zero_division=0) if X_test else "N/A (small dataset)"

    return {
        "accuracy":  round(accuracy * 100, 1),
        "report":    report,
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "classes":   _labels,
    }


def detect_intent(user_input: str) -> str:
    """Detect intent using ML model, fall back to rules if model not trained."""
    global _pipeline
    if _pipeline is None:
        train_intent_model()

    try:
        return _pipeline.predict([user_input])[0]
    except Exception:
        return _rule_based_intent(user_input)


def _rule_based_intent(text: str) -> str:
    text = text.lower()
    if ("ربح" in text or "دفع" in text) and "عميل" in text:
        return "client_profit"
    if "مورد" in text and ("رفع" in text or "سعر" in text or "تغير" in text):
        return "supplier_check"
    if ("اقترح" in text or "احسب" in text) and "سعر" in text:
        return "suggest_price"
    if "هامش" in text or "خاسر" in text or "ضعيف" in text:
        return "low_margin"
    if "أفضل" in text or "أكثر" in text and "ربح" in text:
        return "top_products"
    if "إجمالي" in text or "مجموع" in text:
        return "total_profit"
    return "unknown"


def extract_name_after_keyword(user_input: str, keyword: str):
    pattern = rf"{keyword}\s+([^\s،,؟?]+)"
    match   = re.search(pattern, user_input)
    return match.group(1) if match else None


def normalize_text(text: str) -> str:
    """Basic Arabic/English text normalization."""
    text = text.strip().lower()
    text = re.sub(r"[،,؟?!.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_confidence(user_input: str) -> dict:
    """Return intent probabilities for all classes."""
    global _pipeline
    if _pipeline is None:
        train_intent_model()
    probs   = _pipeline.predict_proba([user_input])[0]
    classes = _pipeline.classes_
    return dict(sorted(zip(classes, probs), key=lambda x: -x[1]))
