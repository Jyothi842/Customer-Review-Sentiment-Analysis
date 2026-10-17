"""
code.py  -  Train the Customer Review Sentiment model.

Run:  python code.py
Output: model/sentiment_model.joblib  +  model/metrics.json
"""
import glob
import json
import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_FILE = "CustomerReviewSentiment.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def find_data_file() -> str:
    """Use DATA_FILE, or any *CustomerReviewSentiment*.csv in this folder."""
    if os.path.exists(DATA_FILE):
        return DATA_FILE
    matches = glob.glob("*CustomerReviewSentiment*.csv") + glob.glob("data/*.csv")
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {DATA_FILE}. Put the CSV next to code.py.")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML, collapse whitespace. Keep ! and ? (useful sentiment cues)."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9!?'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["review", "sentiment"]).drop_duplicates(subset=["review"])
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df["clean_review"] = df["review"].apply(clean_text)
    df = df[df["clean_review"].str.len() > 0]
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def main():
    path = find_data_file()
    df = load_data(path)
    print(f"Loaded {len(df)} reviews from {path}")
    print(df["sentiment"].value_counts(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["sentiment"],
        test_size=0.2, random_state=42, stratify=df["sentiment"],
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, preds, labels=labels)

    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, preds))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {"accuracy": acc, "labels": labels, "confusion_matrix": cm.tolist(), "report": report},
            f, indent=2,
        )
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
