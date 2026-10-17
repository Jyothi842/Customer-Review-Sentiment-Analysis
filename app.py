"""
app.py  -  Streamlit web app for Customer Review Sentiment Analysis.

Run:  streamlit run app.py     (train first with: python code.py)
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

import importlib.util

# NOTE: "code" clashes with Python's built-in `code` module, so load code.py by path.
_spec = importlib.util.spec_from_file_location(
    "train_code", os.path.join(os.path.dirname(os.path.abspath(__file__)), "code.py")
)
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
MODEL_PATH, METRICS_PATH, clean_text = _train.MODEL_PATH, _train.METRICS_PATH, _train.clean_text

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")

EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😞"}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict(model, texts):
    cleaned = [clean_text(t) for t in texts]
    labels = model.predict(cleaned)
    probs = model.predict_proba(cleaned)
    return labels, probs


st.title("💬 Customer Review Sentiment Analyzer")
st.caption("Classifies reviews as positive, neutral or negative.")

model = load_model()
if model is None:
    st.error("Model not found. Run `python code.py` first to train it.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Single review", "Batch (CSV)", "Model info"])

# ---- Single review ----
with tab1:
    text = st.text_area("Enter a customer review", height=150,
                        placeholder="The movie was amazing, I loved every minute!")
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter a review.")
        else:
            labels, probs = predict(model, [text])
            label = labels[0]
            st.subheader(f"{EMOJI.get(label, '')} {label.capitalize()}")
            st.bar_chart(pd.Series(probs[0], index=model.classes_, name="Confidence"))

# ---- Batch ----
with tab2:
    st.write("Upload a CSV containing a **review** column.")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up)
        if "review" not in df.columns:
            st.error("CSV must have a 'review' column.")
        else:
            df = df.dropna(subset=["review"]).copy()
            labels, probs = predict(model, df["review"].tolist())
            df["predicted_sentiment"] = labels
            df["confidence"] = probs.max(axis=1).round(3)
            st.dataframe(df.head(100), use_container_width=True)
            st.bar_chart(df["predicted_sentiment"].value_counts())
            st.download_button("Download results", df.to_csv(index=False).encode(),
                               "sentiment_results.csv", "text/csv")

# ---- Model info ----
with tab3:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        st.metric("Test accuracy", f"{m['accuracy']:.2%}")
        st.write("Confusion matrix (rows = actual, columns = predicted)")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]))
        st.write("Per-class report")
        st.dataframe(pd.DataFrame(m["report"]).T.round(3))
    else:
        st.info("No metrics file found. Run `python code.py`.")
