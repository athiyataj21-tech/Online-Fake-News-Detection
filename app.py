import streamlit as st
import joblib
import re

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ==============================
# Load Model & Vectorizer
# ==============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ==============================
# Clean Text
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================
# UI
# ==============================
st.title("📰 Fake News Detection System")
st.caption("⚠️ Predictions are based on language patterns, not factual verification.")

st.markdown(
    """
    Enter a news article below and check whether it is **Fake** or **Real**.  
    Model used: **Logistic Regression + TF-IDF**
    """
)

news_input = st.text_area(
    "📝 Paste News Text Here",
    height=220,
    placeholder="Enter the full news article text..."
)

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some news text!")
    else:
        cleaned_text = clean_text(news_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        probability = model.predict_proba(vectorized_text)[0]

        st.subheader("📌 Prediction Result")

        # Threshold-based decision
        if probability[0] > 0.6:
            st.error("Fake News 🔴")
        elif probability[1] > 0.6:
            st.success("Real News 🟢")
        else:
            st.warning("⚠️ Uncertain Prediction")

        st.subheader("📊 Confidence Score")
        st.info(f"Fake News 🔴: **{probability[0]*100:.2f}%**")
        st.info(f"Real News 🟢: **{probability[1]*100:.2f}%**")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("Fake News Detection Project | Logistic Regression + TF-IDF")
