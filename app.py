import streamlit as st
import re
import string
import joblib

# --- Step 1: Define cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# --- Step 2: Load trained model and vectorizer ---
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# --- Step 3: Prediction function ---
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# --- Step 4: Streamlit UI ---
st.title("📦 Flipkart Customer Sentiment Analyzer")

st.markdown("Enter a customer support message below to predict the sentiment:")

user_input = st.text_area("Customer Remark", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a customer remark.")
    else:
        sentiment = predict_sentiment(user_input)
        st.success(f"**Predicted Sentiment:** {sentiment}")
