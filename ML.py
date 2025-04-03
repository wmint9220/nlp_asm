import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load('model.keras')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define a function to predict the sentiment
def predict_sentiment(text):
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    # Predict the sentiment (0: negative, 1: positive)
    sentiment = model.predict(text_vectorized)[0]
    return "Positive" if sentiment == 1 else "Negative"

# Streamlit UI
st.title('Financial Sentiment Analysis')

st.write("""
    This app predicts the sentiment of a given financial sentence or news.
    It can classify the sentiment as either Positive or Negative based on financial content.
""")

# User input
user_input = st.text_area("Enter a financial sentence or report:")

if user_input:
    # Run the sentiment prediction when the user provides input
    sentiment = predict_sentiment(user_input)
    st.write(f"Sentiment: {sentiment}")
