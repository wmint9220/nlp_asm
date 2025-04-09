import streamlit as st
import tensorflow as tf
import urllib.request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization (if needed)

# Download the model from GitHub (replace URL with your model's raw file URL)
url = "https://raw.githubusercontent.com/wmint9220/nlp_asm/main/model.keras"
urllib.request.urlretrieve(url, "model.keras")

# Now load the model
model = tf.keras.models.load_model('model.keras')

# Preprocess the input text before passing it to the model
def predict_sentiment(text):
     processed_text = preprocess(text)
     prediction = model.predict(processed_text)  # Get prediction from model
     sentiment = np.argmax(prediction, axis=1)  # Assuming classification model (e.g., 0 = negative, 1 = positive)
     return sentiment

# Streamlit UI
st.title('Financial Sentiment Analysis')

user_input = st.text_area("Enter a financial sentence:")

if user_input:
    sentiment = predict_sentiment(user_input)
    sentiment_label = "Positive" if sentiment == 1 else "Negative"  # Adjust based on your model's output
    st.write(f"Sentiment: {sentiment_label}")
