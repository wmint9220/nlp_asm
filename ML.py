import streamlit as st
import tensorflow as tf
import urllib.request

# Download the model from GitHub (replace URL with your model's raw file URL)
url = "https://raw.githubusercontent.com/wmint9220/nlp_asm/main/model.keras"
urllib.request.urlretrieve(url, "model.keras")

# Now load the model
model = tf.keras.models.load_model('model.keras')

# Load the model
model = tf.keras.models.load_model('model.keras')

# Create a function for making predictions
def predict_sentiment(text):
    # Assuming you have preprocessing steps here before passing it to the model
    processed_text = preprocess(text)  # Replace with your preprocessing function
    prediction = model.predict(processed_text)
    return prediction

# Streamlit UI
st.title('Financial Sentiment Analysis')

user_input = st.text_area("Enter a financial sentence:")

if user_input:
    sentiment = predict_sentiment(user_input)
    st.write(f"Sentiment: {sentiment}")
