import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
model = joblib.load("sentiment_model.pkl")
vectorised = joblib.load("tfidf_vectorizer.pkl")
# App title
st.title("Twitter Sentiment Analyzer 🚀")

# Input box
tweet = st.text_input("Enter a tweet:")

# When user clicks the button
if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("⚠️ Please enter a tweet to analyze.")
    else:
        # Vectorize and predict
        vector = vectorised.transform([tweet])
        prediction = model.predict(vector)[0]

        # Show result
        if prediction == 4:
            st.success("✅ Positive Sentiment")
        elif prediction == 0:
            st.error("❌ Negative Sentiment")
        else:
            st.info("🤔 Uncertain or neutral sentiment")