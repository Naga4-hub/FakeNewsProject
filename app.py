import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

# Load saved model and vectorizer
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

stop_words = stopwords.words('english')

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article or headline to check if it's **REAL or FAKE**.")

user_input = st.text_area("📝 Paste your news content here:", "")

if st.button("Check Now"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        if prediction[0] == 0:
            st.error("🚫 This news is predicted as: **FAKE**")
        else:
            st.success("✅ This news is predicted as: **REAL**")
