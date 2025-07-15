 📰 Fake News Detection App (Streamlit + Machine Learning)

A simple and accurate machine learning project to classify news articles as **REAL** or **FAKE** using text classification and Natural Language Processing (NLP) techniques.

---
 📌 Project Overview

This app allows users to enter a news headline or article and receive an instant prediction on whether it is real or fake. It uses a Logistic Regression model trained on a dataset of real and fake news articles and is deployed using **Streamlit**.

---

## 🧠 Technologies Used

- **Python**
- **Pandas** – data handling
- **scikit-learn** – machine learning model
- **nltk** – for stopword removal
- **TF-IDF Vectorizer** – for text feature extraction
- **Streamlit** – web app interface

---
 📂 Files Included

- `main.py` – Model training script
- `app.py` – Streamlit app for user interaction
- `fake_news_model.pkl` – Trained Logistic Regression model
- `tfidf_vectorizer.pkl` – Trained TF-IDF transformer
- `README.md` – Project details

> *Note: You can get the dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).*

---
 🚀 How to Run Locally

1. Clone this repo:
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector

2.Intall Depencies
pip install pandas scikit-learn nltk streamlit

3.Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py

4.🧪 Sample Use
Paste a news sentence like:

arduino
Copy
Edit
"Aliens landed in Delhi and hacked the metro system."
Click Check Now → It will tell you: 🚫 FAKE or ✅ REAL

🧑‍💻 Author
Nagalakshmi D
📍 Tiruchirappalli, Tamil Nadu
📧 nagadhamo123@gmail.com
🔗 LinkedIn

⭐ If You Liked It...
Please ⭐ star the repo and share feedback!
Connect with me for collaborations or fresher roles in software, AI/ML, or web development.

yaml
Copy
Edit
