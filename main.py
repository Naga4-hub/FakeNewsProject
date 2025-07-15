import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

print("🔍 Starting fake news detection script...")

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine both
data = pd.concat([fake, true], axis=0)
data = data[['text', 'label']]  # Use article text

# Prepare data
stop_words = stopwords.words('english')
X = data['text']
y = data['label']

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.25, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))

import pickle

# Save the trained model
with open("fake_news_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the fitted TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("✅ Model and vectorizer saved successfully!")

