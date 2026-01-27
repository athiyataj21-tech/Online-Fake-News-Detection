# =============================
# Fake News Detection Project
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =============================
# Load Dataset
# =============================
df = pd.read_csv("Fake.csv")   # change path if required

print("Dataset Loaded Successfully")
print(df.head())
print(df.info())

# =============================
# Data Cleaning
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# Map labels
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

# =============================
# Visualization 1: Class Distribution
# =============================
plt.figure()
df["label"].value_counts().plot(kind="bar")
plt.title("Class Distribution (Fake vs Real News)")
plt.xlabel("Label (0 = Fake, 1 = Real)")
plt.ylabel("Count")
plt.show()

# =============================
# Visualization 2: Text Length Distribution
# =============================
df["text_length"] = df["text"].apply(len)

plt.figure()
sns.histplot(df["text_length"], bins=50)
plt.title("Distribution of News Text Length")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

# =============================
# Train-Test Split
# =============================
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# TF-IDF Vectorization
# =============================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =============================
# Model Training
# =============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# =============================
# Predictions
# =============================
y_pred = model.predict(X_test_tfidf)

# =============================
# Evaluation Metrics
# =============================
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =============================
# Visualization 3: Confusion Matrix
# =============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# =============================
# Visualization 4: Important Words (TF-IDF)
# =============================
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_real = np.argsort(coefficients)[-10:]
top_fake = np.argsort(coefficients)[:10]

plt.figure()
plt.barh(
    [feature_names[i] for i in top_real],
    coefficients[top_real]
)
plt.title("Top Words Predicting REAL News")
plt.xlabel("Coefficient Value")
plt.show()

plt.figure()
plt.barh(
    [feature_names[i] for i in top_fake],
    coefficients[top_fake]
)
plt.title("Top Words Predicting FAKE News")
plt.xlabel("Coefficient Value")
plt.show()

print("\nProject Execution Completed Successfully!")
