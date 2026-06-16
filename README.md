# 📰 Fake News Detection Using Machine Learning

## 📖 Overview

Fake news and misinformation spread rapidly through digital platforms, making it difficult for users to distinguish between reliable and misleading information. This project uses Machine Learning and Natural Language Processing (NLP) techniques to automatically classify news articles as **Real** or **Fake**.

The system preprocesses textual data, converts it into numerical features using TF-IDF Vectorization, and trains a classification model to detect misinformation with high accuracy.

---

## 🎯 Problem Statement

The increasing volume of online news has made manual verification difficult. This project aims to build an intelligent system capable of identifying fake news articles based on their textual content, helping users make informed decisions.

---

## 🚀 Features

- Automated Fake News Classification
- Text Cleaning and Preprocessing
- TF-IDF Feature Extraction
- Machine Learning Model Training
- Model Evaluation and Performance Analysis
- Prediction on Custom News Articles
- Saved Model and Vectorizer for Reuse

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Pickle

### Tools
- Jupyter Notebook
- Git
- GitHub

---

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── Online_Fake_News_Detection.ipynb
├── fake_news_file.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── dataset/
│   └── news_data.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

### 1. Data Collection
- Load labeled datasets containing real and fake news articles.

### 2. Data Preprocessing
- Remove punctuation
- Convert text to lowercase
- Remove stopwords
- Clean URLs and special characters

### 3. Feature Engineering
- Convert text into numerical representations using TF-IDF Vectorization.

### 4. Model Training
- Train machine learning classification models on processed news data.

### 5. Model Evaluation
- Evaluate performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

### 6. Prediction
- Predict whether a given news article is **Real** or **Fake**.

---

## 📊 Results

The model successfully identifies fake and real news articles using NLP-based feature extraction and machine learning techniques.

| Metric | Score |
|----------|----------|
| Accuracy | Replace with your score |
| Precision | Replace with your score |
| Recall | Replace with your score |
| F1-Score | Replace with your score |

> Update the table with your actual model performance metrics.

---

## ▶️ Installation & Execution

### Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
```

### Navigate to the Project Directory

```bash
cd fake-news-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook Online_Fake_News_Detection.ipynb
```

### Run Python Script

```bash
python fake_news_file.py
```

---

## 💡 Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Natural Language Processing (NLP)
- Feature Engineering
- Machine Learning
- Model Evaluation
- Python Programming
- Git & GitHub

---

## 🔮 Future Improvements

- Deploy the model using Streamlit
- Implement Deep Learning models such as LSTM and BERT
- Real-time News Verification
- URL-based News Detection
- API Integration for News Platforms

---

## 👩‍💻 Author

### Athiya Taj

BE (ECE) graduate with a strong interest in Data Science, Machine Learning, and Python Development. Passionate about solving real-world problems through data-driven solutions and continuously expanding technical skills through hands-on projects.

- 📍 Bengaluru, India
- 📧 athiyataj9@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/athiya-taj-a17416394
- 🐙 GitHub: https://github.com/athiyataj21-tech

---

## 📜 License

This project is intended for educational and learning purposes.
