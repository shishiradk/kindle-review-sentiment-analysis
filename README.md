```
# 📘 Kindle Review Sentiment Analysis using NLP Techniques

This project performs binary sentiment classification (Positive vs Negative) on Kindle reviews using NLP techniques like Bag of Words (BoW), TF-IDF, and Word2Vec, followed by training with a Naive Bayes classifier.

---

## 📁 Dataset

- **File Used:** `all_kindle_review.csv`
- **Columns Used:**
  - `reviewText`: User review
  - `rating`: Star rating (used for labeling)

---

## ✅ Project Pipeline

### 1. Data Preprocessing

- Removed missing values
- Converted ratings:
  - Ratings < 3 → 0 (Negative)
  - Ratings ≥ 3 → 1 (Positive)
- Cleaned review text by:
  - Lowercasing
  - Removing special characters, URLs, and HTML tags
  - Removing stopwords
  - Normalizing whitespaces
  - Lemmatizing words using WordNet

### 2. Tokenization

- Tokenized each review using `nltk.word_tokenize()` for Word2Vec model training.

---

## 🧠 Feature Extraction

### 🔹 Bag of Words (BoW)
- Used `CountVectorizer` to create document-term matrix.

### 🔹 TF-IDF
- Used `TfidfVectorizer` to capture word importance based on frequency.

### 🔹 Word2Vec
- Trained a Word2Vec model using `gensim`
- Averaged word vectors to represent each review numerically

---

## 🤖 Model

- **Algorithm:** Gaussian Naive Bayes (`GaussianNB`)
- Trained 3 separate models:
  - On BoW features
  - On TF-IDF features
  - On Word2Vec features

---

## 📊 Results

### ✅ BOW Accuracy
```
Accuracy: [YOUR BOW ACCURACY HERE]
```

### ✅ TF-IDF Accuracy
```
Accuracy: [YOUR TF-IDF ACCURACY HERE]
```

### ✅ Word2Vec Accuracy
```
Accuracy: [YOUR WORD2VEC ACCURACY HERE]
Confusion Matrix:
[[TN FP]
 [FN TP]]
Classification Report:
[YOUR CLASSIFICATION REPORT HERE]
```

---

## 🛠 Libraries Used

- `pandas`, `numpy`
- `sklearn` (for model training and evaluation)
- `gensim` (for Word2Vec)
- `nltk` (for stopwords, tokenization, lemmatization)
- `beautifulsoup4`, `lxml` (for HTML tag removal)

---

## 📦 Installation

```bash
# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run

Ensure your dataset `all_kindle_review.csv` is in the working directory, then run your script:

```bash
python your_script_name.py
```

---

## 📌 To Do (Optional)

- Test more classifiers (Logistic Regression, SVM)
- Use pre-trained embeddings like GloVe
- Build a frontend using Streamlit/Flask
- Add cross-validation and hyperparameter tuning

---

## 📧 Contact

Feel free to connect for feedback or collaboration!
```
