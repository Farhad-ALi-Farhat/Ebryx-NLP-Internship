# NLP Fundamentals & Text Preprocessing — IMDB Sentiment Analysis

This project applies fundamental Natural Language Processing techniques to the IMDB Movie Reviews dataset. It covers the complete pipeline from raw text preprocessing to numerical feature extraction and sentiment classification using Logistic Regression.

---

## Objectives

- Clean and preprocess raw text data
- Tokenize text using NLTK
- Remove stopwords, punctuation, numbers, and HTML tags
- Compare NLTK and spaCy lemmatization
- Represent text using Bag-of-Words (BoW)
- Explore unigram and bigram features
- Represent text using TF-IDF
- Train a Logistic Regression sentiment classifier
- Evaluate the model using Accuracy, Precision, Recall, and F1-score
- Understand sparse and high-dimensional text representations

---

## Dataset

**IMDB Dataset of 50K Movie Reviews**

- 50,000 movie reviews
- 25,000 positive reviews
- 25,000 negative reviews
- Binary sentiment classification

Dataset:  
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-moviereviews

---

## Workflow

```text
IMDB Reviews
     ↓
Train/Test Split
     ↓
Text Cleaning
     ├── Lowercasing
     ├── HTML Removal
     ├── Number Removal
     ├── Punctuation Removal
     ├── Tokenization
     └── Stopword Removal
     ↓
Lemmatization
     ├── NLTK
     └── spaCy
     ↓
Feature Extraction
     ├── Bag-of-Words
     ├── N-grams
     └── TF-IDF
     ↓
Logistic Regression
     ↓
Model Evaluation
```

---

## Feature Representations

### Bag-of-Words

`CountVectorizer` was used to convert the processed reviews into word-count vectors.

The unigram representation produced:

**78,626 features**

Adding bigrams increased the feature space to:

**2,322,567 features**

This demonstrates how quickly the feature space can grow when N-grams are introduced.

### TF-IDF

`TfidfVectorizer` was used to convert the processed reviews into TF-IDF representations.

The unigram TF-IDF representation contained the same **78,626 features** as the BoW representation, but assigned weighted values instead of raw word counts.

---

## Model

A Logistic Regression classifier was trained using the TF-IDF features.

### Pipeline

```text
TF-IDF Features
      ↓
Logistic Regression
      ↓
Sentiment Prediction
```

### Results

| Metric | Score |
|---|---:|
| Accuracy | 89.61% |
| Precision | 88.58% |
| Recall | 90.94% |
| F1 Score | 89.75% |

---

## Key Takeaways

- Raw text must be transformed into numerical representations before it can be used by traditional machine learning models.
- BoW represents documents using word frequencies but does not capture semantic relationships.
- TF-IDF weights words according to their importance within individual documents and the overall corpus.
- N-grams can capture short sequences of words but substantially increase the feature space.
- BoW and TF-IDF produce sparse, high-dimensional representations.
- TF-IDF combined with Logistic Regression provides a strong classical baseline for sentiment classification.
- Lemmatization can reduce different forms of words to a common base form.

---

## Technologies

- Python
- Jupyter Notebook
- NumPy
- Pandas
- NLTK
- spaCy
- Scikit-learn
- Matplotlib

---

## Author

**Farhad Ali**
