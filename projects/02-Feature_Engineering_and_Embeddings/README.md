# Word Embeddings & SMS Spam Detection

This project explores **word embeddings and text representation techniques** using the SMS Spam Collection dataset.

The project covers training a custom **Word2Vec** model using Gensim, exploring pre-trained **GloVe** embeddings, visualizing word embeddings using **PCA**, and comparing **TF-IDF** and **Word2Vec** features for SMS spam classification.

---

## Overview

The main objective of this project is to understand how different text representation techniques affect the performance of a machine learning model for spam detection.

Two primary approaches are compared:

- **TF-IDF** — traditional sparse text representation
- **Word2Vec** — dense word embedding representation

The project also explores pre-trained **GloVe** embeddings to understand how general-purpose word representations capture semantic relationships.

---

## Objectives

- Preprocess and clean SMS text data
- Train a custom Word2Vec model using Gensim
- Explore learned word embeddings using nearest-word similarity
- Calculate cosine similarity between word vectors
- Load and explore pre-trained GloVe embeddings
- Visualize word embeddings using PCA
- Build a spam detection model using TF-IDF features
- Build a spam detection model using Word2Vec features
- Compare TF-IDF and Word2Vec representations for spam detection

---

## Dataset

The project uses the **SMS Spam Collection Dataset**, containing SMS messages labeled as either:

- `ham` — legitimate message
- `spam` — unwanted/spam message

### Dataset Source

[Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

After cleaning and removing duplicate messages, the dataset contains **5,169 unique SMS messages**.

---

## Project Workflow

```text
SMS Spam Collection
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Train / Test Split
        │
        ├───────────────────────┐
        ▼                       ▼
     Word2Vec                  TF-IDF
        │                       │
        ▼                       ▼
Word Embedding Features    TF-IDF Features
        │                       │
        ▼                       ▼
Logistic Regression       Logistic Regression
        │                       │
        └───────────┬───────────┘
                    ▼
          Performance Comparison
```

---

## Text Preprocessing

The SMS messages are cleaned before generating text representations.

The preprocessing steps include:

- Converting text to lowercase
- Removing non-alphanumeric characters
- Normalizing whitespace
- Tokenizing messages into words

The cleaned tokens are then used for training the Word2Vec model.

---

## Word2Vec

A custom Word2Vec model is trained using **Gensim**.

### Configuration

| Parameter | Value |
|---|---:|
| Vector Size | 100 |
| Window | 5 |
| Minimum Count | 2 |
| Architecture | Skip-gram |
| Negative Sampling | Enabled |
| Epochs | 20 |

The model learns a 100-dimensional vector representation for words appearing in the SMS corpus.

### Word Similarity

The learned embeddings are explored using:

- `most_similar()`
- Cosine similarity

These operations are used to examine relationships between words learned from the SMS corpus.

---

## GloVe Embeddings

The project also loads pre-trained **GloVe Wiki Gigaword 100-dimensional embeddings**.

GloVe provides general-purpose word representations learned from a large external corpus.

The embeddings are explored using:

- Nearest-word queries
- Cosine similarity
- Semantic relationship exploration

This provides a practical comparison between:

- Domain-specific Word2Vec embeddings trained on SMS messages
- General-purpose pre-trained GloVe embeddings

---

## Embedding Visualization

The Word2Vec embeddings are originally represented in 100 dimensions.

To visualize them, **Principal Component Analysis (PCA)** is used to reduce the embeddings to two dimensions.

```text
100-Dimensional Word Vectors
            │
            ▼
           PCA
            │
            ▼
      2-Dimensional Space
            │
            ▼
       Scatter Plot
```

A small selection of representative words is visualized to make the learned embedding space easier to inspect.

---

## Spam Detection

The project compares two different feature representations using **Logistic Regression**.

### TF-IDF

TF-IDF represents each SMS using a sparse vector based on the importance of words within the dataset.

### Word2Vec

Word2Vec provides dense word-level representations.

Since a message contains multiple words, the Word2Vec vectors of the words in each message are averaged to create a single **100-dimensional message representation**.

These message-level vectors are then provided to Logistic Regression for classification.

---

## Results

The two approaches were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| TF-IDF | 96.62% | **98.98%** | 74.05% | 84.72% |
| **Word2Vec** | **97.97%** | 95.83% | **87.79%** | **91.63%** |

### Results Analysis

The Word2Vec-based model achieved the higher overall performance.

Compared with TF-IDF:

- Accuracy increased from **96.62% → 97.97%**
- Recall increased from **74.05% → 87.79%**
- F1-score increased from **84.72% → 91.63%**

TF-IDF achieved higher precision:

- **98.98%** vs **95.83%**

However, Word2Vec achieved substantially better recall and F1-score, indicating that it was better at identifying spam messages while maintaining strong overall classification performance.

For this dataset, **Word2Vec provided the better overall representation for the spam detection task**.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Gensim
- Word2Vec
- GloVe
- Logistic Regression
- PCA
- Jupyter Notebook

---

## Key Takeaways

- **TF-IDF** provides a strong traditional baseline for text classification.
- **Word2Vec** creates dense representations that capture relationships between words.
- Pre-trained **GloVe** embeddings provide useful general-purpose semantic representations.
- PCA can be used to visualize high-dimensional word embeddings.
- Averaging Word2Vec word vectors provides a simple way to create fixed-size message representations.
- For this SMS spam dataset, Word2Vec achieved a better overall classification result than TF-IDF.
- **F1-score and recall** were particularly improved with the Word2Vec representation.

---

## References

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013.
- Pennington, Socher, and Manning, *GloVe: Global Vectors for Word Representation*, 2014.
- SMS Spam Collection Dataset — Kaggle.
