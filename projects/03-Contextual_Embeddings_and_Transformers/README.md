# Transformers in NLP

This project covers the fundamentals and practical implementation of **Transformer-based Natural Language Processing (NLP)** using PyTorch and Hugging Face.

The goal is to understand how Transformers work, how pretrained models can be used for NLP tasks, and how models such as DistilBERT can be fine-tuned for downstream applications.

---

## Topics Covered

* Transformer architecture
* Self-attention
* Multi-head self-attention
* Contextual embeddings
* Hugging Face Transformers
* Hugging Face Datasets
* Pretrained language models
* Transfer learning
* Fine-tuning
* Transformer-based text classification
* Named Entity Recognition (NER)
* Attention visualization
* Feature extraction from Transformer models
* Dimensionality reduction with PCA

---

## Tools & Libraries

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* scikit-learn
* NumPy
* Matplotlib

---

## Hands-On Work

### 1. Hugging Face Pipelines

Used the Hugging Face `pipeline()` API for common NLP tasks.

#### Sentiment Analysis

Used a pretrained Transformer model to classify text as positive or negative.

#### Named Entity Recognition

Used a pretrained NER model to identify entities such as people, organizations, and locations in text.

---

### 2. DistilBERT Fine-Tuning

Fine-tuned **DistilBERT** on the IMDb sentiment dataset.

The workflow included:

1. Loading the IMDb dataset
2. Tokenizing the text using a DistilBERT tokenizer
3. Preparing the dataset for PyTorch
4. Fine-tuning DistilBERT
5. Evaluating the trained model
6. Running inference on new text

### Result

**Test Accuracy: 84.83%**

This demonstrated the effectiveness of transfer learning: instead of training a language model from scratch, a pretrained Transformer can be adapted to a specific NLP task.

---

## Attention Visualization

Attention maps were visualized to inspect how Transformer attention mechanisms distribute attention across tokens.

This provides an intuitive view of the self-attention mechanism and helps connect the mathematical concept of attention with the behavior of an actual pretrained Transformer.

---

## Frozen DistilBERT Embeddings

In addition to fine-tuning, DistilBERT was also used as a **feature extractor**.

The Transformer was kept frozen and its contextual representations were extracted as **768-dimensional feature vectors**.

These embeddings were then used with traditional and neural classifiers.

### Linear Classifier

A linear classifier was trained using the frozen DistilBERT embeddings.

**Test Accuracy: 81.04%**

### Small Neural Network

A small feed-forward neural network was trained on the same 768-dimensional embeddings.

Architecture:

```text
768
 ↓
Linear(768 → 256)
 ↓
ReLU
 ↓
Dropout(0.3)
 ↓
Linear(256 → 64)
 ↓
ReLU
 ↓
Dropout(0.3)
 ↓
Linear(64 → 2)
```

**Test Accuracy: 80.68%**

The neural network did not outperform the simpler linear classifier, showing that adding nonlinear layers on top of the frozen representations does not necessarily improve performance.

---

## PCA Analysis

PCA was applied to the 768-dimensional DistilBERT embeddings to examine their structure and visualize them in lower-dimensional space.

### 2D PCA

```text
Original shape: (5000, 768)
PCA shape:      (5000, 2)
```

Explained variance:

```text
Component 1: 9.748%
Component 2: 7.089%

Total: 16.84%
```

The 2D representation therefore captures only a portion of the information contained in the original 768-dimensional embeddings.

### 95% Variance

The number of PCA components required to retain approximately 95% of the variance was:

```text
240 components
```

This demonstrates that although the embeddings can be visualized in 2D, their useful information is distributed across a substantially higher-dimensional space.

---

## Results Summary

| Experiment        | Representation   | Model               | Test Accuracy |
| ----------------- | ---------------- | ------------------- | ------------: |
| Frozen embeddings | DistilBERT 768-D | Linear classifier   |    **81.04%** |
| Frozen embeddings | DistilBERT 768-D | Small NN            |    **80.68%** |
| Fine-tuning       | DistilBERT       | Classification head |    **84.83%** |

The comparison demonstrates an important distinction:

**Feature extraction**

```text
Text → Frozen DistilBERT → 768-D embedding → Classifier
```

versus

**Fine-tuning**

```text
Text → DistilBERT → Classification head
          ↑
     Model weights adapt
```

Fine-tuning produced the best result because the Transformer representations were allowed to adapt specifically to the sentiment classification task.

---

## Key Takeaways

* Transformers use **self-attention** to model relationships between tokens.
* **Multi-head attention** allows the model to learn different types of relationships simultaneously.
* Transformer embeddings are **contextual**, meaning the representation of a word depends on its surrounding context.
* Pretrained models such as DistilBERT provide powerful representations without requiring language-model training from scratch.
* Hugging Face makes pretrained Transformer models accessible through high-level APIs such as `pipeline()`.
* Frozen Transformer embeddings can work well with conventional machine-learning classifiers.
* Fine-tuning generally allows the model to adapt its representations to the target task.
* PCA is useful for exploring high-dimensional embeddings, but a 2D visualization does not preserve all of their information.
* Attention visualization provides a way to inspect the internal behavior of Transformer models.

---

## Reference

Hugging Face Course:

https://huggingface.co/course/chapter1/3?fw=pt

---

## Author

Farhad Ali
