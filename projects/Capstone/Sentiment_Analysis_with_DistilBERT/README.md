# NLP Capstone — Sentiment Analysis with DistilBERT

## Overview

This capstone project implements an end-to-end **Natural Language Processing (NLP)** workflow for binary sentiment classification using a pretrained **DistilBERT Transformer** model.

The project covers the complete pipeline:

**Text → Cleaning → Tokenization → Transformer Fine-Tuning → Evaluation → Explainability → Deployment**

The trained model classifies movie reviews as either **Positive** or **Negative** and is deployed through an interactive **Gradio** interface.

---

## Objectives

* Build an end-to-end Transformer-based NLP pipeline
* Fine-tune a pretrained DistilBERT model for sentiment analysis
* Evaluate the model using standard classification metrics
* Analyze model predictions using a confusion matrix
* Visualize Transformer attention
* Deploy the trained model through an interactive Gradio application

---

## Dataset

The project uses the **IMDb Movie Reviews** dataset.

The original dataset contains:

| Split        | Samples |
| ------------ | ------: |
| Training     |  25,000 |
| Test         |  25,000 |
| Unsupervised |  50,000 |

The original training set was split into training and validation subsets:

| Split      | Samples | Purpose                        |
| ---------- | ------: | ------------------------------ |
| Training   |  20,000 | Model fine-tuning              |
| Validation |   5,000 | Model selection and monitoring |
| Test       |  25,000 | Final evaluation               |

Both supervised classes are perfectly balanced.

* `0` → Negative
* `1` → Positive

The unsupervised portion of the IMDb dataset was not used.

---

## NLP Pipeline

### 1. Text Cleaning

The reviews contain HTML formatting such as:

```text
<br />
```

These tags were removed while preserving the actual review content.

Only basic cleaning was performed because Transformer models work best when the natural linguistic structure of the text is preserved.

The following preprocessing operations were intentionally avoided:

* Stopword removal
* Stemming
* Lemmatization
* Manual lowercasing
* Aggressive punctuation removal

DistilBERT's tokenizer handles the appropriate text normalization itself.

---

### 2. Tokenization

The `distilbert-base-uncased` tokenizer was used.

Each review was:

* Tokenized into WordPiece tokens
* Truncated to a maximum of **128 tokens**
* Padded to the same length
* Converted into `input_ids`
* Provided with an `attention_mask`

The resulting model inputs contain:

```text
input_ids
attention_mask
label
```

---

## Model

### DistilBERT

The project uses:

**`distilbert-base-uncased`**

DistilBERT was selected instead of full BERT because it provides a significantly lighter Transformer architecture while retaining strong performance for text classification.

Model characteristics:

| Property                     |      Value |
| ---------------------------- | ---------: |
| Transformer layers           |          6 |
| Hidden size                  |        768 |
| Attention heads              |         12 |
| Total parameters             | 66,955,010 |
| Trainable parameters         | 66,955,010 |
| Number of classes            |          2 |
| Maximum sequence length used |        128 |

A new classification head was added for the two sentiment classes and fine-tuned together with the pretrained model.

---

## Training

The model was fine-tuned using the Hugging Face `Trainer`.

### Training configuration

| Parameter             | Value      |
| --------------------- | ---------- |
| Model                 | DistilBERT |
| Dataset               | IMDb       |
| Epochs                | 2          |
| Learning rate         | `2e-5`     |
| Training batch size   | 4          |
| Evaluation batch size | 4          |
| Gradient accumulation | 4          |
| Effective batch size  | 16         |
| Weight decay          | 0.01       |
| Sequence length       | 128        |
| Mixed precision       | FP16       |
| Best-model metric     | F1         |

Training was performed using a GPU:

**NVIDIA GeForce MX330**

The model checkpoint with the best validation F1-score was retained.

---

## Validation Results

The training and validation results were:

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall |     F1 |
| ----: | ------------: | --------------: | -------: | --------: | -----: | -----: |
|     1 |        1.3201 |          0.3024 |   87.08% |    86.50% | 87.88% | 87.18% |
|     2 |        0.8437 |          0.3461 |   87.52% |    86.70% | 88.64% | 87.66% |

The best model was selected using validation F1.

---

## Final Test Results

The saved best model was evaluated once on the previously unseen test set containing **25,000 reviews**.

| Metric    | Test Score |
| --------- | ---------: |
| Accuracy  | **88.04%** |
| Precision | **87.07%** |
| Recall    | **89.34%** |
| F1-score  | **88.19%** |
| Loss      | **0.3309** |

### Final Result

> **DistilBERT achieved 88.04% accuracy and 88.19% F1-score on the held-out IMDb test set.**

The test performance was slightly higher than validation performance, indicating that there was no obvious degradation in generalization.

---

## Confusion Matrix

The final confusion matrix was:

```text
                  Predicted
                  Negative  Positive

Actual Negative     10842      1658
Actual Positive      1332     11168
```

### Interpretation

* **10,842** negative reviews were correctly classified.
* **11,168** positive reviews were correctly classified.
* **1,658** negative reviews were incorrectly classified as positive.
* **1,332** positive reviews were incorrectly classified as negative.

The model correctly classified:

```text
10,842 + 11,168 = 22,010
```

out of:

```text
25,000
```

test examples, producing the reported **88.04% accuracy**.

---

## Explainability — Attention Visualization

To investigate how the Transformer processes text, attention weights were extracted from the model.

DistilBERT contains:

* 6 Transformer layers
* 12 attention heads per layer
* 72 attention heads in total

Attention visualization initially required changing the attention implementation from **SDPA** to **eager attention**, because SDPA does not provide the attention weights needed for visualization in this configuration.

The model was therefore loaded with:

```python
attn_implementation="eager"
```

Attention from the `[CLS]` token was examined for a selected attention head.

An example produced high attention values for tokens such as:

```text
silly
painfully
cannot
cheap
```

These tokens were potentially relevant to the negative sentiment prediction.

### Important limitation

Attention weights should **not** be interpreted as definitive feature importance or a causal explanation of a prediction.

They provide insight into what a particular attention mechanism focuses on, but they do not necessarily explain the complete reasoning behind the model's classification.

---

## Model Deployment

The trained model was deployed using **Gradio**.

The application accepts a movie review and returns the predicted sentiment probabilities.

Example:

```text
Input:
This movie was absolutely fantastic. I loved every minute of it.

Prediction:
POSITIVE — 99%
NEGATIVE — 1%
```

Another example:

```text
Input:
The movie was boring, predictable, and painfully slow.

Prediction:
NEGATIVE — 100%
POSITIVE — 0%
```

A mixed-sentiment example:

```text
Input:
It wasn't terrible, but it wasn't particularly good either.

Prediction:
NEGATIVE — 97%
POSITIVE — 3%
```

The final example also demonstrates that model confidence should not automatically be interpreted as certainty, particularly for ambiguous or mixed-sentiment text.

---

## Saved Model

The fine-tuned model was saved locally as:

```text
distilbert-imdb-final/
```

The directory contains the trained model weights and tokenizer configuration required to reload the model without retraining.

The model can be loaded using:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

save_path = "./distilbert-imdb-final"

tokenizer = AutoTokenizer.from_pretrained(save_path)

model = AutoModelForSequenceClassification.from_pretrained(
    save_path
)
```

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* NumPy
* Matplotlib
* Gradio

---

## Key Concepts Demonstrated

This capstone brings together the major concepts covered throughout the NLP and Transformers work:

* NLP preprocessing
* Text classification
* Tokenization
* WordPiece subword tokenization
* Transformer architectures
* BERT-style models
* DistilBERT
* Transfer learning
* Fine-tuning
* Attention mechanisms
* Attention visualization
* Model evaluation
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrices
* Model deployment
* Interactive NLP applications

---

## Limitations

Several limitations should be considered:

1. **Sequence length**

   Reviews were limited to 128 tokens. Longer reviews are truncated, meaning some potentially useful information may not reach the model.

2. **Dataset domain**

   The model was trained on movie reviews and may not perform equally well on other types of text, such as tweets, product reviews, customer support messages, or formal documents.

3. **Binary sentiment**

   The model only predicts positive or negative sentiment. It does not explicitly classify neutral or mixed sentiment.

4. **Attention interpretation**

   Attention visualization provides useful insight but should not be treated as a definitive explanation of model decisions.

5. **Confidence**

   A high probability does not necessarily mean the prediction is correct. The mixed-sentiment example demonstrates that the model can be highly confident about ambiguous text.

6. **Hardware constraints**

   Fine-tuning was performed on an NVIDIA GeForce MX330 with limited VRAM, requiring a small batch size and relatively short sequence length.

---

## Conclusion

This project demonstrates a complete modern NLP workflow using a pretrained Transformer.

Starting from raw IMDb movie reviews, the text was cleaned and tokenized before being passed to DistilBERT. The pretrained model was fine-tuned for binary sentiment classification and evaluated using both validation and held-out test data.

The final model achieved:

**88.04% test accuracy**
**88.19% test F1-score**

Attention visualization provided an additional perspective into Transformer behavior, while the Gradio interface demonstrated how the trained model could be presented as an interactive NLP application.

Overall, the project demonstrates the complete progression from **raw text to a trained, evaluated, explainable, and deployable Transformer-based NLP system**.
