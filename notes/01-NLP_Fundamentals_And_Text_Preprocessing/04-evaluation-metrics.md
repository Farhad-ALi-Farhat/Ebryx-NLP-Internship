# Evaluation Metrics for NLP Tasks

When building an NLP model, we need metrics to determine **how well its predictions match the true labels**.

For many NLP tasks such as:

* Sentiment classification
* Spam detection
* Topic classification
* Intent classification
* Named Entity Recognition

we can use **Precision, Recall, and F1 score**.

These metrics are based on the **confusion matrix**.

---

## Confusion Matrix

For a binary classification problem, predictions can be divided into four categories:

|                        |     Actual Positive |     Actual Negative |
| ---------------------- | ------------------: | ------------------: |
| **Predicted Positive** |  True Positive (TP) | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) |  True Negative (TN) |

### True Positive (TP)

The model predicts positive, and the actual class is positive.

```text
Actual:    Positive
Predicted: Positive
```

### False Positive (FP)

The model predicts positive, but the actual class is negative.

```text
Actual:    Negative
Predicted: Positive
```

### False Negative (FN)

The model predicts negative, but the actual class is positive.

```text
Actual:    Positive
Predicted: Negative
```

### True Negative (TN)

The model predicts negative, and the actual class is negative.

```text
Actual:    Negative
Predicted: Negative
```

These four values form the basis of Precision and Recall.

---

# 1. Precision

**Precision** answers:

> Of everything the model predicted as positive, how many were actually positive?

The formula is:

$$
Precision = \frac{TP}{TP + FP}
$$

For example, suppose a spam classifier predicts that **100 emails are spam**.

Of those:

* 90 are actually spam → TP = 90
* 10 are legitimate → FP = 10

Then:

$$
Precision = \frac{90}{90 + 10} = 0.90
$$

So the model has **90% precision**.

### In NLP

For a sentiment classifier, precision for the positive class asks:

> Of all reviews predicted as positive, how many were actually positive?

High precision means **few false positive predictions**.

---

# 2. Recall

**Recall** answers:

> Of all the examples that were actually positive, how many did the model correctly identify?

The formula is:

$$
Recall = \frac{TP}{TP + FN}
$$

Suppose there are actually **100 spam emails**, and the model correctly identifies 90 of them.

Then:

* TP = 90
* FN = 10

Therefore:

$$
Recall = \frac{90}{90 + 10} = 0.90
$$

The model has **90% recall**.

### In NLP

For spam detection, recall asks:

> Of all the actual spam messages, how many did the model successfully detect?

High recall means **few false negatives**.

---

# Precision vs Recall

The easiest way to remember the difference:

### Precision

> **When the model says YES, how often is it correct?**

$$
Precision = \frac{TP}{TP+FP}
$$

Focuses on **false positives**.

### Recall

> **Of all the actual YES cases, how many did the model find?**

$$
Recall = \frac{TP}{TP+FN}
$$

Focuses on **false negatives**.

A useful memory trick:

```text
Precision → Predicted Positives
Recall    → Actual Positives
```

---

# Precision–Recall Trade-off

Precision and recall often move in opposite directions when changing the classification threshold.

For example, imagine a spam classifier.

If we make the classifier **very strict** about calling something spam:

```text
Fewer messages → classified as spam
```

This can increase precision because fewer legitimate messages are incorrectly labeled as spam.

However, the model may miss more actual spam:

```text
Recall ↓
```

If we make the classifier more permissive:

```text
More messages → classified as spam
```

we may detect more actual spam:

```text
Recall ↑
```

but also incorrectly classify more legitimate messages:

```text
Precision ↓
```

This creates the **precision-recall trade-off**.

---

# 3. F1 Score

Sometimes we want a single metric that balances precision and recall.

That's where the **F1 score** comes in.

F1 is the **harmonic mean** of precision and recall:

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

It can also be written directly using the confusion-matrix values:

$$
F1 = \frac{2TP}{2TP + FP + FN}
$$

Suppose:

$$
Precision = 0.80
$$

and

$$
Recall = 0.60
$$

Then:

$$
F1 =
2 \times
\frac{0.80 \times 0.60}
{0.80 + 0.60}
$$

$$
F1 \approx 0.686
$$

So the F1 score is approximately **0.69**.

The harmonic mean is useful because a very high precision combined with very low recall will still result in a relatively low F1 score.

---

# Why Not Just Use Accuracy?

Accuracy is:

$$
Accuracy =
\frac{TP + TN}
{TP + TN + FP + FN}
$$

It measures the proportion of **all predictions that are correct**.

The problem is that accuracy can be misleading when classes are **imbalanced**.

Imagine a sentiment dataset containing:

```text
950 Negative
50 Positive
```

A model that predicts **Negative for every review** achieves:

$$
Accuracy = \frac{950}{1000} = 95%
$$

That sounds excellent.

But the model completely fails to identify the positive reviews.

Precision and recall expose this kind of problem much better.

---

# Choosing the Right Metric

The appropriate metric depends on what matters most for the task.

| Metric        | Focus                                | Useful When                              |
| ------------- | ------------------------------------ | ---------------------------------------- |
| **Precision** | Avoiding FP                          | Positive predictions need to be reliable |
| **Recall**    | Avoiding FN                          | Missing positive cases is costly         |
| **F1**        | Balance between precision and recall | Both FP and FN matter                    |
| **Accuracy**  | Overall correctness                  | Dataset is reasonably balanced           |

### Spam Detection

If incorrectly labeling legitimate messages as spam is particularly problematic, **precision** becomes important.

### Sentiment Classification

If both false positives and false negatives matter, **F1** can provide a useful single summary.

### Imbalanced NLP Dataset

If one class is much more common than another, accuracy alone may give a misleading picture. Precision, recall, and F1 are generally more informative.

---

# Multiclass NLP Tasks

Many NLP problems have more than two classes.

For example:

```text
Positive
Negative
Neutral
```

or:

```text
Sports
Politics
Technology
Business
Entertainment
```

Precision, recall, and F1 can be calculated **per class** and then aggregated.

Common averaging strategies include:

### Macro Average

Calculates the metric independently for each class and then gives every class equal weight.

Useful when every class is considered equally important.

### Weighted Average

Calculates the metric for each class and weights each result according to the number of samples in that class.

Useful when class sizes are different and we want the overall result to reflect the dataset distribution.

### Micro Average

Aggregates the underlying TP, FP, and FN across classes before calculating the metric.

This gives greater influence to classes with more samples.

---

# NLP Example

Suppose we build a sentiment classifier:

```text
Input Review
      ↓
NLP Model
      ↓
Positive / Negative
```

The test set produces:

```text
TP = 80
FP = 10
FN = 20
TN = 90
```

### Precision

$$
Precision = \frac{80}{80+10}
$$

$$
Precision \approx 0.889
$$

### Recall

$$
Recall = \frac{80}{80+20}
$$

$$
Recall = 0.80
$$

### F1

$$
F1 =
2 \times
\frac{0.889 \times 0.80}
{0.889 + 0.80}
$$

$$
F1 \approx 0.842
$$

Therefore:

```text
Precision ≈ 0.889
Recall    = 0.800
F1        ≈ 0.842
```

The model is more precise than it is comprehensive: when it predicts positive, it is usually correct, but it still misses some actual positive examples.

---

# Metric Selection by Task

There is no universally "best" metric. The correct choice depends on the consequences of different types of errors.

```text
                    What matters most?
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
     Avoid FP          Avoid FN        Balance Both
          ↓                ↓                ↓
     Precision           Recall             F1
```

For practical NLP projects, it is usually useful to report **multiple metrics** rather than relying on a single number.

---

# Key Takeaways

* **Precision** measures how reliable positive predictions are.
* **Recall** measures how many actual positive examples the model finds.
* **F1** balances precision and recall using their harmonic mean.
* **Accuracy** measures overall correctness but can be misleading with imbalanced datasets.
* Precision is especially concerned with **false positives**.
* Recall is especially concerned with **false negatives**.
* F1 is useful when we want to balance both types of errors.
* For multiclass NLP tasks, metrics can be aggregated using **macro, weighted, or micro averaging**.
* The best metric depends on the **specific NLP task and the cost of different errors**.

---

## Reference

[Google Machine Learning Crash Course — Classification: Accuracy, Precision, Recall, and Related Metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy)
