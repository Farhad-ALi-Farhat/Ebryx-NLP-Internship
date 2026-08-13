# Limitations of BoW & TF-IDF — Context Loss

BoW (Bag-of-Words) and TF-IDF are useful techniques for converting text into numerical features, but they have an important limitation: **they lose much of the context and semantic meaning of language**.

---

## 1. Loss of Word Order

Consider:

```text
The dog chased the cat.
The cat chased the dog.
```

Both sentences contain the same words with the same frequencies.

A basic BoW representation would therefore be identical:

```text
["the", "dog", "chased", "cat"]
→ [2, 1, 1, 1]
```

TF-IDF has the same fundamental limitation. It changes the importance assigned to words but does not inherently encode their order.

Therefore, these methods cannot distinguish between:

```text
dog → chased → cat
```

and:

```text
cat → chased → dog
```

This is an example of **context loss**.

---

## 2. No Understanding of Semantics

Consider:

```text
I purchased a car.
I bought an automobile.
```

Humans recognize that these sentences have very similar meanings.

BoW and TF-IDF, however, treat:

```text
purchased ≠ bought
car ≠ automobile
```

as different features.

They do not inherently know that these words are semantically related.

---

## 3. No Inherent Relationship Between Similar Words

Suppose the vocabulary contains:

```text
king
queen
```

A BoW representation might look like:

```text
king  → [0, 1, 0, 0, ...]
queen → [0, 0, 1, 0, ...]
```

The vectors are simply positions in the vocabulary.

There is no inherent notion that:

* king and queen are related
* king and man are related
* king and queen are more similar than king and banana

This differs from word embeddings, where semantic relationships can be represented through positions in a continuous vector space.

---

## 4. Polysemy

A single word can have multiple meanings.

Consider:

```text
I deposited money in the bank.

The airplane landed near the river bank.
```

The word **bank** has different meanings in the two sentences.

BoW and TF-IDF assign the same feature to `bank` in both cases:

```text
bank → same feature
```

They do not inherently change the representation based on surrounding words.

Contextual embeddings, such as those produced by transformer models, can represent the word differently depending on its context.

---

## 5. High-Dimensional Sparse Vectors

Suppose a corpus contains:

```text
50,000 unique words
```

A document could be represented using a 50,000-dimensional vector:

```text
[0, 0, 0, 0, 1, 0, 0, ..., 0]
```

Only a small fraction of the values may be non-zero.

This creates:

* **High dimensionality**
* **Sparse representations**
* Large feature spaces for large vocabularies

Embeddings address this by representing text using relatively small **dense vectors**.

For example:

```text
50,000-dimensional sparse vector
              ↓
      300-dimensional
        dense vector
```

---

## 6. Vocabulary and OOV Problems

BoW and TF-IDF depend on the vocabulary extracted from the training corpus.

For example:

```text
Training vocabulary:
computer
laptop
phone
```

A new document might contain:

```text
smartphone
```

If `smartphone` was not present in the training vocabulary, it may become an **out-of-vocabulary (OOV)** word.

Traditional word-level representations generally cannot infer its meaning simply from the fact that it contains the substring `phone`.

Methods such as **FastText** improve this by using subword information.

---

## 7. TF-IDF Does Not Solve Context Loss

TF-IDF is an improvement over raw word counts because it measures the relative importance of words.

It essentially answers:

> How important is this word to this document compared with the rest of the corpus?

It does **not** answer:

> What does this word mean given the surrounding words?

Therefore:

```text
BoW
 ↓
Word frequency

TF-IDF
 ↓
Word importance

Word Embeddings
 ↓
Semantic relationships

Contextual Embeddings
 ↓
Meaning depends on context
```

---

## 8. Overall Comparison

| Representation      | Word Order | Semantic Relationships | Context-Dependent Meaning | Representation |
| ------------------- | ---------- | ---------------------- | ------------------------- | -------------- |
| BoW                 | ❌          | ❌                      | ❌                         | Sparse         |
| TF-IDF              | ❌          | ❌                      | ❌                         | Sparse         |
| Word2Vec            | Limited    | ✅                      | ❌                         | Dense          |
| GloVe               | Limited    | ✅                      | ❌                         | Dense          |
| FastText            | Limited    | ✅                      | ❌                         | Dense          |
| BERT / Transformers | ✅ Better   | ✅                      | ✅                         | Dense          |

---

## Key Takeaway

The progression of text representations can be summarized as:

```text
Raw Text
   ↓
BoW
   ↓
Word Counts
   ↓
TF-IDF
   ↓
Word Importance
   ↓
Word Embeddings
   ↓
Semantic Relationships
   ↓
Contextual Embeddings
   ↓
Meaning Depends on Context
```

The fundamental motivation for embeddings is:

> **Instead of representing words as independent features, represent them as dense vectors that can capture relationships and semantic information.**
