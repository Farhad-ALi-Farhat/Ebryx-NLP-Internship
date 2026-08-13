# Distributed Representations: Word2Vec, GloVe, FastText

Distributed representations are a major step beyond traditional text representations such as **Bag-of-Words (BoW)** and **TF-IDF**.

Instead of representing each word as a sparse vector based on vocabulary positions, distributed representations encode words as **dense vectors** whose dimensions collectively capture useful linguistic patterns and relationships.

---

## 1. What Is a Distributed Representation?

In BoW and TF-IDF, a word can be represented using a large sparse vector where each dimension corresponds to a vocabulary item.

For example:

```text
cat → [0, 0, 1, 0, 0, 0, ...]
dog → [0, 1, 0, 0, 0, 0, ...]
```

These representations tell us which vocabulary positions are present, but they do not inherently encode semantic relationships.

A distributed representation instead represents a word using a relatively small **dense vector**:

```text
cat → [0.21, -0.45, 0.73, 0.12, ...]
dog → [0.24, -0.41, 0.69, 0.15, ...]
```

The meaning is **distributed across multiple dimensions**.

No individual dimension necessarily represents a specific concept. Instead, the combination of dimensions captures patterns learned from the training corpus.

---

# 2. The Distributional Hypothesis

Word embeddings are largely based on the **distributional hypothesis**:

> Words that occur in similar contexts tend to have similar meanings.

For example:

```text
The cat drank milk.
The cat chased the mouse.

The dog drank milk.
The dog chased the mouse.
```

`cat` and `dog` appear in similar contexts.

Therefore, an embedding model can learn representations where:

```text
similarity(cat, dog) → high
```

while an unrelated word such as `airplane` may have a less similar representation.

This idea forms the foundation of **Word2Vec, GloVe, and FastText**.

---

# 3. Word2Vec

**Word2Vec** is a family of neural-network-based techniques for learning word embeddings.

Instead of explicitly defining the meaning of a word, Word2Vec learns representations by using words and their surrounding context.

There are two main architectures:

```text
Word2Vec
├── CBOW
└── Skip-gram
```

---

## 3.1 CBOW — Continuous Bag of Words

CBOW predicts a **target word from its surrounding context**.

For example:

```text
The cat drinks milk
```

If `cat` is the target:

```text
Context → Target

The + drinks → cat
```

More generally:

```text
surrounding words
       ↓
     model
       ↓
   target word
```

The model is trained on many such examples and learns word representations that help it predict likely target words.

### CBOW Characteristics

* Predicts the target word from surrounding context.
* Generally faster to train than Skip-gram.
* Works particularly well with frequent words.
* Learns representations from local context.

---

# 4. Skip-gram

Skip-gram reverses the CBOW task.

Instead of:

```text
Context → Target
```

it learns:

```text
Target → Context
```

For example:

```text
The cat drinks milk
```

If `cat` is the target:

```text
             cat
              ↓
       ┌──────┼──────┐
       ↓      ↓      ↓
      The   drinks   ...
```

The model attempts to predict words that occur around `cat`.

### Skip-gram Characteristics

* Predicts surrounding context words from a target word.
* Often performs well with smaller datasets.
* Can represent rare words better than CBOW.
* Generally requires more computation than CBOW.

---

# 5. How Word2Vec Learns Meaning

Word2Vec does not contain a predefined dictionary of word meanings.

Instead, it learns vectors through a prediction task.

After training, the learned parameters can be used as word embeddings.

This can result in meaningful relationships between vectors.

For example, a trained embedding space may exhibit relationships such as:

```text
king - man + woman ≈ queen
```

This is not a guaranteed linguistic rule. It demonstrates that relationships can emerge naturally in the learned vector space.

---

# 6. GloVe

**GloVe** stands for:

> **Global Vectors for Word Representation**

GloVe learns word vectors using **global word co-occurrence statistics**.

The central idea is:

> Words that have similar patterns of co-occurrence with other words should have similar representations.

For example:

```text
cat → milk, mouse, animal, pet
dog → milk, bone, animal, pet
```

Because `cat` and `dog` have similar co-occurrence patterns, their vectors can become similar.

---

# 7. Word2Vec vs GloVe

The main conceptual difference is the information they emphasize:

| Word2Vec                        | GloVe                                |
| ------------------------------- | ------------------------------------ |
| Prediction-based                | Co-occurrence-based                  |
| Learns from context windows     | Uses global co-occurrence statistics |
| CBOW / Skip-gram                | Global matrix-style objective        |
| Neural prediction objective     | Factorization-style objective        |
| Strong semantic representations | Strong semantic representations      |

A useful mental model is:

```text
Word2Vec:
"Can I predict nearby words?"

GloVe:
"What words tend to occur together across the corpus?"
```

Both can produce high-quality **static word embeddings**.

---

# 8. FastText

**FastText** was developed by Facebook AI Research.

Its major improvement over traditional word-level embeddings is the use of **subword information**.

Instead of treating a word as an indivisible unit, FastText represents it using smaller character n-grams.

For example:

```text
playing
```

can be represented using subword components such as:

```text
play
lay
ayi
yin
ing
...
```

The exact n-grams depend on the implementation and boundary markers.

---

# 9. Why Subword Information Matters

Consider:

```text
play
playing
played
player
playful
```

These words share meaningful character patterns.

FastText can use these shared subword components when constructing word representations.

This makes FastText particularly useful for:

* Rare words
* Morphologically rich languages
* Unseen words
* Words with meaningful prefixes and suffixes
* Some spelling variations

---

# 10. FastText and OOV Words

Suppose a word was not explicitly encountered during training:

```text
unhappiness
```

A traditional word-level embedding may not have a vector for it.

FastText can construct a representation using subword information:

```text
unhappiness
      ↓
character/subword representations
      ↓
combined representation
      ↓
word vector
```

This gives FastText an important advantage over traditional Word2Vec and GloVe embeddings.

---

# 11. Static vs Contextual Embeddings

Word2Vec, GloVe, and FastText are generally **static embeddings**.

A word typically has one learned vector regardless of the sentence in which it appears.

Consider:

```text
I deposited money in the bank.

I sat beside the river bank.
```

The word `bank` has different meanings in these sentences.

Traditional static embeddings generally represent:

```text
bank → one vector
```

rather than dynamically changing the representation based on the surrounding sentence.

This limitation leads to the development of **contextual embeddings**, such as those used by BERT and other transformer models.

---

# 12. Comparison

| Feature                   | Word2Vec   | GloVe      | FastText   |
| ------------------------- | ---------- | ---------- | ---------- |
| Dense vectors             | ✅          | ✅          | ✅          |
| Semantic relationships    | ✅          | ✅          | ✅          |
| Local context             | ✅          | Indirectly | ✅          |
| Global statistics         | Indirectly | ✅          | Indirectly |
| Subword information       | ❌          | ❌          | ✅          |
| Handles rare words well   | Somewhat   | Somewhat   | ✅          |
| Better OOV handling       | ❌          | ❌          | ✅          |
| Context-dependent vectors | ❌          | ❌          | ❌          |
| Static embeddings         | ✅          | ✅          | ✅          |

---

# 13. Overall Progression

The development of text representations can be viewed as:

```text
                         TEXT
                           │
            ┌──────────────┴──────────────┐
            ↓                             ↓
       BoW / TF-IDF                Word Embeddings
            │                             │
         Sparse                          Dense
            │                             │
     Word frequency              Semantic information
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
                Word2Vec                GloVe                FastText
                    │                     │                     │
               Prediction             Co-occurrence          Subwords
                    │                     │                     │
                    └─────────────────────┼─────────────────────┘
                                          ↓
                                  Static Embeddings
                                          │
                                          ↓
                              Contextual Embeddings
                                          │
                                          ↓
                                   BERT / Transformers
```

---

# 14. Key Takeaways

### BoW / TF-IDF

> Which words are present, and how important are they?

### Word2Vec

> Which words occur in similar contexts?

### GloVe

> How do words co-occur across the corpus?

### FastText

> What can the word's subword structure tell us?

### Contextual Embeddings

> What does this word mean **in this particular context**?

The major conceptual progression is:

```text
Sparse lexical features
        ↓
Dense semantic representations
        ↓
Context-aware representations
```

Word2Vec, GloVe, and FastText form the bridge between **traditional NLP feature engineering** and **modern transformer-based NLP**.
