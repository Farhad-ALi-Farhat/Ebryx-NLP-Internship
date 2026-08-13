# Word Embeddings: Vectors & Cosine Similarity

Word embeddings represent words as **dense numerical vectors** in a multidimensional space.

Once words are represented as vectors, we can compare those vectors to measure how similar their meanings or usage patterns are.

One of the most commonly used measures for this is **cosine similarity**.

---

# 1. Words as Vectors

Suppose an embedding model produces the following vectors:

```text
cat → [0.8, 0.2, 0.6, 0.1]
dog → [0.7, 0.3, 0.5, 0.2]
car → [-0.2, 0.9, -0.4, 0.7]
```

Each word is represented as a point/vector in a high-dimensional space.

Real-world embeddings are usually much larger:

```text
Word2Vec → 100, 200, 300 dimensions
GloVe    → 50, 100, 200, 300 dimensions
```

The individual dimensions usually do not have simple human-readable meanings.

Instead, the **combination of dimensions** encodes patterns learned from the training corpus.

---

# 2. Why Compare Embedding Vectors?

If two words occur in similar contexts, their vectors may be located close to each other in embedding space.

For example:

```text
             dog
            /
           /
         cat


                         car
```

Conceptually:

```text
similarity(cat, dog) → high
similarity(cat, car) → lower
```

We therefore need a mathematical method to compare vectors.

A common choice is **cosine similarity**.

---

# 3. Cosine Similarity

Cosine similarity measures the **angle between two vectors** rather than directly comparing their magnitudes.

For vectors $\mathbf{A}$ and $\mathbf{B}$:

$\text{cosine similarity}$
========================

$$
\frac{\mathbf{A}\cdot\mathbf{B}}
{|\mathbf{A}||\mathbf{B}|}
$$

The dot product is:


$\mathbf{A}\cdot\mathbf{B}$
=========================

$$
\sum_{i=1}^{n} A_iB_i
$$

The magnitude of a vector is:


$|\mathbf{A}|$
============

$$
\sqrt{\sum_{i=1}^{n} A_i^2}
$$

Therefore:


$\text{cosine similarity}$
========================

$$
\frac{\sum_{i=1}^{n}A_iB_i}
{\sqrt{\sum_{i=1}^{n}A_i^2}
\sqrt{\sum_{i=1}^{n}B_i^2}}
$$

---

# 4. Interpreting Cosine Similarity

Cosine similarity generally ranges from **-1 to 1**.

| Value | Interpretation                                      |
| ----: | --------------------------------------------------- |
|   `1` | Vectors point in the same direction                 |
|   `0` | Vectors are perpendicular / directionally unrelated |
|  `-1` | Vectors point in opposite directions                |

For many word-embedding applications, similarity values are primarily positive.

For example:

```text
cosine(cat, dog)      = 0.85
cosine(cat, kitten)   = 0.91
cosine(cat, airplane) = 0.12
```

A higher value indicates greater directional similarity.

---

# 5. Simple Example

Consider two vectors:

```text
A = [1, 2]
B = [2, 4]
```

`B` is simply a scaled version of `A`.

Both vectors point in exactly the same direction.

Therefore:

$$
\text{cosine similarity}(A,B)=1
$$

This demonstrates an important property of cosine similarity:

> It focuses on the **direction** of vectors rather than their magnitude.

---

# 6. Cosine Similarity in NLP

Suppose an embedding model produces:

```text
cat → [0.8, 0.2, 0.6]
dog → [0.7, 0.3, 0.5]
car → [-0.1, 0.9, -0.4]
```

We can calculate:

```text
similarity(cat, dog)
similarity(cat, car)
similarity(dog, car)
```

We would generally expect:

```text
cat ↔ dog
```

to have a higher similarity than:

```text
cat ↔ car
```

because the embedding model may have learned that `cat` and `dog` appear in similar linguistic contexts.

---

# 7. Finding Similar Words

Cosine similarity can be used to find the words closest to a given word in embedding space.

For example:

```text
Query:
"king"

      ↓

Compare with every word
in the vocabulary

      ↓

Top similar words:

queen
prince
royal
monarch
...
```

This is essentially a **nearest-neighbor search** over the embedding space.

---

# 8. Cosine Similarity vs Euclidean Distance

Another way to compare vectors is Euclidean distance.


$d(A,B)$
======
$$
\sqrt{\sum_{i=1}^{n}(A_i-B_i)^2}
$$

The main difference is:

### Euclidean Distance

Measures the actual distance between two points.

### Cosine Similarity

Measures the angle between two vectors.

For example:

```text
A = [1, 2]
B = [2, 4]
```

The vectors have different magnitudes, but they point in exactly the same direction.

Therefore:

```text
Cosine similarity = 1
```

This makes cosine similarity useful for many embedding-based comparisons.

---

# 9. Similarity Does Not Mean Synonymy

A high cosine similarity does **not necessarily mean that two words are synonyms**.

For example:

```text
doctor
hospital
```

may have high similarity because they frequently occur in related contexts.

However:

```text
doctor ≠ hospital
```

They are related concepts rather than synonyms.

This follows from the distributional hypothesis:

> Words that occur in similar contexts tend to have similar representations.

---

# 10. Vector Relationships

Word embeddings can also encode relationships through vector operations.

A famous example is:

$$
\text{king} - \text{man} + \text{woman}
\approx
\text{queen}
$$

The idea is that relationships between words can emerge from the geometry of the embedding space.

This is not a guaranteed mathematical rule for every embedding model or dataset. It is an example of the kinds of relationships that can emerge from training.

---

# 11. Practical Python Example

Using scikit-learn:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

cat = np.array([[0.8, 0.2, 0.6]])
dog = np.array([[0.7, 0.3, 0.5]])

similarity = cosine_similarity(cat, dog)

print(similarity)
```

The output will be a value close to `1`, indicating that the two vectors point in similar directions.

---

# 12. Overall Workflow

```text
Raw Text
   ↓
Train / Load Embedding Model
   ↓
Words become vectors
   ↓
Embedding Space
   ↓
Compare vectors
   ↓
Cosine Similarity
   ↓
Find semantically related words
```

---

# 13. Key Takeaways

* **Word embeddings** represent words as dense numerical vectors.
* Each word is represented by a point in a high-dimensional embedding space.
* Similar words can have similar vector representations.
* **Cosine similarity** measures the directional similarity between two vectors.
* Cosine similarity ranges from `-1` to `1`.
* A value closer to `1` indicates greater directional similarity.
* High similarity does not necessarily mean two words are synonyms.
* Embedding spaces can capture semantic and linguistic relationships.
* Cosine similarity is widely used for comparing embeddings.

The central idea is:

> **Words become vectors, and the geometry of those vectors can be used to measure relationships between words.**
