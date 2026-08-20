# Semantic Relationships and Vector Arithmetic

Word embeddings can represent more than simple word similarity.

Because words are represented as vectors in a multidimensional space, relationships between words can sometimes be captured by the **geometry of that space**.

This allows us to perform operations such as **vector arithmetic** and solve analogy-style problems.

---

# 1. Similarity vs Relationship

Cosine similarity can tell us whether two words have similar representations.

For example:

```text
cat ↔ dog
```

may have high similarity because both are animals and occur in similar contexts.

However, embeddings can also capture relationships such as:

```text
king ↔ queen
man ↔ woman
Paris ↔ France
Rome ↔ Italy
```

These aren't necessarily synonyms.

They represent different types of semantic relationships:

```text
king   → queen      = gender relationship
man    → woman      = gender relationship

Paris  → France     = capital → country
Rome   → Italy      = capital → country
```

---

# 2. Vector Arithmetic

One of the most famous examples of word embedding arithmetic is:

$$
\text{king} - \text{man} + \text{woman}
\approx
\text{queen}
$$

Using vectors:

$$
\vec{king}-\vec{man}+\vec{woman}
\approx
\vec{queen}
$$

The intuition is:

```text
king
 ↓
remove the "man" component
 ↓
add the "woman" component
 ↓
queen
```

This demonstrates that certain relationships can be represented through directions in the embedding space.

---

# 3. Relationship as a Vector Difference

Suppose:

```text
A → B
```

represents some semantic relationship.

That relationship can be represented by:

$$
\vec{B}-\vec{A}
$$

For example:

$$
\vec{king}-\vec{man}
$$

may represent a particular relationship.

If another pair has a similar relationship:

$$
\vec{queen}-\vec{woman}
$$

then we may find:

$$
\vec{king}-\vec{man}
\approx
\vec{queen}-\vec{woman}
$$

Therefore:

$$
\vec{king}-\vec{man}+\vec{woman}
\approx
\vec{queen}
$$

---

# 4. Geometric Interpretation

The actual embedding space may have hundreds of dimensions, but we can visualize the idea in a simplified space:

```text
          queen
            ↑
            |
           king
            |
            |
           man
            |
            ↓
          woman
```

The important concept is not the exact visual arrangement.

Instead, think of a semantic relationship as a **direction**:

```text
word A ─────────────→ word B
          relationship
```

If another pair exhibits a similar relationship:

```text
word C ─────────────→ word D
          relationship
```

then:

$$
\vec{B}-\vec{A}
\approx
\vec{D}-\vec{C}
$$

This is the basic geometric intuition behind embedding analogies.

---

# 5. Capital-Country Relationships

Another common example involves geographical relationships.

Suppose:

```text
Paris → France
Rome  → Italy
```

A trained embedding space may approximately represent:

$$
\vec{Paris}-\vec{France}
\approx
\vec{Rome}-\vec{Italy}
$$

This suggests that the relationship:

```text
capital → country
```

can sometimes appear as a similar direction.

Therefore:

$$
\vec{Paris}-\vec{France}+\vec{Italy}
\approx
\vec{Rome}
$$

Again, this is an approximate property rather than a guaranteed mathematical rule.

---

# 6. Analogy Tasks

Vector arithmetic can be used for analogy questions.

The classic structure is:

```text
A : B :: C : ?
```

This means:

> A is to B as C is to what?

For example:

```text
king : queen :: man : ?
```

We can calculate:

$$
\vec{king}-\vec{man}+\vec{woman}
$$

The resulting vector should ideally be close to:

```text
queen
```

---

# 7. Finding the Answer

The resulting vector does not necessarily correspond exactly to an existing word.

Instead, we search for the vocabulary vector that is most similar to it.

Conceptually:

```text
king - man + woman
          ↓
      target vector
          ↓
  compare with vocabulary
          ↓
    cosine similarity
          ↓
    nearest vectors
          ↓
        queen
```

For example:

```text
target vector

queen       → 0.92
princess    → 0.84
royal       → 0.71
monarch     → 0.68
...
```

The word with the highest similarity can be selected as the predicted answer.

---

# 8. Semantic Relationship ≠ Synonymy

A high embedding similarity does not necessarily mean two words are synonyms.

For example:

```text
doctor ↔ hospital
```

may have high similarity because they frequently occur in related contexts.

However:

```text
doctor ≠ hospital
```

They are related concepts rather than synonyms.

Similarly:

```text
Paris ↔ France
```

represents a strong semantic relationship, but the words have completely different meanings.

Therefore, embeddings can capture:

* Similarity
* Relatedness
* Associations
* Analogies
* Semantic relationships

not just synonymy.

---

# 9. Why Does Vector Arithmetic Work?

Word embeddings are trained on large amounts of text.

During training, words that occur in related contexts tend to develop related representations.

As a result, certain linguistic patterns can emerge as approximately consistent directions in the vector space.

For example:

$$
\vec{king}-\vec{man}
\approx
\vec{queen}-\vec{woman}
$$

The model was not necessarily explicitly instructed:

> "Create a gender dimension."

Instead, statistical patterns in the training data can cause such relationships to emerge naturally.

This is an example of an **emergent property of the learned representation**.

---

# 10. Limitations

Vector arithmetic is useful, but it is not perfect.

## 10.1 Not Every Relationship Is Linear

Some semantic relationships cannot be represented accurately using a single vector direction.

---

## 10.2 Training Data Bias

Embeddings learn patterns from their training data.

If the training corpus contains social or cultural biases, those patterns can also appear in the embedding space.

---

## 10.3 Different Models Produce Different Results

Word2Vec, GloVe, and FastText trained on different datasets can produce different embedding spaces.

Therefore, a relationship that works well in one model may not work as well in another.

---

## 10.4 Static Embedding Limitation

Traditional Word2Vec, GloVe, and FastText embeddings are generally **static**.

A word typically has one vector regardless of its context.

For example:

```text
I deposited money in the bank.

I sat beside the river bank.
```

The word `bank` has two different meanings, but a traditional static embedding generally provides one vector:

```text
bank → one vector
```

This limitation motivates the use of **contextual embeddings** such as BERT.

---

# 11. Overall Progression

The concepts can be connected as follows:

```text
Word
 ↓
Embedding Vector
 ↓
Position in Vector Space
 ↓
Cosine Similarity
 ↓
Semantic Similarity
 ↓
Vector Differences
 ↓
Semantic Relationships
 ↓
Vector Arithmetic
 ↓
Analogy / Relationship Prediction
```

---

# 12. Key Takeaways

* Word embeddings represent words as vectors.
* The geometry of the vector space can encode semantic relationships.
* Vector subtraction can represent relationships between words.
* Vector addition can transfer those relationships to other words.
* Cosine similarity can be used to find the nearest vocabulary words.
* Analogy problems can be expressed using vector arithmetic.
* Semantic relatedness does not necessarily mean synonymy.
* Vector arithmetic is approximate and depends heavily on the training data and embedding model.
* Static embeddings have limitations because a word generally has one vector regardless of context.

The famous example:

$$
\boxed{
\text{king} - \text{man} + \text{woman}
\approx
\text{queen}
}
$$

should be understood as an example of **linguistic relationships emerging as approximately linear patterns in a learned embedding space**.
