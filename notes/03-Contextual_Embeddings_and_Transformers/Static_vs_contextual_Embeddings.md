# Why Context Matters: Static vs Contextual Embeddings

The key problem with **static embeddings** is that they assign **one fixed vector to each word**, regardless of how that word is used.

## 1. Static Embeddings

Models like **Word2Vec, GloVe, and FastText** learn a single representation for each word.

For example:

```text
bank → [0.21, -0.14, 0.73, ...]
```

That same vector is used in:

> I deposited money in the bank.

and:

> We sat beside the river bank.

But **bank** has two different meanings:

```text
bank₁ → financial institution
bank₂ → side of a river
```

A static embedding cannot represent these meanings separately.

```text
"money in the bank"
          ↓
      bank → V

"river bank"
          ↓
      bank → V
```

The vector `V` is essentially the same.

---

## 2. Why Context Matters

Consider:

> The **bat** flew out of the cave.

and:

> He swung the **bat** at the ball.

The word `bat` refers to:

```text
Sentence 1 → animal
Sentence 2 → sports equipment
```

But a static embedding gives:

```text
bat → [same vector]
```

Therefore, the embedding itself doesn't tell us which meaning is intended.

---

## 3. Contextual Embeddings

**Contextual embeddings** solve this by making the representation depend on the surrounding words.

Conceptually:

```text
The bat flew out of the cave.
                 ↓
            Transformer
                 ↓
          bat → vector A
```

while:

```text
He swung the bat at the ball.
                 ↓
            Transformer
                 ↓
          bat → vector B
```

Therefore:

$$
\text{vector A} \neq \text{vector B}
$$

The model can encode the meaning of `bat` **within its particular context**.

---

## 4. Static vs Contextual Representation

### Static Embeddings

The mapping is approximately:

$$
f(\text{word}) \rightarrow \text{vector}
$$

For example:

$$
f(\text{bank}) = V_{bank}
$$

regardless of the sentence.

### Contextual Embeddings

The mapping becomes:

$$
f(\text{word}, \text{context}) \rightarrow \text{vector}
$$

So:

$$
f(\text{bank}, \text{financial context}) = V_1
$$

while:

$$
f(\text{bank}, \text{river context}) = V_2
$$

This is the fundamental difference.

---

## 5. Context Captures Relationships

Context isn't only about different meanings. It also helps capture **relationships between words**.

Consider:

> The dog chased the cat.

The representation of `dog` can incorporate information from:

```text
dog
 ↓
chased
 ↓
cat
```

Now change the sentence:

> The cat chased the dog.

The same words are present, but their **roles have changed**.

A contextual model can represent these differences because each token's representation is influenced by the other tokens.

---

## 6. Comparison

| Feature                            | Static Embedding          | Contextual Embedding |
| ---------------------------------- | ------------------------- | -------------------- |
| Vector depends on context?         | ❌ No                      | ✅ Yes                |
| One vector per word?               | ✅ Generally               | ❌ No                 |
| Handles multiple meanings well?    | ❌ Limited                 | ✅ Much better        |
| Captures contextual relationships? | Limited                   | Strong               |
| Examples                           | Word2Vec, GloVe, FastText | BERT, RoBERTa        |
| Uses surrounding tokens?           | ❌ No                      | ✅ Yes                |

---

## 7. Core Idea

**Static embeddings answer:**

> What does this word generally mean?

**Contextual embeddings answer:**

> What does this word mean **here**, in this sentence?

This ability to adapt a token's representation to its surrounding context is one of the major reasons **Transformer-based models** such as BERT became so powerful for NLP.

### Evolution

```text
Bag of Words
     ↓
TF-IDF
     ↓
Word2Vec / GloVe / FastText
     ↓
Contextual Embeddings
     ↓
Transformers
     ↓
BERT / GPT / T5
```
