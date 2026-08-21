# Self-Attention and Transformer Architecture

## 1. Why Self-Attention?

Before Transformers, sequence models such as **RNNs, LSTMs, and GRUs** processed text sequentially.

For example:

```text
The → cat → sat → on → the → mat
```

This made it difficult to efficiently capture relationships between distant words.

Consider:

> The animal didn't cross the street because **it** was tired.

To understand `it`, the model needs to connect it with `animal`.

**Self-attention** allows every token to directly consider other tokens in the sequence.

```text
The      ────────────────┐
animal ──────────────────┤
didn't ──────────────────┤
cross ───────────────────┤
street ──────────────────┤
it ──────────────────────┤
was ─────────────────────┤
tired ───────────────────┘
             ↓
        Self-Attention
```

---

## 2. What Is Self-Attention?

Self-attention is a mechanism that allows each token to determine **which other tokens are important for understanding it**.

For example:

> The cat sat on the mat because **it** was tired.

When processing `it`, attention can assign different importance to the other tokens:

```text
it
│
├── The       low
├── cat       high
├── sat       medium
├── on        low
├── the       low
├── mat       medium
├── because   low
└── tired     high
```

The model learns these relationships automatically.

The result is a new representation of `it` that incorporates relevant information from the surrounding context.

---

## 3. Query, Key, and Value

Self-attention uses three representations:

* **Query (Q)** — what information this token is looking for
* **Key (K)** — what information each token offers for matching
* **Value (V)** — the information that gets passed forward

A simplified view:

```text
Token
  │
  ├────────→ Query
  ├────────→ Key
  └────────→ Value
```

Attention first compares the query with the keys to determine relevance.

Conceptually:

```text
Query
  │
  ↓
Compare with Keys
  │
  ↓
Attention Scores
  │
  ↓
Softmax
  │
  ↓
Weighted Values
  │
  ↓
New Representation
```

---

## 4. Scaled Dot-Product Attention

The standard attention equation is:


$Attention(Q,K,V)$
================

$$
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Where:

* $Q$ = Query matrix
* $K$ = Key matrix
* $V$ = Value matrix
* $d_k$ = dimensionality of the keys

The important intuition is:

> **Compare queries with keys → determine importance → use those importance scores to combine values.**

---

## 5. Self-Attention Example

Consider:

> The dog chased the cat.

For the token `dog`, the model calculates relationships with the other tokens:

```text
             dog
              │
       ┌──────┼──────┐
       ↓      ↓      ↓
      The   chased   cat
       │      │       │
      low    high   medium
```

The exact attention patterns are **learned by the model**.

The resulting representation of `dog` contains information not only about `dog`, but also about its context.

This is what makes the resulting representation **contextual**.

---

## 6. Multi-Head Attention

Transformers don't rely on a single attention mechanism.

They use **Multi-Head Attention**.

```text
                 Input
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
     Head 1      Head 2     Head 3
        │          │          │
        ↓          ↓          ↓
   Relationship  Relationship Relationship
        │          │          │
        └──────────┼──────────┘
                   ↓
             Concatenate
                   ↓
             Linear Layer
```

Different attention heads can learn different types of relationships.

For example, one head might focus more on:

```text
subject ↔ verb
```

while another might capture:

```text
pronoun ↔ noun
```

and another:

```text
word ↔ nearby context
```

The model learns what each head should specialize in.

---

## 7. Why Positional Information Is Needed

There's an important problem.

Self-attention looks at tokens simultaneously, so the model needs some way to know their **order**.

These sentences contain the same words:

```text
The dog chased the cat.
The cat chased the dog.
```

But they have different meanings.

Therefore Transformers add **positional information** to token representations.

Conceptually:

```text
Token        Position
----------------------
The             1
dog             2
chased          3
the             4
cat             5
```

Different Transformer architectures use different techniques for representing position, including:

* Learned positional embeddings
* Sinusoidal positional encodings
* Rotary Positional Embeddings (RoPE)

---

## 8. Transformer Encoder Block

A simplified Transformer encoder block looks like this:

```text
Input
  │
  ↓
Multi-Head Self-Attention
  │
  ↓
Add & Normalize
  │
  ↓
Feed-Forward Network
  │
  ↓
Add & Normalize
  │
  ↓
Output
```

The **Feed-Forward Network (FFN)** applies additional learned transformations to each token's representation.

The attention mechanism primarily allows tokens to **communicate with each other**, while the feed-forward network further processes the resulting representations.

---

## 9. Multiple Transformer Layers

A Transformer isn't just one block.

Many blocks are stacked together:

```text
Input Tokens
     ↓
┌─────────────────────┐
│ Transformer Layer 1 │
└─────────────────────┘
     ↓
┌─────────────────────┐
│ Transformer Layer 2 │
└─────────────────────┘
     ↓
┌─────────────────────┐
│ Transformer Layer 3 │
└─────────────────────┘
     ↓
        ...
     ↓
┌─────────────────────┐
│ Transformer Layer N │
└─────────────────────┘
     ↓
Contextual Representations
```

As information passes through the layers, the model can build increasingly sophisticated representations.

---

## 10. Original Transformer Architecture

The original Transformer architecture contains two major components:

```text
                Transformer
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
       Encoder               Decoder
          │                     │
          ↓                     ↓
   Understand Input       Generate Output
```

### Encoder

The encoder processes the input and produces contextual representations.

Useful for tasks such as:

* Classification
* Named Entity Recognition
* Semantic similarity
* Feature extraction

### Decoder

The decoder generates output tokens.

Useful for:

* Text generation
* Translation
* Summarization
* Conversational AI

---

## 11. Encoder-Only vs Decoder-Only vs Encoder-Decoder

Modern Transformer models can use different parts of the architecture.

### Encoder-Only

Examples:

* BERT
* RoBERTa

```text
Input
  ↓
Encoder
  ↓
Contextual Representations
  ↓
Task
```

Primarily focused on **understanding text**.

### Decoder-Only

Examples:

* GPT-style models

```text
Input
  ↓
Decoder
  ↓
Next-Token Prediction
  ↓
Generated Text
```

Primarily focused on **generating text**.

### Encoder-Decoder

Examples:

* T5
* BART

```text
Input
  ↓
Encoder
  ↓
Representation
  ↓
Decoder
  ↓
Output
```

Useful for sequence-to-sequence tasks.

---

## 12. Why Transformers Were a Major Breakthrough

Compared with RNNs/LSTMs:

| Feature                    | RNN/LSTM  | Transformer   |
| -------------------------- | --------- | ------------- |
| Sequential processing      | ✅         | ❌             |
| Self-attention             | ❌         | ✅             |
| Long-range relationships   | Difficult | Strong        |
| Parallel processing        | Limited   | Excellent     |
| Training efficiency        | Lower     | Higher        |
| Contextual representations | Yes       | Very powerful |
| Foundation of modern LLMs  | ❌         | ✅             |

The major conceptual shift was:

### RNN/LSTM

```text
token → token → token → token
              ↓
         hidden state
```

### Transformer

```text
token ←→ token ←→ token ←→ token
   ↕        ↕        ↕
token ←→ token ←→ token ←→ token

        Self-Attention
```

Tokens can directly interact with other tokens instead of relying on information being passed sequentially through hidden states.

---

## 13. Overall Transformer Pipeline

At a high level:

```text
Raw Text
   ↓
Tokenization
   ↓
Token IDs
   ↓
Token Embeddings
   +
Positional Information
   ↓
Multi-Head Self-Attention
   ↓
Add & Normalize
   ↓
Feed-Forward Network
   ↓
Add & Normalize
   ↓
Repeat Transformer Layers
   ↓
Contextual Representations
   ↓
Task-Specific Output
```

---

## 14. Big Picture

The progression is:

```text
Static Embeddings
      ↓
"word" → fixed vector

Contextual Embeddings
      ↓
"word + context" → context-dependent vector

Transformers
      ↓
Self-attention allows tokens to
interact with one another

BERT / GPT / T5
      ↓
Large pretrained Transformer models
```

### Key Takeaway

> **Self-attention allows every token to selectively use information from other tokens, producing representations whose meaning depends on the entire context.**

This ability to model relationships between tokens efficiently is one of the main reasons **Transformers became the foundation of modern NLP and Large Language Models (LLMs).**
