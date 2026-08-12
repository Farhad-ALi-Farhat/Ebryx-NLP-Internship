# Text Preprocessing

Raw text is difficult for machine-learning models to process directly. **Text preprocessing** transforms raw language into a more structured representation that can be used by NLP models.

A typical classical NLP pipeline looks like:

```text
Raw Text
   ↓
Tokenization
   ↓
Normalization
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
N-grams
   ↓
BoW / TF-IDF
   ↓
Numerical Features
   ↓
ML Model
```

Not every NLP project needs every step. Modern Transformer pipelines also handle preprocessing differently.

---

## 1. Tokenization

**Tokenization** is the process of splitting text into smaller units called **tokens**.

For example:

```text
"I love natural language processing."
```

can be tokenized into:

```text
["I", "love", "natural", "language", "processing", "."]
```

### Word Tokenization

Splits text into individual words or tokens.

```text
"I enjoy NLP"
        ↓
["I", "enjoy", "NLP"]
```

### Sentence Tokenization

Splits text into individual sentences.

```text
"I love NLP. It is fascinating!"
```

becomes:

```text
[
    "I love NLP.",
    "It is fascinating!"
]
```

### Subword Tokenization

Modern Transformer models commonly use **subword tokenization**.

A word can be represented using multiple smaller pieces.

Conceptually:

```text
"unhappiness"
      ↓
["un", "happiness"]
```

The exact splitting depends on the tokenizer.

Tokenization is important because models ultimately operate on **tokens rather than raw text**.

---

# 2. Lemmatization

**Lemmatization** reduces words to their **base or dictionary form**, called a lemma.

Examples:

```text
running  → run
ran      → run
cars     → car
```

Lemmatization attempts to consider the **meaning and grammatical role** of a word.

### Lemmatization vs Stemming

These are often confused.

**Stemming** usually applies simple rules:

```text
studies  → studi
running  → run
```

The result does not necessarily have to be a valid word.

**Lemmatization** attempts to produce a meaningful dictionary form:

```text
studies  → study
running  → run
```

Therefore:

> **Stemming is generally faster and simpler, while lemmatization is more linguistically informed.**

---

# 3. Stopwords

**Stopwords** are common words that are sometimes removed because they may contribute relatively little information for a particular NLP task.

Examples in English include:

```text
the
is
a
an
and
of
to
in
```

For example:

```text
"The cat is sitting on the table."
```

could become:

```text
["cat", "sitting", "table"]
```

after stopword removal.

### Why Remove Stopwords?

In traditional NLP approaches, removing very frequent words can:

* Reduce the number of features
* Reduce computational cost
* Remove some low-information words
* Make classical models more efficient

However, **stopwords should not automatically be removed**.

Consider:

```text
"I do not like this movie."
```

If `"not"` is removed:

```text
"I do like this movie."
```

The meaning changes completely.

Therefore, whether stopwords should be removed depends on the task.

---

# 4. Bag of Words (BoW)

**Bag of Words (BoW)** represents a document using the frequency of its words.

The key idea is:

> **Represent a document based on which words occur and how frequently they occur, while ignoring word order.**

Consider:

```text
Document 1: "I love cats"
Document 2: "I love dogs"
```

Vocabulary:

```text
["I", "love", "cats", "dogs"]
```

BoW representation:

```text
Document 1 → [1, 1, 1, 0]
Document 2 → [1, 1, 0, 1]
```

Each position corresponds to a word in the vocabulary.

### Example with Word Counts

Suppose:

```text
Document:
"cat dog cat"
```

Vocabulary:

```text
["cat", "dog"]
```

BoW becomes:

```text
[2, 1]
```

because:

```text
cat → 2
dog → 1
```

### Limitation of BoW

BoW ignores word order.

Consider:

```text
"The dog chased the cat."
"The cat chased the dog."
```

The words are almost identical, but the meanings are different.

BoW largely treats them as equivalent because it doesn't encode relationships between words.

This limitation motivates techniques such as **N-grams and word embeddings**.

---

# 5. TF-IDF

**TF-IDF** stands for:

> **Term Frequency–Inverse Document Frequency**

It improves upon simple word-count representations by considering not only how frequently a word occurs in a document, but also **how common that word is across the entire collection of documents**.

The intuition is:

> A word is important if it appears frequently in a particular document but doesn't appear frequently across all documents.

---

## Term Frequency (TF)

TF measures how frequently a term occurs in a document.

A common formulation is:

$$
TF(t,d) = \frac{\text{Number of occurrences of }t\text{ in }d}{\text{Total number of terms in }d}
$$

For example:

```text
Document:
"cat cat dog"
```

For `cat`:

$$
TF(cat) = \frac{2}{3}
$$

For `dog`:

$$
TF(dog) = \frac{1}{3}
$$

---

## Inverse Document Frequency (IDF)

IDF measures how rare a term is across the entire collection of documents.

A common formulation is:

$$
IDF(t,D) = \log\left(\frac{N}{df(t)}\right)
$$

where:

* $N$ = total number of documents
* $df(t)$ = number of documents containing term $t$

A word appearing in many documents receives a **lower IDF**.

A rare word receives a **higher IDF**.

---

## TF-IDF Score

The two components are combined:

$$
TFIDF(t,d,D) = TF(t,d) \times IDF(t,D)
$$

A word receives a high TF-IDF score when it:

* Appears frequently in a particular document
* Is relatively uncommon across the overall corpus

This makes TF-IDF useful for identifying **important words within documents**.

### BoW vs TF-IDF

| Feature                         | BoW | TF-IDF |
| ------------------------------- | --- | ------ |
| Word frequency                  | ✓   | ✓      |
| Considers corpus-wide frequency | ✗   | ✓      |
| Downweights common words        | ✗   | ✓      |
| Captures semantic meaning       | ✗   | ✗      |
| Considers word order            | ✗   | ✗      |

TF-IDF is still a **sparse lexical representation**. It does not understand that:

```text
"car"
```

and

```text
"automobile"
```

have similar meanings.

That limitation leads us toward **word embeddings**.

---

# 6. N-grams

An **N-gram** is a sequence of **N consecutive tokens**.

N-grams allow classical NLP models to capture some information about word order.

## Unigram

A unigram contains one token.

```text
"I love NLP"
```

produces:

```text
["I", "love", "NLP"]
```

## Bigram

A bigram contains two consecutive tokens:

```text
["I love", "love NLP"]
```

## Trigram

A trigram contains three consecutive tokens:

```text
["I love NLP"]
```

For:

```text
"I really love NLP"
```

we get:

### Unigrams

```text
I
really
love
NLP
```

### Bigrams

```text
I really
really love
love NLP
```

### Trigrams

```text
I really love
really love NLP
```

### Why Are N-grams Useful?

Consider:

```text
"I love this movie"
```

A unigram model sees:

```text
I
love
this
movie
```

A bigram model additionally captures:

```text
"I love"
"love this"
"this movie"
```

This provides some information about local word order.

N-grams are commonly combined with **BoW or TF-IDF**:

```text
Text
 ↓
Tokenization
 ↓
N-gram Generation
 ↓
TF-IDF
 ↓
Feature Matrix
 ↓
Machine Learning Model
```

---

# Putting Everything Together

A traditional NLP pipeline might look like:

```text
Raw Text
    ↓
Tokenization
    ↓
Stopword Removal
    ↓
Lemmatization
    ↓
N-gram Generation
    ↓
BoW / TF-IDF
    ↓
Numerical Feature Matrix
    ↓
Machine Learning Model
```

For example:

```text
"I really loved these movies!"
```

could be processed approximately as:

```text
Raw text
    ↓
"I really loved these movies!"
    ↓
Tokenization
    ↓
["I", "really", "loved", "these", "movies"]
    ↓
Stopword removal
    ↓
["really", "loved", "movies"]
    ↓
Lemmatization
    ↓
["really", "love", "movie"]
    ↓
TF-IDF
    ↓
[0.31, 0.74, 0.62, ...]
```

The exact result depends on the preprocessing library, vocabulary, corpus, and configuration.

---

# Traditional NLP vs Modern NLP

These techniques are fundamental for understanding **classical NLP**, but not every modern NLP system uses this exact pipeline.

### Traditional NLP

```text
Text
 ↓
Tokenization
 ↓
Cleaning
 ↓
BoW / TF-IDF
 ↓
ML Algorithm
```

### Modern Transformer-Based NLP

```text
Text
 ↓
Subword Tokenization
 ↓
Token IDs
 ↓
Embeddings
 ↓
Transformer
 ↓
Contextual Representations
 ↓
Task Output
```

This distinction becomes important when moving from:

```text
TF-IDF
   ↓
Word Embeddings
   ↓
Contextual Embeddings
   ↓
Transformers
   ↓
Large Language Models
```

---

# Key Takeaways

* **Tokenization** splits text into smaller units.
* **Lemmatization** converts words into meaningful base forms.
* **Stopword removal** removes selected high-frequency words when appropriate.
* **BoW** represents text using word frequencies.
* **TF-IDF** assigns greater importance to words that are frequent in a document but uncommon across the corpus.
* **N-grams** capture local sequences of words.
* BoW and TF-IDF are **sparse lexical representations**.
* BoW and TF-IDF do not capture deep semantic relationships.
* These limitations motivate the transition to **dense word embeddings**.
