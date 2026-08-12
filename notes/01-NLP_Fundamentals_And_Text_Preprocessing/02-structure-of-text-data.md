# Structure of Text Data

Before applying NLP techniques, we need to understand how **text is structured**. NLP systems typically break text into progressively smaller units, with three important concepts being **corpus, sentences, and tokens**.

---

## 1. Corpus

A **corpus** is a collection of text documents used for NLP analysis or model training.

For example, a sentiment-analysis dataset might contain thousands of reviews:

```text
Corpus
│
├── Review 1: "This movie was excellent."
├── Review 2: "The acting was terrible."
├── Review 3: "I really enjoyed it."
└── ...
```

A corpus can contain:

* A collection of documents
* News articles
* Product reviews
* Social media posts
* Books
* Transcripts
* Conversations

In machine learning, a corpus is often divided into:

```text
Corpus
   │
   ├── Training Data
   ├── Validation Data
   └── Test Data
```

**Example:** The collection of movie reviews used to train a sentiment classifier is a corpus.

---

## 2. Sentence

A **sentence** is a linguistic unit that generally expresses a complete thought and is usually separated by punctuation such as `.`, `?`, or `!`.

Example:

```text
"The movie was excellent. The acting was amazing!"
```

contains two sentences:

```text
Sentence 1 → "The movie was excellent."

Sentence 2 → "The acting was amazing!"
```

Breaking a document into individual sentences is called **sentence segmentation** or **sentence tokenization**.

This is useful because NLP models often need to process text at the sentence level.

---

## 3. Token

A **token** is an individual unit produced when text is divided into smaller pieces.

Depending on the tokenizer and NLP system, tokens can be:

* Words
* Punctuation
* Subwords
* Characters

For example:

```text
"The movie was excellent!"
```

A simple word-level tokenizer might produce:

```text
["The", "movie", "was", "excellent", "!"]
```

Each element is a **token**.

Modern Transformer models generally use **subword tokenization**, so a word may be split into multiple tokens.

For example, a tokenizer could represent:

```text
"unhappiness"
```

as something conceptually similar to:

```text
["un", "happiness"]
```

The exact tokens depend on the tokenizer.

---

## Hierarchy of Text

These concepts can be viewed hierarchically:

```text
Corpus
  │
  ├── Document
  │     │
  │     ├── Sentence
  │     │      ├── Token
  │     │      ├── Token
  │     │      └── Token
  │     │
  │     └── Sentence
  │            ├── Token
  │            └── Token
  │
  └── Document
```

For example:

```text
Corpus
   ↓
Document
   ↓
Sentence
   ↓
Tokens
```

Consider:

```text
"Natural language processing is fascinating. It allows computers to work with human language."
```

We can break it down as:

```text
Corpus
└── Document
    ├── Sentence 1
    │   ├── Natural
    │   ├── language
    │   ├── processing
    │   ├── is
    │   └── fascinating
    │
    └── Sentence 2
        ├── It
        ├── allows
        ├── computers
        ├── to
        ├── work
        ├── with
        ├── human
        └── language
```

---

## Why This Structure Matters

NLP operations often happen at different levels:

| Level        | Example               | Common Tasks                         |
| ------------ | --------------------- | ------------------------------------ |
| **Corpus**   | Collection of reviews | Training, vocabulary analysis        |
| **Document** | One review/article    | Classification, summarization        |
| **Sentence** | One sentence          | Sentence classification, translation |
| **Token**    | Word/subword          | Embeddings, language modeling        |

Understanding this hierarchy is important because **tokenization is the bridge between raw human language and numerical representations**.

Eventually, our pipeline will become:

```text
Raw Text
   ↓
Documents
   ↓
Sentences
   ↓
Tokens
   ↓
Token IDs
   ↓
Embeddings
   ↓
NLP Model
```

Transformers do not directly process raw text. They process **numerical representations of tokens**, which makes understanding tokens and tokenization essential for understanding modern NLP.

---

## Key Takeaways

* A **corpus** is a collection of text used for NLP analysis or model training.
* A **document** is an individual piece of text within a corpus.
* A **sentence** is a linguistic unit within a document.
* A **token** is a smaller unit produced by a tokenizer.
* Tokens can represent words, punctuation, subwords, or characters depending on the tokenization method.
* Modern Transformer models commonly use **subword tokenization**.
* The general hierarchy is:

```text
Corpus
  ↓
Document
  ↓
Sentence
  ↓
Token
```

* The NLP processing pipeline eventually converts:

```text
Raw Text → Tokens → Token IDs → Embeddings → Model
```
