# Keyword Search
## TF-IDF:
TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical formula used to rank the importance of a word in a document relative to a collection of documents (corpus). In the context of notes creation, it helps you automatically extract tags, summarize content, and build an intelligent search engine.
### 1. The Intuition: "Rarity = Relevance"
If you search your notes for "Neural Networks," every note containing those words is a candidate. However, if a note mentions "Neural" 20 times, it’s likely more relevant than a note that mentions it once.

* The Catch: Common words like "the" or "is" appear frequently everywhere.
* The Solution: TF-IDF rewards words that are frequent in one specific note but rare across your entire notebook.

### 2. The Formulas
The final score is the product of two distinct calculations:

   1. Term Frequency (TF): Measures local density.
      
   $$\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in doc } d}{\text{Total words in doc } d}$$ 
   
   2. Inverse Document Frequency (IDF): Measures global rarity.  \
      
   $$\text{IDF}(t, D) = \log\left(\frac{\text{Total number of docs } N}{\text{Docs containing term } t + 1}\right)$$ 
   
   3. TF-IDF Weight:
      
   $$\text{Score} = \text{TF} \times \text{IDF}$$ 
   

### 3. When to Use vs. Avoid

| Scenario | Action | Reason |
|---|---|---|
| Keyword Tagging | Use | Excellent for finding unique "signature words" for a note. |
| Search Ranking | Use | Ranks results by relevance rather than just date or title. |
| Finding Similarity | Use | Allows you to find "Related Notes" using vector math. |
| Small Note Count | Avoid | IDF needs a diverse corpus to know what a "rare" word is. |
| Deep Meaning | Avoid | It doesn't understand synonyms (e.g., "Car" vs "Automobile"). |
| Context/Order | Avoid | It treats text as a "Bag of Words," ignoring sentence structure. |
### 4. Implementation
```python
import math
import re
from collections import Counter

def calculate_tfidf(corpus):
    # 1. Preprocessing: Lowercase, remove punctuation, and tokenize
    processed_docs = [re.sub(r'[^\w\s]', '', doc.lower()).split() for doc in corpus]
    num_docs = len(processed_docs)

    # 2. Compute Term Frequency (TF)
    # TF = (count of word in doc) / (total words in doc)
    tf_scores = []
    for doc in processed_docs:
        counts = Counter(doc)
        total_words = len(doc)
        tf_scores.append({word: count / total_words for word, count in counts.items()})

    # 3. Compute Inverse Document Frequency (IDF)
    # IDF = log(Total Docs / Docs containing the word)
    unique_words = set(word for doc in processed_docs for word in doc)
    idf_scores = {}
    for word in unique_words:
        docs_with_word = sum(1 for doc in processed_docs if word in doc)
        # Adding 1 to denominator to avoid division by zero (smoothing)
        idf_scores[word] = math.log(num_docs / (1 + docs_with_word))

    # 4. Compute Final TF-IDF
    # TF-IDF = TF * IDF
    tfidf_final = []
    for doc_tf in tf_scores:
        doc_tfidf = {word: tf_val * idf_scores[word] for word, tf_val in doc_tf.items()}
        tfidf_final.append(doc_tfidf)

    return tfidf_final

# --- Main Execution ---
if __name__ == "__main__":
    example_corpus = [
        "The sun is shining in the sky.",
        "The sky is blue and bright.",
        "The sun is a bright star."
    ]

    results = calculate_tfidf(example_corpus)

    # Display results
    for i, doc_scores in enumerate(results):
        print(f"\nDocument {i+1} Scores:")
        # Sort by score descending to see most 'important' words first
        for word, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {word:10}: {score:.4f}")
```
