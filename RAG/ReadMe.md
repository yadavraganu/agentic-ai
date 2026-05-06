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
   2. Inverse Document Frequency (IDF): Measures global rarity.  
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
