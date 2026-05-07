Text splitting (or chunking) is the process of breaking large documents into smaller, manageable pieces to fit within a Large Language Model's (LLM) context window or to improve the accuracy of vector database searches.
### Important Points to Consider

* Context Preservation: If a chunk is too small, it loses its meaning; if it’s too large, it might exceed the model's limits or dilute the focus of search results.
* Chunk Size: This is the maximum length of each chunk (e.g., 500 characters or tokens).
* Chunk Overlap: To avoid cutting a sentence or concept in half, you "overlap" the end of one chunk with the start of the next (typically 10-20% of the chunk size).
* Metadata: Attaching source info (like page numbers or headers) to each chunk helps the model track where information originated.

### Types of Text Splitters

| Type | When to Use | When to Avoid |
|---|---|---|
| Recursive Character | The Best Default. Ideal for most text (articles, reports) because it tries to keep paragraphs and sentences together. | Rarely needs avoiding, though it might be slower than basic character splitting. |
| Token-Based | When you have a strict token limit for your LLM (e.g., GPT-4) and want to maximize input space. | Avoid if you need human-readable splits, as it can cut text in the middle of words or sentences. |
| Semantic | For complex content where topics change frequently within paragraphs; it uses AI to group sentences by meaning. | Avoid if you are on a tight budget (higher computational cost) or if document structure is very clear. |
| Character-Based | For simple, unstructured data like chat logs where structural integrity isn't a priority. | Avoid in production RAG. It often cuts words in half, leading to poor search results. |
| Structure-Aware | For files with specific formatting like Markdown, HTML, or Code (Python, JS). | Avoid for standard plain text, as it will look for tags or headers that aren't there. |
