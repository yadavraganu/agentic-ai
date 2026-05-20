## Why Vector Databases Bypass Pure KNN
KNN (K-Nearest Neighbors) is not used by default in vector databases because it is too slow for production applications.
While KNN is 100% accurate, it uses a brute-force linear scan (also called "exact search"). This means to find a match, it must calculate the distance between your query vector and every single vector stored in the database.
Vector databases avoid pure KNN for three primary reasons:
### 1. Terrible Computational Scaling ($O(N)$ Time Complexity)
* The Problem: If you have 100,000 vectors, KNN performs 100,000 distance calculations per query. If you scale to 100 million vectors, it performs 100 million calculations.
* The Result: As your database grows, your search latency increases linearly. A query that takes 2 milliseconds on a small dataset will eventually take several seconds or minutes on a large enterprise dataset.
### 2. High Dimensionality Latency
Vector embeddings (from models like OpenAI, Cohere, or BERT) typically have between 768 and 1536+ dimensions. Computing the mathematical distance (like Euclidean distance or Cosine similarity) across 1,536 dimensions for millions of rows causes CPU usage to skyrocket, grinding database throughput to a halt.
### 3. Real-Time Production Demands
Production systems (like search engines, chatbots, or recommendation systems) require queries to return results within 10 to 50 milliseconds. Pure KNN cannot meet this requirement at scale, which is why vector databases trade a fraction of a percent of accuracy to use ANN (Approximate Nearest Neighbors) algorithms instead.
### When Vector Databases Do Use KNN
Even though ANN is the default, modern vector databases still include a pure KNN feature for specific scenarios:
* Small Datasets: If you are storing fewer than 10,000 to 50,000 vectors, a brute-force KNN scan is fast enough and guarantees 100% perfect accuracy.
* Hard Pre-Filtering: If your metadata query restricts the search to only a handful of documents (e.g., searching only documents owned by "User_XYZ"), the database will often run a fast KNN scan on just that tiny subset.
* Ground Truth Benchmarking: Developers run KNN on a sample dataset to get a perfect baseline result, allowing them to measure exactly how accurate their ANN algorithms are performing.
