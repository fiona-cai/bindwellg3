import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load chunked document.....

with open("processed_document.json", "r") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(texts)} chunks")


# 2. Load embedding model !

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts, show_progress_bar=True)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")


# 3. Build FAISS index!

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built")

# 4. Query function !

def retrieve_chunks(question, k=5):
    query_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results


# 5. Test query

if __name__ == "__main__":
    question = "What pesticide restrictions apply under the 2026 PGP?"

    results = retrieve_chunks(question, k=5)

    print("\nTop retrieved chunks:\n")
    for i, chunk in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(chunk["text"][:500])
        print()
