import json
import faiss
import numpy as np
# from sentence_transformers import SentenceTransformer
import os

from langchain_openai import OpenAIEmbeddings


# 1. Load chunked document.....

chunked_file_path = "heading-chunks.json"
# chunked_file_path = "chunks_char.json"

with open(chunked_file_path, "r") as f:
    chunks = json.load(f)

# Load or Build FAISS index!
embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
)
save_faiss_index_path = "embed_faiss_index.bin"
index = None

# Load from path index created all ready
if (os.path.exists(save_faiss_index_path)):
    index = faiss.read_index(save_faiss_index_path)
else:

    # texts = [chunk["text"] for chunk in chunks]
    texts = [chunk["content"] for chunk in chunks]

    # texts = [chunk for chunk in chunks]

    print(f"Loaded {len(texts)} chunks")

    # 2. Load embedding model !

    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = embedding_model.embed_documents(texts)

    # Convert to numpy array
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, save_faiss_index_path)

print("FAISS index built")

# 4. Query function !

def retrieve_chunks(question, k=10):
    # query_embedding = model.encode([question]).astype("float32")
    query_embedding = np.array(embedding_model.embed_query(question)).astype("float32").reshape(1, -1)
    # faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results


# 5. Test query

if __name__ == "__main__":
    # question = "What pesticide restrictions apply under the 2026 PGP?"
    #question = "For what activities are eligible for the Pesticide General Permit?"

    question = "What are rules for mosquito control?"

    results = retrieve_chunks(question, k=5)
    
    print(f"Question asked: {question}")

    print("\nTop retrieved chunks:\n")
    for i, chunk in enumerate(results):
        print(f"--- Result {i+1} ---")
        # print(chunk)
        print(chunk["content"])
        print()
