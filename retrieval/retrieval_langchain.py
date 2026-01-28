import json
# import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from langchain_classic.retrievers import EnsembleRetriever
import os

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

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
save_faiss_index_name = "embed_faiss_store_langchain.bin"
faiss_store = None

docs = [Document(page_content=chunk["content"], metadata=chunk["metadata"]) for chunk in chunks]

# texts = [chunk for chunk in chunks]

print(f"Loaded {len(docs)} docs")

# Load from path index created all ready
if (os.path.exists(save_faiss_index_name + ".faiss")):
    # index = faiss.read_index(save_faiss_index_name)
    faiss_store = FAISS.load_local(".", embedding_model, 
                                    index_name=save_faiss_index_name, 
                                    allow_dangerous_deserialization=True)
    print("FAISS index loaded")
else:

    # texts = [chunk["text"] for chunk in chunks]
    # 2. Load embedding model !

    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # embeddings = model.encode(texts, show_progress_bar=True)

    # embeddings = embedding_model.embed_documents(texts)

    # # Convert to numpy array
    # embeddings = np.array(embeddings).astype("float32")
    # dimension = embeddings.shape[1]
    # index = faiss.IndexFlatL2(dimension)
    # index.add(embeddings)

    # faiss.write_index(index, save_faiss_index_name)

    faiss_store = FAISS.from_documents(docs, embedding_model)

    faiss_store.save_local(".", save_faiss_index_name)

    print("FAISS index built")

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 5

# 4. Query function !

def retrieve_chunks(question, k=10):
    # query_embedding = model.encode([question]).astype("float32")
    bm25.k = k
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25, 
                    faiss_store.as_retriever(
                        search_kwargs={
                            "k": k,
                            "score_threshold": 0.2
                        })],
        weights=[0.2, 0.8] 
    )

    # results = hybrid_retriever.invoke(question)

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_retriever
    )

    results = compression_retriever.invoke(
        question
    )

    # print(results)

    # contents = [p.page_content for p in results]

    return results


# 5. Test query

if __name__ == "__main__":
    # question = "What pesticide restrictions apply under the 2026 PGP?"
    #question = "For what activities are eligible for the Pesticide General Permit?"

    question = "What are options I have if I can't get a PGP permit?"
    # question = "What are alternative permits I can get and their requirements?"
    #question = "What are rules for mosquito control?"

    results = retrieve_chunks(question, k=20)
    
    print(f"Question asked: {question}")

    print("\nTop retrieved chunks:\n")
    for i, chunk in enumerate(results):
        print(f"--- Result {i+1} ---")
        # print(chunk)
        print(chunk.page_content)
        print()
