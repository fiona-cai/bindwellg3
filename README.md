# EPA Consultant Agent

RAG-based Q&A agent over EPA documents (e.g. NPDES Pesticide General Permit), with hybrid retrieval, reranking, and table support.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

From the project root, start the server:
```bash
python api.py
```
A FastAPI server will start. Once you see startup complete, open **http://127.0.0.1:8001** in your browser. (Host and port can be changed in `config.py`.)

Wait until
```
INFO:     Application startup complete.
``` 
is shown.

> **Note:** The FastAPI server serves the frontend (index.html, style.css, script.js). You do not need to open `index.html` separately.

> **OpenAI API key** is required; set it in your environment before running.

## Codebase overview

🔍 **ML and Retrieval**

Go to:
[retrieval/retrieval_langchain.py](retrieval/retrieval_langchain.py)
- Ingests all documents into FAISS Index
- Uses Hybrid Search to first find the top 30-50 documents relevant to the query
     - Keyword - BM25
     - Semantic - OpenAIEmbeddings and FAISS Cosine Similarity search
     - MMR - to reduce redundacy in passage retrieved to get a broad coverage of topic
- Then passes through a reranker to reembed passages in the context of the query - query may allow passages to derive new meaning
     - experimented with a lightweight, fast reranker (FlashRerank) and a larger, more accurate reranker (BGE Reranker)
     - Notable improvement in performance with previously ranked 5th or below documents moving to 1st and 2nd spot (desired)

💻 **Frontend code** - [frontend/index.html](frontend/index.html)

🔧 **Backend code** - [api.py](api.py)
- contains call to LLM to synthesize response from the retrieved excerpts
- call for retrieving documents (method exposed from retrieval)
- various endpoints to facilitate communication with frontend
Ex. POST /api/chat - for LLM based chat responses
POST /api/ask - for raw retrieved text excerpts
GET /api/tables - to list tables in document

📄 **Data and document processing**

[data/chunk_by_heading.py](data/chunk_by_heading.py) — chunks the document by title/heading
- we reasoned this would be the most effective way to avoid data fragmentation as well as keep reasonable chunk size

[data/document_processor.py](data/document_processor.py) — extracts tables to JSON (headers and rows).

[data/prepare_table_data.py](data/prepare_table_data.py) — appends table data to text chunks.

✅ **Evaluation** — [retrieval/retrieval_eval.py](retrieval/retrieval_eval.py)
- Uses deepeval over 20+ questions in [pgp_test_questions.json](pgp_test_questions.json)


| Metric            | Percentage of Tests |
| ------------------| --------------------|
| Answer Relevancy  | 92.31%              |
| Faithfulness      | 96.15%              |


⭐ Answer Relevancy - how well can the LLM answer the question with knowledge of retrieved text

⭐ Faithfulness - how well grounded LLM responses are in terms of retrieved text
