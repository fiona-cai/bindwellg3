## Installation of Dependencies

To install the required libaries to run the project, run:
```
pip install -r requirements.txt
```

## Usage Details

To view our application and chat with our agent, run in the root directory:
```
python api.py
```
and a FastAPI server will launch. When the server process completes loading, you can click on the link
to launch the app. The link should be 127.0.0.1:8001. (APP_HOST and APP_PORT can be changed in ```config.py```)

Wait until
```
INFO:     Application startup complete.
``` 
is shown.

> [!CAUTION]
> You do not need to launch the index.html file seperately for the frontend. The FastAPI server serves the index.html, style.css, and script.js files.

> [!NOTE]
> You will need to create an OpenAI API Key to use this project.

## Information about codebase

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

📄 **Data and Document Processing**

[chunk_by_heading.py](data/chunk_by_heading.py) - Chunks the document by title heading
- we reasoned this would be the most effective way to avoid data fragmentation as well as keep reasonable chunk size

[document_processor.py](data/document_processor.py) - Gets all tables and outputs JSON with table headers and rows as lists with their indexes matching for each column

[prepare_table_data.py](data/prepare_table_data.py) - appends to table data to rest of text chunks

✅ **Evaluation** [retrieval_eval.py](retrieval_eval.py)
- Uses deepeval to test over a set of 20+ questions we gathered in [pgp_test_questions.json](pgp_test_questions.json)


| Metric            | Percentage of Tests |
| ------------------| --------------------|
| Answer Relevancy  | 92.31%              |
| Faithfulness      | 96.15%              |


⭐ Answer Relevancy - how well can the LLM answer the question with knowledge of retrieved text

⭐ Faithfulness - how well grounded LLM responses are in terms of retrieved text
