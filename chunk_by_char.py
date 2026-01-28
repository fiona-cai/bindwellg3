from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

file_path = "./2026-pgp.pdf"
out_path = "./chunks_char.json"

loader = PyPDFLoader(file_path)

docs = loader.load()
print(f"Loaded {len(docs)} pages")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_texts = []
for i, document in enumerate(docs):
	chunks = text_splitter.split_text(document.page_content)
	print(f"Page {i+1}: {len(chunks)} chunks")
	all_texts.extend(chunks)

print(f"Total chunks: {len(all_texts)}")

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_texts, f, indent=4)
