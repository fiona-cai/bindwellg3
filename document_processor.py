from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "./2026-pgp.pdf"

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
if all_texts:
	print(all_texts[65])
