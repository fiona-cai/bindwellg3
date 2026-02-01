from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangChainDocument

import json
import re

file_path = "../2026-pgp.pdf"
out_path = "./heading-chunks-w-title.json"

def chunk_by_heading():
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    # Combine all pages into one string so we can split by heading patterns like "1.0", "2.0", etc.
    full_text = "\n\n".join([d.page_content for d in docs])

    # Split on lines that start with a numbering pattern (e.g. 1.0, 2.0, 3.1)
    # The lookahead keeps the heading at the start of each split segment.
    sections = [s.strip() for s in re.split(r'(?m)^(?=\d+\.\d+)', full_text) if s.strip()]

    print(f"Found {len(sections)} sections based on headings")

    # Convert to LangChain Document objects with basic metadata
    lc_documents = []
    for i, sec in enumerate(sections):
        first_line = sec.split("\n")[0]
        title = re.split(r'(?<=\D)\.', first_line)[0]
        print(f"Title: {title}")
        meta = {"source": file_path, "section_index": i + 1, "heading_title": title}
        lc_documents.append({"content": sec, "metadata": meta})
        # lc_documents.append(LangChainDocument(page_content=sec, metadata=meta))

    print(f"Created {len(lc_documents)} LangChain Document objects")
    # if lc_documents:
    #     print(lc_documents[100].page_content[:1000])

    with open(out_path, 'w', encoding='utf-8') as json_file:
        json.dump(lc_documents, json_file, indent=4)


if __name__ == "__main__":
    chunk_by_heading()
