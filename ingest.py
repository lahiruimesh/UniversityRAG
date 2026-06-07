# ingest.py

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/academic calender 2026.pdf")

documents = loader.load()

print("Pages:", len(documents))

print(documents[0].page_content[:500])