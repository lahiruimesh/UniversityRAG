from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --------------------
# 1. Load PDF
# --------------------
loader = PyPDFLoader("data/academic calender 2026.pdf")
documents = loader.load()

print(f"Pages loaded: {len(documents)}")

# --------------------
# 2. Split into chunks
# --------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")

# --------------------
# 3. Create embedding model
# --------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------
# 4. Store in ChromaDB
# --------------------
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"   # <-- saves locally
)

db.persist()

print("\nVector DB created and saved successfully!")