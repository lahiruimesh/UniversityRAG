from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("data/academic calender 2026.pdf")
documents = loader.load()

print(f"Pages loaded: {len(documents)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0].page_content)

# Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert first chunk to vector
vector = embedding_model.embed_query(
    chunks[0].page_content
)

print(f"\nVector length: {len(vector)}")

# Convert user query to vector
query_vector = embedding_model.embed_query(
    "When is my database lecture?"
)

print(f"Query vector size: {len(query_vector)}")