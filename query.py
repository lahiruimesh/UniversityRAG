from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# -------------------------
# 1. Embedding model
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 2. Load ChromaDB
# -------------------------
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# -------------------------
# 3. User question
# -------------------------
query = "When is my exam schedule?"

# -------------------------
# 4. Retrieve relevant chunks
# -------------------------
results = db.similarity_search(query, k=3)

# Combine context
context = "\n\n".join([doc.page_content for doc in results])

# -------------------------
# 5. Load LLaMA 3
# -------------------------
llm = Ollama(model="llama3")

# -------------------------
# 6. Create prompt
# -------------------------
prompt = f"""
You are a helpful university assistant.

Use ONLY the context below to answer the question.
If answer is not in context, say "I don't know".

Context:
{context}

Question:
{query}

Answer in a clear and short way:
"""

# -------------------------
# 7. Get response
# -------------------------
response = llm.invoke(prompt)

# -------------------------
# 8. Output
# -------------------------
print("\n===== AI ANSWER =====\n")
print(response)