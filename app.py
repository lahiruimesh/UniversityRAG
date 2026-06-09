from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

app = FastAPI(
    title="Faculty AI Assistant"
)

# -----------------------
# Load once
# -----------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

llm = Ollama(model="llama3")

# -----------------------
# Request schema
# -----------------------

class QuestionRequest(BaseModel):
    question: str

# -----------------------
# Health Check
# -----------------------

@app.get("/")
def root():
    return {"message": "Faculty AI Assistant Running"}

# -----------------------
# Ask Endpoint
# -----------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    query = request.question

    results = db.similarity_search(query, k=8)

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
You are a university assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)

    return {
        "question": query,
        "answer": answer
    }