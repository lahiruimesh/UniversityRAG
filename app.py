from fastapi import FastAPI
from pydantic import BaseModel

from llama_cpp import Llama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="University RAG Assistant")

# -------------------
# EMBEDDINGS
# -------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# VECTOR DB
# -------------------
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# -------------------
# LLM (llama.cpp)
# -------------------
llm = Llama(
    model_path="models/llama3.q4.gguf",
    n_ctx=4096,
    n_threads=6
)

# -------------------
# REQUEST MODEL
# -------------------
class Question(BaseModel):
    question: str


# -------------------
# HEALTH CHECK
# -------------------
@app.get("/")
def home():
    return {"status": "RAG system running"}

# -------------------
# ASK API
# -------------------
@app.post("/ask")
def ask(req: Question):

    query = req.question

    # retrieval
    results = db.similarity_search(query, k=3)
    context = "\n\n".join([r.page_content for r in results])

    # prompt
    prompt = f"""
You are a university assistant.

Use ONLY context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    # inference
    response = llm(
        prompt,
        max_tokens=300,
        temperature=0.2,
        stop=["</s>"]
    )

    answer = response["choices"][0]["text"]

    return {
        "question": query,
        "answer": answer.strip()
    }