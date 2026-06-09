from llama_cpp import Llama

llm = Llama(
    model_path="models/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning.i1-Q4_K_M (3).gguf",
    n_ctx=4096,
    n_threads=6
)

response = llm(
    "Q: What is RAG in AI?\nA:",
    max_tokens=200,
    stop=["Q:"]
)

print(response["choices"][0]["text"])