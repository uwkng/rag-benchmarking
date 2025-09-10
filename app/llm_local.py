from langchain_ollama import ChatOllama

def get_local_llm(model: str = "mistral", temperature: float = 0.2):
    return ChatOllama(model=model, temperature=temperature)