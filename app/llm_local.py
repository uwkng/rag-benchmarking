from langchain_community.llms import Ollama

def get_local_llm(model: str = "mistral", temperature: float = 0.2):
    return Ollama(model=model, temperature=temperature)