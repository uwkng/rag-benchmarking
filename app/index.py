import json
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHUNK_PATH = "data/chunks.jsonl"
PERSIST_DIR_PATH = "chroma_index"
EMBEDDING_MODEL = "all-mpnet-base-v2"

def load_chunks(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts

if __name__ == "__main__":
    Path(PERSIST_DIR_PATH).mkdir(parents=True, exist_ok=True)
    texts = load_chunks(CHUNK_PATH)

    embedding_model = HuggingFaceEmbeddings()

    vectorstore = Chroma.from_texts(
        texts,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR_PATH
    )