import json
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma

# Get the chunked data
texts = []
with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])

# Define the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create the embeddings
embeddings = model.encode(texts)

# Building the index
vectorstore = Chroma.from_documents(
    texts,
    embedding=embeddings,
    persist_directory="chroma_index"
)