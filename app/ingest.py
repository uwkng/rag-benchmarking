import json
from pathlib import Path
from langchain_community.document_loaders import JSONLoader
from pprint import pprint

# Loading

loader = JSONLoader(
    file_path="./data/mini_wikipedia_corpus.jsonl",
    jq_schema='.passage',
    text_content=True,
    json_lines=True
)

data = loader.load()

texts = [doc.page_content for doc in data]

# Checking the chunk size
for i, t in enumerate(texts[:10]):
    print(f"Chunk {i}: {len(t)} characters")

# Save chunks
with open("data/chunks.jsonl", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")