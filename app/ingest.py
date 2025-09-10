import json
from pathlib import Path
from langchain_community.document_loaders import JSONLoader

class CorpusIngestor:

    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load(self):
        loader = JSONLoader(
            file_path=str(self.input_path),
            jq_schema='.passage', 
            text_content=True,
            json_lines=True,
        )
        docs = loader.load()
        return docs

    def to_texts(self, docs):
        return [doc.page_content for doc in docs]

    def save_chunks(self, texts):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    def run(self, preview: int = 10):
        docs = self.load()
        texts = self.to_texts(docs)

        for i, t in enumerate(texts[:preview]):
            print(f"Chunk {i}: {len(t)} characters")

        self.save_chunks(texts)
        return len(texts)


def ingest_corpus(
    input_path: str = "data/mini_wikipedia_corpus.jsonl",
    output_path: str = "data/chunks.jsonl",
):
    ingestor = CorpusIngestor(input_path, output_path)
    return ingestor.run()

if __name__ == "__main__":
    n = ingest_corpus()
    print(f"Ingest complete: {n} chunks saved.")