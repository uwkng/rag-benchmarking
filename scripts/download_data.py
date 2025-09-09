
from datasets import load_dataset
import json
from pathlib import Path

def save_jsonl(dataset, path, text_field=None):
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            if text_field and isinstance(item, dict) and text_field in item:
                rec = {"text": item[text_field]}
            else:
                rec = item
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    Path("data").mkdir(parents=True, exist_ok=True)

    print("Downloading corpus...")
    corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    corpus_split = list(corpus.values())[0]
    save_jsonl(corpus_split, "data/mini_wikipedia_corpus.jsonl", text_field="text")
    print(f"Saved {corpus_split.num_rows} rows to data/mini_wikipedia_corpus.jsonl")

    print("Downloading question-answer set...")
    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    qa_split = list(qa.values())[0]
    save_jsonl(qa_split, "data/mini_wikipedia_qa_jsonl")
    print(f"Saved {qa_split.num_rows} rows to data/mini_wikipedia_qa_jsonl")