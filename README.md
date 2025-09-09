# RAG Benchmarking

This repository explores **Retrieval-Augmented Generation (RAG)** through two implementations:

1. **Baseline RAG**: Classic pipeline with loader, chunking embeddings, vector store and retrieval.
2. **GraphRAG**: Extended pipeline that integrates graph structures for multi-hop reasoning and entity relations.

The project aims to compare both approaches in terms of **accuracy, faithfulness and retrieval quality**, using the **RAG Mini-Wikipedia dataset (Hugging Face)** for evaluation.

---

## Quickstart

Clone the repository and set up a Python environment:

```bash
git clone https://github.com/<your-username>/rag-benchmarking.git
cd rag-benchmarking

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

## Dataset

RAG Mini-Wikipedia (https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)

- Small, open-domain QA dataset based on Wikipedia.
- Contains ~918 question-answer pairs (test split) and ~3,200 cleaned text passages in English.
- Licensed under CC-BY-3.0.
- Well-suited for quick RAG prototyping and evaluation thanks to its compact size, realistic QA setup and open availability.

## Roadmap

1. Implement Baseline RAG pipeline end-to-end
2. Add Evalutation with metrics (faithfulness, context precision/recall, answer relevancy)
3. Extend to GraphRAG
4. Provide a Benchmark comparison between Baseline and GraphRAG

## License

MIT License - free to use and modify.
