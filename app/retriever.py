from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

PERSIST_DIR = "chroma_index"
SEARCH_TYPE = "similarity"
SEARCH_KWARGS = {"k": 2}

def get_embeddings():
    """Get the embedding model - fallback if app.embeddings doesn't exist"""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever():
    """Load persisted Chroma vectorstore and return a retriever"""
    persist_path = Path(PERSIST_DIR)
    
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma index directory '{PERSIST_DIR}' not found. "
            "Run the indexing script first to create the vectorstore."
        )
    
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs=SEARCH_KWARGS
    )
    
    return retriever