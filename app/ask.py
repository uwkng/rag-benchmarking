import sys
from .rag_chain import build_chain

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the capital of Uruguay?"
    print(f"Question: {q}")
    try:
        chain = build_chain()
        answer = chain.invoke(q)
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error: {e}")