from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.retriever import get_retriever
from app.prompts import rag_prompt
from app.llm_local import get_local_llm

def build_chain():
    retriever = get_retriever()
    llm = get_local_llm(model="mistral", temperature=0.2)
    
    return (
        {"context": retriever | (lambda docs: "\n\n".join(
            f"[{i+1}] {d.page_content.strip().replace('\n',' ')}" for i, d in enumerate(docs)
        )), "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
