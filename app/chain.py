import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

load_dotenv()

FOLLOWUP_SIGNALS = [
    " it ", " its ", " that ", " this ", " those ", " these ",
    "what about", "how about", "tell me more", "what else",
    "also ", "same ", "and the ", "any other"
]


def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def build_rag_chain(vectorstore, chunks: list[Document]) -> tuple:
    chroma = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    bm25   = BM25Retriever.from_documents(chunks)
    bm25.k = 5

    return chroma, bm25


def is_followup(query: str, chat_history: list) -> bool:
    if not chat_history:
        return False
    q = f" {query.lower()} "
    return any(sig in q for sig in FOLLOWUP_SIGNALS)


def rewrite_query(query: str, chat_history: list) -> str:
    llm = get_llm()

    last_msgs    = chat_history[-2:]
    history_text = ""
    for msg in last_msgs:
        role          = "U" if isinstance(msg, HumanMessage) else "A"
        history_text += f"{role}: {msg.content[:100]}\n"

    resp = llm.invoke(
        f"History:\n{history_text}\nRewrite as standalone question: '{query}'\nReturn ONLY the rewritten question."
    )
    return resp.content.strip().strip('"')


def _merge_docs(chroma_docs: list[Document], bm25_docs: list[Document]) -> list[Document]:
    # combine both results, deduplicate by first 80 chars
    seen = set()
    merged = []
    for doc in chroma_docs + bm25_docs:
        key = doc.page_content.strip()[:80]
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def truncate_context(docs: list[Document], max_chars: int = 400) -> str:
    parts = []
    for doc in docs:
        content = doc.page_content.strip()
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        page = doc.metadata.get("page", "?")
        parts.append(f"[P{page}] {content}")
    return "\n".join(parts)


def get_answer(retrievers: tuple, query: str, chat_history: list = None) -> dict:
    if chat_history is None:
        chat_history = []

    chroma, bm25 = retrievers
    llm          = get_llm()

    # rewrite only if it looks like a follow-up
    if is_followup(query, chat_history):
        search_query = rewrite_query(query, chat_history)
    else:
        search_query = query

    # run both retrievers and merge results
    chroma_docs = chroma.invoke(search_query)
    bm25_docs   = bm25.invoke(search_query)
    docs        = _merge_docs(chroma_docs, bm25_docs)

    context = truncate_context(docs, max_chars=400)
    sources = list(set([
        f"Page {d.metadata.get('page', '?')} ({d.metadata.get('type', 'text')})"
        for d in docs
    ]))

    # last 1 exchange for history
    history_text = "None"
    if chat_history:
        lines = []
        for msg in chat_history[-2:]:
            role = "User" if isinstance(msg, HumanMessage) else "Bot"
            lines.append(f"{role}: {msg.content[:80]}")
        history_text = "\n".join(lines)

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""You are an IIT syllabus assistant. Use the context below to answer.
Be specific with course codes and credits. If the answer isn't in the context, say you don't know.

History: {history}
Context: {context}
Q: {question}
A:"""
    )

    answer = (prompt | llm).invoke({
        "history":  history_text,
        "context":  context,
        "question": query
    }).content

    return {
        "answer":          answer,
        "sources":         sources,
        "rewritten_query": search_query if search_query != query else None
    }