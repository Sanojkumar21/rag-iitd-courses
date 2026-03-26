import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from app.extract import extract_pdf
from app.chunker import chunk_documents
from app.vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists
from app.chain import build_rag_chain, get_answer

st.set_page_config(page_title="IIT Syllabus Chatbot", page_icon="🎓", layout="wide")
st.title("🎓 IIT Course Syllabus Chatbot")
st.caption("Ask anything about your IIT courses — subjects, credits, schedules, electives!")

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Syllabus")

    # find pdf in data/ folder
    pdf_path = None
    if os.path.exists("data"):
        pdfs = [f for f in os.listdir("data") if f.endswith(".pdf")]
        if pdfs:
            pdf_path = os.path.join("data", pdfs[0])
            st.caption(f"PDF: `{pdfs[0]}`")
        else:
            st.warning("no PDF found in data/ folder")

    # auto-load if vectorstore already exists
    if "chain" not in st.session_state and vectorstore_exists():
        with st.spinner("loading vector store..."):
            vs = load_vectorstore()
            chroma = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 15}
            )
            st.session_state["chain"] = (chroma, None)
        st.info("loaded from disk — click Process for full hybrid search")

    # process button
    if pdf_path and st.button("Process PDF", type="primary"):
        with st.spinner("extracting..."):
            documents = extract_pdf(pdf_path)
            st.info(f"extracted {len(documents)} sections")

        with st.spinner("chunking..."):
            chunks = chunk_documents(documents)
            st.info(f"created {len(chunks['search_chunks'])} chunks")

        with st.spinner("building vector store..."):
            vs, search_chunks = build_vectorstore(chunks)
            st.session_state["chain"] = build_rag_chain(vs, search_chunks)

        st.success("ready!")

    st.divider()
    st.markdown("**example questions:**")
    st.markdown("- What subjects are in semester 3?")
    st.markdown("- How many credits for Institute Core?")
    st.markdown("- What electives are available?")
    st.markdown("- What are the B.Tech requirements?")
    st.divider()

    if st.button("Clear Chat"):
        st.session_state["messages"]     = []
        st.session_state["chat_history"] = []
        st.rerun()

# ── session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ── chat display ──────────────────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("rewritten_query"):
                st.caption(f"interpreted as: \"{msg['rewritten_query']}\"")
            if msg.get("sources"):
                with st.expander("sources"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")

# ── chat input ────────────────────────────────────────────────────────────────

if query := st.chat_input("ask about your IIT courses..."):
    if "chain" not in st.session_state:
        st.warning("click 'Process PDF' first!")
    else:
        with st.chat_message("user"):
            st.write(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                result    = get_answer(
                    st.session_state["chain"],
                    query,
                    chat_history=st.session_state["chat_history"]
                )
                answer    = result["answer"]
                sources   = result["sources"]
                rewritten = result.get("rewritten_query")

            st.write(answer)
            if rewritten:
                st.caption(f"interpreted as: \"{rewritten}\"")
            if sources:
                with st.expander("sources"):
                    for src in sources:
                        st.write(f"- {src}")

        st.session_state["messages"].append({
            "role":            "assistant",
            "content":         answer,
            "sources":         sources,
            "rewritten_query": rewritten
        })

        st.session_state["chat_history"].append(HumanMessage(content=query))
        st.session_state["chat_history"].append(AIMessage(content=answer))

        if len(st.session_state["chat_history"]) > 20:
            st.session_state["chat_history"] = st.session_state["chat_history"][-20:]

if "chain" not in st.session_state:
    st.info("click 'Process PDF' in the sidebar to get started!")