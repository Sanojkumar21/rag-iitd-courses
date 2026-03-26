from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# parent chunks are bigger — they go to the LLM as context
# child chunks are smaller — they're what actually gets searched in chroma
PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE  = 400
OVERLAP           = 50


def _make_splitter(size: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )


def _fix_table(doc: Document) -> Document:
    # wrapping with a label helps the embedding model understand it's tabular data
    doc.page_content = f"Table:\n{doc.page_content}"
    doc.metadata["type"] = "table"
    return doc


def _split_into_parents(text_docs: list[Document]) -> list[Document]:
    splitter = _make_splitter(PARENT_CHUNK_SIZE)
    parents  = splitter.split_documents(text_docs)

    for idx, p in enumerate(parents):
        p.metadata["type"]         = "parent_chunk"
        p.metadata["parent_index"] = idx
        p.metadata["source_page"]  = p.metadata.get("page", "unknown")

    return parents


def _split_into_children(parents: list[Document]) -> list[Document]:
    splitter = _make_splitter(CHILD_CHUNK_SIZE)
    children = []

    for p in parents:
        kids = splitter.split_documents([p])
        for kid in kids:
            kid.metadata["type"]         = "child_chunk"
            kid.metadata["parent_index"] = p.metadata["parent_index"]
            kid.metadata["source_page"]  = p.metadata.get("source_page", "unknown")
        children.extend(kids)

    return children


def chunk_documents(docs: list[Document]) -> dict:
    text_docs  = [d for d in docs if d.metadata.get("type") == "text"]
    table_docs = [d for d in docs if d.metadata.get("type") == "table"]

    tables  = [_fix_table(d) for d in table_docs]
    parents = _split_into_parents(text_docs)
    children = _split_into_children(parents)

    # children + tables go into chroma for search
    # parents + tables go to the LLM when a chunk is matched
    search_chunks  = children + tables
    context_chunks = parents  + tables

    print(f"parents : {len(parents)} | children : {len(children)} | tables : {len(tables)}")
    print(f"search pool : {len(search_chunks)} chunks")

    return {
        "search_chunks":  search_chunks,   # embed these in chromadb
        "context_chunks": context_chunks,  # return these to the LLM
    }