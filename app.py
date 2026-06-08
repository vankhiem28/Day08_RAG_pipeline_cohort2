"""Streamlit UI for the Day 8 RAG pipeline."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.services.indexing_service import IndexingService
from src.task10_generation import generate_with_citation


st.set_page_config(
    page_title="Drug Law RAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        background: #f7f7f4;
        border-right: 1px solid #deded7;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .5rem;
        margin: .5rem 0 1rem;
    }
    .metric-box {
        border: 1px solid #deded7;
        border-radius: 6px;
        padding: .55rem .7rem;
        background: #ffffff;
    }
    .metric-label {
        color: #62625b;
        font-size: .78rem;
        margin-bottom: .15rem;
    }
    .metric-value {
        color: #1f2933;
        font-size: .95rem;
        font-weight: 650;
        overflow-wrap: anywhere;
    }
    .source-row {
        border-left: 3px solid #4f6f52;
        padding: .15rem 0 .15rem .75rem;
        margin: .25rem 0 .5rem;
    }
    .source-meta {
        color: #62625b;
        font-size: .82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_indexing_service() -> IndexingService:
    return IndexingService()


def ensure_index() -> int:
    chunks = get_indexing_service().load_or_build_chunks()
    return len(chunks)


def rebuild_index() -> int:
    chunks = get_indexing_service().build_index(force=True)
    return len(chunks)


def render_metrics(result: dict[str, Any]) -> None:
    sources = result.get("sources", [])
    retrieval_source = result.get("retrieval_source", "none")
    best_score = max((float(item.get("score", 0.0)) for item in sources), default=0.0)
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-box">
            <div class="metric-label">Retrieval</div>
            <div class="metric-value">{retrieval_source}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Sources</div>
            <div class="metric-value">{len(sources)}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Best score</div>
            <div class="metric-value">{best_score:.3f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(result: dict[str, Any]) -> None:
    sources = result.get("sources", [])
    if not sources:
        st.info("Không có source chunk.")
        return

    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        citation = f"[{metadata.get('citation_id', index)}]"
        title = metadata.get("title") or metadata.get("source") or "Nguồn"
        score = float(source.get("score", 0.0))
        chunk_id = metadata.get("chunk_id", "unknown")
        article = metadata.get("article", "")
        retrieval_method = source.get("retrieval_method", "")
        with st.expander(f"{citation} {title}", expanded=index == 1):
            st.markdown(
                f"""
                <div class="source-row">
                  <div class="source-meta">score={score:.3f} | chunk_id={chunk_id} | {article} | {retrieval_method}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(source.get("content", ""))


def ask(question: str, top_k: int) -> dict[str, Any]:
    return generate_with_citation(question, top_k=top_k)


if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.header("RAG Controls")
    top_k = st.slider("Top K", min_value=1, max_value=8, value=5, step=1)
    chunk_count = ensure_index()
    st.metric("Indexed chunks", chunk_count)
    if st.button("Rebuild index", use_container_width=True):
        chunk_count = rebuild_index()
        st.success(f"Indexed {chunk_count} chunks")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


st.title("Drug Law RAG")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["result"]["answer"])
        render_metrics(item["result"])
        render_sources(item["result"])


question = st.chat_input("Nhập câu hỏi về pháp luật ma túy hoặc dữ liệu báo chí")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = ask(question, top_k=top_k)
        st.write(result["answer"])
        render_metrics(result)
        render_sources(result)

    st.session_state.history.append({"question": question, "result": result})
