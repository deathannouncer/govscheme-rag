import requests
import streamlit as st

API_URL = "http://localhost:8000/query"
CACHE_CLEAR_URL = "http://localhost:8000/cache"

st.set_page_config(page_title="Govt Scheme RAG", page_icon="🏛️", layout="centered")

with st.sidebar:
    st.title("🏛️ Govt Scheme RAG")
    st.caption(
        "Agentic hybrid-search RAG over Indian government scheme data - "
        "query rewriting, vector + full-text hybrid retrieval, RRF fusion, "
        "cross-encoder reranking, and an agent loop that decides whether "
        "to retrieve again before answering."
    )
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear server cache"):
        try:
            requests.delete(CACHE_CLEAR_URL, timeout=5)
            st.success("Cache cleared")
        except requests.RequestException as e:
            st.error(f"Couldn't reach the API: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict], cached: bool, latency_ms: float):
    label = f"Sources ({'cached' if cached else 'fresh'} - {latency_ms:.0f}ms)"
    with st.expander(label):
        for s in sources:
            if s.get("official_link"):
                st.markdown(f"- [{s['scheme_name']}]({s['official_link']})")
            else:
                st.markdown(f"- {s['scheme_name']}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"], msg.get("cached", False), msg.get("latency_ms", 0))

question = st.chat_input("Ask about an Indian government scheme...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching schemes and generating an answer..."):
            try:
                resp = requests.post(API_URL, json={"question": question}, timeout=300).json()
            except requests.RequestException as e:
                st.error(f"Couldn't reach the API at {API_URL} - is uvicorn running? ({e})")
                st.stop()

        st.markdown(resp["answer"])
        render_sources(resp["sources"], resp["cached"], resp["latency_ms"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resp["answer"],
            "sources": resp["sources"],
            "cached": resp["cached"],
            "latency_ms": resp["latency_ms"],
        }
    )
