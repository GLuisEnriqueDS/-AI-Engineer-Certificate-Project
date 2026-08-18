import contextlib
import io

import streamlit as st
from google.genai import errors as genai_errors

from graph.pipeline import COLLECTION_NAME, build_rag_graph, initial_state, load_resources
from llm.answer import GEMINI_MODEL
from retrieval.retrieval_methods import EMBEDDING_MODEL, HYBRID_RERANK_KEY, RERANKER_MODEL

st.set_page_config(page_title="KFF ACA Assistant", page_icon="🩺", layout="wide")

@st.cache_resource(show_spinner="Cargando modelos (embeddings, cross-encoder)...")
def get_graph():
    resources = load_resources()
    return build_rag_graph(resources)


def format_answer(result):
    """Misma logica que print_result() en main.py, pero devuelve markdown
    para un st.chat_message en vez de imprimir a consola."""
    status = result["status"]
    parts = [result["answer"]]

    if status == "rejected_query":
        return f"**Pregunta rechazada:** {result['rejection_reason']}"
    if status == "no_relevant_context":
        parts.append("\n\n*(no se encontro contexto relevante tras reformular la pregunta)*")
    elif status == "low_confidence":
        parts.append(f"\n\n*(respuesta con baja confianza segun el juez, score={result.get('answer_score')})*")

    if result["documents"]:
        seen_titles = set()
        sources = []
        for doc in result["documents"]:
            title = doc.metadata.get("title", "?")
            url = doc.metadata.get("url", "?")
            if title not in seen_titles:
                sources.append(f"- [{title}]({url})")
                seen_titles.add(title)
        if sources:
            parts.append("\n\n**Fuentes**\n" + "\n".join(sources))

    return "".join(parts)


def run_turn(graph, question, chat_history):
    """Corre el grafo capturando los print() de debug de cada nodo (en vez
    de que vayan a la consola, quedan disponibles para el panel lateral)."""
    debug_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(debug_buffer):
            result = graph.invoke(initial_state(question, chat_history))
        return result, debug_buffer.getvalue(), None
    except genai_errors.APIError as exc:
        error_msg = f"Error de la API de Gemini ({exc.code} {exc.status}): {exc.message}"
        return None, debug_buffer.getvalue(), error_msg


def render_sidebar(debug_log):
    with st.sidebar:
        st.header("Stack")
        st.markdown(
            f"""
- **LLM**: `{GEMINI_MODEL}` (Gemini via Vertex AI)
- **Embeddings**: `{EMBEDDING_MODEL}`
- **Reranker**: `{RERANKER_MODEL}`
- **Retrieval usado**: {HYBRID_RERANK_KEY}
- **Vector DB**: ChromaDB (`{COLLECTION_NAME}`)
- **Orquestacion**: LangGraph
- **Meta-queries**: function-calling nativo de Gemini
  (count / recent / summary de un articulo, sin retrieval)
"""
        )
        st.divider()

        st.header("Debug")
        if not debug_log:
            st.caption("Todavia no hiciste ninguna pregunta.")
        else:
            last_question, last_trace = debug_log[-1]
            st.caption(f"Ultimo turno: *{last_question}*")
            st.code(last_trace or "(sin salida)", language="text")


def render_welcome():
    st.title("🩺 KFF ACA Assistant")
    with st.expander("👋 About this assistant", expanded=len(st.session_state.messages) == 0):
        st.markdown(
            """
Hi, I'm your health market news assistant, specialized in the
Affordable Care Act (ACA).

I read through Kaiser Family Foundation (KFF) articles so you don't have
to dig through them yourself. Here's what I can currently do:

- 📊&nbsp;&nbsp;Tell you how many articles I have on file
- 🆕&nbsp;&nbsp;Show you the most recently published articles
- 📝&nbsp;&nbsp;Summarize a specific article, if you tell me its name
- 💬&nbsp;&nbsp;Answer questions about ACA topics -- premiums, coverage,
  policy changes, and more -- based on those articles

I also keep track of our conversation, so feel free to ask follow-up
questions naturally (e.g. *"what about the states involved?"*).

*Note: I always answer in English, since that's the language of my
source articles.*
"""
        )


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "debug_log" not in st.session_state:
        st.session_state.debug_log = []

    render_welcome()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Escribi tu pregunta sobre el ACA...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                graph = get_graph()
                result, debug_trace, error = run_turn(graph, question, st.session_state.chat_history)

            st.session_state.debug_log.append((question, debug_trace))

            if error:
                st.error(error)
                answer_md = f"⚠️ {error}"
            else:
                answer_md = format_answer(result)
                st.markdown(answer_md)
                if result["status"] != "rejected_query":
                    st.session_state.chat_history.append((question, result["answer"]))

        st.session_state.messages.append({"role": "assistant", "content": answer_md})

    render_sidebar(st.session_state.debug_log)


if __name__ == "__main__":
    main()
