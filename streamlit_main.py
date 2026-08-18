import contextlib
import io
import uuid

import streamlit as st
from google.genai import errors as genai_errors

from graph.pipeline import COLLECTION_NAME, build_rag_graph, initial_state, load_resources
from llm.answer import GEMINI_MODEL
from llm.observability import ENABLED as LANGFUSE_ENABLED
from llm.observability import flush as langfuse_flush
from llm.observability import observe, score, trace_session
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


def run_turn(graph, question, chat_history, session_id):
    """Corre el grafo capturando los print() de debug de cada nodo (en vez
    de que vayan a la consola, quedan disponibles para el panel lateral), y
    lo envuelve en un trace de Langfuse (root span + session_id de esta
    sesion de Streamlit) igual que main.py."""
    debug_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(debug_buffer):
            with observe("span", "rag_turn") as root:
                root.update(input={"question": question})
                with trace_session(session_id=session_id, user_id="streamlit-user", trace_name="rag_turn"):
                    result = graph.invoke(initial_state(question, chat_history))
                root.update(output={"status": result["status"], "answer": result["answer"]})
                trace_id = root.trace_id
        return result, debug_buffer.getvalue(), None, trace_id
    except genai_errors.APIError as exc:
        error_msg = f"Error de la API de Gemini ({exc.code} {exc.status}): {exc.message}"
        return None, debug_buffer.getvalue(), error_msg, None
    finally:
        langfuse_flush()


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
- **Tracing**: {"Langfuse ✅" if LANGFUSE_ENABLED else "Langfuse deshabilitado (faltan claves en .env)"}
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


def render_message(message):
    """Dibuja un mensaje del chat. Si es del assistant y tiene trace_id
    (turno instrumentado con Langfuse), agrega botones de feedback 👍/👎 --
    una sola vez por mensaje, usando session_state.feedback_given para no
    mostrarlos de nuevo (ni permitir votar dos veces) despues del rerun que
    dispara cada click."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        trace_id = message.get("trace_id")
        if not trace_id:
            return

        vote = st.session_state.feedback_given.get(trace_id)
        if vote:
            st.caption("Gracias por tu feedback 👍" if vote == "up" else "Gracias por tu feedback 👎")
            return

        col_up, col_down, _ = st.columns([1, 1, 8])
        if col_up.button("👍", key=f"fb_up_{trace_id}"):
            score("user_feedback", 1, trace_id, data_type="NUMERIC", comment="👍")
            langfuse_flush()
            st.session_state.feedback_given[trace_id] = "up"
            st.rerun()
        if col_down.button("👎", key=f"fb_down_{trace_id}"):
            score("user_feedback", 0, trace_id, data_type="NUMERIC", comment="👎")
            langfuse_flush()
            st.session_state.feedback_given[trace_id] = "down"
            st.rerun()


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
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}

    render_welcome()

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Escribi tu pregunta sobre el ACA...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Pensando..."):
            graph = get_graph()
            result, debug_trace, error, trace_id = run_turn(
                graph, question, st.session_state.chat_history, st.session_state.session_id
            )

        st.session_state.debug_log.append((question, debug_trace))

        if error:
            answer_md = f"⚠️ {error}"
        else:
            answer_md = format_answer(result)
            if result["status"] != "rejected_query":
                st.session_state.chat_history.append((question, result["answer"]))

        st.session_state.messages.append({"role": "assistant", "content": answer_md, "trace_id": trace_id})
        st.rerun()

    render_sidebar(st.session_state.debug_log)


if __name__ == "__main__":
    main()
