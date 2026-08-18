import uuid

from google.genai import errors as genai_errors

from graph.pipeline import build_rag_graph, initial_state, load_resources
from llm.observability import flush as langfuse_flush
from llm.observability import observe, trace_session


def print_result(result):
    status = result["status"]
    print(f"\n=== Estado: {status} ===")

    if status == "rejected_query":
        print(f"Pregunta rechazada: {result['rejection_reason']}")
        return

    print(result["answer"])

    if status == "no_relevant_context":
        print("(no se encontro contexto relevante tras reformular la pregunta)")
    elif status == "low_confidence":
        print(f"(respuesta con baja confianza segun el juez, score={result.get('answer_score')})")

    if result["documents"]:
        print("\n--- Fuentes ---")
        seen_titles = set()
        for doc in result["documents"]:
            title = doc.metadata.get("title", "?")
            url = doc.metadata.get("url", "?")
            if title not in seen_titles:
                print(f"  - {title}\n    {url}")
                seen_titles.add(title)


def main():
    resources = load_resources()
    graph = build_rag_graph(resources)
    print("Listo. Escribe una pregunta (o 'salir' para terminar).\n")

    chat_history = []
    session_id = str(uuid.uuid4())

    while True:
        question = input("Pregunta> ").strip()
        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            break

        print("# --DEBUG-- #")
        try:
            with observe("span", "rag_turn") as root:
                root.update(input={"question": question})
                with trace_session(session_id=session_id, user_id="cli-user", trace_name="rag_turn"):
                    result = graph.invoke(initial_state(question, chat_history))
                root.update(output={"status": result["status"], "answer": result["answer"]})
        except genai_errors.APIError as exc:
            print(f"\nError de la API de Gemini ({exc.code} {exc.status}): {exc.message}")
            print("No se pudo responder esta pregunta. Probá de nuevo en unos segundos.\n")
            continue
        finally:
            langfuse_flush()

        print_result(result)
        print()

        if result["status"] != "rejected_query":
            chat_history.append((question, result["answer"]))


if __name__ == "__main__":
    main()
