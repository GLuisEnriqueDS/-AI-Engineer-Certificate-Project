from google.genai import errors as genai_errors

from graph.pipeline import build_rag_graph, initial_state, load_resources


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

    while True:
        question = input("Pregunta> ").strip()
        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            break

        print("# --DEBUG-- #")
        try:
            result = graph.invoke(initial_state(question, chat_history))
        except genai_errors.APIError as exc:
            print(f"\nError de la API de Gemini ({exc.code} {exc.status}): {exc.message}")
            print("No se pudo responder esta pregunta. Probá de nuevo en unos segundos.\n")
            continue

        print_result(result)
        print()

        if result["status"] != "rejected_query":
            chat_history.append((question, result["answer"]))


if __name__ == "__main__":
    main()
