import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import types

import langchain_core  
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, message=".*allowed_objects.*")
from langgraph.graph import END, StateGraph  # noqa: E402

from llm.answer import answer_with_llm, build_context, call_gemini, load_gemini_client
from llm.observability import observe
from retrieval.query_filters import build_where_filter, get_known_categories
from retrieval.retrieval_methods import (
    COLLECTION_NAME,
    HYBRID_RERANK_KEY,
    build_retrievers,
    load_models,
)
from tools.meta_queries import count_articles, get_article_excerpt, list_recent_articles, load_articles

MAX_RETRIEVAL_ATTEMPTS = 2
MAX_ANSWER_ATTEMPTS = 2
ANSWER_VALID_THRESHOLD = 4
MAX_HISTORY_TURNS = 3

JUDGE_CONFIG = types.GenerateContentConfig(temperature=0.0)

INTENT_TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="count_articles",
        description="Returns how many articles are indexed in the knowledge base. "
                    "Use for questions like 'how many articles do you have'.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="list_recent_articles",
        description="Returns the N most recently published articles (title, date, URL). "
                    "Use for questions like 'what are the most recent articles'.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "n": types.Schema(
                    type="INTEGER",
                    description="How many recent articles to return. Default 5.",
                )
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_article_excerpt",
        description="Returns the short editorial summary (excerpt) of ONE specific article "
                    "explicitly named or quoted in the question. Use only when the user "
                    "names a specific article and asks for its summary -- not for general "
                    "questions about ACA topics.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(
                    type="STRING",
                    description="The article title as mentioned in the question.",
                )
            },
            required=["title"],
        ),
    ),
    types.FunctionDeclaration(
        name="answer_from_articles",
        description="Use for any other question about the content of the ACA articles "
                    "(facts, explanations, analysis, opinions) that is not a count, not a "
                    "request for the most recent articles, and not a summary of one named "
                    "article.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
])
INTENT_CONFIG = types.GenerateContentConfig(temperature=0.0, tools=[INTENT_TOOLS])

CONDENSE_QUESTION_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up into a standalone question that can be understood without the history (resolve pronouns like "it"/"they" and implicit references to what was just discussed).

If the follow-up question is already standalone and does not depend on the history, return it unchanged.

Chat history:
{history}

Follow-up question: {question}

Respond with ONLY the standalone question, no other text."""

QUERY_VALIDATION_PROMPT = """You are a gatekeeper for a question-answering system about the Affordable Care Act (ACA), based on KFF (Kaiser Family Foundation) articles.

Decide if the following question is well-formed and about health policy, health insurance, or the ACA. Reject only for these reasons: empty input, a greeting or chit-chat, gibberish, or a topic unrelated to health policy/insurance.

Do NOT reject based on whether you think the data exists or is recent enough -- you do not know what is in the article database, and it is updated over time. Do NOT reason about which years are "future" or "past". A question mentioning any year, including years after your training cutoff, is in scope: whether the data actually exists is decided later, by retrieval, not by you.

Question: {question}

Respond in EXACTLY this format, with no other text:
Verdict: VALID
Reason: <one short sentence>

or:
Verdict: INVALID
Reason: <one short sentence>"""

RETRIEVAL_GRADING_PROMPT = """You are grading whether the retrieved context is relevant enough to answer the question below. It does not need to contain the full answer, only to be on-topic and useful as a starting point.

Context:
{context}

Question: {question}

Respond in EXACTLY this format, with no other text:
Verdict: RELEVANT
or:
Verdict: NOT_RELEVANT"""

QUERY_TRANSFORM_PROMPT = """The question below did not retrieve relevant context from a knowledge base about the Affordable Care Act (ACA). Rewrite it to be more likely to match relevant documents: expand acronyms, drop conversational filler, make implicit topics explicit. Keep it a single question, in English.

Original question: {question}

Respond with ONLY the rewritten question, no other text."""

ANSWER_VALIDATION_RUBRIC = """Rate the ANSWER on a scale of 1 to 5 for how well it is grounded in the CONTEXT.
1 = Completely false or fabricated, not supported by the context at all
2 = Mostly incorrect, with at most one correct detail
3 = Partially correct, but missing key information or containing errors
4 = Mostly correct and complete, with minor issues
5 = Fully correct, complete, and fully supported by the context

Context:
{context}

Question: {question}

Answer: {answer}

Respond in EXACTLY this format, with no other text:
Score: X"""

FALLBACK_ANSWER = "No tengo informacion suficiente en los articulos indexados para responder esto con confianza."


class RAGState(TypedDict):
    question: str
    chat_history: list[tuple[str, str]]
    standalone_question: str
    query_valid: bool
    rejection_reason: str
    intent: str
    meta_match: dict | None
    transformed_question: str
    where_filter: dict | None
    documents: list
    retrieval_valid: bool
    retrieval_attempts: int
    answer: str
    answer_valid: bool
    answer_score: int | None
    answer_attempts: int
    status: str


@dataclass
class Resources:
    """Todo lo pesado (modelos, clientes) se carga una sola vez y viaja
    por closure a los nodos -- LangGraph solo le pasa el state a cada
    nodo, asi que los recursos no pueden ir ahi sin ensuciar el schema."""

    chroma_client: object
    embeddings: object
    cross_encoder: object
    gemini_client: object
    known_categories: list
    articles: list


def load_resources():
    print("Cargando modelos (embeddings, cross-encoder)...")
    chroma_client, embeddings, cross_encoder = load_models()
    known_categories = get_known_categories(chroma_client, COLLECTION_NAME)
    articles = load_articles(chroma_client, COLLECTION_NAME)
    gemini_client = load_gemini_client()
    return Resources(chroma_client, embeddings, cross_encoder, gemini_client, known_categories, articles)


def _ask_judge(gemini_client, prompt, name="judge"):
    response = call_gemini(
        gemini_client, name=name, model="gemini-2.5-flash", contents=prompt, config=JUDGE_CONFIG
    )
    return response.text.strip()


def _parse_verdict(text, valid_token, invalid_token):
    match = re.search(rf"Verdict:\s*({valid_token}|{invalid_token})", text, re.IGNORECASE)
    if not match:
        # Sin veredicto parseable, no asumimos lo mejor: tratamos como
        # invalido/no-relevante, igual que un fallback de seguridad.
        return False
    return match.group(1).upper() == valid_token


def _extract_score(text):
    match = re.search(r"\bScore:\s*([1-5])\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def build_rag_graph(resources: Resources):
    def condense_question(state: RAGState) -> dict:
        with observe("span", "condense_question") as span:
            span.update(input={"question": state["question"], "history_turns": len(state["chat_history"])})
            history = state["chat_history"]
            if not history:
                print("[condense_question] sin historial previo, la pregunta ya es standalone")
                span.update(output={"standalone_question": state["question"]})
                return {"standalone_question": state["question"]}

            recent_history = history[-MAX_HISTORY_TURNS:]
            history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in recent_history)
            print(f"[condense_question] reescribiendo '{state['question']}' con {len(recent_history)} turno(s) previo(s)...")
            prompt = CONDENSE_QUESTION_PROMPT.format(history=history_text, question=state["question"])
            rewritten = _ask_judge(resources.gemini_client, prompt, name="condense_question")
            print(f"[condense_question] -> '{rewritten}'")
            span.update(output={"standalone_question": rewritten})
            return {"standalone_question": rewritten}

    def classify_intent(state: RAGState) -> dict:
        with observe("span", "classify_intent") as span:
            question = state["standalone_question"]
            span.update(input={"question": question})
            print(f"[classify_intent] Gemini decidiendo que tool usar para '{question}'...")
            response = call_gemini(
                resources.gemini_client, name="classify_intent",
                model="gemini-2.5-flash", contents=question, config=INTENT_CONFIG,
            )

            calls = response.function_calls
            if not calls:
                print("[classify_intent] -> sin tool call, content_query (sigue al gatekeeper)")
                span.update(output={"intent": ""})
                return {"intent": "", "meta_match": None}

            call = calls[0]
            args = dict(call.args or {})
            print(f"[classify_intent] -> tool call: {call.name}({args})")

            if call.name == "count_articles":
                span.update(output={"intent": "count"})
                return {"intent": "count", "meta_match": None}

            if call.name == "list_recent_articles":
                span.update(output={"intent": "recent", "n": args.get("n", 5)})
                return {"intent": "recent", "meta_match": {"n": int(args.get("n", 5))}}

            if call.name == "get_article_excerpt":
                result = get_article_excerpt(resources.articles, args.get("title", ""))
                if result is None:
                    print(f"[classify_intent] -> get_article_excerpt no encontro '{args.get('title')}' indexado, cae a content_query")
                    span.update(output={"intent": "", "unmatched_title": args.get("title")})
                    return {"intent": "", "meta_match": None}
                span.update(output={"intent": "summary", "matched_title": result["title"]})
                return {"intent": "summary", "meta_match": result}

            # "answer_from_articles" u otra cosa inesperada -> content_query
            span.update(output={"intent": ""})
            return {"intent": "", "meta_match": None}

    def handle_meta_query(state: RAGState) -> dict:
        with observe("span", "handle_meta_query") as span:
            intent = state["intent"]
            span.update(input={"intent": intent})
            print(f"[handle_meta_query] resolviendo '{intent}' directo contra la coleccion, sin retrieval...")

            if intent == "count":
                total = count_articles(resources.articles)
                answer = f"There are {total} articles indexed."
            elif intent == "recent":
                n = state["meta_match"]["n"]
                recent = list_recent_articles(resources.articles, n=n)
                lines = [f"- {a['title']} ({a['date'][:10]})\n  {a['url']}" for a in recent]
                answer = "The most recent articles are:\n" + "\n".join(lines)
            else:  # "summary"
                meta = state["meta_match"]
                answer = f"{meta['title']}\n{meta.get('excerpt', '')}\n\nSource: {meta['url']}"

            print(f"[handle_meta_query] -> respuesta generada ({len(answer)} caracteres)")
            span.update(output={"answer": answer})
            return {"answer": answer}

    def validate_query(state: RAGState) -> dict:
        with observe("span", "validate_query") as span:
            print(f"[validate_query] recepcionista: revisando '{state['standalone_question']}'...")
            question = state["standalone_question"].strip()
            span.update(input={"question": question})
            if not question:
                print("[validate_query] -> RECHAZADA (pregunta vacia)")
                span.update(output={"valid": False, "reason": "Pregunta vacia."})
                return {"query_valid": False, "rejection_reason": "Pregunta vacia."}

            prompt = QUERY_VALIDATION_PROMPT.format(question=question)
            output = _ask_judge(resources.gemini_client, prompt, name="validate_query")
            valid = _parse_verdict(output, "VALID", "INVALID")
            reason_match = re.search(r"Reason:\s*(.+)", output)
            reason = reason_match.group(1).strip() if reason_match else "Pregunta fuera del alcance del sistema."
            print(f"[validate_query] -> {'VALIDA, pasa al archivista' if valid else f'RECHAZADA ({reason})'}")

            span.update(output={"valid": valid, "reason": reason})
            return {"query_valid": valid, "rejection_reason": "" if valid else reason}

    def retrieve(state: RAGState) -> dict:
        with observe("span", "retrieve") as span:
            question = state.get("transformed_question") or state["standalone_question"]
            intento = state["retrieval_attempts"] + 1
            print(f"[retrieve] archivista: buscando (intento {intento}) con '{question}'...")
            where_filter = build_where_filter(question, resources.known_categories)
            print(f"[retrieve] filtro de metadata detectado: {where_filter or '(ninguno)'}")
            span.update(input={"question": question, "attempt": intento, "where_filter": where_filter})

            retrievers, _ = build_retrievers(
                resources.chroma_client, resources.embeddings, resources.cross_encoder, where_filter
            )
            if retrievers is None:
                print("[retrieve] -> 0 documentos (el filtro no matchea nada indexado)")
                span.update(output={"doc_count": 0})
                return {"where_filter": where_filter, "documents": []}

            docs = retrievers[HYBRID_RERANK_KEY].invoke(question)
            print(f"[retrieve] -> {len(docs)} documentos traidos")
            span.update(output={"doc_count": len(docs)})
            return {"where_filter": where_filter, "documents": docs}

    def grade_documents(state: RAGState) -> dict:
        with observe("span", "grade_documents") as span:
            print("[grade_documents] supervisor: revisando si los documentos sirven...")
            if not state["documents"]:
                print("[grade_documents] -> NOT_RELEVANT (no hay documentos)")
                span.update(output={"relevant": False})
                span.score_trace(name="retrieval_relevant", value=0, data_type="BOOLEAN")
                return {"retrieval_valid": False}

            context = build_context(state["documents"])
            prompt = RETRIEVAL_GRADING_PROMPT.format(context=context, question=state["standalone_question"])
            output = _ask_judge(resources.gemini_client, prompt, name="grade_documents")
            relevant = _parse_verdict(output, "RELEVANT", "NOT_RELEVANT")
            print(f"[grade_documents] -> {'RELEVANT, pasa al redactor' if relevant else 'NOT_RELEVANT'}")
            span.update(output={"relevant": relevant})
            span.score_trace(name="retrieval_relevant", value=1 if relevant else 0, data_type="BOOLEAN")
            return {"retrieval_valid": relevant}

    def transform_query(state: RAGState) -> dict:
        with observe("span", "transform_query") as span:
            print(f"[transform_query] traductor: reformulando '{state['standalone_question']}'...")
            span.update(input={"question": state["standalone_question"]})
            prompt = QUERY_TRANSFORM_PROMPT.format(question=state["standalone_question"])
            rewritten = _ask_judge(resources.gemini_client, prompt, name="transform_query")
            print(f"[transform_query] -> nueva version: '{rewritten}'")
            span.update(output={"rewritten": rewritten})
            return {
                "transformed_question": rewritten,
                "retrieval_attempts": state["retrieval_attempts"] + 1,
            }

    def generate(state: RAGState) -> dict:
        with observe("span", "generate") as span:
            intento = state["answer_attempts"] + 1
            print(f"[generate] redactor: escribiendo respuesta (intento {intento})...")
            span.update(input={"question": state["standalone_question"], "attempt": intento})
            answer = answer_with_llm(resources.gemini_client, state["documents"], state["standalone_question"])
            print(f"[generate] -> respuesta generada ({len(answer)} caracteres)")
            span.update(output={"answer": answer})
            return {"answer": answer, "answer_attempts": state["answer_attempts"] + 1}

    def validate_answer(state: RAGState) -> dict:
        with observe("span", "validate_answer") as span:
            print("[validate_answer] editor: chequeando la respuesta contra el contexto...")
            context = build_context(state["documents"])
            prompt = ANSWER_VALIDATION_RUBRIC.format(
                context=context, question=state["standalone_question"], answer=state["answer"]
            )
            output = _ask_judge(resources.gemini_client, prompt, name="validate_answer")
            score = _extract_score(output)
            valid = score is not None and score >= ANSWER_VALID_THRESHOLD
            print(f"[validate_answer] -> score={score} ({'aprobada' if valid else 'rechazada'}, umbral={ANSWER_VALID_THRESHOLD})")
            span.update(output={"score": score, "valid": valid})
            if score is not None:
                span.score_trace(name="answer_groundedness", value=score, data_type="NUMERIC")
            return {
                "answer_score": score,
                "answer_valid": valid,
            }

    def finalize_rejected(state: RAGState) -> dict:
        print("[finalize_rejected] recepcion de salida: entregando rechazo")
        return {"status": "rejected_query"}

    def finalize_no_context(state: RAGState) -> dict:
        print("[finalize_no_context] recepcion de salida: sin contexto util, entregando fallback")
        return {"answer": FALLBACK_ANSWER, "status": "no_relevant_context"}

    def finalize_low_confidence(state: RAGState) -> dict:
        print("[finalize_low_confidence] recepcion de salida: entregando respuesta con advertencia")
        return {"status": "low_confidence"}

    def finalize_ok(state: RAGState) -> dict:
        print("[finalize_ok] recepcion de salida: entregando respuesta aprobada")
        return {"status": "ok"}

    def route_after_query_validation(state: RAGState) -> str:
        decision = "retrieve" if state["query_valid"] else "reject"
        print(f"[route_after_query_validation] -> {decision}")
        return decision

    def route_after_intent(state: RAGState) -> str:
        decision = "meta" if state["intent"] else "content"
        print(f"[route_after_intent] -> {decision}")
        return decision

    def route_after_grading(state: RAGState) -> str:
        if state["retrieval_valid"]:
            decision = "generate"
        elif state["retrieval_attempts"] < MAX_RETRIEVAL_ATTEMPTS:
            decision = "transform"
        else:
            decision = "no_context"
        print(f"[route_after_grading] intentos={state['retrieval_attempts']}/{MAX_RETRIEVAL_ATTEMPTS} -> {decision}")
        return decision

    def route_after_validation(state: RAGState) -> str:
        if state["answer_valid"]:
            decision = "ok"
        elif state["answer_attempts"] < MAX_ANSWER_ATTEMPTS:
            decision = "retry"
        else:
            decision = "low_confidence"
        print(f"[route_after_validation] intentos={state['answer_attempts']}/{MAX_ANSWER_ATTEMPTS} -> {decision}")
        return decision

    workflow = StateGraph(RAGState)

    workflow.add_node("condense_question", condense_question)
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("handle_meta_query", handle_meta_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("validate_answer", validate_answer)
    workflow.add_node("finalize_rejected", finalize_rejected)
    workflow.add_node("finalize_no_context", finalize_no_context)
    workflow.add_node("finalize_low_confidence", finalize_low_confidence)
    workflow.add_node("finalize_ok", finalize_ok)

    workflow.set_entry_point("condense_question")
    workflow.add_edge("condense_question", "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"meta": "handle_meta_query", "content": "validate_query"},
    )
    workflow.add_edge("handle_meta_query", "finalize_ok")
    workflow.add_conditional_edges(
        "validate_query",
        route_after_query_validation,
        {"retrieve": "retrieve", "reject": "finalize_rejected"},
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "transform": "transform_query", "no_context": "finalize_no_context"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "validate_answer")
    workflow.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {"ok": "finalize_ok", "retry": "generate", "low_confidence": "finalize_low_confidence"},
    )
    workflow.add_edge("finalize_rejected", END)
    workflow.add_edge("finalize_no_context", END)
    workflow.add_edge("finalize_low_confidence", END)
    workflow.add_edge("finalize_ok", END)

    return workflow.compile()


def initial_state(question, chat_history=None):
    """Estado inicial para invocar el grafo con una pregunta nueva.

    chat_history: lista de (pregunta, respuesta) de turnos anteriores en la
    misma sesion de CLI -- el grafo no la acumula solo, hay que pasarsela
    en cada invocacion (ver graph_main.py)."""
    return {
        "question": question,
        "chat_history": chat_history or [],
        "standalone_question": "",
        "query_valid": True,
        "rejection_reason": "",
        "intent": "",
        "meta_match": None,
        "transformed_question": "",
        "where_filter": None,
        "documents": [],
        "retrieval_valid": False,
        "retrieval_attempts": 0,
        "answer": "",
        "answer_valid": False,
        "answer_score": None,
        "answer_attempts": 0,
        "status": "",
    }
