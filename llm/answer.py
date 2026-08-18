import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from llm.observability import observe

GEMINI_MODEL = "gemini-2.5-flash"
TOP_K_CONTEXT = 3

RETRYABLE_STATUS_CODES = {429, 500, 503}
MAX_RETRIES = 4
BASE_DELAY_SECONDS = 2

GENERATION_CONFIG = types.GenerateContentConfig(temperature=0.15, top_p=0.9)

PROMPT_TEMPLATE_1 = """You are an assistant that answers questions using ONLY the context provided, extracted from KFF (Kaiser Family Foundation) articles about the Affordable Care Act.

Always answer in English, regardless of the language of the question. Be direct and concise: lead with the specific fact, number, or name requested, in as few words as possible, then add at most one short sentence of supporting detail if needed. Do not restate the question or add filler. When stating a specific fact, number, or name, reuse the exact wording from the context instead of paraphrasing it.

If the context does not contain enough information to answer, say so explicitly instead of making something up.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_2 = """You are an assistant that answers questions using ONLY the context provided, extracted from KFF (Kaiser Family Foundation) articles about the Affordable Care Act.

Always answer in English, regardless of the language of the question. Give a thorough, explicit answer: explain the reasoning and relevant context behind the fact, not just the fact itself, developing it in as many sentences as needed for a complete answer. When stating a specific fact, number, or name, reuse the exact wording from the context instead of paraphrasing it, and explicitly point out which part of the context it comes from (e.g. "According to [title], ...").

If the context does not contain enough information to answer, say so explicitly instead of making something up.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE = PROMPT_TEMPLATE_2


def load_gemini_client():
    load_dotenv()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["VERTEX_CREDENTIALS_JSON"]

    return genai.Client(
        vertexai=True,
        project=os.environ["VERTEX_PROJECT_ID"],
        location=os.environ["VERTEX_LOCATION"],
    )


def _response_output(response):
    """Texto de la respuesta, o la(s) tool call(s) si no hay texto (caso de
    classify_intent, que usa function-calling y no genera texto)."""
    try:
        if response.text:
            return response.text
    except Exception:
        pass
    calls = getattr(response, "function_calls", None)
    if calls:
        return [{"name": c.name, "args": dict(c.args or {})} for c in calls]
    return None


def call_gemini(client, name="gemini_call", **kwargs):
    """generate_content con retry + backoff exponencial (+jitter) ante 429
    (RESOURCE_EXHAUSTED, cuota) y errores transitorios de servidor (500/503).

    Punto de entrada unico al SDK: graph/pipeline.py (_ask_judge,
    classify_intent) y answer_with_llm en este modulo pasan por aca, para no
    repetir la logica de retry en cada nodo del grafo. Sin acceso para subir
    la cuota de Vertex AI, reintentar con espera es la unica mitigacion
    disponible del lado del codigo.

    Tambien es el unico lugar que envuelve la llamada como "generation" de
    Langfuse (ver llm/observability.py) -- asi las 3 formas de llamar a
    Gemini en el proyecto (_ask_judge, classify_intent, answer_with_llm)
    quedan trazadas con tokens reales de response.usage_metadata, sin
    duplicar esa logica en cada una."""
    last_error = None
    with observe("generation", name, model=kwargs.get("model")) as gen:
        gen.update(input=kwargs.get("contents"))
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(**kwargs)
                usage = response.usage_metadata
                gen.update(
                    output=_response_output(response),
                    usage_details={
                        "input": usage.prompt_token_count,
                        "output": usage.candidates_token_count,
                        "total": usage.total_token_count,
                    }
                    if usage
                    else None,
                )
                return response
            except (genai_errors.ClientError, genai_errors.ServerError) as exc:
                if exc.code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                    raise
                last_error = exc

            delay = BASE_DELAY_SECONDS * (2**attempt) + random.uniform(0, 1)
            print(f"[call_gemini] {last_error.code} {last_error.status} -- reintentando en {delay:.1f}s (intento {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(delay)


def build_context(docs):
    parts = []
    for doc in docs:
        title = doc.metadata.get("title", "?")
        url = doc.metadata.get("url", "?")
        parts.append(f"[{title}]({url})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_with_llm(client, docs, question, prompt_template=PROMPT_TEMPLATE):
    context = build_context(docs)
    prompt = prompt_template.format(context=context, question=question)
    response = call_gemini(
        client, name="generate", model=GEMINI_MODEL, contents=prompt, config=GENERATION_CONFIG
    )
    return response.text
