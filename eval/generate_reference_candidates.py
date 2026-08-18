
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from llm.answer import answer_with_llm, load_gemini_client
from retrieval.retrieval_methods import HYBRID_KEY, build_retrievers, load_models

DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.json"
CANDIDATES_PATH = DATA_DIR / "reference_candidates_round2.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EXTRA_ROUNDS = 3
THRESHOLD = 0.60


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def semantic_similarity(embedder, a, b):
    emb = embedder.encode([a, b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def main():
    dataset = load_golden_dataset()
    pending = [item for item in dataset if len(item["Respuestas"]) == 1]
    print(f"{len(pending)} preguntas con 1 sola referencia (de {len(dataset)} totales)")

    print("Construyendo retriever hibrido...")
    client, embeddings, cross_encoder = load_models()
    retrievers, _ = build_retrievers(client, embeddings, cross_encoder)
    retriever = retrievers[HYBRID_KEY]

    print("Cargando cliente Gemini y modelo de embeddings...")
    llm_client = load_gemini_client()
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    candidates_by_question = []

    for i, item in enumerate(pending, start=1):
        question = item["Pregunta"]
        original = item["Respuestas"][0]
        print(f"[{i}/{len(pending)}] {question}")

        docs = retriever.invoke(question)
        accepted = []
        seen_texts = set()

        for round_n in range(1, EXTRA_ROUNDS + 1):
            candidate = answer_with_llm(llm_client, docs, question)
            score = semantic_similarity(embedder, candidate, original)
            print(f"  ronda {round_n}: semantic={score:.3f}")
            if (
                score > THRESHOLD
                and candidate.strip() != original.strip()
                and candidate.strip() not in seen_texts
            ):
                accepted.append({"text": candidate, "semantic": score})
                seen_texts.add(candidate.strip())

        if accepted:
            candidates_by_question.append(
                {
                    "question": question,
                    "original": original,
                    "candidates": accepted,
                }
            )

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates_by_question, f, ensure_ascii=False, indent=2)

    total_candidates = sum(len(q["candidates"]) for q in candidates_by_question)
    print(f"\n{len(candidates_by_question)} preguntas con candidatos aceptados")
    print(f"{total_candidates} candidatos totales para revisar")
    print(f"Guardado en {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
