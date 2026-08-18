import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer

from llm.answer import answer_with_llm, load_gemini_client
from retrieval.retrieval_methods import HYBRID_KEY, build_retrievers, load_models

DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.json"
RESULTS_PATH = DATA_DIR / "eval_results.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def normalize_text(text):
    """Normalizacion estilo SQuAD: minusculas, sin puntuacion, sin articulos."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def best_scores(prediction, references, rouge, embedder):
    """Puntua contra cada referencia y devuelve el maximo por metrica
    (estilo SQuAD), junto con cual referencia gano cada metrica."""
    best = {"f1": 0.0, "rougeL": 0.0, "semantic": -1.0}
    best_ref = {"f1": None, "rougeL": None, "semantic": None}

    pred_emb = embedder.encode([prediction], normalize_embeddings=True)[0]

    for reference in references:
        f1 = token_f1(prediction, reference)
        if f1 > best["f1"]:
            best["f1"], best_ref["f1"] = f1, reference

        rouge_l = rouge.compute(predictions=[prediction], references=[reference])["rougeL"]
        if rouge_l > best["rougeL"]:
            best["rougeL"], best_ref["rougeL"] = rouge_l, reference

        ref_emb = embedder.encode([reference], normalize_embeddings=True)[0]
        semantic = float(np.dot(pred_emb, ref_emb))
        if semantic > best["semantic"]:
            best["semantic"], best_ref["semantic"] = semantic, reference

    return best, best_ref


def run_evaluation(dataset):
    print("Construyendo retriever hibrido...")
    client, embeddings, cross_encoder = load_models()
    retrievers, _ = build_retrievers(client, embeddings, cross_encoder)
    retriever = retrievers[HYBRID_KEY]

    print("Cargando cliente Gemini y modelo de embeddings...")
    client = load_gemini_client()
    rouge = evaluate.load("rouge")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    rows = []
    for i, item in enumerate(dataset, start=1):
        question = item["Pregunta"]
        references = item["Respuestas"]
        print(f"[{i}/{len(dataset)}] {question}")

        docs = retriever.invoke(question)
        prediction = answer_with_llm(client, docs, question)

        best, best_ref = best_scores(prediction, references, rouge, embedder)

        rows.append(
            {
                "question": question,
                "expected": references[0],
                "observed": prediction,
                "f1": best["f1"],
                "rougeL": best["rougeL"],
                "semantic": best["semantic"],
                "n_references": len(references),
                "best_reference": {
                    "f1": best_ref["f1"],
                    "rougeL": best_ref["rougeL"],
                    "semantic": best_ref["semantic"],
                },
            }
        )

    return rows


def save_results(rows):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado en {RESULTS_PATH}")


def main():
    dataset = load_golden_dataset()
    rows = run_evaluation(dataset)
    save_results(rows)


if __name__ == "__main__":
    main()
