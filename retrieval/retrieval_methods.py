#python retrieval/retrieval_methods.py

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # los modelos ya estan cacheados; evita el HEAD a huggingface.co

import chromadb
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from retrieval.query_filters import build_where_filter, get_known_categories

CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "kff_aca_articles"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K = 5

VECTOR_KEY = "1) Vectorial"
BM25_KEY = "2) BM25"
HYBRID_KEY = "3) Hibrido (vector + BM25)"
HYBRID_RERANK_KEY = "4) Hibrido + reranking"

SCORED_METHODS = {VECTOR_KEY, BM25_KEY}


def load_models():
    """Carga lo pesado (cliente Chroma + modelos) una sola vez."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    return client, embeddings, cross_encoder


def load_documents(client, where_filter=None):
    """Trae los chunks de Chroma (filtrados o no) como Document de LangChain."""
    raw_collection = client.get_collection(name=COLLECTION_NAME)
    data = raw_collection.get(where=where_filter) if where_filter else raw_collection.get()

    return [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]


def build_retrievers(client, embeddings, cross_encoder, where_filter=None):
    """Construye los 4 retrievers, opcionalmente acotados a `where_filter`."""
    documents = load_documents(client, where_filter)
    if where_filter and not documents:
        # No hay chunks que cumplan el filtro: no tiene sentido buscar en
        # todo el resto de la coleccion como si el filtro no existiera.
        return None, None

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": K, "filter": where_filter})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = K

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    reranker = CrossEncoderReranker(model=cross_encoder, top_n=K)
    hybrid_rerank_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )

    retrievers = {
        VECTOR_KEY: vector_retriever,
        BM25_KEY: bm25_retriever,
        HYBRID_KEY: hybrid_retriever,
        HYBRID_RERANK_KEY: hybrid_rerank_retriever,
    }

    # search_fn(query) -> lista de (doc, score), solo para los metodos que
    # exponen un score real (vectorial y BM25).
    scored_search_fns = {
        VECTOR_KEY: lambda query: vector_store.similarity_search_with_relevance_scores(
            query, k=K, filter=where_filter
        ),
        BM25_KEY: lambda query: bm25_search_with_scores(bm25_retriever, query, k=K),
    }

    return retrievers, scored_search_fns


def bm25_search_with_scores(bm25_retriever, query, k=K):
    tokenized_query = bm25_retriever.preprocess_func(query)
    scores = bm25_retriever.vectorizer.get_scores(tokenized_query)
    ranked = sorted(zip(bm25_retriever.docs, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


def print_results(method_name, docs):
    print(f"\n--- {method_name} ---")
    if not docs:
        print("  (sin resultados)")
        return
    for rank, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "?")
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  {rank}. {title}\n     {snippet}...")


def print_scored_results(method_name, doc_score_pairs):
    print(f"\n--- {method_name} ---")
    if not doc_score_pairs:
        print("  (sin resultados)")
        return
    for rank, (doc, score) in enumerate(doc_score_pairs, start=1):
        title = doc.metadata.get("title", "?")
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  {rank}. (score={score:.4f}) {title}\n     {snippet}...")


def compare_methods(retrievers, scored_search_fns, question):
    """Corre los 4 metodos, imprime cada uno y devuelve {metodo: [docs]}."""
    results = {}

    for method_name, search_fn in scored_search_fns.items():
        doc_score_pairs = search_fn(question)
        print_scored_results(method_name, doc_score_pairs)
        results[method_name] = [doc for doc, _ in doc_score_pairs]

    for method_name, retriever in retrievers.items():
        if method_name in SCORED_METHODS:
            continue
        docs = retriever.invoke(question)
        print_results(method_name, docs)
        results[method_name] = docs

    return results


def main():
    print("Cargando modelos (embeddings, cross-encoder)...")
    client, embeddings, cross_encoder = load_models()
    known_categories = get_known_categories(client, COLLECTION_NAME)
    print(f"Categorias conocidas: {', '.join(known_categories)}")
    print("Listo. Escribe una pregunta (o 'salir' para terminar).\n")

    while True:
        query = input("Pregunta> ").strip()
        if not query:
            continue
        if query.lower() in {"salir", "exit", "quit"}:
            break

        where_filter = build_where_filter(query, known_categories)
        print(f"Filtro detectado: {where_filter or '(ninguno)'}")

        retrievers, scored_search_fns = build_retrievers(
            client, embeddings, cross_encoder, where_filter
        )
        if retrievers is None:
            print(f"Sin articulos indexados que cumplan el filtro {where_filter}.")
            print()
            continue

        compare_methods(retrievers, scored_search_fns, query)
        print()


if __name__ == "__main__":
    main()
