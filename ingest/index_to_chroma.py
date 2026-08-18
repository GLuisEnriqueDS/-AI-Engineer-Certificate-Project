import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # el modelo ya esta cacheado; evita el HEAD a huggingface.co

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTENT_JSON = PROJECT_ROOT / "data" / "kff_aca_articles_content.json"
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "kff_aca_articles"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def chunk_text(text):
    return _splitter.split_text(text)


def build_chunks(articles):
    ids, documents, metadatas = [], [], []

    for article in articles:
        pieces = chunk_text(article["content"])
        for idx, piece in enumerate(pieces):
            ids.append(f"{article['id']}-{idx}")
            documents.append(piece)
            metadatas.append(
                {
                    "article_id": article["id"],
                    "title": article["title"],
                    "url": article["url"],
                    "date": article["date"],
                    "chunk_index": idx,
                    "year": article["year"],
                    "category": article["category"],
                    "primary_topic": article["primary_topic"],
                    "excerpt": article["excerpt"],
                }
            )

    return ids, documents, metadatas


def get_indexed_article_ids(collection):
    result = collection.get(include=["metadatas"])
    return {m["article_id"] for m in result["metadatas"]}


def upsert_in_batches(collection, ids, documents, metadatas, batch_size=5000):
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"  Embebiendo chunks {start + 1}-{end} de {total}...")
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embebe y reemplaza todos los articulos, incluso los ya indexados.",
    )
    args = parser.parse_args()

    with open(CONTENT_JSON, encoding="utf-8") as f:
        articles = json.load(f)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    if args.force:
        pending = articles
    else:
        indexed_ids = get_indexed_article_ids(collection)
        pending = [a for a in articles if a["id"] not in indexed_ids]
        skipped = len(articles) - len(pending)
        if skipped:
            print(f"{skipped} articulo(s) ya indexado(s), se saltean (usar --force para re-embeberlos)")

    if not pending:
        print("Nada nuevo para indexar.")
        return

    if args.force:
        # Antes de reemplazar, borramos los chunks viejos de estos articulos:
        # si el nuevo contenido genera menos chunks que antes, un upsert solo
        # dejaria chunks huerfanos con indices que ya no existen.
        for article in pending:
            collection.delete(where={"article_id": article["id"]})

    ids, documents, metadatas = build_chunks(pending)
    print(f"{len(pending)} articulo(s) -> {len(documents)} chunks")

    upsert_in_batches(collection, ids, documents, metadatas)
    print(f"Guardado en Chroma ({CHROMA_DIR}), coleccion '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
