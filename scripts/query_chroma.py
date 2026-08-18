import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "kff_aca_articles"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#python scripts/query_chroma.py list_articles
#python scripts/query_chroma.py chunks 123
#python scripts/query_chroma.py full 456
#python scripts/query_chroma.py search "tu pregunta" 5
#python scripts/query_chroma.py delete_collection si

def get_collection():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def count():
    collection = get_collection()
    print(f"Total de chunks en la coleccion: {collection.count()}")


def peek(limit=3):
    collection = get_collection()
    preview = collection.peek(limit=int(limit))
    for doc_id, doc, meta in zip(preview["ids"], preview["documents"], preview["metadatas"]):
        print(f"{doc_id} | {meta['title']} | {doc[:80]}...")


def list_articles():
    collection = get_collection()
    all_items = collection.get()
    seen = {}
    for meta in all_items["metadatas"]:
        seen[meta["article_id"]] = meta["title"]
    for article_id, title in seen.items():
        print(f"  [{article_id}] {title}")


def chunks(article_id):
    collection = get_collection()
    filtered = collection.get(where={"article_id": int(article_id)})
    pairs = sorted(
        zip(filtered["metadatas"], filtered["documents"]),
        key=lambda x: x[0]["chunk_index"],
    )
    for meta, doc in pairs:
        print(f"{meta['article_id']}-{meta['chunk_index']}: {doc[:100]}...")


def full(article_id):
    collection = get_collection()
    result = collection.get(where={"article_id": int(article_id)})
    if not result["ids"]:
        print(f"No se encontro ningun chunk para article_id={article_id}")
        return

    pairs = sorted(
        zip(result["metadatas"], result["documents"]),
        key=lambda x: x[0]["chunk_index"],
    )
    print("Titulo:", pairs[0][0]["title"])
    print("URL:", pairs[0][0]["url"])
    print("Numero de chunks:", len(pairs))
    print()
    print("\n".join(doc for _, doc in pairs))


def search(pregunta, n_results=3):
    collection = get_collection()
    result = collection.query(query_texts=[pregunta], n_results=int(n_results))
    for doc_id, meta, dist in zip(
        result["ids"][0], result["metadatas"][0], result["distances"][0]
    ):
        print(f"  distancia={dist:.3f} | {doc_id} | {meta['title']}")


def delete_collection(confirm=""):
    if confirm != "si":
        print(f"Esto borra la coleccion '{COLLECTION_NAME}' completa de {CHROMA_DIR}.")
        print("Para confirmar: python query_chroma.py delete_collection si")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Coleccion '{COLLECTION_NAME}' eliminada de {CHROMA_DIR}.")


FUNCTIONS = {
    "count": count,
    "peek": peek,
    "list_articles": list_articles,
    "chunks": chunks,
    "full": full,
    "search": search,
    "delete_collection": delete_collection,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in FUNCTIONS:
        print(__doc__)
        return

    func_name = sys.argv[1]
    args = sys.argv[2:]
    FUNCTIONS[func_name](*args)


if __name__ == "__main__":
    main()
