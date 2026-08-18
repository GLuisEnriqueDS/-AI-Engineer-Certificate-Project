import difflib

def load_articles(client, collection_name):
    """{article_id: metadata} -> lista de metadata, tomando el primer chunk
    de cada articulo (alcanza porque title/url/date/excerpt se repiten
    identicos en todos los chunks de un mismo articulo). Pensado para
    llamarse una vez al arrancar, no en cada pregunta."""
    collection = client.get_collection(name=collection_name)
    data = collection.get(include=["metadatas"])
    seen = {}
    for meta in data["metadatas"]:
        seen.setdefault(meta["article_id"], meta)
    return list(seen.values())


def count_articles(articles):
    return len(articles)


def list_recent_articles(articles, n=5):
    recent = sorted(articles, key=lambda meta: meta["date"], reverse=True)
    return [
        {"title": meta["title"], "url": meta["url"], "date": meta["date"]}
        for meta in recent[:n]
    ]


def find_article_by_title(articles, title_query):
    """Matchea title_query (el titulo tal como lo extrajo Gemini de la
    pregunta, no necesariamente textual/exacto) contra los titulos
    indexados: primero por contencion literal (en cualquier direccion),
    despues por similitud aproximada (difflib)."""
    if not title_query or not title_query.strip():
        return None

    query_lower = title_query.lower().strip()

    for meta in articles:
        title_lower = meta["title"].lower()
        if query_lower in title_lower or title_lower in query_lower:
            return meta

    titles = [meta["title"] for meta in articles]
    close = difflib.get_close_matches(title_query, titles, n=1, cutoff=0.5)
    if not close:
        return None
    return next(meta for meta in articles if meta["title"] == close[0])


def get_article_excerpt(articles, title_query):
    meta = find_article_by_title(articles, title_query)
    if meta is None:
        return None
    return {"title": meta["title"], "url": meta["url"], "excerpt": meta.get("excerpt", "")}
