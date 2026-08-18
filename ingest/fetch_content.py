import argparse
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTICLES_JSON = DATA_DIR / "kff_aca_articles.json"
OUTPUT_JSON = DATA_DIR / "kff_aca_articles_content.json"
POST_URL_TEMPLATE = "https://www.kff.org/wp-json/wp/v2/posts/{id}"
TOP_N = 408 # Desde el 2020
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def make_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def load_top_articles(path, top_n):
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)
    articles.sort(key=lambda a: a["date"], reverse=True)
    return articles[:top_n]


def load_existing_content(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def strip_boilerplate(text):
    text = _strip_header(text)
    text = _strip_footer(text)
    return text


def _strip_header(text):
    marker = "Copy Link"
    pos = text.find(marker)
    if pos == -1:
        return text

    rest = text[pos + len(marker):]
    extra_marker = "\nAdd KFF on Google"
    if rest.startswith(extra_marker):
        rest = rest[len(extra_marker):]
    return rest.strip()


def _strip_footer(text):
    # "More On" siempre marca el inicio de la nube de tags al final.
    marker = "More On"
    pos = text.find(marker)
    if pos == -1:
        return text
    return text[:pos].strip()


def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup.select("script, style"):
        tag.decompose()

    # El bloque de "Related Content" viene embebido al final del contenido
    for related in soup.select(".wp-block-kff-shared-related-content"):
        related.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return strip_boilerplate("\n".join(lines))


def fetch_content(session, article_id):
    response = session.get(
        POST_URL_TEMPLATE.format(id=article_id),
        params={"_fields": "content"},
        timeout=30,
    )
    response.raise_for_status()
    raw_html = response.json()["content"]["rendered"]
    return clean_html(raw_html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-descarga el contenido de todos los articulos, incluso los ya presentes en el JSON.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Cantidad de articulos mas recientes a procesar (por defecto: todos).",
    )
    args = parser.parse_args()

    articles = load_top_articles(ARTICLES_JSON, args.top_n)
    existing_by_id = {a["id"]: a for a in load_existing_content(OUTPUT_JSON)}
    session = make_session()
    results = []
    fetched = 0
    reused = 0

    for i, article in enumerate(articles, start=1):
        cached = existing_by_id.get(article["id"])
        if cached and not args.force:
            print(f"[{i}/{len(articles)}] {article['title']} (ya descargado, se reusa)")
            results.append(cached)
            reused += 1
            continue

        print(f"[{i}/{len(articles)}] {article['title']}")
        text = fetch_content(session, article["id"])
        results.append(
            {
                "id": article["id"],
                "title": article["title"],
                "url": article["url"],
                "date": article["date"],
                "year": article["year"],
                "category": article["category"],
                "primary_topic": article["primary_topic"],
                "excerpt": article["excerpt"],
                "content": text,
            }
        )
        fetched += 1
        time.sleep(1)  # ser cortes con el servidor

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"\nGuardado contenido de {len(results)} articulos en {OUTPUT_JSON} "
        f"({fetched} descargados, {reused} reusados)"
    )


if __name__ == "__main__":
    main()
