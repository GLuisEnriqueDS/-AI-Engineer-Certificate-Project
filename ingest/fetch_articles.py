import csv
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.kff.org/wp-json/wp/v2/posts"
CATEGORIES_URL = "https://www.kff.org/wp-json/wp/v2/categories"
CATEGORY_ID = 3962798  # "Affordable Care Act"
PER_PAGE = 100
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "kff_aca_articles.csv"
OUTPUT_JSON = DATA_DIR / "kff_aca_articles.json"
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
        backoff_factor=2,  # 2s, 4s, 8s, 16s, 32s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


def clean_excerpt(raw_html):
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ").strip()
    return " ".join(text.split())


def fetch_category_map(session):
    """Trae todas las categorias del sitio y arma un dict {id: name}."""
    category_map = {}
    page = 1

    while True:
        response = session.get(
            CATEGORIES_URL,
            params={"per_page": PER_PAGE, "page": page, "_fields": "id,name"},
            timeout=30,
        )
        if response.status_code == 400:
            break
        response.raise_for_status()

        batch = response.json()
        if not batch:
            break

        category_map.update({item["id"]: item["name"] for item in batch})

        total_pages = int(response.headers.get("X-WP-TotalPages", page))
        if page >= total_pages:
            break
        page += 1

    return category_map


def fetch_page(session, page, max_attempts=5):
    params = {
        "categories": CATEGORY_ID,
        "per_page": PER_PAGE,
        "page": page,
        "_fields": "id,link,title,date,categories,primary_topic,excerpt",
    }
    for attempt in range(1, max_attempts + 1):
        try:
            return session.get(BASE_URL, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            wait = 2 * attempt
            print(f"  Error en pagina {page} (intento {attempt}/{max_attempts}): {exc}")
            if attempt == max_attempts:
                raise
            time.sleep(wait)


def fetch_all_articles():
    articles = []
    session = make_session()
    category_map = fetch_category_map(session)
    page = 1

    while True:
        response = fetch_page(session, page)

        if response.status_code == 400:
            # WordPress devuelve 400 cuando "page" excede el total de paginas
            break
        response.raise_for_status()

        batch = response.json()
        if not batch:
            break

        for item in batch:
            other_category_ids = [cid for cid in item["categories"] if cid != CATEGORY_ID]
            category = (
                category_map.get(other_category_ids[0], str(other_category_ids[0]))
                if other_category_ids
                else item["primary_topic"]
            )
            articles.append(
                {
                    "id": item["id"],
                    "title": item["title"]["rendered"],
                    "url": item["link"],
                    "date": item["date"],
                    "year": int(item["date"][:4]),
                    "category": category,
                    "primary_topic": item["primary_topic"],
                    "excerpt": clean_excerpt(item["excerpt"]["rendered"]),
                }
            )

        total_pages = int(response.headers.get("X-WP-TotalPages", page))
        print(f"Pagina {page}/{total_pages} -> {len(batch)} articulos")

        if page >= total_pages:
            break

        page += 1
        time.sleep(1)  # ser cortes con el servidor

    return articles


def save_csv(articles, path):
    fieldnames = ["id", "title", "url", "date", "year", "category", "primary_topic", "excerpt"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)


def save_json(articles, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    articles = fetch_all_articles()
    print(f"\nTotal de articulos obtenidos: {len(articles)}")

    save_csv(articles, OUTPUT_CSV)
    save_json(articles, OUTPUT_JSON)
    print(f"Guardado en {OUTPUT_CSV} y {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
