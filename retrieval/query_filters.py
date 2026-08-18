import re
from datetime import date

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

def get_known_categories(client, collection_name):
    collection = client.get_collection(name=collection_name)
    data = collection.get(include=["metadatas"])
    return sorted({meta["category"] for meta in data["metadatas"] if meta.get("category")})


def extract_year(question):
    match = YEAR_PATTERN.search(question)
    if not match:
        return None
    year = int(match.group())
    return min(year, date.today().year)


def extract_category(question, known_categories):
    question_lower = question.lower()
    for category in known_categories:
        if category.lower() in question_lower:
            return category
    return None


def build_where_filter(question, known_categories):
    year = extract_year(question)
    category = extract_category(question, known_categories)

    conditions = []
    if year is not None:
        conditions.append({"year": year})
    if category is not None:
        conditions.append({"category": category})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
