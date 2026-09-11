import httpx


OPEN_LIBRARY_API = "https://openlibrary.org/search.json"


def search_books(topic: str, limit: int = 5) -> str:
    """
    Search Open Library for books related to a topic.

    Open Library is a free public API and does not require an API key.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                OPEN_LIBRARY_API,
                params={
                    "q": topic,
                    "limit": limit,
                    "fields": "title,author_name,first_publish_year,key",
                },
            )

            response.raise_for_status()
            data = response.json()

        docs = data.get("docs", [])

        if not docs:
            return f"No books found for '{topic}'."

        lines = [f"Recommended books related to '{topic}':", ""]

        for index, book in enumerate(docs[:limit], start=1):
            title = book.get("title", "Unknown title")

            authors = book.get("author_name", [])
            author = ", ".join(authors[:2]) if authors else "Unknown author"

            year = book.get("first_publish_year")
            year_text = f" ({year})" if year else ""

            key = book.get("key", "")
            link = f"https://openlibrary.org{key}" if key else ""

            lines.append(f"{index}. {title}")
            lines.append(f"   Author: {author}{year_text}")

            if link:
                lines.append(f"   {link}")

        return "\n".join(lines)

    except httpx.HTTPError as exc:
        return f"Book search failed: {exc}"