import httpx


STACK_EXCHANGE_API = "https://api.stackexchange.com/2.3"
GITHUB_API = "https://api.github.com"


def _normalize_topic(topic: str) -> str:
    """Normalize a user topic for API searches."""
    return topic.strip().lower()


def _normalize_stackoverflow_tag(topic: str) -> str:
    """
    Convert a simple topic into a likely Stack Overflow tag.
    """
    return (
        topic.strip()
        .lower()
        .replace(" ", "-")
        .replace("_", "-")
    )


def _search_github(
    client: httpx.Client,
    topic: str,
    limit: int = 5,
) -> list[dict]:
    """
    Search public GitHub repositories related to a topic.
    """
    response = client.get(
        f"{GITHUB_API}/search/repositories",
        params={
            "q": topic,
            "sort": "stars",
            "order": "desc",
            "per_page": limit,
        },
        headers={
            "Accept": "application/vnd.github+json",
        },
    )

    response.raise_for_status()

    return response.json().get("items", [])


def _search_stackoverflow(
    client: httpx.Client,
    topic: str,
    limit: int = 5,
) -> list[dict]:
    """
    Get recent Stack Overflow questions for a topic tag.
    """
    tag = _normalize_stackoverflow_tag(topic)

    response = client.get(
        f"{STACK_EXCHANGE_API}/questions",
        params={
            "site": "stackoverflow",
            "order": "desc",
            "sort": "activity",
            "tagged": tag,
            "pagesize": limit,
        },
    )

    response.raise_for_status()

    return response.json().get("items", [])


def get_current_technology(
    topic: str,
    limit: int = 5,
) -> str:
    """
    Research a technology topic using public GitHub and
    Stack Overflow data.

    This is intended to give the agent evidence about:
    - active open-source projects
    - commonly discussed technical topics
    - developer activity
    """
    normalized_topic = _normalize_topic(topic)

    if not normalized_topic:
        return "Technology topic cannot be empty."

    try:
        with httpx.Client(timeout=15.0) as client:
            github_repositories = _search_github(
                client,
                normalized_topic,
                limit,
            )

            stackoverflow_questions = _search_stackoverflow(
                client,
                normalized_topic,
                limit,
            )

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code

        if status == 403:
            return (
                "Technology research was rate-limited by a public API. "
                "Try again later."
            )

        return (
            f"Technology research API returned HTTP {status}."
        )

    except httpx.RequestError as exc:
        return (
            f"Technology research failed because of a network error: {exc}"
        )

    lines = [
        f"Current technology research for: {topic}",
        "",
        "GitHub open-source activity",
        "----------------------------",
    ]

    if github_repositories:
        for index, repository in enumerate(
            github_repositories[:limit],
            start=1,
        ):
            name = repository.get(
                "full_name",
                "Unknown repository",
            )

            stars = repository.get(
                "stargazers_count",
                0,
            )

            description = repository.get(
                "description",
                "",
            ) or "No description available."

            url = repository.get(
                "html_url",
                "",
            )

            lines.append(
                f"{index}. {name} — {stars:,} stars"
            )
            lines.append(
                f"   {description}"
            )

            if url:
                lines.append(
                    f"   {url}"
                )

    else:
        lines.append(
            "No GitHub repositories were found."
        )

    lines.extend(
        [
            "",
            "Recent Stack Overflow activity",
            "-------------------------------",
        ]
    )

    if stackoverflow_questions:
        for index, question in enumerate(
            stackoverflow_questions[:limit],
            start=1,
        ):
            title = question.get(
                "title",
                "Untitled question",
            )

            link = question.get(
                "link",
                "",
            )

            score = question.get(
                "score",
                0,
            )

            answers = question.get(
                "answer_count",
                0,
            )

            lines.append(
                f"{index}. {title}"
            )
            lines.append(
                f"   Score: {score} | Answers: {answers}"
            )

            if link:
                lines.append(
                    f"   {link}"
                )

    else:
        lines.append(
            "No recent Stack Overflow activity was found."
        )

    lines.extend(
        [
            "",
            "Interpretation guidance",
            "----------------------",
            (
                "GitHub results indicate active open-source interest "
                "and should not be treated as a definitive ranking."
            ),
            (
                "Stack Overflow activity indicates developer discussion "
                "and should not by itself be treated as proof that a "
                "technology is the newest or best choice."
            ),
        ]
    )

    return "\n".join(lines)