"""
Project recommendations backed by a free, open-source catalog:
https://github.com/florinpop17/app-ideas

No API key and no signup. Data is read as plain text from GitHub's raw file host.

This module is designed to be used as an MCP tool by Road Map Planner.
The source, catalog, ranking logic, and recommendation quality are preserved.
"""

import json
import re
from typing import Any

import httpx


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ★★★  SOURCE — these lines point the code at the open-source repo  ★★★ ║
# ╚═══════════════════════════════════════════════════════════════════════╝

SOURCE_REPO = "florinpop17/app-ideas"       # ★ repo
SOURCE_BRANCH = "master"                    # ★ branch

# ★ Human-readable GitHub repository page
SOURCE_URL = f"https://github.com/{SOURCE_REPO}"

# ★ Raw GitHub file host
RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_BRANCH}"
)

# ★ Catalog: Name | Description | Tier
CATALOG_URL = f"{RAW_BASE_URL}/README.md"

# ─────────────────────────────────────────────────────────────────────────


TIERS = {
    "beginner": "1-Beginner",
    "intermediate": "2-Intermediate",
    "advanced": "3-Advanced",
}


# Words that say nothing about the domain,
# so they are ignored when matching the goal.
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "want",
    "become",
    "learn",
    "job",
    "career",
    "developer",
    "engineer",
    "programmer",
    "professional",
    "how",
    "make",
}


# Expected README table format:
# | [Project Name](./path/to/project.md) | Description | 1-Beginner |
_ROW = re.compile(
    r"^\|\s*"
    r"(?P<name>[^|]+?)"
    r"\s*\|\s*"
    r"(?P<desc>[^|]+?)"
    r"\s*\|\s*"
    r"(?P<tier>[123]-\w+)"
    r"\s*\|?\s*$"
)

# Markdown link:
# [Title](https://example.com)
_LINK = re.compile(
    r"\[(?P<title>[^\]]+)\]\((?P<href>[^)\s]+)\)"
)


def _get_text(url: str) -> str:
    """
    Download plain text from GitHub.

    Raises an exception when the request fails so the calling function
    can return a structured JSON error instead of crashing the MCP server.
    """
    with httpx.Client(
        timeout=20.0,
        follow_redirects=True,
    ) as client:

        response = client.get(
            url,
            headers={
                "User-Agent": "road-map-planner-project-recommender"
            },
        )

    response.raise_for_status()
    return response.text


def _load_catalog() -> list[dict[str, Any]]:
    """
    Parse the README table into:

    {
        "name": ...,
        "description": ...,
        "tier": ...,
        "path": ...
    }
    """

    # ★ SOURCE FETCH #1: the catalog
    text = _get_text(CATALOG_URL)

    projects: list[dict[str, Any]] = []

    for line in text.splitlines():

        match = _ROW.match(line.strip())

        if not match:
            continue

        name_cell = match["name"]

        link = _LINK.search(name_cell)

        title = link["title"] if link else name_cell

        path = (
            re.sub(r"^\./", "", link["href"])
            if link
            else None
        )

        projects.append(
            {
                "name": title.replace("🌟", "").strip(),
                "description": match["desc"].strip(),
                "tier": match["tier"],
                "path": path,
            }
        )

    return projects


def _section(markdown: str, heading: str) -> str:
    """
    Extract a Markdown section such as:

    ## Description

    text...

    ## User Stories
    """

    pattern = rf"^##\s+{re.escape(heading)}[^\n]*\n(.*?)(?=^##\s|\Z)"

    match = re.search(
        pattern,
        markdown,
        re.S | re.M | re.I,
    )

    return match.group(1).strip() if match else ""


def _bullets(block: str, max_items: int) -> list[str]:
    """
    Extract bullet-point items from a Markdown section.
    """

    items: list[str] = []

    for line in block.splitlines():

        line = line.strip()

        if not re.match(r"^[-*]\s", line):
            continue

        # Remove bullet, optional checkbox and bold markers.
        line = re.sub(
            r"^[-*]\s*(\[[ xX]\]\s*)?",
            "",
            line,
        )

        line = line.replace("**", "")

        line = re.sub(
            r"^User [Ss]tory:?\s*",
            "",
            line,
        )

        if line:
            items.append(line)

    return items[:max_items]


def _load_details(project: dict[str, Any]) -> dict[str, Any]:
    """
    Fetch one project's Markdown page and pull out the useful sections.

    This is intentionally kept separate from catalog loading so a failure
    on one project's page does not destroy the entire recommendation result.
    """

    if not project["path"]:
        return {}

    # ★ SOURCE FETCH #2: one project page
    markdown = _get_text(
        f"{RAW_BASE_URL}/{project['path']}"
    )

    description = _section(
        markdown,
        "Description",
    )

    resources = _LINK.findall(
        _section(
            markdown,
            "Useful links",
        )
    )

    return {
        "details": description[:600],
        "user_stories": _bullets(
            _section(markdown, "User Stories"),
            8,
        ),
        "bonus_features": _bullets(
            _section(markdown, "Bonus features"),
            5,
        ),
        "resources": [
            {
                "title": title,
                "url": url,
            }
            for title, url in resources[:5]
        ],
    }


def _tokens(text: str) -> set[str]:
    """
    Convert text into simple normalized tokens.

    Plural words ending with 's' are reduced to their singular form.
    """

    return {
        word.rstrip("s")
        for word in re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
        if len(word) > 2 and word not in STOPWORDS
    }


def _rank(
    projects: list[dict[str, Any]],
    goal: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Rank projects by the number of overlapping tokens between:

    user's goal
    +
    project name/description
    """

    goal_tokens = _tokens(goal)

    def score(project: dict[str, Any]) -> int:
        project_text = (
            f"{project['name']} {project['description']}"
        )

        project_tokens = _tokens(project_text)

        return len(
            goal_tokens & project_tokens
        )

    ranked = sorted(
        projects,
        key=score,
        reverse=True,
    )

    matched = [
        project
        for project in ranked
        if score(project) > 0
    ]

    return matched, ranked


def project_recommendations(
    goal: str,
    experience_level: str = "beginner",
    limit: int = 3,
) -> str:
    """
    MCP-compatible project recommendation tool for Road Map Planner.

    Parameters
    ----------
    goal:
        User's career, education, technology, or project goal.

    experience_level:
        beginner | intermediate | advanced

    limit:
        Number of project recommendations to return.
        Maximum allowed value is 5.

    Returns
    -------
    str
        JSON string containing project recommendations and source data.
    """

    # ------------------------------------------------------------
    # 1. Validate goal
    # ------------------------------------------------------------

    goal = goal.strip()

    if not goal:
        return json.dumps(
            {
                "found": False,
                "message": "No goal was provided.",
            },
            indent=2,
        )

    # ------------------------------------------------------------
    # 2. Validate experience level
    # ------------------------------------------------------------

    normalized_experience = (
        experience_level.strip().lower()
    )

    tier = TIERS.get(
        normalized_experience
    )

    if tier is None:
        return json.dumps(
            {
                "found": False,
                "message": (
                    "experience_level must be one of: "
                    f"{', '.join(TIERS)}."
                ),
            },
            indent=2,
        )

    # ------------------------------------------------------------
    # 3. Limit recommendations
    # ------------------------------------------------------------

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 3

    limit = max(
        1,
        min(limit, 5),
    )

    # ------------------------------------------------------------
    # 4. Load catalog
    # ------------------------------------------------------------

    try:
        catalog = _load_catalog()

    except Exception as exc:

        return json.dumps(
            {
                "found": False,
                "source": SOURCE_REPO,
                "source_url": SOURCE_URL,
                "error": (
                    f"Unable to contact GitHub: {exc}"
                ),
            },
            indent=2,
        )

    # ------------------------------------------------------------
    # 5. Filter by experience tier
    # ------------------------------------------------------------

    tier_projects = [
        project
        for project in catalog
        if project["tier"] == tier
    ]

    if not tier_projects:

        return json.dumps(
            {
                "found": False,
                "source": SOURCE_REPO,
                "source_url": SOURCE_URL,
                "message": (
                    f"No projects found for tier {tier}."
                ),
            },
            indent=2,
        )

    # ------------------------------------------------------------
    # 6. Rank projects according to user's goal
    # ------------------------------------------------------------

    matched, ranked = _rank(
        tier_projects,
        goal,
    )

    goal_matched = bool(matched)

    # Prefer goal-matched projects.
    # Fall back to ranked projects if there are no direct matches.
    selected = (
        matched or ranked
    )[:limit]

    # ------------------------------------------------------------
    # 7. Load detailed information
    # ------------------------------------------------------------

    projects: list[dict[str, Any]] = []

    for project in selected:

        try:
            extra = _load_details(project)

        except Exception:
            # Keep the catalog entry even when its detail page fails.
            extra = {}

        project_url = (
            f"{SOURCE_URL}/blob/"
            f"{SOURCE_BRANCH}/"
            f"{project['path']}"
            if project["path"]
            else SOURCE_URL
        )

        projects.append(
            {
                "name": project["name"],
                "summary": project["description"],
                "tier": project["tier"],
                "url": project_url,
                **extra,
            }
        )

    # ------------------------------------------------------------
    # 8. Provide additional ideas from the same tier
    # ------------------------------------------------------------

    selected_names = {
        project["name"]
        for project in selected
    }

    other_ideas = [
        {
            "name": project["name"],
            "summary": project["description"],
        }
        for project in tier_projects
        if project["name"] not in selected_names
    ][:15]

    # ------------------------------------------------------------
    # 9. Return structured data to the AI agent
    # ------------------------------------------------------------

    result = {
        "found": True,

        # ★ Source information
        "source": SOURCE_REPO,
        "source_url": SOURCE_URL,

        # ★ User request information
        "goal": goal,
        "experience_level": normalized_experience,

        # ★ Matching information
        "goal_matched": goal_matched,

        # ★ Selected recommendations
        "projects": projects,

        # ★ Other projects from the same difficulty tier
        "other_ideas_in_tier": other_ideas,

        # ★ Instructions for Road Map Planner's LLM
        "instruction_to_agent": (
            "Recommend the projects above, adapting them to "
            "the user's goal. "

            "The catalog is a general list of app ideas, "
            "not profession-specific. "

            "If goal_matched is false, say so and adapt the "
            "closest ideas to the goal's domain instead of "
            "implying they were written for it. "

            "Do not invent details that are not in the "
            "project data."
        ),
    }

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )


# ================================================================
# Backward compatibility
# ================================================================
#
# If your existing Road Map Planner code imports:
#
#     recommend_projects
#
# it will continue to work.
#
# If your MCP server exposes:
#
#     project_recommendations
#
# that name is also available.
# ================================================================

recommend_projects = project_recommendations