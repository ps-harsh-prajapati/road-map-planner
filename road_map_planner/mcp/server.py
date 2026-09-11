from mcp.server.fastmcp import FastMCP

from road_map_planner.mcp.tools.books import search_books
from road_map_planner.mcp.tools.projects import recommend_projects
from road_map_planner.mcp.tools.roadmap import build_roadmap_structure
from road_map_planner.mcp.tools.technology import get_current_technology
from road_map_planner.mcp.tools.timeline import create_preparation_timeline


mcp = FastMCP("road-map-planner")


@mcp.tool()
def health_check() -> str:
    """Check whether the Road Map Planner MCP server is running."""
    return "Road Map Planner MCP server is healthy."


@mcp.tool()
def technology_research(topic: str, limit: int = 5) -> str:
    """
    Research current technology activity related to a topic.
    """
    return get_current_technology(
        topic=topic,
        limit=limit,
    )


@mcp.tool()
def book_recommendations(
    topic: str,
    limit: int = 5,
) -> str:
    """
    Find books related to a learning topic.
    """
    return search_books(
        topic=topic,
        limit=limit,
    )


@mcp.tool()
def project_recommendations(
    goal: str,
    experience_level: str = "beginner",
    limit: int = 3,
) -> str:
    """
    Recommend practical real-world projects based on the user's
    goal and experience level.
    """
    return recommend_projects(
        goal=goal,
        experience_level=experience_level,
        limit=limit,
    )


@mcp.tool()
def preparation_timeline(
    target_months: int,
    hours_per_day: float,
    topics: list[str],
) -> str:
    """
    Create a preparation timeline based on available study time,
    target duration, and learning topics.
    """
    return create_preparation_timeline(
        target_months=target_months,
        hours_per_day=hours_per_day,
        topics=topics,
    )


@mcp.tool()
def roadmap_structure(
    goal: str,
    roadmap_type: str = "career",
    experience_level: str = "beginner",
    target_months: int = 6,
    hours_per_day: float = 2.0,
) -> str:
    """
    Create a structured roadmap skeleton using the user's goal,
    experience level, target duration, and study time.
    """
    return build_roadmap_structure(
        goal=goal,
        roadmap_type=roadmap_type,
        experience_level=experience_level,
        target_months=target_months,
        hours_per_day=hours_per_day,
    )


if __name__ == "__main__":
    mcp.run()