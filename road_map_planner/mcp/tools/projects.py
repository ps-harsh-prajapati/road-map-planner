from typing import List


PROJECTS = {
    "ai engineer": [
        {
            "name": "Document Q&A Assistant",
            "level": "Intermediate",
            "skills": [
                "Python",
                "LLMs",
                "embeddings",
                "RAG",
                "vector databases",
                "FastAPI",
            ],
            "description": (
                "Build an application that lets users upload documents "
                "and ask questions about their contents."
            ),
        },
        {
            "name": "AI Agent Task Planner",
            "level": "Intermediate",
            "skills": [
                "Python",
                "LLMs",
                "tool calling",
                "MCP",
                "API integration",
            ],
            "description": (
                "Build an agent that decides which tools to use to "
                "complete multi-step user tasks."
            ),
        },
        {
            "name": "Recommendation System",
            "level": "Intermediate",
            "skills": [
                "Python",
                "machine learning",
                "feature engineering",
                "model evaluation",
            ],
            "description": (
                "Build a recommendation engine for books, courses, "
                "movies, or products."
            ),
        },
    ],
    "machine learning": [
        {
            "name": "End-to-End ML Prediction System",
            "level": "Beginner",
            "skills": [
                "Python",
                "pandas",
                "scikit-learn",
                "model evaluation",
            ],
            "description": (
                "Train a machine learning model and expose it through "
                "a simple API."
            ),
        },
        {
            "name": "Customer Churn Prediction",
            "level": "Intermediate",
            "skills": [
                "Python",
                "data preprocessing",
                "classification",
                "model evaluation",
            ],
            "description": (
                "Predict which customers are likely to stop using "
                "a service."
            ),
        },
    ],
    "data science": [
        {
            "name": "Business Analytics Dashboard",
            "level": "Beginner",
            "skills": [
                "Python",
                "pandas",
                "data visualization",
                "SQL",
            ],
            "description": (
                "Analyze a real-world dataset and create a dashboard "
                "that explains important business trends."
            ),
        },
        {
            "name": "Customer Segmentation",
            "level": "Intermediate",
            "skills": [
                "Python",
                "pandas",
                "clustering",
                "data visualization",
            ],
            "description": (
                "Group customers into meaningful segments using "
                "unsupervised learning."
            ),
        },
    ],
    "full stack developer": [
        {
            "name": "Job Application Tracker",
            "level": "Beginner",
            "skills": [
                "HTML",
                "CSS",
                "JavaScript",
                "backend development",
                "database",
            ],
            "description": (
                "Build a web application for tracking job applications, "
                "interviews, and application status."
            ),
        },
        {
            "name": "Real-Time Collaboration App",
            "level": "Advanced",
            "skills": [
                "frontend development",
                "backend development",
                "WebSockets",
                "authentication",
                "databases",
            ],
            "description": (
                "Build a collaborative application where multiple "
                "users can update shared data in real time."
            ),
        },
    ],
    "cybersecurity": [
        {
            "name": "Security Log Analyzer",
            "level": "Beginner",
            "skills": [
                "Python",
                "log analysis",
                "regular expressions",
                "security fundamentals",
            ],
            "description": (
                "Build a tool that analyzes application or server logs "
                "and highlights suspicious patterns."
            ),
        },
        {
            "name": "Network Monitoring Dashboard",
            "level": "Intermediate",
            "skills": [
                "networking",
                "Python",
                "monitoring",
                "data visualization",
            ],
            "description": (
                "Create a dashboard that monitors network activity "
                "and highlights unusual behavior."
            ),
        },
    ],
}


def _find_projects(goal: str) -> List[dict]:
    normalized_goal = goal.lower().strip()

    for category, projects in PROJECTS.items():
        if category in normalized_goal:
            return projects

    matching_projects = []

    for category, projects in PROJECTS.items():
        category_words = category.split()

        if any(word in normalized_goal for word in category_words):
            matching_projects.extend(projects)

    return matching_projects


def recommend_projects(
    goal: str,
    experience_level: str = "beginner",
    limit: int = 3,
) -> str:
    """
    Recommend practical projects based on a career or education goal.
    """
    projects = _find_projects(goal)

    if not projects:
        return (
            f"No predefined projects were found for '{goal}'. "
            "The agent should use the user's goal to create "
            "custom project ideas."
        )

    level_order = {
        "beginner": 1,
        "intermediate": 2,
        "advanced": 3,
    }

    requested_level = level_order.get(
        experience_level.lower(),
        1,
    )

    suitable_projects = [
        project
        for project in projects
        if level_order.get(project["level"].lower(), 1)
        <= requested_level + 1
    ]

    if not suitable_projects:
        suitable_projects = projects

    selected_projects = suitable_projects[:limit]

    lines = [
        f"Project recommendations for: {goal}",
        f"Experience level: {experience_level}",
        "",
    ]

    for index, project in enumerate(selected_projects, start=1):
        lines.append(f"{index}. {project['name']}")
        lines.append(f"   Level: {project['level']}")
        lines.append(f"   {project['description']}")
        lines.append(
            "   Skills: " + ", ".join(project["skills"])
        )
        lines.append("")

    return "\n".join(lines)