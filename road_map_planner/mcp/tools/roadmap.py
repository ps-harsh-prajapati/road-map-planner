from math import ceil


def build_roadmap_structure(
    goal: str,
    roadmap_type: str = "career",
    experience_level: str = "beginner",
    target_months: int = 6,
    hours_per_day: float = 2.0,
) -> str:
    """
    Build a structured roadmap skeleton based on the user's
    goal and constraints.

    This tool does not decide the exact technologies.
    The agent uses other MCP tools for current technology,
    books, and projects.
    """

    if not goal.strip():
        return "Goal cannot be empty."

    if roadmap_type.lower() not in {"career", "education"}:
        return "roadmap_type must be 'career' or 'education'."

    if target_months <= 0:
        return "target_months must be greater than 0."

    if hours_per_day <= 0:
        return "hours_per_day must be greater than 0."

    normalized_level = experience_level.lower().strip()

    valid_levels = {
        "beginner",
        "intermediate",
        "advanced",
    }

    if normalized_level not in valid_levels:
        normalized_level = "beginner"

    total_hours = target_months * 30 * hours_per_day
    weeks = target_months * 4

    # Use five broad phases, while ensuring short roadmaps
    # do not receive more phases than available months.
    phase_count = min(target_months, 5)

    base_months = target_months // phase_count
    extra_months = target_months % phase_count

    phase_months: list[int] = []

    for index in range(phase_count):
        months = base_months

        if index < extra_months:
            months += 1

        phase_months.append(max(1, months))

    phases: list[str] = []

    phase_names = [
        "Foundation",
        "Core Skills",
        "Intermediate Development",
        "Projects & Practical Experience",
        "Career / Exam Preparation",
    ]

    for index, months in enumerate(phase_months):
        phase_name = phase_names[index]

        phases.append(
            f"{phase_name}: {months} month(s)"
        )

    if roadmap_type.lower() == "education":
        outcome = (
            "Build the required academic knowledge, practice skills, "
            "complete projects or assignments, and prepare for assessment."
        )
    else:
        outcome = (
            "Build job-relevant skills, complete portfolio projects, "
            "demonstrate practical ability, and prepare for interviews."
        )

    lines = [
        "Roadmap Structure",
        "",
        f"Goal: {goal}",
        f"Roadmap type: {roadmap_type}",
        f"Experience level: {normalized_level}",
        f"Target duration: {target_months} month(s)",
        f"Study time: {hours_per_day:g} hour(s) per day",
        f"Estimated total study time: {total_hours:g} hours",
        f"Estimated duration in weeks: {weeks}",
        "",
        "Suggested phases:",
        "",
    ]

    for index, phase in enumerate(phases, start=1):
        lines.append(f"{index}. {phase}")

    lines.extend(
        [
            "",
            "Recommended effort allocation:",
            "Foundation and fundamentals: 20%",
            "Core and intermediate skills: 30%",
            "Hands-on projects: 30%",
            "Revision and career/exam preparation: 20%",
            "",
            f"Expected outcome: {outcome}",
            "",
            "Important:",
            (
                "The agent should use technology research, project "
                "recommendations, books, and timeline information "
                "to fill these phases with specific content."
            ),
        ]
    )

    return "\n".join(lines)