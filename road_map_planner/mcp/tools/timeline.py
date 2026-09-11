from math import ceil


def create_preparation_timeline(
    target_months: int,
    hours_per_day: float,
    topics: list[str],
) -> str:
    """
    Create a practical preparation timeline from the user's
    available time and learning topics.
    """
    if target_months <= 0:
        return "Target duration must be greater than 0 months."

    if hours_per_day <= 0:
        return "Study hours per day must be greater than 0."

    if not topics:
        return "At least one learning topic is required."

    weeks = target_months * 4
    hours_per_week = hours_per_day * 7
    total_hours = target_months * 30 * hours_per_day

    topics_per_phase = max(1, ceil(len(topics) / target_months))

    lines = [
        "Preparation Timeline",
        "",
        f"Duration: {target_months} month(s)",
        f"Study time: {hours_per_day:g} hour(s) per day",
        f"Estimated total study time: {total_hours:g} hours",
        f"Estimated study time per week: {hours_per_week:g} hours",
        "",
    ]

    topic_index = 0

    for month in range(1, target_months + 1):
        start_topic = topic_index
        end_topic = min(
            start_topic + topics_per_phase,
            len(topics),
        )

        month_topics = topics[start_topic:end_topic]

        if month_topics:
            topic_index = end_topic
        else:
            month_topics = [
                "Revision",
                "Practice",
            ]

        lines.append(f"Month {month}")
        lines.append("-" * 40)

        for topic in month_topics:
            lines.append(f"Focus: {topic}")

        if month < target_months:
            lines.append("Practice: Build small exercises and review concepts.")
        else:
            lines.append(
                "Final preparation: Complete projects, revise, "
                "and practice interview/exam questions."
            )

        lines.append("")

    lines.extend(
        [
            f"Total estimated weeks: {weeks}",
            "",
            "Recommended weekly pattern:",
            "• 60% learning and hands-on practice",
            "• 25% project work",
            "• 15% revision and review",
        ]
    )

    return "\n".join(lines)