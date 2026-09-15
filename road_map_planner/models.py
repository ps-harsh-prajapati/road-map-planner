from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    goal: str
    roadmap_type: str = Field(
        description="career or education"
    )
    current_skills: list[str] = Field(default_factory=list)
    experience_level: str = "beginner"
    available_hours_per_day: float = 2.0
    target_months: int = 6


class RoadmapPhase(BaseModel):
    phase: str
    duration: str
    topics: list[str]
    projects: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)


class Roadmap(BaseModel):
    title: str
    goal: str
    roadmap_type: str
    phases: list[RoadmapPhase]
    recommended_books: list[str] = Field(default_factory=list)
    recommended_projects: list[str] = Field(default_factory=list)
    current_technologies: list[str] = Field(default_factory=list)