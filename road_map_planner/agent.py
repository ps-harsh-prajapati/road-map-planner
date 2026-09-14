import asyncio
import json
import os
import sys
import traceback
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from road_map_planner.prompts import SYSTEM_PROMPT


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b",
)

MAX_AGENT_TURNS = 6


# ============================================================
# Utility helpers
# ============================================================

def _format_exception(
    exc: BaseException,
) -> str:
    """
    Convert normal exceptions and ExceptionGroup errors
    into readable text.
    """

    if isinstance(exc, BaseExceptionGroup):
        lines = [
            f"{type(exc).__name__}: {exc}"
        ]

        for index, nested in enumerate(
            exc.exceptions,
            start=1,
        ):
            lines.append(
                f"\nNested error {index}:"
            )
            lines.append(
                _format_exception(nested)
            )

        return "\n".join(lines)

    return f"{type(exc).__name__}: {exc}"


def _tool_schema(
    tool: Any,
) -> dict[str, Any]:
    """Convert an MCP tool to a compact LLM schema."""

    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema or {},
    }


# ============================================================
# MCP discovery
# ============================================================

async def _get_mcp_tools(
    session: ClientSession,
) -> list[dict[str, Any]]:
    """Discover tools exposed by the MCP server."""

    response = await session.list_tools()

    if response is None:
        raise RuntimeError(
            "MCP server returned no tool-list response."
        )

    if not getattr(
        response,
        "tools",
        None,
    ):
        raise RuntimeError(
            "MCP server returned an empty tool list."
        )

    return [
        _tool_schema(tool)
        for tool in response.tools
    ]


# ============================================================
# LLM prompt construction
# ============================================================

def _build_system_message(
    tools: list[dict[str, Any]],
) -> str:
    """Build a compact system prompt for the Road Map Planner agent."""

    tool_lines: list[str] = []

    for tool in tools:
        name = tool["name"]
        description = tool.get(
            "description",
            "",
        )

        schema = tool.get(
            "input_schema",
            {},
        )

        properties = schema.get(
            "properties",
            {},
        )

        arguments = ", ".join(
            properties.keys()
        )

        tool_lines.append(
            f"- {name}({arguments}): {description}"
        )

    available_tools = "\n".join(tool_lines)

    return f"""
You are Road Map Planner, an autonomous career and education planning agent.

{SYSTEM_PROMPT}

==================================================
AVAILABLE MCP TOOLS
==================================================

{available_tools}

==================================================
YOUR JOB
==================================================

Understand the user's request, decide which MCP tools are useful,
call those tools when necessary, inspect their results, and continue
until you have enough information to produce a useful final roadmap.

You decide:
- which tool to call
- when to call it
- whether another tool is needed
- when to stop

Do NOT blindly call every tool.

==================================================
VERY IMPORTANT JSON FORMAT
==================================================

You MUST return ONLY ONE valid JSON object.

There are ONLY TWO valid values for "action":

1. "tool"
2. "final"

NEVER put a tool name directly inside "action".

==================================================
CORRECT TOOL CALL
==================================================

Example:

{{
  "action": "tool",
  "tool_name": "roadmap_structure",
  "arguments": {{
    "goal": "Become a professional AI engineer",
    "roadmap_type": "career",
    "experience_level": "beginner",
    "target_months": 6,
    "hours_per_day": 2
  }}
}}

Notice:

"action" = "tool"

"tool_name" = the actual MCP tool name

==================================================
CORRECT FINAL RESPONSE
==================================================

{{
  "action": "final",
  "answer": "Complete roadmap..."
}}

==================================================
INCORRECT RESPONSES
==================================================

DO NOT generate:

{{
  "action": "roadmap_structure"
}}

DO NOT generate:

{{
  "action": "technology_research"
}}

DO NOT generate:

{{
  "action": "book_recommendations"
}}

The tool name ALWAYS belongs inside:

"tool_name"

and "action" MUST be:

"tool"

==================================================
TOOL SELECTION
==================================================

Use roadmap_structure when you need a structured roadmap based on
the user's goal, experience, duration, and study time.

Use technology_research when current technology information would
improve the answer.

Use book_recommendations when books or learning resources are useful.

Use project_recommendations when practical projects would help.

Use preparation_timeline when the user provides or needs a study
duration and schedule.

Use only the tools that are relevant.

==================================================
AFTER A TOOL RESULT
==================================================

After receiving a tool result:

1. Read the result.
2. Use it as context.
3. Decide whether another tool is needed.
4. Do not repeat the same tool unnecessarily.
5. When enough information has been collected, return "final".

==================================================
FULL ROADMAP REQUIREMENTS
==================================================

For a complete career or education roadmap, the final answer should
normally contain:

1. Goal
2. Starting Point / Assumptions
3. Skill Gaps
4. Roadmap Phases
5. Current Technologies
6. Books / Resources
7. Real-World Projects
8. Preparation Timeline
9. Expected Outcome

Do not return a one-line answer for a full roadmap.

==================================================
FINAL RULE
==================================================

Return ONLY valid JSON.

For tool calls:

{{
  "action": "tool",
  "tool_name": "EXACT_TOOL_NAME",
  "arguments": {{}}
}}

For the final answer:

{{
  "action": "final",
  "answer": "ready"
}}
"""

# ============================================================
# Ollama
# ============================================================

async def _call_ollama(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask Ollama for the next agent action."""

    payload = {
    "model": OLLAMA_MODEL,
    "keep_alive": "10m",
    "messages": [
        {
            "role": "system",
            "content": _build_system_message(
                tools
            ),
        },
        *messages,
    ],
    "stream": False,
    "format": "json",
    "options": {
        "temperature": 0,
        "num_predict": 4096,
    },
}

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,
                write=30.0,
                pool=30.0,
            )
        ) as client:

            response = await client.post(
                OLLAMA_URL,
                json=payload,
            )

            response.raise_for_status()

    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Could not connect to Ollama.\n"
            f"URL: {OLLAMA_URL}\n"
            f"Model: {OLLAMA_MODEL}"
        ) from exc

    except httpx.ReadTimeout as exc:
        raise RuntimeError(
            "Ollama took too long to produce a response.\n"
            f"Model: {OLLAMA_MODEL}\n"
            "The local 8B model is running on CPU, so "
            "generation can take several minutes."
        ) from exc

    except httpx.HTTPStatusError as exc:
        detail = ""

        try:
            detail = exc.response.text.strip()
        except Exception:
            pass

        message = (
            f"Ollama returned HTTP "
            f"{exc.response.status_code}."
        )

        if detail:
            message += f"\nResponse: {detail}"

        raise RuntimeError(
            message
        ) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            "Ollama network request failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    data = response.json()

    content = (
        data
        .get("message", {})
        .get("content")
    )

    if not content:
        raise RuntimeError(
            "Ollama returned no message content."
        )

    try:
        decision = json.loads(
            content
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The model returned invalid JSON.\n\n"
            f"Raw response:\n{content}"
        ) from exc

    if not isinstance(
        decision,
        dict,
    ):
        raise RuntimeError(
            "The model response must be a JSON object."
        )

    return decision


# ============================================================
# MCP tool execution
# ============================================================

async def _call_mcp_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    Safely execute an MCP tool.

    Every stage is protected because some MCP failures can
    otherwise escape as ExceptionGroup/TaskGroup errors.
    """

    try:
        result = await session.call_tool(
            tool_name,
            arguments,
        )

    except Exception as exc:
        return (
            f"MCP tool '{tool_name}' failed.\n"
            f"{_format_exception(exc)}"
        )

    if result is None:
        return (
            f"MCP tool '{tool_name}' returned None."
        )

    content_items = getattr(
        result,
        "content",
        None,
    )

    if content_items is None:
        return (
            f"MCP tool '{tool_name}' returned "
            "a result without content."
        )

    if (
        getattr(
            result,
            "isError",
            False,
        )
    ):
        return (
            f"MCP tool '{tool_name}' "
            "reported an error."
        )

    parts: list[str] = []

    try:
        for content in content_items:

            if content is None:
                continue

            text = getattr(
                content,
                "text",
                None,
            )

            if text is not None:
                parts.append(
                    str(text)
                )
            else:
                parts.append(
                    str(content)
                )

    except Exception as exc:
        return (
            f"MCP tool '{tool_name}' returned "
            "content that could not be processed.\n"
            f"{_format_exception(exc)}"
        )

    if not parts:
        return (
            f"MCP tool '{tool_name}' "
            "returned no readable content."
        )

    return "\n".join(parts)


# ============================================================
# Roadmap validation
# ============================================================

def _is_complete_roadmap(
    answer: str,
) -> bool:
    """
    Check whether the model produced a substantial roadmap.

    The check is intentionally flexible because a local LLM may use
    different headings such as "Tech Stack" instead of
    "Current Technologies".
    """

    answer = answer.strip()

    # A very short response is definitely not a roadmap.
    if len(answer) < 700:
        return False

    answer_lower = answer.lower()

    section_groups = [
        # Goal
        (
            "goal",
            "objective",
            "target",
        ),

        # Starting point
        (
            "starting point",
            "current level",
            "assumptions",
            "background",
        ),

        # Skills
        (
            "skill gap",
            "skills to learn",
            "skills",
            "prerequisites",
        ),

        # Roadmap
        (
            "roadmap",
            "phases",
            "learning path",
            "learning stages",
        ),

        # Technology
        (
            "current technolog",
            "technologies",
            "tech stack",
            "tools and technologies",
        ),

        # Resources
        (
            "books",
            "resources",
            "learning resources",
        ),

        # Projects
        (
            "projects",
            "real-world projects",
            "portfolio",
        ),

        # Timeline
        (
            "timeline",
            "schedule",
            "month 1",
            "week 1",
        ),

        # Outcome
        (
            "expected outcome",
            "outcome",
            "result",
            "career readiness",
        ),
    ]

    matches = 0

    for group in section_groups:
        if any(
            keyword in answer_lower
            for keyword in group
        ):
            matches += 1

    # Require most major sections, not every exact heading.
    return matches >= 7

def _normalize_decision(
    decision: dict[str, Any],
    available_tools: set[str],
) -> dict[str, Any]:
    """
    Normalize tool decisions produced by the local LLM.

    The model should return:

    {
        "action": "tool",
        "tool_name": "preparation_timeline",
        "arguments": {...}
    }

    But smaller local models sometimes return:

    {
        "action": "preparation_timeline",
        "arguments": {...}
    }

    or use a synonymous action name such as:

    {
        "action": "start_study_plan",
        "arguments": {...}
    }
    """

    action = decision.get("action")

    # ---------------------------------------------------------
    # Correct tool format
    # ---------------------------------------------------------

    if action == "tool":
        return decision

    # ---------------------------------------------------------
    # Direct MCP tool name in action
    # ---------------------------------------------------------

    if (
        isinstance(action, str)
        and action in available_tools
    ):
        return {
            "action": "tool",
            "tool_name": action,
            "arguments": decision.get(
                "arguments",
                {},
            ),
        }

    # ---------------------------------------------------------
    # Common aliases produced by smaller local models
    # ---------------------------------------------------------

    aliases = {
        "start_study_plan": "preparation_timeline",
        "study_plan": "preparation_timeline",
        "create_study_plan": "preparation_timeline",
        "build_study_plan": "preparation_timeline",

        "research_technology": "technology_research",
        "current_technology": "technology_research",
        "technology_search": "technology_research",

        "find_books": "book_recommendations",
        "book_search": "book_recommendations",
        "recommend_books": "book_recommendations",

        "find_projects": "project_recommendations",
        "recommend_projects": "project_recommendations",
        "project_search": "project_recommendations",

        "build_roadmap": "roadmap_structure",
        "create_roadmap": "roadmap_structure",
        "plan_roadmap": "roadmap_structure",
    }

    if (
        isinstance(action, str)
        and action in aliases
    ):
        normalized_tool = aliases[action]

        if normalized_tool in available_tools:
            return {
                "action": "tool",
                "tool_name": normalized_tool,
                "arguments": decision.get(
                    "arguments",
                    {},
                ),
            }

    # ---------------------------------------------------------
    # Leave unknown actions unchanged so the caller can
    # report a useful error.
    # ---------------------------------------------------------

    return decision
def _trim_tool_result(
    result: str,
    max_chars: int = 5000,
) -> str:
    """
    Limit the amount of tool output sent to the local LLM.

    Large API responses can make later reasoning steps extremely
    slow on CPU-only local models.
    """

    if len(result) <= max_chars:
        return result

    return (
        result[:max_chars]
        + "\n\n[Tool result truncated for the local LLM.]"
    )

async def _generate_final_answer(
    user_request: str,
    tool_results: list[dict[str, str]],
) -> str:
    """
    Generate the final roadmap using a compact context.

    This avoids sending the full agent history and MCP schemas
    to the local 7B model for the final response.
    """

    sections: list[str] = []

    for item in tool_results:
        sections.append(
            f"""
TOOL: {item["tool"]}

RESULT:
{item["result"]}
""".strip()
        )

    combined_results = "\n\n".join(
        sections
    )

    prompt = f"""
You are the final answer writer for Road Map Planner.

USER REQUEST:
{user_request}

You have already collected information from MCP tools.

TOOL RESULTS:
{combined_results}

Create the final roadmap.

The roadmap should include:

1. Goal
2. Starting Point / Assumptions
3. Skill Gaps
4. Roadmap Phases
5. Current Technologies
6. Books / Resources
7. Real-World Projects
8. Preparation Timeline
9. Expected Outcome

Important:

- Use the tool results as factual context.
- Do not invent information that contradicts the tool results.
- Make the plan practical and actionable.
- Respect the user's available time.
- Do not mention MCP, internal tools, or agent reasoning.
- Do not output JSON.
- Return only the final human-readable roadmap.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "keep_alive": "10m",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional roadmap writer. "
                    "Write a concise but complete roadmap."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 3000,
        },
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,
                write=30.0,
                pool=30.0,
            )
        ) as client:

            response = await client.post(
                OLLAMA_URL,
                json=payload,
            )

            response.raise_for_status()

    except httpx.ReadTimeout as exc:
        raise RuntimeError(
            "The final roadmap generation timed out. "
            "The local model is running on CPU."
        ) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Final roadmap generation failed: {exc}"
        ) from exc

    data = response.json()

    answer = (
        data
        .get("message", {})
        .get("content", "")
    )

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty final roadmap."
        )

    return answer.strip()
def _validate_roadmap_request(user_request: str) -> tuple[bool, str]:
    """Reject empty, traceback/error, and clearly unrelated input before MCP/LLM."""
    text = user_request.strip()

    if not text:
        return False, "Please enter a roadmap request."

    lowered = text.lower()

    # Do not send pasted application errors/tracebacks into the planner.
    error_markers = (
        "traceback (most recent call last)",
        "httpx.readtimeout",
        "runtimeerror:",
        "exception:",
        "filenotfounderror:",
        "valueerror:",
        "typeerror:",
        "keyerror:",
        "attributeerror:",
        "modulenotfounderror:",
        "connectionerror:",
        "stack trace",
    )

    if any(marker in lowered for marker in error_markers):
        return (
            False,
            "It looks like you pasted an error or traceback. "
            "Please enter a roadmap request instead.\n\n"
            "Example: I want to become an AI engineer in 6 months."
        )

    # A lightweight intent check prevents arbitrary text from reaching MCP.
    roadmap_terms = (
        "roadmap", "learn", "learning", "become", "career", "career path",
        "skill", "skills", "study", "prepare", "preparation", "education",
        "developer", "engineer", "programmer", "job", "technology",
        "technologies", "project", "projects", "course", "interview",
        "machine learning", "artificial intelligence", "ai engineer",
        "data scientist", "software engineer", "web developer", "backend",
        "frontend", "devops", "cloud", "cybersecurity",
    )

    if not any(term in lowered for term in roadmap_terms):
        return (
            False,
            "I can help you create a career, education, or skill roadmap. "
            "Please describe what you want to learn or become.\n\n"
            "Example: I want to become an AI engineer in 6 months."
        )

    return True, ""


async def run_agent(
    user_request: str,
) -> str:
    """Run the Road Map Planner agent with input validation and MCP tools."""

    # Validate BEFORE starting MCP or calling Ollama.
    is_valid, validation_message = _validate_roadmap_request(user_request)
    if not is_valid:
        return validation_message

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "road_map_planner.mcp.server",
        ],
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await _get_mcp_tools(session)
                tool_names = {tool["name"] for tool in tools}

                print("[Agent] MCP connected.")
                print(
                    "[Agent] Available tools: "
                    + ", ".join(sorted(tool_names))
                )

                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": user_request.strip()}
                ]

                tool_history: list[dict[str, Any]] = []
                tool_results: list[dict[str, str]] = []

                for turn in range(1, MAX_AGENT_TURNS + 1):
                    print(f"[Agent] Step {turn}/{MAX_AGENT_TURNS}")

                    decision_messages = list(messages)

                    if tool_history:
                        decision_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "TOOLS ALREADY USED:\n"
                                    + json.dumps(tool_history[-3:], indent=2)
                                ),
                            }
                        )

                    decision = await _call_ollama(
                        messages=decision_messages,
                        tools=tools,
                    )
                    decision = _normalize_decision(decision, tool_names)
                    action = decision.get("action")

                    # ------------------------------------------------------
                    # FINAL
                    # ------------------------------------------------------
                    if action == "final":
                        print("[Agent] Generating final roadmap...")

                        answer = await _generate_final_answer(
                            user_request=user_request.strip(),
                            tool_results=tool_results,
                        )

                        if not answer or not answer.strip():
                            raise RuntimeError(
                                "Failed to generate final roadmap."
                            )

                        print("[Agent] Finished.")
                        return answer.strip()

                    # ------------------------------------------------------
                    # TOOL
                    # ------------------------------------------------------
                    if action != "tool":
                        raise RuntimeError(
                            f"Invalid agent action: {action}"
                        )

                    tool_name = decision.get("tool_name")
                    arguments = decision.get("arguments", {})

                    if not isinstance(tool_name, str) or not tool_name:
                        raise RuntimeError(
                            "Agent did not provide a valid tool name."
                        )

                    if not isinstance(arguments, dict):
                        raise RuntimeError(
                            "Tool arguments must be a JSON object."
                        )

                    if tool_name not in tool_names:
                        raise RuntimeError(
                            "Agent selected unavailable MCP tool: "
                            f"{tool_name}"
                        )

                    print(f"[Agent] Calling: {tool_name}")
                    print(
                        "[Agent] Arguments: "
                        + json.dumps(arguments, ensure_ascii=False)
                    )

                    result = await _call_mcp_tool(
                        session,
                        tool_name,
                        arguments,
                    )
                    result = _trim_tool_result(result, max_chars=5000)

                    if len(result) > 12000:
                        result = (
                            result[:12000]
                            + "\n\n[Tool result truncated for the local LLM.]"
                        )

                    tool_results.append(
                        {"tool": tool_name, "result": result}
                    )

                    print("[Agent] Result received.")

                    tool_history.append(
                        {
                            "tool": tool_name,
                            "arguments": arguments,
                        }
                    )

                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": json.dumps(decision),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "MCP TOOL RESULT\n\n"
                                    f"Tool: {tool_name}\n\n"
                                    f"{result}\n\n"
                                    "Use this result as factual context. "
                                    "Do not repeat unnecessary details."
                                ),
                            },
                        ]
                    )

                raise RuntimeError(
                    "Agent reached its maximum "
                    f"of {MAX_AGENT_TURNS} steps without producing "
                    "a complete roadmap."
                )

    except BaseExceptionGroup as exc:
        raise RuntimeError(
            "MCP/agent operation failed:\n\n" + _format_exception(exc)
        ) from exc


# ============================================================
# Synchronous wrapper
# ============================================================

def run(
    user_request: str,
) -> str:
    """Synchronous entry point used by the CLI and frontend."""

    return asyncio.run(
        run_agent(
            user_request
        )
    )


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":

    print(
        "╭────────────────────────────────────╮"
    )
    print(
        "│ 🗺️ Road Map Planner Agent         │"
    )
    print(
        "╰────────────────────────────────────╯"
    )
    print()

    request = input(
        "Enter your roadmap request:\n> "
    ).strip()

    if not request:
        print(
            "No request provided."
        )
        sys.exit(0)

    try:
        answer = run(
            request
        )

        print()
        print(
            "=" * 70
        )
        print(
            answer
        )
        print(
            "=" * 70
        )

    except KeyboardInterrupt:
        print(
            "\nAgent stopped."
        )

    except Exception as exc:
        print(
            "\nAgent error:"
        )

        print(
            _format_exception(exc)
        )

        traceback.print_exc()

        sys.exit(1)