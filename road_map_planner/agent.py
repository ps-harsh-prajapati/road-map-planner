import asyncio
import json
import os
import sys
import traceback
from typing import Any, Optional

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
    "qwen2.5:1.5b",
)

MAX_AGENT_TURNS = 6


# ============================================================
# Utility helpers
# ============================================================

def _format_exception(exc: BaseException) -> str:
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


def _tool_schema(tool: Any) -> dict[str, Any]:
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
    """Build the system prompt for the Road Map Planner agent."""

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

Understand the user's request and decide which MCP tools are useful.

You decide:
- which tool to call
- when to call it
- whether another tool is needed
- when to stop

Do NOT blindly call every available tool.

==================================================
VERY IMPORTANT JSON FORMAT
==================================================

You MUST return ONLY ONE valid JSON object.

There are ONLY TWO valid values for "action":

1. "tool"
2. "final"

NEVER put an MCP tool name directly inside "action".

==================================================
TOOL CALL FORMAT
==================================================

For every MCP tool call, return EXACTLY this structure:

{{
  "action": "tool",
  "tool_name": "EXACT_TOOL_NAME",
  "arguments": {{}}
}}

Example:

{{
  "action": "tool",
  "tool_name": "project_recommendations",
  "arguments": {{
    "topic": "machine learning",
    "experience_level": "beginner"
  }}
}}

IMPORTANT:

- "action" MUST be exactly "tool"
- "tool_name" MUST contain the MCP tool name
- "arguments" MUST be a JSON object
- Use only tool names from AVAILABLE MCP TOOLS

==================================================
FINAL DECISION FORMAT
==================================================

When enough information has been collected, return EXACTLY:

{{
  "action": "final"
}}

The final decision must contain nothing else.

DO NOT include:
- answer
- roadmap
- goal
- resources
- learning_phases
- projects
- timeline
- explanations
- markdown

==================================================
INCORRECT FORMAT
==================================================

These are WRONG:

{{
  "action": "project_recommendations"
}}

{{
  "action": "technology_research"
}}

{{
  "action": "roadmap_structure"
}}

These are also WRONG:

{{
  "action": "final",
  "answer": "..."
}}

{{
  "action": "final",
  "roadmap": {{
    ...
  }}
}}

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

Use only the tools relevant to the request.

==================================================
AFTER A TOOL RESULT
==================================================

After receiving a tool result:

1. Read the result.
2. Use it as context.
3. Decide whether another tool is needed.
4. Do not repeat the same tool unnecessarily.
5. If enough information is available, return:

{{
  "action": "final"
}}

==================================================
FINAL RULE
==================================================

Return ONLY valid JSON.

For a tool call:

{{
  "action": "tool",
  "tool_name": "EXACT_TOOL_NAME",
  "arguments": {{}}
}}

For completion:

{{
  "action": "final"
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
                "content": _build_system_message(tools),
            },
            *messages,
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 128,
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
            "The local model may be slow on CPU."
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

        raise RuntimeError(message) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            "Ollama network request failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Ollama returned an invalid HTTP JSON response."
        ) from exc

    content = (
        data
        .get("message", {})
        .get("content")
    )

    if not content:
        raise RuntimeError(
            "Ollama returned no message content."
        )

    content = str(content).strip()

    try:
        decision = json.loads(content)

    except json.JSONDecodeError as exc:
        normalized_content = content.lower()

        if (
            '"action"' in normalized_content
            and '"final"' in normalized_content
        ):
            print(
                "[Agent] Recovered truncated final decision."
            )

            return {
                "action": "final"
            }

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
# Decision normalization
# ============================================================

def _normalize_decision(
    decision: Any,
    available_tool_names: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Normalize different local-model decision formats into
    one canonical schema used by run_agent().

    Canonical tool format:

    {
        "action": "tool",
        "tool_name": "project_recommendations",
        "arguments": {}
    }

    Canonical final format:

    {
        "action": "final"
    }
    """

    if not isinstance(decision, dict):
        raise RuntimeError(
            f"Model decision must be a JSON object, "
            f"got {type(decision).__name__}."
        )

    action = str(
        decision.get("action", "")
    ).strip().lower()

    # ============================================================
    # FINAL
    # ============================================================

    if action == "final":
        return {
            "action": "final"
        }

    # ============================================================
    # STANDARD TOOL CALL
    # ============================================================

    if action in {
        "tool",
        "tool_call",
        "call_tool",
    }:

        tool_name = (
            decision.get("tool_name")
            or decision.get("tool")
            or decision.get("name")
        )

        if not tool_name:
            raise RuntimeError(
                "Model requested a tool call "
                "but did not provide a tool name."
            )

        tool_name = str(tool_name).strip()

        if (
            available_tool_names is not None
            and tool_name not in available_tool_names
        ):
            raise RuntimeError(
                f"Model requested unknown MCP tool: "
                f"{tool_name}. "
                f"Available tools: "
                f"{sorted(available_tool_names)}"
            )

        arguments = (
            decision.get("arguments")
            or decision.get("args")
            or decision.get("parameters")
            or {}
        )

        if not isinstance(arguments, dict):
            arguments = {}

        return {
            "action": "tool",
            "tool_name": tool_name,
            "arguments": arguments,
        }

    # ============================================================
    # LOCAL MODEL FALLBACK
    # ============================================================
    #
    # Some local models return:
    #
    # {
    #     "action": "project_recommendations"
    # }
    #
    # If action matches a real MCP tool, convert it safely.
    #

    if (
        available_tool_names is not None
        and action in available_tool_names
    ):

        arguments = (
            decision.get("arguments")
            or decision.get("args")
            or decision.get("parameters")
            or {}
        )

        if not isinstance(arguments, dict):
            arguments = {}

        print(
            f"[Agent] Normalized direct tool action "
            f"'{action}' -> tool call."
        )

        return {
            "action": "tool",
            "tool_name": action,
            "arguments": arguments,
        }

    # ============================================================
    # INVALID
    # ============================================================

    raise RuntimeError(
        f"Invalid agent action: {action!r}. "
        "Expected 'tool', 'final', or a valid MCP tool name."
    )


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

    if getattr(
        result,
        "isError",
        False,
    ):
        parts: list[str] = []

        for content in content_items:
            text = getattr(
                content,
                "text",
                None,
            )

            if text:
                parts.append(
                    str(text)
                )

        error_detail = "\n".join(parts)

        if error_detail:
            return (
                f"MCP tool '{tool_name}' "
                f"reported an error:\n{error_detail}"
            )

        return (
            f"MCP tool '{tool_name}' reported an error."
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
# Tool-result trimming
# ============================================================

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


# ============================================================
# Final roadmap generation
# ============================================================

async def _generate_final_answer(
    user_request: str,
    tool_results: list[dict[str, str]],
) -> str:
    """
    Generate the actual human-readable roadmap.

    This is separate from the agent decision call.
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

    if not combined_results:
        combined_results = (
            "No MCP tool results were collected. "
            "Use the user's request and clearly stated assumptions."
        )

    prompt = f"""
You are the final answer writer for Road Map Planner.

USER REQUEST:
{user_request}

You have already collected information from MCP tools.

TOOL RESULTS:
{combined_results}

Create the final roadmap for the user.

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

IMPORTANT:

- Return ONLY human-readable Markdown.
- Do NOT return JSON.
- Do NOT return Python code.
- Do NOT mention MCP.
- Do NOT mention internal tools.
- Do NOT mention agent reasoning.
- Use the user's requested goal as the main focus.
- Use tool results as factual context.
- Do not contradict information found in the tool results.
- Make the plan practical and actionable.
- Respect the user's available time.
- If the user did not specify a duration, choose a reasonable progression.
- Keep explanations concise.
- Use clear headings and bullet points.
- Include concrete projects.
- Include a realistic timeline.
- Do not put the roadmap inside a JSON object.

Now write the complete roadmap.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "keep_alive": "10m",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional roadmap writer. "
                    "Write a concise but complete roadmap in Markdown. "
                    "Never output JSON."
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
            "num_predict": 1200,
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
            "Could not connect to Ollama while generating "
            "the final roadmap."
        ) from exc

    except httpx.ReadTimeout as exc:
        raise RuntimeError(
            "The final roadmap generation timed out. "
            f"The local model ({OLLAMA_MODEL}) may be slow on CPU."
        ) from exc

    except httpx.HTTPStatusError as exc:
        detail = ""

        try:
            detail = exc.response.text.strip()
        except Exception:
            pass

        message = (
            "Ollama failed while generating the final roadmap. "
            f"HTTP {exc.response.status_code}."
        )

        if detail:
            message += f"\nResponse: {detail}"

        raise RuntimeError(message) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Final roadmap generation failed: {exc}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Ollama returned an invalid response while "
            "generating the final roadmap."
        ) from exc

    answer = (
        data
        .get("message", {})
        .get("content", "")
    )

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty final roadmap."
        )

    answer = str(answer).strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty final roadmap."
        )

    return answer


# ============================================================
# Main agent
# ============================================================

async def run_agent(
    user_request: str,
) -> str:
    """
    Run the autonomous Road Map Planner agent.

    User
      ↓
    LLM decision
      ↓
    MCP tool
      ↓
    Tool result
      ↓
    LLM decision
      ↓
    Another tool OR final
      ↓
    Final-answer writer
      ↓
    Human-readable roadmap
    """

    if not user_request.strip():
        raise ValueError(
            "User request cannot be empty."
        )

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
        async with stdio_client(
            server_params,
        ) as (
            read,
            write,
        ):

            async with ClientSession(
                read,
                write,
            ) as session:

                await session.initialize()

                tools = await _get_mcp_tools(
                    session
                )

                tool_names = {
                    tool["name"]
                    for tool in tools
                }

                print(
                    "[Agent] MCP connected."
                )

                print(
                    "[Agent] Available tools: "
                    + ", ".join(
                        sorted(tool_names)
                    )
                )

                messages: list[
                    dict[str, Any]
                ] = [
                    {
                        "role": "user",
                        "content": user_request,
                    }
                ]

                tool_history: list[
                    dict[str, Any]
                ] = []

                tool_results: list[
                    dict[str, str]
                ] = []

                for turn in range(
                    1,
                    MAX_AGENT_TURNS + 1,
                ):

                    print(
                        f"[Agent] Step "
                        f"{turn}/{MAX_AGENT_TURNS}"
                    )

                    decision_messages = list(
                        messages
                    )

                    if tool_history:
                        recent_tools = tool_history[-3:]

                        decision_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "TOOLS ALREADY USED:\n"
                                    + json.dumps(
                                        recent_tools,
                                        indent=2,
                                    )
                                    + "\n\n"
                                    "Choose another relevant tool "
                                    "only if needed. Otherwise return "
                                    'exactly {"action":"final"}.'
                                ),
                            }
                        )

                    decision = await _call_ollama(
                        messages=decision_messages,
                        tools=tools,
                    )

                    decision = _normalize_decision(
                        decision,
                        tool_names,
                    )

                    action = decision.get(
                        "action"
                    )

                    # ==================================================
                    # FINAL
                    # ==================================================

                    if action == "final":

                        print(
                            "[Agent] Decision: final"
                        )

                        print(
                            "[Agent] Generating final roadmap..."
                        )

                        try:
                            answer = await _generate_final_answer(
                                user_request=user_request,
                                tool_results=tool_results,
                            )

                        except Exception as exc:
                            raise RuntimeError(
                                "Failed to generate the final roadmap.\n"
                                f"{_format_exception(exc)}"
                            ) from exc

                        answer = answer.strip()

                        if not answer:
                            raise RuntimeError(
                                "Final roadmap generation returned "
                                "an empty answer."
                            )

                        print(
                            "[Agent] Finished."
                        )

                        return answer

                    # ==================================================
                    # TOOL
                    # ==================================================

                    if action != "tool":
                        raise RuntimeError(
                            "Invalid normalized agent action: "
                            f"{action}"
                        )

                    # IMPORTANT:
                    # This now matches _normalize_decision().
                    tool_name = decision.get(
                        "tool_name"
                    )

                    arguments = decision.get(
                        "arguments",
                        {},
                    )

                    if not isinstance(
                        tool_name,
                        str,
                    ):
                        raise RuntimeError(
                            "Agent did not provide "
                            "a valid tool name."
                        )

                    if not isinstance(
                        arguments,
                        dict,
                    ):
                        raise RuntimeError(
                            "Tool arguments must be "
                            "a JSON object."
                        )

                    if tool_name not in tool_names:
                        raise RuntimeError(
                            "Agent selected unavailable "
                            f"MCP tool: {tool_name}"
                        )

                    print(
                        f"[Agent] Calling: "
                        f"{tool_name}"
                    )

                    print(
                        "[Agent] Arguments: "
                        + json.dumps(
                            arguments,
                            ensure_ascii=False,
                        )
                    )

                    result = await _call_mcp_tool(
                        session,
                        tool_name,
                        arguments,
                    )

                    result = _trim_tool_result(
                        result,
                        max_chars=5000,
                    )

                    tool_results.append(
                        {
                            "tool": tool_name,
                            "result": result,
                        }
                    )

                    print(
                        "[Agent] Result received."
                    )

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
                                "content": json.dumps(
                                    decision,
                                    ensure_ascii=False,
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "MCP TOOL RESULT\n\n"
                                    f"Tool: {tool_name}\n\n"
                                    f"{result}\n\n"
                                    "Use this result as factual context. "
                                    "Do not repeat unnecessary details. "
                                    "Decide whether another relevant tool "
                                    "is needed. "
                                    "If enough information is now available, "
                                    'return exactly {"action":"final"}.'
                                ),
                            },
                        ]
                    )

                # ======================================================
                # MAXIMUM TURNS REACHED
                # ======================================================

                print(
                    "[Agent] Maximum tool steps reached. "
                    "Generating final roadmap from collected results..."
                )

                answer = await _generate_final_answer(
                    user_request=user_request,
                    tool_results=tool_results,
                )

                answer = answer.strip()

                if not answer:
                    raise RuntimeError(
                        "Final roadmap generation returned "
                        "an empty answer."
                    )

                print(
                    "[Agent] Finished."
                )

                return answer

    except BaseExceptionGroup as exc:
        raise RuntimeError(
            "MCP/agent operation failed:\n\n"
            + _format_exception(exc)
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