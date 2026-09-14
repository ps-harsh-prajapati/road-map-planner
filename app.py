import os
import subprocess
import sys

from click import command
import streamlit as st


PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

AGENT_TIMEOUT = 1200


st.set_page_config(
    page_title="Road Map Planner",
    page_icon="🗺️",
    layout="wide",
)


def initialize_state() -> None:
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_header() -> None:
    """Render the application header."""
    st.title("🗺️ Road Map Planner")
    st.caption(
        "Plan your career or education journey with a local AI agent."
    )


def render_sidebar() -> None:
    """Render sidebar content."""
    with st.sidebar:
        st.header("Road Map Planner")

        st.write(
            "Your local AI planning assistant can help with:"
        )

        st.markdown(
            """
            - 🎯 Career roadmaps
            - 🎓 Education roadmaps
            - 🧠 Skill-gap analysis
            - ⚙️ Current technologies
            - 📚 Book recommendations
            - 🛠️ Real-world projects
            - 📅 Preparation timelines
            """
        )

        st.divider()

        st.subheader("Example")

        st.code(
            "I want to become a professional AI engineer. "
            "I know Python and basic ML. "
            "I have 6 months and can study 2 hours a day."
        )

        st.divider()

        if st.button(
            "Clear conversation",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()


def render_messages() -> None:
    """Render previous conversation."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def run_agent_process(
    user_request: str,
) -> str:
    """
    Run the existing agent in a completely separate Python process.

    The subprocess is isolated from Streamlit and runs the exact same
    agent entry point that works from the terminal.
    """

    command = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        "-m",
        "road_map_planner.agent",
    ]

    env = os.environ.copy()

    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        process = subprocess.run(
            command,
            input=user_request + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
            env=env,
            timeout=AGENT_TIMEOUT,
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "The agent timed out after "
            f"{AGENT_TIMEOUT} seconds."
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            f"Could not start the agent process: {exc}"
        ) from exc

    stdout = process.stdout or ""
    stderr = process.stderr or ""

    if process.returncode != 0:
        details = stderr.strip()

        if not details:
            details = stdout.strip()

        raise RuntimeError(
            "Agent process failed with exit code "
            f"{process.returncode}.\n\n"
            f"{details}"
        )

    if not stdout.strip():
        raise RuntimeError(
            "Agent completed but returned no output."
        )

    return extract_final_answer(stdout)

def extract_final_answer(output: str,) -> str:
    """
    Extract the final roadmap from the agent CLI output.

    The agent prints a separator around the final answer.
    """

    separator = "=" * 70

    if separator in output:
        sections = output.split(separator)

        if len(sections) >= 3:
            answer = sections[-2].strip()

            if answer:
                return answer

    # Fallback: remove common CLI messages and return the remaining output.
    lines = output.splitlines()

    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if stripped.startswith(
            "[Agent]"
        ):
            continue

        if stripped.startswith(
            "╭"
        ):
            continue

        if stripped.startswith(
            "│"
        ):
            continue

        if stripped.startswith(
            "╰"
        ):
            continue

        if stripped.startswith(
            "Enter your roadmap request:"
        ):
            continue

        if stripped == user_request_placeholder():
            continue

        cleaned_lines.append(line)

    answer = "\n".join(
        cleaned_lines
    ).strip()

    if answer:
        return answer

    return output


def user_request_placeholder() -> str:
    """Return an empty placeholder used by the output cleaner."""
    return ""


def generate_answer(user_request: str,) -> str:
    """Generate the roadmap using the existing agent."""
    return run_agent_process(
        user_request
    )


def main() -> None:
    """Run the Streamlit frontend."""

    initialize_state()

    render_sidebar()
    render_header()
    render_messages()

    prompt = st.chat_input(
        "What roadmap would you like to build?"
    )

    if not prompt:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Agent is researching and building your roadmap..."
        ):
            try:
                answer = generate_answer(
                    prompt
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as exc:

                st.error(
                    "I couldn't generate the roadmap."
                )

                with st.expander(
                    "Technical details"
                ):
                    st.code(
                        f"{type(exc).__name__}: {exc}"
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I couldn't generate the roadmap."
                        ),
                    }
                )


if __name__ == "__main__":
    main()