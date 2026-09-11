from road_map_planner.agent import run


def main() -> None:
    print("╭────────────────────────────────────╮")
    print("│ 🗺️ Road Map Planner               │")
    print("│ Your career & education agent      │")
    print("╰────────────────────────────────────╯")
    print()
    print("Tell me what roadmap you want to build.")
    print("Type 'exit' to quit.")
    print()

    while True:
        try:
            user_request = input("You: ").strip()

            if not user_request:
                continue

            if user_request.lower() in {"exit", "quit"}:
                print("Goodbye! 👋")
                break

            print()
            print("Road Map Planner is thinking...")
            print()

            answer = run(user_request)

            print(answer)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break

        except Exception as exc:
            print()
            print(f"Error: {exc}")
            print()


if __name__ == "__main__":
    main()