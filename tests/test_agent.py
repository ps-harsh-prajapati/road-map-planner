import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_tools() -> None:
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "road_map_planner.mcp.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.list_tools()

            tool_names = {
                tool.name
                for tool in response.tools
            }

            print("Available MCP tools:")
            for name in sorted(tool_names):
                print(f"  - {name}")

            expected_tools = {
                "health_check",
                "technology_research",
                "book_recommendations",
                "project_recommendations",
                "preparation_timeline",
            }

            missing_tools = expected_tools - tool_names

            if missing_tools:
                raise AssertionError(
                    f"Missing MCP tools: {sorted(missing_tools)}"
                )

            result = await session.call_tool(
                "health_check",
                {},
            )

            print()
            print("health_check result:")

            for content in result.content:
                text = getattr(content, "text", None)

                if text is not None:
                    print(text)
                else:
                    print(content)


if __name__ == "__main__":
    asyncio.run(test_mcp_tools())