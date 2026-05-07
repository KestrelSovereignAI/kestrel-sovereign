"""
Example: Using Agent Tools
Demonstrates web search and feedback tools
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def main():
    """Demonstrate tool usage"""

    print("🔧 Kestrel Agent Tools Example")
    print("=" * 50)

    # Example 1: Web Search Tool
    print("\n1. Web Search Tool")
    print("-" * 50)

    from kestrel_sdk.tools.base import get_web_search_tool

    search_tool = get_web_search_tool()

    if search_tool.enabled:
        print("✅ Web search is available")

        # Perform a search
        query = "latest developments in AI agents"
        print(f"\n🔍 Searching for: '{query}'")

        result = await search_tool.search(
            query=query,
            max_results=3,
            include_answer=True
        )

        if result["success"]:
            print(f"\n📊 AI Summary: {result['answer']}\n")
            print(f"Found {len(result['results'])} results:\n")

            for i, item in enumerate(result['results'], 1):
                print(f"{i}. {item['title']}")
                print(f"   URL: {item['url']}")
                print(f"   {item['content'][:150]}...\n")
        else:
            print(f"❌ Search failed: {result['error']}")
    else:
        print("❌ Web search not available (missing TAVILY_API_KEY)")

    # Example 2: Feedback Tool (requires database)
    print("\n2. Feedback Tool")
    print("-" * 50)

    database_url = os.getenv("DATABASE_URL")

    if database_url:
        print("✅ Database available for feedback tool")

        import asyncpg
        from kestrel_sdk.tools.base import get_feedback_tool, FeedbackType, FeedbackSeverity

        # Create connection pool
        pool = await asyncpg.create_pool(database_url)

        feedback_tool = get_feedback_tool(pool)

        # Create dummy IDs (in real usage, these come from the database)
        import uuid
        companion_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())

        # Record an observation
        print("\n📝 Recording observation...")
        feedback_id = await feedback_tool.record_feedback(
            companion_id=companion_id,
            user_id=user_id,
            feedback_type=FeedbackType.OBSERVATION,
            title="Example observation",
            content="This is an example of agent self-observation",
            severity=FeedbackSeverity.INFO,
            source="agent",
            tags=["example", "demo"]
        )
        print(f"✅ Recorded feedback with ID: {feedback_id}")

        # Record a tool usage
        print("\n📊 Recording tool usage...")
        usage_id = await feedback_tool.record_tool_usage(
            companion_id=companion_id,
            user_id=user_id,
            tool_name="example_tool",
            success=True,
            input_params={"query": "test"},
            output_result={"status": "success"},
            execution_time_ms=150,
            conversation_context="Example tool usage"
        )
        print(f"✅ Recorded tool usage with ID: {usage_id}")

        # Get feedback entries
        print("\n📋 Retrieving feedback...")
        entries = await feedback_tool.get_feedback(
            companion_id=companion_id,
            user_id=user_id,
            limit=5
        )
        print(f"Found {len(entries)} feedback entries")

        for entry in entries:
            print(f"\n  [{entry['severity'].upper()}] {entry['title']}")
            print(f"  Type: {entry['feedback_type']} | Source: {entry['source']}")
            print(f"  {entry['content'][:100]}...")

        # Get tool stats
        print("\n📊 Tool usage statistics...")
        stats = await feedback_tool.get_tool_stats(
            companion_id=companion_id,
            user_id=user_id,
            days=7
        )

        if stats["tools"]:
            for tool in stats["tools"]:
                success_rate = (tool["successful_uses"] / tool["total_uses"] * 100)
                print(f"\n  🔧 {tool['tool_name']}")
                print(f"     Uses: {tool['total_uses']} | Success: {success_rate:.1f}%")
                print(f"     Avg time: {tool['avg_execution_time_ms']:.0f}ms")
        else:
            print("  No tool usage data yet")

        # Clean up
        await pool.close()
    else:
        print("❌ Database not available (missing DATABASE_URL)")

    print("\n" + "=" * 50)
    print("✅ Example complete!")
    print("\nTo use these tools in production:")
    print("1. Set TAVILY_API_KEY for web search")
    print("2. Set DATABASE_URL for feedback tools")
    print("3. Use chat commands: !search, !feedback, !tools")


if __name__ == "__main__":
    asyncio.run(main())
