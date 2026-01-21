#!/usr/bin/env python3
"""Run Deep Research demo with Parasail API.

Usage:
    python run_demo.py "Your research question here"
    
    # Or use the default demo question:
    python run_demo.py
"""

import os
import sys
import asyncio

# Ensure API keys are set (must be provided via environment)
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Set it to your Parasail API key: export OPENAI_API_KEY='your-key'")
    sys.exit(1)
if not os.environ.get("TAVILY_API_KEY"):
    print("Error: TAVILY_API_KEY environment variable not set")
    print("Get one at https://tavily.com and set: export TAVILY_API_KEY='your-key'")
    sys.exit(1)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from deep_research.research_agent_full import deep_researcher_builder

# Default demo question (shorter for faster testing)
DEFAULT_QUESTION = """
Write a brief analysis of the current state of open-source AI agents in 2025,
focusing on the key frameworks and their adoption.
"""

async def run_research(question: str):
    """Run the full deep research pipeline."""
    print("\n" + "=" * 70)
    print("🔬 ThinkDepth.ai Deep Research with Parasail API")
    print("=" * 70)
    print(f"\n📋 Research Question:\n{question.strip()}\n")
    print("-" * 70)
    print("⏳ Starting research (this may take 5-15 minutes)...")
    print("-" * 70 + "\n")
    
    # Set up the agent with memory
    checkpointer = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    
    # Configure the run
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
    
    # Run the research
    result = await full_agent.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=thread
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("📄 FINAL REPORT")
    print("=" * 70 + "\n")
    
    final_report = result.get("final_report", "No report generated")
    print(final_report)
    
    # Save report to file
    output_file = "research_report.md"
    with open(output_file, "w") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"## Question\n{question.strip()}\n\n")
        f.write(f"## Report\n{final_report}\n")
    
    print("\n" + "-" * 70)
    print(f"✅ Report saved to: {output_file}")
    print("-" * 70 + "\n")
    
    return result

async def main():
    """Main entry point."""
    # Get question from command line or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = DEFAULT_QUESTION
    
    try:
        await run_research(question)
    except KeyboardInterrupt:
        print("\n\n⚠️ Research interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
