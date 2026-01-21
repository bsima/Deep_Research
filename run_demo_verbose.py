#!/usr/bin/env python3
"""Run Deep Research demo with verbose progress output."""

import os
import sys
import asyncio
from datetime import datetime

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

def log(msg):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

async def run_research(question: str):
    """Run the full deep research pipeline with streaming."""
    log("🚀 Starting Deep Research with Parasail API")
    log(f"📋 Question: {question[:100]}...")
    
    checkpointer = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
    
    log("⏳ Running research pipeline (streaming events)...")
    
    final_result = None
    async for event in full_agent.astream_events(
        {"messages": [HumanMessage(content=question)]},
        config=thread,
        version="v2"
    ):
        kind = event.get("event", "")
        name = event.get("name", "")
        
        if kind == "on_chain_start" and name:
            log(f"  → Starting: {name}")
        elif kind == "on_chain_end" and name:
            log(f"  ✓ Completed: {name}")
            if name == "LangGraph":
                final_result = event.get("data", {}).get("output", {})
        elif kind == "on_tool_start":
            tool_name = event.get("name", "unknown")
            log(f"  🔧 Tool: {tool_name}")
    
    if final_result:
        report = final_result.get("final_report", "No report")
        log("=" * 60)
        log("📄 FINAL REPORT")
        log("=" * 60)
        print(report[:2000] + "..." if len(report) > 2000 else report)
        
        with open("research_report.md", "w") as f:
            f.write(f"# Research Report\n\n{report}\n")
        log("✅ Report saved to research_report.md")
    else:
        log("⚠️ No final result captured")

async def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What are the top 3 open-source AI agent frameworks in 2025? Brief overview only."
    
    try:
        await run_research(question)
    except Exception as e:
        log(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
