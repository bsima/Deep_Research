#!/usr/bin/env python3
"""Test script to verify Parasail integration with Deep Research."""

import os
import asyncio

# Ensure API keys are set (must be provided via environment)
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Set it to your Parasail API key: export OPENAI_API_KEY='your-key'")
    exit(1)
if not os.environ.get("TAVILY_API_KEY"):
    print("Error: TAVILY_API_KEY environment variable not set")
    print("Get one at https://tavily.com and set: export TAVILY_API_KEY='your-key'")
    exit(1)

from langchain_core.messages import HumanMessage

async def test_basic_model():
    """Test basic model connectivity."""
    print("=" * 60)
    print("Test 1: Basic Model Connectivity")
    print("=" * 60)
    
    from deep_research.config import get_model_string, get_model_kwargs
    from langchain.chat_models import init_chat_model
    
    model = init_chat_model(model=get_model_string(), **get_model_kwargs())
    response = await model.ainvoke([HumanMessage(content="Say 'Parasail integration works!' in exactly those words.")])
    print(f"Response: {response.content}")
    print("✓ Basic model test passed\n")

async def test_tavily_search():
    """Test Tavily search integration."""
    print("=" * 60)
    print("Test 2: Tavily Search")
    print("=" * 60)
    
    from deep_research.utils import tavily_search
    
    result = tavily_search.invoke({"query": "What is LangGraph?"})
    print(f"Search result preview: {result[:500]}...")
    print("✓ Tavily search test passed\n")

async def test_research_agent():
    """Test a single research agent."""
    print("=" * 60)
    print("Test 3: Research Agent (Single Query)")
    print("=" * 60)
    
    from deep_research.research_agent import researcher_agent
    
    result = await researcher_agent.ainvoke({
        "researcher_messages": [
            HumanMessage(content="What are the key features of LangGraph for building AI agents? Keep it brief.")
        ],
        "research_topic": "LangGraph features"
    })
    
    print(f"Compressed research preview: {result.get('compressed_research', 'N/A')[:500]}...")
    print("✓ Research agent test passed\n")

async def main():
    """Run all tests."""
    print("\n🚀 Testing Deep Research with Parasail API\n")
    
    try:
        await test_basic_model()
        await test_tavily_search()
        await test_research_agent()
        
        print("=" * 60)
        print("✅ All tests passed! Ready to run full demo.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
