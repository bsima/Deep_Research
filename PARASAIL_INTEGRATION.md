feat: Integrate ThinkDepth.ai Deep Research with Parasail API

This commit adapts the ThinkDepth.ai Deep Research system to work with
Parasail's OpenAI-compatible API, enabling state-of-the-art deep research
capabilities using Parasail's hosted models.

## Overview

ThinkDepth.ai Deep Research is ranked #1 on the DeepResearch Bench leaderboard,
outperforming Google Gemini 2.5 Pro, OpenAI, and Anthropic Claude deep research.
This integration allows Parasail customers to leverage this powerful research
system through the Parasail API.

## Architecture

The system uses a multi-agent architecture built on LangGraph:

1. **Scoping Phase**: Clarifies user request and generates research brief
2. **Draft Generation**: Creates initial draft report
3. **Supervisor Agent**: Coordinates multiple parallel researcher agents
4. **Researcher Agents**: Perform web searches via Tavily and synthesize findings
5. **Final Report**: Generates comprehensive report with citations

## Changes Made

### New Files

- `src/config.py`: Centralized Parasail API configuration
  - API base URL: https://api.parasail.io/v1
  - Default model: parasail-glm47 (best for research tasks)
  - Fast model: parasail-mistral-small-32-24b (for summarization)
  - Environment variable support for runtime configuration

- `test_parasail.py`: Connectivity and integration tests
  - Tests basic model connectivity
  - Tests Tavily search integration
  - Tests single research agent workflow

- `run_demo.py`: Simple demo runner script
- `run_demo_verbose.py`: Demo with streaming progress output

- `.gitignore`: Standard Python gitignore

### Modified Files

- `src/utils.py`:
  - Integrated Parasail config for model initialization
  - Added async parallel summarization (`summarize_webpage_content_async`)
  - Rewrote `process_search_results` for parallel processing
  - Added SKIP_LLM_SUMMARIZATION env var for 10x faster searches
  - Reduced default search results from 3 to 2 for speed

- `src/research_agent.py`:
  - Updated all model initializations to use Parasail config

- `src/research_agent_full.py`:
  - Updated writer model to use Parasail config

- `src/research_agent_scope.py`:
  - Updated model and creative_model to use Parasail config

- `src/multi_agent_supervisor.py`:
  - Updated supervisor model to use Parasail config
  - Made max_researcher_iterations configurable via env var
  - Made max_concurrent_researchers configurable via env var

## Performance Optimizations

The original implementation was slow due to sequential LLM calls for
summarizing each search result. We implemented several optimizations:

| Optimization | Speedup | Environment Variable |
|--------------|---------|---------------------|
| Skip LLM summarization | ~10x per search | SKIP_LLM_SUMMARIZATION=1 |
| Faster summarization model | ~2x per search | Built-in |
| Parallel async summarization | ~3x per search | Automatic |
| Reduced search results | ~33% fewer calls | Built-in (2 vs 3) |
| Configurable iterations | Variable | MAX_RESEARCHER_ITERATIONS |
| Configurable parallelism | Variable | MAX_CONCURRENT_RESEARCHERS |

### Recommended Settings for Different Use Cases

**Maximum Quality (slow, ~10-20 min):**
```bash
# Default settings, full LLM summarization
uv run python run_demo.py "Your research question"
```

**Balanced (medium, ~5-10 min):**
```bash
export MAX_RESEARCHER_ITERATIONS=8
uv run python run_demo_verbose.py "Your research question"
```

**Fast Demo (quick, ~2-5 min):**
```bash
export SKIP_LLM_SUMMARIZATION=1
export MAX_RESEARCHER_ITERATIONS=5
export MAX_CONCURRENT_RESEARCHERS=1
uv run python run_demo_verbose.py "Your research question"
```

## Usage

### Prerequisites

1. Parasail API key (set as OPENAI_API_KEY)
2. Tavily API key (set as TAVILY_API_KEY)
3. Python 3.11+ with uv package manager

### Quick Start

```bash
# Set API keys
export OPENAI_API_KEY="your-parasail-api-key"
export TAVILY_API_KEY="your-tavily-api-key"

# Install dependencies
uv sync

# Run tests
uv run python test_parasail.py

# Run demo
uv run python run_demo_verbose.py "What are the top AI frameworks in 2025?"
```

### Output

The system generates a comprehensive research report saved to
`research_report.md` with:
- Structured sections with headers
- In-depth analysis from multiple sources
- Numbered citations
- Source URLs

## Parasail Models Used

| Purpose | Model | Context | Notes |
|---------|-------|---------|-------|
| Main reasoning | parasail-glm47 | 128k | Best quality for research |
| Fast summarization | parasail-mistral-small-32-24b | 32k | 3x faster, good quality |

Other compatible Parasail models:
- parasail-qwen3-235b-a22b-instruct-2507 (complex reasoning)
- parasail-kimi-k2-instruct (262k context for long documents)
- parasail-deepseek-v32 (general coding/research)

## Technical Details

### API Compatibility

Parasail's API is OpenAI-compatible, requiring only:
1. Set `openai_api_base` to https://api.parasail.io/v1
2. Set `OPENAI_API_KEY` to Parasail API key
3. Use Parasail model names (e.g., parasail-glm47)

### LangChain Integration

The integration uses LangChain's `init_chat_model` with kwargs:
```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="openai:parasail-glm47",
    openai_api_base="https://api.parasail.io/v1"
)
```

### Async Parallel Processing

Search result summarization now uses asyncio.gather for parallel
processing when running outside an existing event loop:
```python
async def _process_search_results_async(unique_results):
    tasks = [summarize_webpage_content_async(content) for content in results]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)
```

## Example Output

For the query "What is LangGraph?", the system generates a ~2000 word
report covering:
- Overview and core purpose
- Fundamental architecture (nodes, edges, state)
- Primary use cases
- Comparison with traditional LLM chains
- Conclusion with citations

## Credits

- Original ThinkDepth.ai Deep Research: https://github.com/thinkdepthai/Deep_Research
- Parasail API: https://parasail.io
- LangGraph: https://github.com/langchain-ai/langgraph
