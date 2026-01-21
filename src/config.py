"""Parasail API Configuration.

Centralized configuration for using Parasail's OpenAI-compatible API
with the Deep Research system.
"""

import os

# Parasail API Configuration
PARASAIL_API_BASE = "https://api.parasail.io/v1"
PARASAIL_MODEL = os.environ.get("PARASAIL_MODEL", "parasail-glm47")

# Fast mode: use a smaller/faster model for summarization
FAST_MODE = os.environ.get("FAST_MODE", "").lower() in ("1", "true", "yes")
PARASAIL_FAST_MODEL = "parasail-mistral-small-32-24b"  # Faster for simple tasks

# Model kwargs to pass to init_chat_model for Parasail
def get_model_kwargs(max_tokens: int | None = None) -> dict:
    """Get kwargs for init_chat_model to use Parasail API.
    
    Args:
        max_tokens: Optional max tokens limit
        
    Returns:
        Dict of kwargs for init_chat_model
    """
    kwargs = {
        "openai_api_base": PARASAIL_API_BASE,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return kwargs

def get_model_string(fast: bool = False) -> str:
    """Get the model string for init_chat_model.
    
    Args:
        fast: Use faster model for simple tasks like summarization
    """
    if fast or FAST_MODE:
        return f"openai:{PARASAIL_FAST_MODEL}"
    return f"openai:{PARASAIL_MODEL}"
