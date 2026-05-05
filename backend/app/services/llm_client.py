"""
LLM Client for Text-to-SQL Chat functionality.
Uses Anthropic Claude models for natural language to SQL conversion.
"""

import os
import re
from pathlib import Path
from anthropic import Anthropic, AsyncAnthropic
from groq import Groq, AsyncGroq
from dotenv import load_dotenv

# Load environment variables from Track-Blueb root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# 1. Initialize Clients
claude_key = os.getenv("ANTHROPIC_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

anthropic_client = None
async_anthropic_client = None
groq_client = None
async_groq_client = None

if claude_key:
    anthropic_client = Anthropic(api_key=claude_key)
    async_anthropic_client = AsyncAnthropic(api_key=claude_key)
    print("[LLM] Claude provider enabled")
elif groq_key:
    groq_client = Groq(api_key=groq_key)
    async_groq_client = AsyncGroq(api_key=groq_key)
    print("[LLM] Groq provider enabled (Fallback)")
else:
    print("[WARNING] No LLM API Keys found (Claude/Groq) - Chat will be disabled")


def _clean_json_response(response_text: str) -> str:
    """
    Robustly extracts and cleans JSON from LLM responses.
    Handles various markdown formats, code blocks, and extra text.
    """
    if not response_text:
        return "{}"
    
    text = response_text.strip()
    
    # Pattern 1: Extract content from ```json ... ``` blocks
    json_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.IGNORECASE)
    if matches:
        text = matches[0].strip()
    else:
        text = re.sub(r'```(?:json)?', '', text, flags=re.IGNORECASE)
        text = text.replace('```', '')
    
    # Pattern 2: Try to find JSON object directly { ... }
    json_object_pattern = r'\{[\s\S]*\}'
    json_match = re.search(json_object_pattern, text)
    if json_match:
        text = json_match.group(0)
    
    text = text.strip()
    text = text.replace('\\"', '"')
    
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]
    
    if not text.startswith('{'):
        print(f"[WARNING] Response doesn't look like JSON: {text[:100]}...")
        return "{}"
    
    return text


def call_llm(prompt: str, model_type: str = 'flash') -> str:
    """
    Calls the specified LLM provider and returns a JSON string (sync).
    Prioritizes Claude, falls back to Groq.
    """
    system_prompt = "Output valid JSON only."
    
    # Case A: Use Claude
    if anthropic_client:
        model = "claude-haiku-4-5-20251001" if model_type == 'flash' else "claude-sonnet-4-5-20250929"
        try:
            message = anthropic_client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return _clean_json_response(message.content[0].text)
        except Exception as e:
            print(f"[ERROR] Claude API Error: {e}")
            # Try falling back to Groq if key exists
            if not groq_client:
                raise Exception(f"Claude API Error: {e}")
    
    # Case B: Use Groq (Default or Fallback)
    if groq_client:
        # Use Llama-3.3-70B for both as it's the strongest SQL generator on Groq
        model = "llama-3.3-70b-versatile"
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                response_format={"type": "json_object"}
            )
            return _clean_json_response(chat_completion.choices[0].message.content)
        except Exception as e:
            print(f"[ERROR] Groq API Error: {e}")
            raise Exception(f"Groq API Error: {e}")

    raise Exception("No LLM provider (Claude/Groq) is configured")


def call_llm_raw(prompt: str, model_type: str = 'flash') -> str:
    """
    Calls the LLM and returns RAW response without JSON cleaning (sync).
    """
    if anthropic_client:
        model = "claude-haiku-4-5-20251001" if model_type == 'flash' else "claude-sonnet-4-5-20250929"
        try:
            message = anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            if not groq_client: raise Exception(f"Claude Error: {e}")

    if groq_client:
        model = "llama-3.3-70b-versatile"
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=1024
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Groq Error: {e}")

    raise Exception("No LLM provider configured")


async def call_llm_raw_async(prompt: str, model_type: str = 'flash') -> str:
    """
    Async version - Calls the LLM and returns RAW response.
    """
    if async_anthropic_client:
        model = "claude-haiku-4-5-20251001" if model_type == 'flash' else "claude-sonnet-4-5-20250929"
        try:
            message = await async_anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            if not async_groq_client: raise Exception(f"Async Claude Error: {e}")

    if async_groq_client:
        model = "llama-3.3-70b-versatile"
        try:
            chat_completion = await async_groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=1024
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Async Groq Error: {e}")

    raise Exception("No LLM provider configured")
