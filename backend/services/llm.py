"""
LLM Service — Wraps Groq API (free tier, Llama 3.3 70B).
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3


def _call_with_retry(messages, temperature=0.2, response_format=None):
    """Call Groq API with automatic retry on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": MODEL,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = _client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                wait = 30 * (attempt + 1)
                if "Please try again in" in error_str:
                    try:
                        time_part = error_str.split("Please try again in")[1].split(".")[0]
                        if "m" in time_part:
                            mins = int(time_part.split("m")[0].strip())
                            wait = (mins + 1) * 60
                        elif "s" in time_part:
                            secs = int(time_part.split("s")[0].strip())
                            wait = secs + 5
                    except (ValueError, IndexError):
                        pass

                print(f"  [LLM] Rate limited. Waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded for LLM call")


async def ask_llm(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return _call_with_retry(messages, temperature=0.2)


async def ask_llm_json(prompt: str, system: str = "") -> dict:
    messages = []
    sys_content = (system + "\n\n" if system else "")
    sys_content += (
        "IMPORTANT: Respond with ONLY valid JSON. "
        "No markdown fences, no explanation, no preamble. Just the JSON object."
    )
    messages.append({"role": "system", "content": sys_content})
    messages.append({"role": "user", "content": prompt})

    text = _call_with_retry(
        messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    text = text.strip()
    return json.loads(text)


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await ask_llm("What is 2+2? Reply in one word.")
        print(f"Text test: {result}")

        result = await ask_llm_json(
            'Return a JSON object with key "answer" and value 4.'
        )
        print(f"JSON test: {result}")

    asyncio.run(main())
