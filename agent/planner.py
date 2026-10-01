"""LLM planner.

Turns a natural-language instruction into a simple JSON operation.
The LLM never writes or executes Python code.
"""

import json
import os

# The system prompt tells the LLM exactly what it is allowed to do.
SYSTEM_PROMPT = """You are a data operation planner.

Your job is to understand the user's natural-language request and convert it into one supported JSON operation.

You do NOT write Python code.
You do NOT execute anything.
Only return valid JSON.

Supported operations:

1. fill_missing   {"operation": "fill_missing", "column": "<name>", "method": "mean" | "median" | "mode"}
2. drop_missing   {"operation": "drop_missing"}
3. normalize      {"operation": "normalize", "column": "<name>", "method": "minmax" | "zscore"}
4. encode         {"operation": "encode", "column": "<name>", "method": "label" | "one_hot"}
5. remove_duplicates {"operation": "remove_duplicates"}
6. lowercase      {"operation": "lowercase", "column": "<name>"}
7. strip_spaces   {"operation": "strip_spaces", "column": "<name>"}
8. statistics     {"operation": "statistics"}

Use the dataset columns provided by the application. Only use column names that actually exist.

If the request is not related to supported operations, return:

{"operation": "unknown"}

Return ONLY the JSON object, nothing else."""


def _build_user_prompt(columns, instruction):
    """Give the LLM the column names plus the user instruction."""
    return (
        f"Dataset columns: {', '.join(columns)}\n\n"
        f"User instruction:\n{instruction}"
    )


def _extract_json(text):
    """Pull the first JSON object out of the model's reply."""
    text = text.strip()
    # Remove markdown code fences if the model added them
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in the model response.")
    return json.loads(text[start:end + 1])


def plan_with_openai(api_key, columns, instruction):
    """Ask OpenAI to plan the operation."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(columns, instruction)},
        ],
    )
    return _extract_json(response.choices[0].message.content)


def plan_with_gemini(api_key, columns, instruction):
    """Ask Gemini to plan the operation."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=SYSTEM_PROMPT + "\n\n" + _build_user_prompt(columns, instruction),
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )
    return _extract_json(response.text)


def plan_operation(provider, api_key, columns, instruction):
    """Route the request to the selected provider and return a dict."""
    if provider == "OpenAI":
        result = plan_with_openai(api_key, columns, instruction)
    elif provider == "Gemini":
        result = plan_with_gemini(api_key, columns, instruction)
    else:
        raise ValueError("Unknown AI provider selected.")

    if not isinstance(result, dict) or "operation" not in result:
        return {"operation": "unknown"}
    return result


def get_api_key(provider):
    """Get the API key from .env based on the selected provider."""
    if provider == "OpenAI":
        return os.getenv("OPENAI_API_KEY")
    return os.getenv("GEMINI_API_KEY")
