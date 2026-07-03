"""Robust JSON parsing utilities for LLM responses."""

import json
import re


def parse_json_robust(content: str) -> dict:
    """Parse JSON content from LLM, attempting basic repairs if needed."""
    content = content.strip()
    
    # 1. Strip markdown code block wrappers
    if content.startswith("```"):
        try:
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        except IndexError:
            pass

    # 2. Try parsing directly
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 3. Try to append a missing closing brace if it starts with { but doesn't end with }
    if content.startswith("{") and not content.endswith("}"):
        try:
            return json.loads(content + "}")
        except json.JSONDecodeError:
            pass

    # 4. Attempt to find the first { and last } to extract JSON substring
    match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 5. Let the standard json module parse it and throw its normal exception if still failing
    return json.loads(content)
