"""
output_parser.py — Parses the LLM's raw text response into a verdict prediction.

Five-tier fallback, strictest to loosest. Never raises, never guesses a verdict
when nothing reliable was found — parse_verdict_response() always returns a dict,
with verdict_pred=None / parse_success=False only if every tier fails.
"""
import json
import re

VALID_VERDICTS = {"AC", "WA", "TLE", "CE", "RE"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*?\}", re.DOTALL)
_TOKEN_RE = re.compile(r"\b(AC|WA|TLE|RE|CE)\b", re.IGNORECASE)


def _from_json_object(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        return None
    return {"verdict_pred": verdict, "reasoning": obj.get("reasoning")}


def parse_verdict_response(raw_text: str) -> dict:
    text = (raw_text or "").strip()

    try:
        parsed = _from_json_object(json.loads(text))
        if parsed:
            return {**parsed, "parse_success": True, "parse_method": "direct_json"}
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            parsed = _from_json_object(json.loads(fence_match.group(1)))
            if parsed:
                return {**parsed, "parse_success": True, "parse_method": "fenced_json"}
        except json.JSONDecodeError:
            pass

    brace_match = _BRACE_RE.search(text)
    if brace_match:
        try:
            parsed = _from_json_object(json.loads(brace_match.group(0)))
            if parsed:
                return {**parsed, "parse_success": True, "parse_method": "extracted_json"}
        except json.JSONDecodeError:
            pass

    token_match = _TOKEN_RE.search(text)
    if token_match:
        return {
            "verdict_pred": token_match.group(1).upper(),
            "reasoning": None,
            "parse_success": True,
            "parse_method": "token_fallback",
        }

    return {
        "verdict_pred": None,
        "reasoning": None,
        "parse_success": False,
        "parse_method": "failed",
    }
