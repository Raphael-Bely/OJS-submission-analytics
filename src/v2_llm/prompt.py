"""
prompt.py — Builds the verdict-prediction prompt sent to the LLM (RQ3).

build_user_prompt() takes ONLY problem_statement, code and language — never the
sampled row's true verdict. Nothing on the path to the prompt may carry the answer.
"""

SYSTEM_PROMPT = (
    "You are an expert competitive-programming judge. Given a problem statement "
    "and a candidate's source code, predict the verdict the automated online "
    "judge would assign. Answer only with the requested JSON object."
)

MAX_CODE_CHARS = 4000

VERDICT_DEFINITIONS = """### Possible verdicts
- AC (Accepted): correct output, within time and memory limits, on every test case.
- WA (Wrong Answer): runs to completion but produces incorrect output on at least one test case.
- TLE (Time Limit Exceeded): does not finish within the time limit on at least one test case.
- RE (Runtime Error): crashes, throws, or exits abnormally on at least one test case.
- CE (Compile Error): does not compile / has a syntax error and never runs."""

# Languages with no separate ahead-of-time compilation stage. Verified against
# this project's own data (src/error_analysis.py's atcoder_error_by_language.csv):
# CE never occurs for either, at any difficulty level - only for compiled
# languages (C, C++, Java, C#, Go, Rust).
INTERPRETED_LANGUAGES = {"python", "ruby"}

_INTERPRETED_LANGUAGE_NOTE = """### Note on this language
{language} is interpreted, and in this dataset {language} submissions are
NEVER judged CE - not even ones with a genuine SyntaxError. A confirmed
example: a submission with an actual unparseable syntax error (a stray colon
on a plain assignment, and a missing colon on an if-statement - it cannot run
at all) is still recorded as RE, not CE. Whatever is wrong with the code -
including a real syntax error - classify it as RE here, never CE."""

REASONING_GUIDANCE = """### Before you decide
- The sample inputs above are illustrative only. The real test suite includes
  hidden cases you cannot see: larger values, boundary conditions, adversarial
  inputs. Do not conclude AC just because the code would produce the right
  output on the samples shown - check what happens on inputs the samples do
  not cover.
- If you find something wrong with the code, decide specifically what it
  causes: a crash/exception during execution (RE), a value that is simply
  wrong (WA), or correct-but-too-slow behavior on large inputs (TLE). These
  are different outcomes - do not default to WA for every bug you find."""


def build_user_prompt(problem_statement: str, code: str, language: str) -> str:
    truncated = len(code) > MAX_CODE_CHARS
    if truncated:
        code = code[:MAX_CODE_CHARS] + "\n... (truncated)"

    language_note = ""
    if language.strip().lower() in INTERPRETED_LANGUAGES:
        language_note = "\n\n" + _INTERPRETED_LANGUAGE_NOTE.format(language=language)

    return f"""### Problem statement
{problem_statement}

### Submission (language: {language})
```
{code}
```

{VERDICT_DEFINITIONS}{language_note}

{REASONING_GUIDANCE}

### Task
Predict the verdict for this submission. Respond with exactly one JSON object, nothing else:
{{"reasoning": "<two to four sentences - address hidden/edge-case inputs, and the specific consequence if you spot a bug>", "verdict": "<AC|WA|TLE|RE|CE>"}}"""
