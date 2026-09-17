"""
run_predictions.py — RQ3 pilot: predicts submission verdicts with a local LLM
and compares accuracy against a majority-class baseline.

Reuses embedding.py's sampling/code-reading and data_loader/difficulty_labeler's
problem loading - same stratified draw (seed=42) as every other RQ2 script.

Usage:
    python src/v2_llm/run_predictions.py <model_path> [device] [language] [problem_id] [n_per_cell]

Example (pilot, matches the NB13 S8 p02659/Python reference point):
    python src/v2_llm/run_predictions.py data/models/Qwen--Qwen2.5-Coder-7B-Instruct mps Python p02659 5

Backend: defaults to the in-process transformers client (LocalHFClient). For a
quantized model via MLX (needs .venv_mlx - see mlx_client.py's docstring for
why it's a separate venv), set LLM_BACKEND=mlx. Not a CLI positional arg,
same reasoning as embedding.py's GCB_POOLING - an experiment, not a routine
knob:
    LLM_BACKEND=mlx python src/v2_llm/run_predictions.py data/models/mlx-community--Qwen2.5-Coder-14B-Instruct-4bit mlx Python p02659 5

Output: data/processed/llm_verdicts/predictions_{model_tag}_{language}_{problem_id}_n{n}.csv
Re-running with the same arguments asks first, if that file already exists:
resume it (default - submissions already present are skipped, same as
before), or start a fresh run in a separately-versioned file (_v2, _v3, ...)
so the old results aren't silently mixed into or mistaken for a new run.
"""
import csv
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))         # sibling modules (prompt, llm_client, output_parser)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ modules (embedding, data_loader, ...)

import prompt as prompt_module
from output_parser import parse_verdict_response

DEFAULT_DEVICE      = "cpu"
DEFAULT_LANGUAGE    = "Python"
DEFAULT_PROBLEM_ID  = "p02659"  # RQ2's reference problem (highest verdict lift, NB11)
DEFAULT_N_PER_CELL  = 20

OUTPUT_DIR = Path("./data/processed/llm_verdicts")
DESCRIPTIONS_DIR = Path("./data/Project_CodeNet/problem_descriptions")

CSV_COLUMNS = [
    "submission_id", "problem_id", "language", "difficulty", "proficiency_group",
    "verdict_true", "verdict_pred", "reasoning", "parse_success", "parse_method",
    "code_truncated", "model_name",
]


def _model_tag(model_path: str) -> str:
    return Path(model_path).name or Path(model_path).parent.name


def _output_path(model_path: str, language: str, problem_id: str, n_per_cell: int) -> Path:
    tag = _model_tag(model_path)
    return OUTPUT_DIR / f"predictions_{tag}_{language}_{problem_id}_n{n_per_cell}.csv"


def _already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return set(pl.read_csv(out_path)["submission_id"].to_list())


def _next_versioned_path(path: Path) -> Path:
    """Next free _vN suffix for `path` (the un-suffixed file counts as the
    implicit v1 and is never touched). foo_n25.csv -> foo_n25_v2.csv, and if
    that's also taken -> foo_n25_v3.csv, etc."""
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _confirm_fresh_or_resume(out_path: Path, n_existing: int) -> bool:
    """Asks whether to resume `out_path` (default) or start a fresh,
    separately-versioned run. Returns True for fresh, False for resume.

    Prompts on /dev/tty rather than plain input() so the question is still
    visible when stdout/stderr are redirected to a log file (e.g.
    `... > run.log 2>&1`, a real case that came up: with print()/input(),
    the prompt text lands in the log instead of the terminal and the run
    just looks silently stuck). Falls back to resuming - not hanging - if
    there's no controlling terminal at all (fully non-interactive/background
    invocation), since a prompt nobody can see or answer must not block forever.
    """
    message = (
        f"\nOutput file already exists: {out_path}\n"
        f"  ({n_existing} predictions already recorded for this exact "
        f"model/language/problem/n_per_cell combination)\n"
        f"Resume and add to this file, or start a fresh run in a new, "
        f"separately-versioned file (_v2, _v3, ...)?\n"
        f"[r]esume / [f]resh (default: resume): "
    )
    try:
        with open("/dev/tty", "r+") as tty:
            tty.write(message)
            tty.flush()
            answer = tty.readline().strip().lower()
    except OSError:
        print(message + "\n[no controlling terminal - defaulting to resume]")
        answer = "r"
    return answer in ("f", "fresh")


def run(model_path: str, device: str, language: str, problem_id: str, n_per_cell: int) -> None:
    from data_loader import load_atcoder_problems, OUT_USER_PROFILES
    from difficulty_labeler import label_abc_problems
    from embedding import sample_submissions, read_source_code
    from problem_parser import extract_problem_statement

    df_atcoder = load_atcoder_problems()
    df_abc     = label_abc_problems(df_atcoder)
    abc_ids    = df_abc["problem_id"].to_list()
    df_users   = pl.read_csv(OUT_USER_PROFILES)

    print(f"Sampling submissions ({language}, {problem_id}, n_per_cell={n_per_cell})...")
    df_sample = sample_submissions(
        abc_ids, df_abc, df_users,
        n_per_cell=n_per_cell, language=language, problem_id=problem_id,
    )

    statement = extract_problem_statement(DESCRIPTIONS_DIR, problem_id)
    if not statement:
        raise RuntimeError(
            f"No usable problem statement found for {problem_id} - check "
            f"{DESCRIPTIONS_DIR / (problem_id + '.html')} exists and is readable."
        )

    out_path = _output_path(model_path, language, problem_id, n_per_cell)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        n_existing = len(_already_done(out_path))
        if _confirm_fresh_or_resume(out_path, n_existing):
            out_path = _next_versioned_path(out_path)
            print(f"Starting fresh: {out_path}")

    done_ids = _already_done(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} submissions already predicted, skipping those.")

    write_header = not out_path.exists()
    backend = os.environ.get("LLM_BACKEND", "hf")
    print(f"Loading model '{model_path}' on device '{device}' (backend={backend})...")
    if backend == "mlx":
        from mlx_client import MLXClient
        client = MLXClient(model_path, device=device)
    elif backend == "hf":
        from llm_client import LocalHFClient
        client = LocalHFClient(model_path, device=device)
    else:
        raise ValueError(f"LLM_BACKEND='{backend}' invalide - 'hf' ou 'mlx' attendu.")

    n_skipped_done, n_missing_code, n_predicted = 0, 0, 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(df_sample.iter_rows(named=True), start=1):
            if row["submission_id"] in done_ids:
                n_skipped_done += 1
                continue

            code = read_source_code(
                row["problem_id"], row["language"], row["submission_id"], row["filename_ext"]
            )
            if not code:
                n_missing_code += 1
                continue

            # SECURITY INVARIANT: only problem_statement/code/language reach the
            # prompt below - row["status_code"] (the true verdict) never does.
            user_prompt = prompt_module.build_user_prompt(statement, code, row["language"])
            raw_response = client.generate(prompt_module.SYSTEM_PROMPT, user_prompt)
            parsed = parse_verdict_response(raw_response)

            writer.writerow({
                "submission_id":      row["submission_id"],
                "problem_id":         row["problem_id"],
                "language":           row["language"],
                "difficulty":         row["difficulty"],
                "proficiency_group":  row["proficiency_group"],
                "verdict_true":       row["status_code"],
                "verdict_pred":       parsed["verdict_pred"],
                "reasoning":          parsed["reasoning"],
                "parse_success":      int(parsed["parse_success"]),
                "parse_method":       parsed["parse_method"],
                "code_truncated":     int(len(code) > prompt_module.MAX_CODE_CHARS),
                "model_name":         model_path,
            })
            f.flush()
            n_predicted += 1
            print(f"  [{i}/{df_sample.height}] {row['submission_id']}: "
                  f"true={row['status_code']}  pred={parsed['verdict_pred']}  "
                  f"({parsed['parse_method']})")

    print(f"\nDone. {n_skipped_done} already done, {n_missing_code} missing source, "
          f"{n_predicted} predicted this run.")
    _summarize(out_path)


def _summarize(out_path: Path) -> None:
    df = pl.read_csv(out_path)
    scored = df.filter(pl.col("parse_success") == 1)
    if scored.height == 0:
        print("No successfully-parsed predictions to summarize.")
        return

    accuracy = (scored["verdict_true"] == scored["verdict_pred"]).mean()
    counts = scored["verdict_true"].value_counts().sort("count", descending=True)
    baseline = counts["count"][0] / scored.height
    parse_rate = df["parse_success"].mean()

    print(f"\n{scored.height} scored predictions (of {df.height} total)")
    print(f"  Accuracy               : {accuracy:.3f}")
    print(f"  Majority-class baseline: {baseline:.3f}")
    print(f"  Lift over baseline     : {accuracy - baseline:+.3f}")
    print(f"  Parse success rate     : {parse_rate:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
        print(__doc__)
        sys.exit(0)

    model_path = sys.argv[1]
    device     = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DEVICE
    language   = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_LANGUAGE
    problem_id = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_PROBLEM_ID
    n_per_cell = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_N_PER_CELL

    run(model_path, device, language, problem_id, n_per_cell)
