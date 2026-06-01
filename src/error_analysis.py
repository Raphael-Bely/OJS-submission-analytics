"""
error_analysis.py — Error distribution analysis by difficulty level.

Methodology: Shimizu et al. (2025) — RQ1.
Analysis unit: the submission (not the user).
For each difficulty level A–F, computes the proportion of each
status type (AC, WA, CE, TLE, RE, Other) across all submissions.
"""
import polars as pl


# Normalization: full status names → short codes
# Values observed in CodeNet AtCoder data.
STATUS_MAP = {
    "Accepted":              "AC",
    "Wrong Answer":          "WA",
    "Compile Error":         "CE",
    "Time Limit Exceeded":   "TLE",
    "Runtime Error":         "RE",
    "Memory Limit Exceeded": "Other",
    "Output Limit Exceeded": "Other",
    "Judge Not Available":   "Other",
}

# Display order for charts and terminal output
STATUS_ORDER = ["AC", "WA", "TLE", "RE", "CE", "Other"]
DIFFICULTY_ORDER = ["A", "B", "C", "D", "E", "F"]


def compute_error_distribution(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes submission status distribution per difficulty letter.

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, status)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty)

    Returns:
        DataFrame with columns:
        difficulty | status_code | n_submissions | pct
        Sorted by difficulty then status_code.
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])

    # Join + normalize status names to short codes
    joined = (
        lazy_subs
        .select(["problem_id", "status"])
        .join(lazy_labels, on="problem_id", how="inner")
        .with_columns(
            pl.col("status")
              .replace(STATUS_MAP, default="Other")
              .alias("status_code")
        )
    )

    # Total submissions per difficulty level (denominator for percentages)
    totals = (
        joined
        .group_by("difficulty")
        .agg(pl.len().alias("total_submissions"))
    )

    # Count per (difficulty, status_code)
    counts = (
        joined
        .group_by(["difficulty", "status_code"])
        .agg(pl.len().alias("n_submissions"))
    )

    # Join + compute percentages
    result = (
        counts
        .join(totals, on="difficulty", how="left")
        .with_columns(
            (pl.col("n_submissions") / pl.col("total_submissions") * 100)
              .round(2)
              .alias("pct")
        )
        .drop("total_submissions")
        .sort(["difficulty", "status_code"])
        .collect()
    )

    # Terminal summary
    total_subs = result["n_submissions"].sum()
    print(f"  Total ABC submissions analyzed: {total_subs:,}\n")

    main_statuses = ["AC", "WA", "TLE", "CE", "RE"]
    header = f"  {'Level':<8}" + "".join(f"{s:>8}%" for s in main_statuses)
    print(header)
    print("  " + "-" * (8 + 9 * len(main_statuses)))

    for diff in DIFFICULTY_ORDER:
        row_data = result.filter(pl.col("difficulty") == diff)
        if row_data.is_empty():
            continue
        pct_map = dict(zip(row_data["status_code"].to_list(), row_data["pct"].to_list()))
        row_str = f"  {diff:<8}" + "".join(f"{pct_map.get(s, 0.0):>8.1f}" for s in main_statuses)
        print(row_str)

    print()
    return result
