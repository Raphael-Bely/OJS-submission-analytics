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


def compute_error_by_group_and_difficulty(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes submission status distribution per (difficulty level x proficiency group).
    Replicates Table 4 of Shimizu et al. (2025).

    Only classified users (G1–G6) are included. Users with no AC (null group)
    are excluded — they cannot be meaningfully assigned to a proficiency level.

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, user_id, status)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty)
        df_users:  User profile DataFrame
                   (required columns: user_id, proficiency_group)

    Returns:
        DataFrame with columns:
        difficulty | proficiency_group | status_code | n_submissions | pct
        Sorted by difficulty, proficiency_group, status_code.
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])

    # Only keep classified users (exclude null proficiency_group)
    lazy_groups = (
        df_users.lazy()
        .select(["user_id", "proficiency_group"])
        .filter(pl.col("proficiency_group").is_not_null())
    )

    # Three-way join: submissions × difficulty labels × user groups
    joined = (
        lazy_subs
        .select(["problem_id", "user_id", "status"])
        .join(lazy_labels, on="problem_id", how="inner")
        .join(lazy_groups, on="user_id", how="inner")
        .with_columns(
            pl.col("status")
              .replace(STATUS_MAP, default="Other")
              .alias("status_code")
        )
    )

    # Total submissions per (difficulty, group) — denominator for percentages
    totals = (
        joined
        .group_by(["difficulty", "proficiency_group"])
        .agg(pl.len().alias("total_submissions"))
    )

    # Count per (difficulty, group, status_code)
    counts = (
        joined
        .group_by(["difficulty", "proficiency_group", "status_code"])
        .agg(pl.len().alias("n_submissions"))
    )

    # Join + compute percentages
    result = (
        counts
        .join(totals, on=["difficulty", "proficiency_group"], how="left")
        .with_columns(
            (pl.col("n_submissions") / pl.col("total_submissions") * 100)
              .round(2)
              .alias("pct")
        )
        .drop("total_submissions")
        .sort(["difficulty", "proficiency_group", "status_code"])
        .collect()
    )

    # Terminal summary — AC rate per (difficulty, group)
    total_subs = result["n_submissions"].sum()
    print(f"  Total submissions analyzed: {total_subs:,}\n")

    groups = sorted(result["proficiency_group"].unique().to_list())
    header = f"  {'Level':<6}" + "".join(f"{g:>9}" for g in groups)
    print(f"  AC rate per group (%):")
    print(header)
    print("  " + "-" * (6 + 9 * len(groups)))

    for diff in DIFFICULTY_ORDER:
        row_parts = []
        for g in groups:
            subset = result.filter(
                (pl.col("difficulty") == diff) &
                (pl.col("proficiency_group") == g) &
                (pl.col("status_code") == "AC")
            )
            val = subset["pct"].item() if not subset.is_empty() else None
            row_parts.append(f"{val:>8.1f}" if val is not None else f"{'—':>9}")
        print(f"  {diff:<6}" + "".join(row_parts))

    print()
    return result
