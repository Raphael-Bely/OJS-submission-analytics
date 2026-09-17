"""
embedding.py — Code embedding pipeline for RQ2.

Samples submissions stratified by (difficulty × status_code), reads source code
from Project CodeNet, embeds with CodeBERT or TF-IDF (baseline).

Usage:
    python src/embedding.py tfidf       # fast baseline, no GPU needed
    python src/embedding.py codebert    # slower, requires: pip install torch transformers

Outputs in data/processed/embeddings/:
    embeddings_{method}.npy   — float32 array, shape (N, D)
    metadata_{method}.csv     — N rows with submission_id, difficulty, status_code, etc.
"""
from pathlib import Path
import difflib
import os
import random
import sys

import numpy as np
import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR   = Path("./data/Project_CodeNet")
PROCESSED_DIR  = Path("./data/processed")
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
METADATA_DIR   = RAW_DATA_DIR / "metadata"

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_CELL    = 500
TARGET_DIFFICULTIES = ["B", "C", "D", "E"]
TARGET_STATUSES     = ["AC", "WA", "TLE", "CE", "RE"]
MAX_TOKENS          = 512

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


# ── Step 1 — Sampling ─────────────────────────────────────────────────────────

def _parse_problem_ids(problem_id: str | None) -> list[str] | None:
    """
    Splits a comma-separated problem_id string into a list (e.g. "p02616,p02642"
    -> ["p02616", "p02642"]). A single id or None/"" pass through unchanged
    (None means "no restriction").
    """
    if not problem_id:
        return None
    return [p.strip() for p in problem_id.split(",") if p.strip()]


def _load_eligible(
    abc_ids: list[str],
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
    language: str | None = None,
    problem_id: str | None = None,
) -> pl.DataFrame:
    """
    Loads all ABC submissions with (submission_id, problem_id, user_id, language,
    filename_ext, status_code, difficulty, proficiency_group).
    Filtered to TARGET_DIFFICULTIES × TARGET_STATUSES.
    If language is given (e.g. "C++"), keeps only submissions whose language
    starts with that prefix (catches C++14, C++17, C++20, etc.).
    If problem_id is given (single id, or comma-separated list), restricts to
    those problem(s) only.
    """
    problem_ids = _parse_problem_ids(problem_id)
    lazy_frames = []
    for pid in abc_ids:
        fp = METADATA_DIR / f"{pid}.csv"
        if fp.exists() and fp.stat().st_size > 50:
            try:
                lf = pl.scan_csv(str(fp), schema_overrides={
                    "submission_id": pl.Utf8,
                    "problem_id":    pl.Utf8,
                    "user_id":       pl.Utf8,
                    "status":        pl.Utf8,
                    "language":      pl.Utf8,
                    "filename_ext":  pl.Utf8,
                })
                lf = lf.select([
                    "submission_id", "problem_id", "user_id",
                    "language", "filename_ext", "status",
                ])
                lazy_frames.append(lf)
            except Exception:
                pass

    if not lazy_frames:
        raise ValueError(f"No submission metadata found in {METADATA_DIR}")

    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])
    lazy_groups = (
        df_users.lazy()
        .select(["user_id", "proficiency_group"])
        .filter(pl.col("proficiency_group").is_not_null())
    )

    lazy = (
        pl.concat(lazy_frames)
        .join(lazy_labels, on="problem_id", how="inner")
        .join(lazy_groups, on="user_id", how="inner")
        .with_columns(
            pl.col("status").replace(STATUS_MAP).fill_null("Other").alias("status_code")
        )
        .filter(pl.col("status_code").is_in(TARGET_STATUSES))
    )
    if not problem_ids:
        lazy = lazy.filter(pl.col("difficulty").is_in(TARGET_DIFFICULTIES))
    if language:
        lazy = lazy.filter(pl.col("language").str.starts_with(language))
    if problem_ids:
        lazy = lazy.filter(pl.col("problem_id").is_in(problem_ids))
    return (
        lazy
        .select([
            "submission_id", "problem_id", "user_id", "difficulty",
            "proficiency_group", "status_code", "language", "filename_ext",
        ])
        .collect()
    )


def sample_submissions(
    abc_ids: list[str],
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
    n_per_cell: int = SAMPLES_PER_CELL,
    seed: int = 42,
    language: str | None = None,
    problem_id: str | None = None,
) -> pl.DataFrame:
    """
    Stratified sample: up to n_per_cell rows per (difficulty × status_code) cell.
    If problem_id is given (single id, or comma-separated list), samples per
    (problem_id × status_code) instead — stratifying by problem prevents the
    largest problem in the list from dominating the pooled sample.
    """
    problem_ids = _parse_problem_ids(problem_id)
    lang_str = f" (langage : {language}*)" if language else ""
    pid_str  = f" ({len(problem_ids)} problèmes)" if problem_ids and len(problem_ids) > 1 \
               else f" (problème : {problem_ids[0]})" if problem_ids else ""
    print(f"Loading submission metadata for sampling{lang_str}{pid_str}...")
    df_all = _load_eligible(abc_ids, df_abc, df_users, language=language, problem_id=problem_id)
    print(f"  Eligible submissions: {df_all.height:,}")

    rng = random.Random(seed)
    sampled = []

    if problem_ids:
        # Un ou plusieurs problèmes fixés → stratifier par (problème × status)
        for pid in problem_ids:
            for status in TARGET_STATUSES:
                cell = df_all.filter(
                    (pl.col("problem_id") == pid) & (pl.col("status_code") == status)
                )
                n = min(n_per_cell, cell.height)
                if n == 0:
                    print(f"  {pid} × {status}: 0 — skipped")
                    continue
                idx = sorted(rng.sample(range(cell.height), n))
                sampled.append(cell[idx])
                print(f"  {pid} × {status}: {n:,}")
    else:
        for diff in TARGET_DIFFICULTIES:
            for status in TARGET_STATUSES:
                cell = df_all.filter(
                    (pl.col("difficulty") == diff) & (pl.col("status_code") == status)
                )
                n = min(n_per_cell, cell.height)
                if n == 0:
                    print(f"  {diff} × {status}: 0 — skipped")
                    continue
                idx = sorted(rng.sample(range(cell.height), n))
                sampled.append(cell[idx])
                print(f"  {diff} × {status}: {n:,}")

    if not sampled:
        raise RuntimeError(
            f"0 soumissions éligibles pour problem_id='{problem_id}'. "
            f"Vérifiez l'ID avec : python src/embedding.py list"
        )
    result = pl.concat(sampled)
    print(f"\n  Total sampled: {result.height:,} submissions")
    return result


# ── Step 2 — Read source code ─────────────────────────────────────────────────

def read_source_code(problem_id: str, language: str, submission_id: str, ext: str) -> str | None:
    """
    Reads from: data/Project_CodeNet/data/{problem_id}/{language}/{submission_id}.{ext}
    Returns None if missing or unreadable.
    """
    path = RAW_DATA_DIR / "data" / problem_id / language / f"{submission_id}.{ext}"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def load_codes(df_sample: pl.DataFrame) -> tuple[list[str], list[bool]]:
    """
    Reads source code for all sampled submissions.
    Returns (codes, valid_mask) where valid_mask[i] = False means file was missing.
    """
    print("Reading source code files...")
    codes, valid_mask = [], []
    for row in df_sample.iter_rows(named=True):
        code = read_source_code(
            row["problem_id"], row["language"], row["submission_id"], row["filename_ext"]
        )
        if code and len(code.strip()) > 10:
            codes.append(code)
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    n_missing = valid_mask.count(False)
    print(f"  Read: {len(codes):,}  Missing/empty: {n_missing:,}")
    return codes, valid_mask


# ── Step 3a — TF-IDF baseline (fast, no GPU) ─────────────────────────────────

def embed_tfidf(codes: list[str]) -> np.ndarray:
    """
    TF-IDF over identifier tokens. Fast baseline, no semantic understanding.
    Returns float32 array of shape (N, 10000).

    Requires: pip install scikit-learn
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("Computing TF-IDF embeddings...")
    vec = TfidfVectorizer(
        token_pattern=r"[a-zA-Z_]\w*",  # identifier-like tokens only
        max_features=10_000,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(codes)
    print(f"  Vocabulary size: {len(vec.vocabulary_):,}")
    return matrix.toarray().astype(np.float32)


# ── Step 3b — CodeBERT family (slow, requires torch + transformers) ──────────

# method -> checkpoint HuggingFace. "graphcodebert" ici est la version naïve :
# même pipeline que codebert (tokenisation brute, token CLS), sans construire le
# graphe de flux de données attendu par le modèle — voir NB12 pour la discussion.
# "graphcodebert_ast" construit le vrai graphe (tree-sitter + DFG.py officiel,
# voir embed_graphcodebert_ast() plus bas et NB13) — même checkpoint, forward
# pass différent.
CODEBERT_CHECKPOINTS = {
    "codebert":      "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
}


def embed_codebert(
    codes: list[str],
    batch_size: int = 16,
    device: str = "cpu",
    checkpoint: str = "microsoft/codebert-base",
) -> np.ndarray:
    """
    Embeds source code with a CodeBERT-family checkpoint (CLS token, 768 dims).
    Returns float32 array of shape (N, 768).

    Requires: pip install transformers torch
    Speed: ~1 submission/sec on CPU, ~50-100/sec on GPU.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {checkpoint} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model     = AutoModel.from_pretrained(checkpoint).to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(codes), batch_size):
        batch   = codes[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded)

        cls_vecs = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_vecs)

        done = min(i + batch_size, len(codes))
        if done % (batch_size * 10) == 0 or done == len(codes):
            print(f"  {done}/{len(codes)}")

    return np.vstack(embeddings).astype(np.float32)


# ── Step 3c — Causal LLM (last-token / mean pooling, no generation) ──────────

def embed_llm_lasttoken(
    codes: list[str],
    checkpoint: str,
    batch_size: int = 16,
    device: str = "cpu",
    pooling: str = "last",
) -> np.ndarray:
    """
    Embeds source code with a causal (decoder-only) LLM checkpoint - a single
    forward pass per batch, no generation. Unlike CodeBERT's CLS token
    (prepended, trained to summarize the whole sequence from the start), a
    causal model has no such token: pooling="last" (default) takes the final
    token's hidden state instead - thanks to causal attention, it has
    attended to every token before it, making it the natural analogue.
    pooling="mean" averages over all real (non-padding) token positions
    instead. Same cls-vs-mean question NB13 S10 already answered for
    GraphCodeBERT, now last-vs-mean since a decoder has no CLS.

    checkpoint: local path (e.g. data/models/Qwen--Qwen2.5-Coder-7B-Instruct)
    or a Hugging Face Hub repo id - passed straight to from_pretrained(), so
    any already-downloaded local model or hub checkpoint works unmodified.

    Returns float32 array of shape (N, hidden_size) - hidden_size depends on
    the checkpoint (e.g. 3584 for Qwen2.5-Coder-7B-Instruct).

    Requires: pip install transformers torch
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if pooling not in ("last", "mean"):
        raise ValueError(f"pooling='{pooling}' invalide - 'last' ou 'mean' attendu.")

    print(f"Loading {checkpoint} on {device} (pooling={pooling})...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Left padding: with pooling="last", this keeps the real last token at
    # position -1 for every row regardless of that sequence's length within
    # the batch. Right padding (the default) would land on a pad token
    # instead for anything shorter than the batch's longest sequence.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # Common for causal-LM tokenizers - no padding needed for single-
        # sequence generation, so none is defined by default.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(codes), batch_size):
        batch   = codes[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)

        last_layer = out.hidden_states[-1]  # (batch, seq_len, hidden)
        if pooling == "last":
            vecs = last_layer[:, -1, :]  # left-padded -> always the real last token
        else:  # mean
            mask    = encoded["attention_mask"].unsqueeze(-1).to(last_layer.dtype)
            summed  = (last_layer * mask).sum(dim=1)
            counts  = mask.sum(dim=1).clamp(min=1)
            vecs    = summed / counts

        # Checkpoints commonly load in bfloat16 (Qwen2.5-Coder's native
        # precision) - numpy has no bf16 type, .numpy() rejects it directly.
        # Cast to fp32 first (harmless: only on an already-small
        # (batch, hidden) tensor, not the full model).
        embeddings.append(vecs.float().cpu().numpy())

        done = min(i + batch_size, len(codes))
        if done % (batch_size * 10) == 0 or done == len(codes):
            print(f"  {done}/{len(codes)}")

    return np.vstack(embeddings).astype(np.float32)


# ── Step 3c — GraphCodeBERT with real data-flow graph (faithful variant) ─────
#
# Ports the extraction + feature-building + graph-guided forward pass from
# the official microsoft/CodeBERT repo (GraphCodeBERT/codesearch:
# parser/DFG.py, run.py, model.py — vendored under src/parser/, MIT license).
# Two deliberate deviations from the literal official code, both forced by
# things only discovered by actually running this against the real
# checkpoint — see _GraphCodeBERTWrapper's docstring for the more important
# one (pooler_output -> CLS of last_hidden_state: this checkpoint ships no
# pooler weights at all). The other is mechanical: np.bool (removed from
# NumPy) -> bool. See NB13.
#
# Unlike embed_codebert(), this can't use the tokenizer's batched
# padding/truncation: GraphCodeBERT's feature format (token ids interleaved
# with synthetic "graph node" positions, plus a full seq_len x seq_len
# attention mask instead of the usual 1D padding mask) has to be built by
# hand, one submission at a time, before anything reaches the GPU — expect
# this to be noticeably slower than embed_codebert() at the same corpus size.

# CODE_LENGTH + DATA_FLOW_LENGTH = MAX_TOKENS (512, the model's real usable
# position budget -- verified: max_position_embeddings=514, minus the pad
# position). Originally split 448/64; measured directly against a 3000-
# submission sample of the actual corpus (see NB13 deep-dive) that the 64-node
# graph budget was truncating 36.8% of submissions, vs. only 7.1% for the
# 448-token code budget -- a lopsided split, not a deliberate tradeoff. 384/128
# was chosen by simulating candidate splits on that same sample and picking
# the one that roughly equalizes both truncation rates (9.8% graph / 11.1%
# code) instead of starving one side. Changing this invalidates any
# previously-generated graphcodebert_ast_*.npy files -- they were embedded
# under the old split and need regenerating to reflect it.
CODE_LENGTH      = 384
DATA_FLOW_LENGTH = MAX_TOKENS - CODE_LENGTH  # 128

# Langages couverts par un DFG_* dans le GraphCodeBERT officiel. C/C++ (le
# langage dominant du corpus AtCoder) n'y figure pas : aucun DFG_cpp n'existe
# en amont, le pré-entraînement graph-guided n'a jamais couvert ce langage.
_TS_GRAMMAR_PACKAGES = {
    "python":     "tree_sitter_python",
    "java":       "tree_sitter_java",
    "ruby":       "tree_sitter_ruby",
    "go":         "tree_sitter_go",
    "php":        "tree_sitter_php",
    "javascript": "tree_sitter_javascript",
}


def _resolve_dfg_language(language: str | None) -> str:
    """
    Maps the CLI 'language' argument (e.g. "Python", used elsewhere as a
    prefix filter against the raw submission language column) to a DFG.py
    language key. Raises with a clear message if there's no official DFG_*
    for it — must fail before sampling starts, not silently fall back to
    the naive pipeline.
    """
    if not language:
        raise ValueError(
            "graphcodebert_ast nécessite un langage explicite (ex. \"Python\") : "
            "le graphe de flux de données est spécifique à un langage."
        )
    dfg_lang = language.strip().lower().split()[0]
    if dfg_lang not in _TS_GRAMMAR_PACKAGES:
        raise ValueError(
            f"Langage '{language}' non supporté par graphcodebert_ast. "
            f"Langages disponibles : {', '.join(sorted(_TS_GRAMMAR_PACKAGES))}. "
            f"C/C++ n'a pas de DFG_* officiel dans GraphCodeBERT (voir src/parser/DFG.py) "
            f"— seule la version naïve ('graphcodebert') fonctionne dessus."
        )
    return dfg_lang


def _get_dfg_parser(dfg_lang: str):
    """
    Builds a tree-sitter Parser + returns the matching DFG_* function for one
    language. Only Python has been exercised end-to-end in this project so
    far (see NB13) — the others are wired the same way but unverified here;
    PHP additionally needs the "<?php ... ?>" wrapping done in
    _extract_dataflow(), matching the official extract_dataflow().

    Requires: pip install tree-sitter tree-sitter-{dfg_lang}
    """
    import importlib

    from tree_sitter import Language, Parser as TSParser

    sys.path.insert(0, str(Path(__file__).parent))
    from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript

    dfg_functions = {
        "python": DFG_python, "java": DFG_java, "ruby": DFG_ruby,
        "go": DFG_go, "php": DFG_php, "javascript": DFG_javascript,
    }
    pkg_name = _TS_GRAMMAR_PACKAGES[dfg_lang]
    try:
        grammar_mod = importlib.import_module(pkg_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Grammaire tree-sitter pour '{dfg_lang}' non installée. "
            f"Lancez : pip install {pkg_name.replace('_', '-')}"
        ) from exc

    ts_parser = TSParser(Language(grammar_mod.language()))
    return ts_parser, dfg_functions[dfg_lang]


def _extract_dataflow(code: str, ts_parser, dfg_func, dfg_lang: str):
    """
    Port of the official extract_dataflow(): strip comments/docstrings, parse
    with tree-sitter, run DFG_{lang}, then keep only edges actually connected
    to something (isolated nodes carry no information — matches upstream's
    own filtering in run.py).
    """
    from parser import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token

    try:
        code = remove_comments_and_docstrings(code, dfg_lang)
    except Exception:
        pass
    if dfg_lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = ts_parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code_lines = code.split("\n")
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
        index_to_code = {}
        for idx, (index, tok) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, tok)
        try:
            dfg, _ = dfg_func(root_node, index_to_code, {})
        except Exception:
            dfg = []
        dfg = sorted(dfg, key=lambda x: x[1])
        indexs = set()
        for d in dfg:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        dfg = [d for d in dfg if d[1] in indexs]
    except Exception:
        code_tokens, dfg = [], []
    return code_tokens, dfg


def _build_graph_features(code: str, tokenizer, ts_parser, dfg_func, dfg_lang: str):
    """
    Port of the official convert_examples_to_features() (code side only — no
    NL/docstring pair here, we only need one embedding per submission):
    subword-tokenizes each leaf token, truncates to leave room for up to
    DATA_FLOW_LENGTH graph nodes, appends the DFG nodes with position_idx=0
    (the marker _GraphCodeBERTWrapper uses to find them), and remaps DFG
    edges from original token indices to the final compact array.
    """
    code_tokens, dfg = _extract_dataflow(code, ts_parser, dfg_func, dfg_lang)
    code_tokens = [
        tokenizer.tokenize("@ " + x)[1:] if idx != 0 else tokenizer.tokenize(x)
        for idx, x in enumerate(code_tokens)
    ]
    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    code_tokens = code_tokens[: CODE_LENGTH + DATA_FLOW_LENGTH - 2 - min(len(dfg), DATA_FLOW_LENGTH)]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]

    dfg = dfg[: CODE_LENGTH + DATA_FLOW_LENGTH - len(code_tokens)]
    position_idx += [0 for _ in dfg]
    code_ids += [tokenizer.unk_token_id for _ in dfg]
    padding_length = CODE_LENGTH + DATA_FLOW_LENGTH - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length

    reverse_index = {x[1]: idx for idx, x in enumerate(dfg)}
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    cls_len = 1
    dfg_to_code = [(a + cls_len, b + cls_len) for a, b in dfg_to_code]

    return code_ids, position_idx, dfg_to_code, dfg_to_dfg


def _build_attn_mask(code_ids, position_idx, dfg_to_code, dfg_to_dfg, tokenizer) -> np.ndarray:
    """
    Port of TextDataset.__getitem__'s mask construction (run.py) — real bool
    dtype instead of the deprecated np.bool the official 2021 code used.
    """
    total_len = CODE_LENGTH + DATA_FLOW_LENGTH
    attn_mask = np.zeros((total_len, total_len), dtype=bool)
    node_index = sum(1 for i in position_idx if i > 1)
    max_length = sum(1 for i in position_idx if i != 1)
    attn_mask[:node_index, :node_index] = True
    for idx, i in enumerate(code_ids):
        if i in (tokenizer.cls_token_id, tokenizer.sep_token_id):
            attn_mask[idx, :max_length] = True
    for idx, (a, b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    for idx, nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(position_idx):
                attn_mask[idx + node_index, a + node_index] = True
    return attn_mask


def compute_dfg_features(df_meta: pl.DataFrame, dfg_lang: str, verbose: bool = False) -> pl.DataFrame:
    """
    For each submission in df_meta (needs problem_id, language, submission_id,
    filename_ext — the same columns sample_submissions() already produces),
    re-parses its source and returns three structural features shown (NB13's
    truncation diagnostics) to correlate with verdict independently of
    whatever the model's embedding itself captures: raw DFG node count, raw
    DFG edge count (both *before* embed_graphcodebert_ast's truncation to
    DATA_FLOW_LENGTH — the model may never see the full graph, but these
    features can still reflect it), and code length in subword tokens.

    CPU-only, no model loaded — reuses the same tree-sitter/DFG.py machinery
    as embed_graphcodebert_ast() but never runs the network, so this is much
    faster and can run over a full multi-thousand corpus in about a minute.

    Returns a DataFrame with submission_id (+ these 3 new columns), meant to
    be joined back onto a metadata/embeddings frame by submission_id.

    Requires: pip install transformers tree-sitter tree-sitter-{dfg_lang}
    (no torch needed — nothing here touches the model).
    """
    from transformers import AutoTokenizer

    ts_parser, dfg_func = _get_dfg_parser(dfg_lang)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    rows = []
    total = df_meta.height
    for i, row in enumerate(df_meta.iter_rows(named=True)):
        code = read_source_code(
            row["problem_id"], row["language"], row["submission_id"], row["filename_ext"]
        )
        if not code:
            rows.append({
                "submission_id": row["submission_id"],
                "dfg_nodes": None, "dfg_edges": None, "code_subword_tokens": None,
            })
            continue

        code_tokens, dfg = _extract_dataflow(code, ts_parser, dfg_func, dfg_lang)
        sub_tokens = [
            tokenizer.tokenize("@ " + x)[1:] if idx != 0 else tokenizer.tokenize(x)
            for idx, x in enumerate(code_tokens)
        ]
        rows.append({
            "submission_id": row["submission_id"],
            "dfg_nodes": len(dfg),
            "dfg_edges": sum(len(x[-1]) for x in dfg),
            "code_subword_tokens": sum(len(t) for t in sub_tokens),
        })

        done = i + 1
        if verbose and (done % 2000 == 0 or done == total):
            print(f"  features DFG : {done}/{total}")

    return pl.DataFrame(rows)


class _GraphCodeBERTWrapper:
    """
    Adapts a plain AutoModel RoBERTa encoder to GraphCodeBERT's graph-guided
    forward pass (official model.py, MIT license): a DFG node's input
    embedding is the mean of the token embeddings it's connected to (nodes
    aren't real vocabulary tokens), attention uses the full 2D mask built
    above instead of a 1D padding mask, and position_ids come from the
    node/token layout instead of a plain range.

    Deviation from official model.py: it returns forward()[1], the pooler
    output. We deliberately don't: microsoft/graphcodebert-base's checkpoint
    (verified directly against its pytorch_model.bin state dict) ships only
    roberta.* and lm_head.* weights — no pooler.*. Official downstream
    scripts get a meaningful pooler by fine-tuning it on a task; used
    zero-shot (our case), it's a randomly-initialized, never-trained
    Linear+Tanh, and reading it off would inject untrained-random noise into
    every embedding. We return the CLS position of last_hidden_state instead
    (pooling="cls", default) — genuinely pretrained weights only, and the
    same readout convention embed_codebert() already uses, so the naive/AST
    comparison isolates the effect of the graph mechanism rather than also
    changing the readout.

    pooling="mean" is an alternative readout, not an official-code deviation
    like the two above — a genuine experiment. It averages last_hidden_state
    over real code-token positions only (excludes CLS/SEP and the synthetic
    graph-node slots), instead of trusting a single designated position (CLS)
    to have organized itself into a good summary. Since nothing here was ever
    fine-tuned for this project's classification tasks, that's not guaranteed
    — averaging spreads the readout across everything the model actually
    attended over. See NB13's mean-pooling section for the comparison against
    pooling="cls".
    """
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, code_ids, attn_mask, position_idx, pooling="cls",
                 cls_token_id=None, sep_token_id=None):
        import torch

        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.embeddings.word_embeddings(code_ids)
        nodes_to_token_mask = (nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask).float()
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = (
            inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
        )
        # transformers' modern masking utils only accept a 2D padding mask or
        # an already-4D (batch, 1, q_len, kv_len) mask (returned untouched,
        # bypassing the incompatible legacy 3D path) — see _preprocess_mask_
        # arguments in transformers.masking_utils. sdpa is forced explicitly
        # (see embed_graphcodebert_ast) so this bool mask keeps PyTorch SDPA's
        # own True=attend semantics, matching attn_mask's construction above;
        # "eager" would instead add the mask as a float bias and silently
        # compute something else.
        out = self.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask.unsqueeze(1),
            position_ids=position_idx,
        )
        if pooling == "cls":
            return out.last_hidden_state[:, 0, :]
        elif pooling == "mean":
            # Real code tokens only: position_idx>=2 already covers CLS/SEP
            # too (see _build_graph_features), so they're explicitly excluded
            # by id instead — same ids _build_attn_mask already special-cases.
            content_mask = (
                position_idx.ge(2) & code_ids.ne(cls_token_id) & code_ids.ne(sep_token_id)
            ).unsqueeze(-1).float()
            summed = (out.last_hidden_state * content_mask).sum(dim=1)
            counts = content_mask.sum(dim=1).clamp(min=1.0)
            return summed / counts
        else:
            raise ValueError(f"pooling inconnu : '{pooling}' (attendu : 'cls' ou 'mean')")


def embed_graphcodebert_ast(
    codes: list[str],
    dfg_lang: str,
    batch_size: int = 16,
    device: str = "cpu",
    checkpoint: str = "microsoft/graphcodebert-base",
    verbose: bool = False,
    pooling: str = "cls",
) -> np.ndarray:
    """
    Embeds source code with GraphCodeBERT's actual graph-guided forward pass:
    tree-sitter -> DFG_{dfg_lang} data-flow graph -> node/token layout with a
    full 2D attention mask -> pooled readout (768 dims). Faithful port of the
    official model.py / run.py (microsoft/CodeBERT, GraphCodeBERT/codesearch,
    MIT license) for the graph mechanism itself — see _GraphCodeBERTWrapper's
    docstring for what pooling="cls" (default) vs "mean" means and why
    neither is official model.py's pooler_output, and the module comment
    above and NB13 for the rest.

    verbose=True prints, for every submission, its token count and DFG node/
    edge count as it's parsed — meant for a small interactive corpus (NB13's
    ~20-submission pilot); left off by default because it'd be one line per
    submission, unreadable on a multi-thousand corpus. Not exposed on the
    CLI for the same reason — call this function directly (e.g. from a
    notebook) if you want it. The batch progress line below always shows the
    aggregate node/edge counts regardless of verbose, at the usual cadence.

    Requires: pip install transformers torch tree-sitter tree-sitter-{dfg_lang}
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    ts_parser, dfg_func = _get_dfg_parser(dfg_lang)

    print(f"Loading {checkpoint} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # sdpa explicitly (not "auto") so the bool 2D->4D attention mask below
    # keeps PyTorch SDPA's True=attend semantics — see _GraphCodeBERTWrapper.
    encoder = AutoModel.from_pretrained(checkpoint, attn_implementation="sdpa").to(device)
    encoder.eval()
    model = _GraphCodeBERTWrapper(encoder)

    n_empty_dfg = 0
    n_nodes_total = 0
    n_edges_total = 0
    embeddings = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i : i + batch_size]
        batch_ids, batch_pos, batch_mask = [], [], []
        for j, code in enumerate(batch):
            code_ids, position_idx, dfg_to_code, dfg_to_dfg = _build_graph_features(
                code, tokenizer, ts_parser, dfg_func, dfg_lang
            )
            n_nodes = len(dfg_to_code)  # DFG.py's "graph nodes" — variables/values kept post-filtering
            n_edges = sum(len(adj) for adj in dfg_to_dfg)  # comesFrom/computedFrom links between them
            n_nodes_total += n_nodes
            n_edges_total += n_edges
            if n_nodes == 0:
                n_empty_dfg += 1
            if verbose:
                n_tokens = sum(1 for p in position_idx if p > 1)
                print(f"    [{i + j + 1}/{len(codes)}] AST -> {n_tokens} tokens de code  |  "
                      f"DFG -> {n_nodes} nœuds, {n_edges} arêtes")
            attn_mask = _build_attn_mask(code_ids, position_idx, dfg_to_code, dfg_to_dfg, tokenizer)
            batch_ids.append(code_ids)
            batch_pos.append(position_idx)
            batch_mask.append(attn_mask)

        code_ids_t     = torch.tensor(batch_ids, dtype=torch.long, device=device)
        position_idx_t = torch.tensor(batch_pos, dtype=torch.long, device=device)
        attn_mask_t    = torch.tensor(np.stack(batch_mask), dtype=torch.bool, device=device)

        with torch.no_grad():
            pooled = model(code_ids_t, attn_mask_t, position_idx_t, pooling=pooling,
                            cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id)

        embeddings.append(pooled.cpu().numpy())

        done = min(i + batch_size, len(codes))
        if done % (batch_size * 10) == 0 or done == len(codes):
            print(f"  {done}/{len(codes)}  (moy. {n_nodes_total / done:.1f} nœuds DFG, "
                  f"{n_edges_total / done:.1f} arêtes / soumission ; {n_empty_dfg} sans graphe jusqu'ici)")

    if n_empty_dfg:
        print(
            f"  Avertissement : {n_empty_dfg}/{len(codes)} soumissions sans nœud DFG "
            f"(code trop simple pour produire une dépendance, ou échec de parsing tree-sitter)"
        )

    return np.vstack(embeddings).astype(np.float32)


# ── Cluster analysis — pairwise code similarity ───────────────────────────────
#
# Not part of the embedding pipeline: a standalone check for whether
# submissions placed close together in a projection (e.g. a dense DBSCAN
# cluster on a t-SNE plot) are actually similar in their *source code*, or
# just coincidentally close in embedding space. Deliberately uses difflib
# rather than the embedding model under investigation, to avoid circularity —
# comparing an embedding's neighborhoods against a metric built from the same
# embedding would just confirm the embedding agrees with itself.

def pairwise_code_similarity(df_meta: pl.DataFrame) -> np.ndarray:
    """
    Comment/docstring-stripped, pairwise difflib.SequenceMatcher ratio for
    every pair of rows in df_meta (needs problem_id, language, submission_id,
    filename_ext — the same columns read_source_code() takes). Order-agnostic:
    the caller decides what group of submissions this is (a cluster, a random
    baseline sample, ...) — this function only ever sees "a list of rows".

    Returns a symmetric (n, n) matrix, diagonal = 1.0. A row/column for a
    submission whose source file is missing or fails to strip is filled with
    NaN rather than dropped, so the matrix shape always matches df_meta and a
    caller averaging over it must consciously decide how to handle NaN
    (np.nanmean) instead of silently comparing groups of different sizes.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from parser import remove_comments_and_docstrings

    n = df_meta.height
    codes: list[str | None] = [None] * n
    for i, row in enumerate(df_meta.iter_rows(named=True)):
        raw = read_source_code(
            row["problem_id"], row["language"], row["submission_id"], row["filename_ext"]
        )
        if raw is None:
            continue
        try:
            codes[i] = remove_comments_and_docstrings(raw, row["language"].strip().lower())
        except Exception:
            codes[i] = raw  # tokenize() can raise on malformed/partial Python source — fall back to raw text rather than dropping the submission

    sim = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        if codes[i] is None:
            continue
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            if codes[j] is None:
                continue
            ratio = difflib.SequenceMatcher(None, codes[i], codes[j]).ratio()
            sim[i, j] = sim[j, i] = ratio
    return sim


# ── Step 4 — Save ─────────────────────────────────────────────────────────────

def save(embeddings: np.ndarray, df_meta: pl.DataFrame, method: str) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_emb  = EMBEDDINGS_DIR / f"embeddings_{method}.npy"
    out_meta = EMBEDDINGS_DIR / f"metadata_{method}.csv"
    np.save(out_emb, embeddings)
    df_meta.write_csv(out_meta)
    print(f"\n  embeddings : {out_emb}  shape={embeddings.shape}")
    print(f"  metadata   : {out_meta}  rows={df_meta.height:,}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    abc_ids: list[str],
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
    method: str = "tfidf",
    n_per_cell: int = SAMPLES_PER_CELL,
    device: str = "cpu",
    batch_size: int = 16,
    seed: int = 42,
    language: str | None = None,
    problem_id: str | None = None,
    checkpoint: str | None = None,
) -> None:
    # Fail fast — a missing/unsupported language for graphcodebert_ast should
    # error before spending minutes loading metadata and reading source files.
    dfg_lang = _resolve_dfg_language(language) if method == "graphcodebert_ast" else None
    if method == "llm" and not checkpoint:
        raise ValueError(
            "method='llm' requires a checkpoint (6th CLI arg) - local path "
            "(e.g. data/models/Qwen--Qwen2.5-Coder-7B-Instruct) or a HF Hub repo id."
        )
    # Not a CLI positional arg (5 is already a lot to remember the order of)
    # — an env var, since this is an experiment, not a routine knob:
    #   GCB_POOLING=mean python src/embedding.py graphcodebert_ast ...
    #   LLM_POOLING=mean python src/embedding.py llm ... <checkpoint>
    if method == "graphcodebert_ast":
        pooling = os.environ.get("GCB_POOLING", "cls")
        if pooling not in ("cls", "mean"):
            raise ValueError(f"GCB_POOLING='{pooling}' invalide — 'cls' ou 'mean' attendu.")
    elif method == "llm":
        pooling = os.environ.get("LLM_POOLING", "last")
        if pooling not in ("last", "mean"):
            raise ValueError(f"LLM_POOLING='{pooling}' invalide — 'last' ou 'mean' attendu.")
    else:
        pooling = "cls"

    df_sample = sample_submissions(
        abc_ids, df_abc, df_users, n_per_cell, seed,
        language=language, problem_id=problem_id,
    )

    codes, valid_mask = load_codes(df_sample)
    df_valid = df_sample.filter(pl.Series("valid", valid_mask))

    if len(codes) == 0:
        raise RuntimeError("No source code files found. Check CodeNet data path.")

    problem_ids = _parse_problem_ids(problem_id)
    lang_tag = "_" + language.lower().replace("+", "p") if language else ""
    if not problem_ids:
        pid_tag = ""
    elif len(problem_ids) == 1:
        pid_tag = "_" + problem_ids[0].replace("/", "-")
    else:
        pid_tag = f"_multi{len(problem_ids)}"
    # Non-default n_per_cell gets its own tag so a small pilot run (e.g. a
    # first graphcodebert_ast smoke test) never overwrites — or gets mistaken
    # for — a full-scale run of the same method/language/problem_id.
    n_tag = f"_n{n_per_cell}" if n_per_cell != SAMPLES_PER_CELL else ""
    # Non-default pooling gets its own tag for the same reason n_tag does —
    # a pooling="mean" run must never collide with the default-pooled file.
    # Default differs by method: "cls" for graphcodebert_ast, "last" for llm.
    # Only methods with an actual pooling choice get a tag at all - tfidf,
    # codebert, graphcodebert always use "cls" with no alternative, so
    # tagging them "_cls" would be noise, not information (bug found the
    # hard way: it was doing exactly that before this fix).
    if method == "graphcodebert_ast":
        pooling_tag = f"_{pooling}" if pooling != "cls" else ""
    elif method == "llm":
        pooling_tag = f"_{pooling}" if pooling != "last" else ""
    else:
        pooling_tag = ""
    # llm needs its own checkpoint in the filename too — unlike the other
    # methods, "llm" alone doesn't say which model, and two different
    # checkpoints must never collide on the same output file.
    checkpoint_tag = f"_{Path(checkpoint).name}" if method == "llm" else ""
    method_tag = f"{method}{checkpoint_tag}{lang_tag}{pid_tag}{n_tag}{pooling_tag}"

    print(f"\nEmbedding with {method_tag}...")
    if method == "tfidf":
        emb = embed_tfidf(codes)
    elif method == "graphcodebert_ast":
        emb = embed_graphcodebert_ast(codes, dfg_lang, batch_size=batch_size, device=device,
                                       checkpoint=CODEBERT_CHECKPOINTS["graphcodebert"], pooling=pooling)
    elif method in CODEBERT_CHECKPOINTS:
        emb = embed_codebert(codes, batch_size=batch_size, device=device,
                              checkpoint=CODEBERT_CHECKPOINTS[method])
    elif method == "llm":
        emb = embed_llm_lasttoken(codes, checkpoint, batch_size=batch_size, device=device, pooling=pooling)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'tfidf', 'codebert', 'graphcodebert', 'graphcodebert_ast', or 'llm'."
        )

    save(emb, df_valid, method_tag)


# ── CLI entry point ───────────────────────────────────────────────────────────

HELP_TEXT = """\
Usage : python src/embedding.py <méthode> [device] [langage] [problem_id] [n_per_cell]

Méthodes :
  tfidf              Baseline lexicale (TF-IDF sur tokens identifiants) — rapide, CPU
  codebert           Encodeur sémantique microsoft/codebert-base (768 dims) — GPU conseillé
  graphcodebert      microsoft/graphcodebert-base, même pipeline que codebert (tokens
                     bruts, token CLS) — version *naïve* : ne construit pas le graphe
                     de flux de données que le modèle sait exploiter, teste seulement
                     si le pré-entraînement structure-aware laisse une trace résiduelle
                     dans les représentations de tokens. Voir NB12.
  graphcodebert_ast  Même checkpoint, mais avec le vrai graphe de flux de données :
                     tree-sitter -> DFG.py officiel (vendored sous src/parser/) ->
                     masque d'attention 2D + position_ids graphe -> CLS de
                     last_hidden_state (checkpoint sans pooler, voir NB13).
                     Nécessite un langage supporté par un DFG_* officiel (python,
                     java, ruby, go, php, javascript — PAS C/C++, qui n'a pas de
                     DFG_cpp en amont). Plus lent que codebert/graphcodebert :
                     construction du graphe en Python, un fichier à la fois.
                     Variable d'environnement GCB_POOLING=mean (défaut : cls) —
                     moyenne sur les tokens de code réels au lieu du seul CLS,
                     expérimental, tag de fichier _mean. Voir NB13.
  llm                LLM causal (decoder-only) quelconque — TF-IDF/CodeBERT
                     encodent, celui-ci ne fait qu'un seul passage forward (pas
                     de génération). Pas de token CLS sur un modèle causal :
                     pooling="last" (défaut) prend le dernier token — grâce à
                     l'attention causale, il a "vu" tout ce qui précède, c'est
                     l'équivalent naturel du CLS. Variable d'environnement
                     LLM_POOLING=mean (défaut : last) pour la moyenne à la
                     place, même question cls-vs-mean que graphcodebert_ast
                     (NB13 S10), tag de fichier _mean. Nécessite un 6e argument
                     positionnel, le checkpoint (voir ci-dessous) — chemin
                     local déjà téléchargé ou id HuggingFace Hub. Variable
                     d'environnement EMBED_BATCH_SIZE (défaut : 16, aussi
                     valable pour codebert/graphcodebert) pour ajuster le
                     débit selon la mémoire disponible.
  list [D]           Liste les problem_id valides (top 30 par volume de soumissions),
                     filtrable par difficulté : python src/embedding.py list D
  help               Affiche cette aide

Arguments positionnels :
  device       cpu (défaut) | mps (Apple Silicon) | cuda (NVIDIA) — ignoré par tfidf
  langage      Préfixe de langage : "C++" (couvre C++14/17/20…), "Python", "Java"…
               "" (chaîne vide) = tous les langages. Obligatoire (et restreint aux
               langages ci-dessus) pour graphcodebert_ast.
  problem_id   Restreint à un ou plusieurs problèmes : un seul id (ex. p02616),
               ou une liste séparée par virgules (ex. p02616,p02642,p02658).
               Sans problem_id : échantillon stratifié difficulté (B–E) × verdict,
               500 soumissions max par case.
               Avec problem_id : stratifié par (problème × verdict) — chaque
               problème listé reçoit son propre quota, qu'il y en ait un ou
               plusieurs, pour que le plus gros ne domine pas l'échantillon.
  n_per_cell   Quota par case (problème × verdict, ou difficulté × verdict) —
               défaut 500. Même tirage stratifié + seed=42 que d'habitude, juste
               un quota plus petit : utile pour un pilote rapide (ex. 5 -> ~20
               soumissions sur un problème à 4 verdicts effectifs) avant de lancer
               le corpus complet. Une valeur non-défaut ajoute un tag _n{N} au nom
               de fichier, pour ne jamais écraser une run à 500/cellule.
  checkpoint   Uniquement pour method=llm (6e argument, obligatoire pour cette
               méthode) : chemin local (ex. data/models/Qwen--Qwen2.5-Coder-7B-
               Instruct) ou id HuggingFace Hub — passé tel quel à
               from_pretrained(). Le nom du checkpoint est repris dans le nom
               de fichier de sortie (deux checkpoints différents ne s'écrasent
               jamais entre eux).

Sorties (data/processed/embeddings/) :
  embeddings_{méthode}[_{langage}][_{problème}][_n{N}].npy   matrice (N, D) float32
  metadata_{méthode}[_{langage}][_{problème}][_n{N}].csv     métadonnées alignées
  Un seul problem_id -> tag exact (ex. _p02616) ; plusieurs -> _multi{N}
  (le problem_id exact de chaque soumission reste dans le CSV de métadonnées).
  Les variantes ne s'écrasent jamais entre elles.

Exemples — un par section du notebook 10_embeddings.ipynb :
  S2  corpus global        python src/embedding.py tfidf
                           python src/embedding.py codebert mps
  S3  un langage           python src/embedding.py tfidf cpu "C++"
                           python src/embedding.py codebert mps "C++"
  S4  un problème          python src/embedding.py list D
                           python src/embedding.py tfidf cpu "" p02616
                           python src/embedding.py codebert mps "" p02616
  S5  problème × langage   python src/embedding.py tfidf cpu "Python" p02616
                           python src/embedding.py codebert mps "Python" p02616

Notebook 11_embeddings_generalization.ipynb — plusieurs problèmes, un langage :
  python src/embedding.py tfidf cpu "Python" "p02658,p02718,p02922,p02659,p02623,p02761,p02642,p02714,p02900,p02616,p02574,p02793"
  python src/embedding.py codebert mps "Python" "p02658,p02718,p02922,p02659,p02623,p02761,p02642,p02714,p02900,p02616,p02574,p02793"

Notebook 12_graphcodebert.ipynb — GraphCodeBERT naïf, même problème/langage que S5 de NB10 :
  python src/embedding.py graphcodebert mps "Python" p02616

Notebook 13_graphcodebert_ast.ipynb — GraphCodeBERT avec le vrai graphe, pilote réduit :
  python src/embedding.py graphcodebert_ast mps "Python" p02659 5

LLM causal, embeddings (RQ2) — checkpoint deja telecharge par src/v2_llm/download_model.py :
  python src/embedding.py llm mps "Python" p02659 5 data/models/Qwen--Qwen2.5-Coder-7B-Instruct
  LLM_POOLING=mean python src/embedding.py llm mps "Python" p02659 5 data/models/Qwen--Qwen2.5-Coder-7B-Instruct
"""


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "tfidf"

    if method in ("help", "-h", "--help"):
        print(HELP_TEXT)
        sys.exit(0)

    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import (
        load_atcoder_problems,
        OUT_USER_PROFILES,
    )
    from difficulty_labeler import label_abc_problems

    df_atcoder = load_atcoder_problems()
    df_abc     = label_abc_problems(df_atcoder)
    abc_ids    = df_abc["problem_id"].to_list()
    df_users   = pl.read_csv(OUT_USER_PROFILES)

    if method == "list":
        # Affiche les problem_ids disponibles avec leur difficulté et nb de soumissions dans metadata
        diff_filter = sys.argv[2] if len(sys.argv) > 2 else None
        rows = df_abc.filter(pl.col("difficulty").is_in(TARGET_DIFFICULTIES))
        if diff_filter:
            rows = rows.filter(pl.col("difficulty") == diff_filter.upper())
        counts = []
        for row in rows.iter_rows(named=True):
            fp = METADATA_DIR / f"{row['problem_id']}.csv"
            if fp.exists():
                try:
                    n = pl.scan_csv(str(fp)).select(pl.len()).collect().item()
                    counts.append((n, row["problem_id"], row["difficulty"]))
                except Exception:
                    pass
        counts.sort(reverse=True)
        print(f"\n{'problem_id':12s}  {'diff':4s}  submissions")
        print("-" * 35)
        for n, pid, diff in counts[:30]:
            print(f"{pid:12s}  {diff:4s}  {n:,}")
        sys.exit(0)

    # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" as fallback
    device     = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    # e.g. "C++" to restrict to C++ variants only (catches C++14, C++17, C++20…)
    language   = sys.argv[3] if len(sys.argv) > 3 else None
    # e.g. "p02402" to restrict to a single problem
    problem_id = sys.argv[4] if len(sys.argv) > 4 else None
    # quota per (problem|difficulty × status) cell — default SAMPLES_PER_CELL
    n_per_cell = int(sys.argv[5]) if len(sys.argv) > 5 else SAMPLES_PER_CELL
    # only meaningful for method="llm" — local path or HF Hub repo id
    checkpoint = sys.argv[6] if len(sys.argv) > 6 else None
    # Not a CLI positional arg — 6 is already a lot to remember the order of,
    # same reasoning as GCB_POOLING/LLM_POOLING. A performance knob, tune it
    # occasionally, don't make everyone scroll past it every run:
    #   EMBED_BATCH_SIZE=64 python src/embedding.py llm ...
    batch_size = int(os.environ.get("EMBED_BATCH_SIZE", 16))

    run_pipeline(abc_ids, df_abc, df_users,
                 method=method, device=device,
                 language=language, problem_id=problem_id,
                 n_per_cell=n_per_cell, checkpoint=checkpoint,
                 batch_size=batch_size)
