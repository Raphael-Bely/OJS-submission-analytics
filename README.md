# OJS Submission Analytics

Research internship project — DDSE Laboratory, Ritsumeikan University  
*Supervised by Prof. Makihara Erina — Developed by Raphael BELY*

---

## Overview

This project analyzes the **AtCoder** submission history from the [Project CodeNet](https://github.com/IBM/Project_CodeNet) dataset (IBM, 2021) to systematically characterize error patterns in Online Judge Systems (OJS) and build an interactive analytics tool.

The work reproduces and extends the findings of:
> Shimizu, S., Makihara, E., & Yoshida, N. (2025). *An Empirical Study of the Error Characteristics in an Online Judge System.* FSE Companion '25.

The project focuses exclusively on **AtCoder** data (1,519 problems, ~12 million submissions), which provides the richest and most structured subset of CodeNet. Integration with AIZU and LeetCode is planned as optional extensions.

---

## Research Questions

### RQ1 — Error Pattern Analysis *(V0 — In Progress)*
> How do programmers' error patterns differ across combinations of problem difficulty, and user proficiency?

Reproducing Shota's proficiency classification (G1–G6) as a baseline, then analyzing error distributions (CE, WA, TLE, RE) across difficulty levels (A–F) and user groups.

### RQ2 — Code-Level Semantic Analysis *(V1 — Planned)*
> Do codes that are close to each other in vector space share similar error patterns?

Using code embeddings (CodeBERT / GraphCodeBERT) to examine whether source codes with short distances in embedding space tend to share the same error characteristics. The tool should allow investigation of whether nearby codes share the same difficulty level or user proficiency group.

### RQ3 — Error Prediction *(V2 — Optional)*
> Does the information visualized by the proposed tool contribute to error prediction?

Distinguishing errors that are easy to predict from those that are not, to shed light on the intrinsic complexity of OJS errors and verify whether insights from RQ1 and RQ2 can function as a predictive model.

---

## Project Architecture

### Iterative Roadmap

```
[ CSV Metadata ]  →  V0: Classical ML & Profiling  →  Dashboard V0  ✅
[ Source Code   ]  →  V1: Embeddings & NLP          →  Dashboard V1  ⏳
[ Code Graphs   ]  →  V2: Deep Learning             →  Predictive Model
```

### Repository Structure

```
src/
├── main.py                   # Pipeline orchestrator (no logic, calls modules in order)
├── data_loader.py            # Single point of access for all raw data I/O
├── difficulty_labeler.py     # Assigns A–F difficulty letters to ABC problems
├── problem_parser.py         # Extracts scores from HTML problem descriptions
├── user_profiling.py         # Builds user profiles and classifies into G1–G6
└── error_analysis.py         # Computes error distributions (by difficulty, group, language)

playground/
├── 00_exploration_score_based_DEPRECATED.ipynb  # Initial exploration (archived)
├── 01_user_classification_G1G6.ipynb            # G1–G6 classification & validation vs Shimizu
├── 02_error_analysis_RQ1.ipynb                  # Error distribution by difficulty (RQ1)
├── 03_error_analysis_by_group_difficulty.ipynb  # Error distribution by difficulty × group (RQ1 extended)
├── 04_language_analysis.ipynb                   # Language distribution & error patterns by language
└── 05_resolution_rate.ipynb                     # Problem resolution rate by difficulty & proficiency group

data/
├── Project_CodeNet/          # Raw dataset (not versioned — 8GB)
└── processed/                # Pipeline outputs (versioned)

docs/
├── planning.md               # Technical specifications & roadmap
└── schedule.md               # 17-week internship timeline
```

---

## Getting Started

### 1. Environment setup

```bash
conda env create -f environment.yml
conda activate codenet
```

### 2. Dataset

Download [Project CodeNet](https://developer.ibm.com/exchanges/data/all/project-codenet/) and extract to `data/Project_CodeNet/`.

### 3. Run the pipeline

```bash
python src/main.py
```

The pipeline runs in phases and produces in `data/processed/`:
- `atcoder_abc_problems_labeled.csv` — 659 ABC problems with difficulty A–F
- `atcoder_user_profiles.csv` — ~96K user profiles with G1–G6 classification
- `atcoder_error_distribution.csv` — error distribution per difficulty level
- `atcoder_error_by_group_difficulty.csv` — error distribution per difficulty × group
- `atcoder_language_distribution.csv` — language distribution per difficulty level
- `atcoder_language_by_group.csv` — language distribution per difficulty × group
- `atcoder_error_by_language.csv` — error distribution per language × difficulty
- `atcoder_resolution_by_difficulty.csv` — resolution, abandon & first try rates per difficulty level
- `atcoder_resolution_by_group.csv` — resolution metrics per difficulty × proficiency group

---

## Current Status — V0

| Step | Status |
|---|---|
| ABC problem labeling (A–F) | ✅ Done |
| User profiling & G1–G6 classification | ✅ Done |
| Error distribution by difficulty (RQ1) | ✅ Done |
| Error distribution by difficulty × group (RQ1 extended) | ✅ Done |
| Language distribution by difficulty & proficiency group | ✅ Done |
| Error patterns by language × difficulty (TLE, AC, CE) | ✅ Done |
| Error patterns by language × proficiency group | ⏳ Planned |
| Resolution rate by difficulty & proficiency group | ✅ Done |
| Streamlit dashboard | ⏳ Planned |
---

## Dataset

**Project CodeNet** — IBM Research (2021)  
AtCoder subset: 1,519 problems · ~12M submissions · ~124K users

---

*Ritsumeikan University — DDSE Laboratory — 2026*
