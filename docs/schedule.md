# OJS Submission Analytics Dashboard — Project Roadmap & Technical Specifications

This repository contains the source code, data pipelines, and interactive analytics dashboard developed during my research internship at Ritsumeikan University (DDSE Laboratory, supervised by Prof. Makihara Erina). The project focuses on analyzing the **AtCoder** submission history from Project CodeNet to systematically characterize error patterns in Online Judge Systems (OJS) and build an interactive analytics tool.

> **Scope confirmed (Prof. Makihara, May 2025):** The project focuses exclusively on **AtCoder** data, which constitutes the majority of CodeNet and provides the difficulty labels required for the analysis. AIZU and LeetCode integration is optional and deferred to V2+.

---

## 0. Research Questions (defined with Prof. Makihara)

* **RQ1 (V0):** How do programmers' error patterns differ across combinations of problem difficulty and user proficiency level? → Reproducing Shimizu *et al.* (2025) proficiency classification as a baseline, then granular analysis by error type (CE, WA, TLE, RE).

* **RQ2 (V1):** Do codes that are close to each other in vector space share similar error patterns? → Code embeddings (CodeBERT / GraphCodeBERT) + topological visualization (t-SNE / UMAP).

* **RQ3 (V2 — optional):** Does the information visualized by the proposed tool contribute to error prediction? → Predictive model distinguishing predictable from unpredictable errors.

---

## 1. Preamble: Development Environment

To guarantee reproducibility, isolation from other laboratory projects, and efficiency when dealing with a massive volume of data (over 12 million rows), the following technical stack has been strictly selected:

### 1.1. IDE (Working Environment)
* **Visual Studio Code (VS Code)**: Chosen as the primary editor for its lightweight nature, robust extension ecosystem, and native integration of **Jupyter Notebooks** (`.ipynb`). Notebooks will be used exclusively for the R&D exploration phase and rapid prototyping before refactoring into pure Python scripts (`.py`).

### 1.2. Application Framework (Project Skeleton)
* **Streamlit**: This open-source framework is used as the core structure for the web application. It allows converting data analysis scripts into interactive and dynamic web applications natively in Python, eliminating the need to develop a complex front-end layer (HTML/CSS/JavaScript).
* **PyTorch / PyTorch Geometric (Planned for V1/V2)**: Used as the Deep Learning framework for tensor manipulation, vector embeddings, and neural network training.

### 1.3. Ecosystem Libraries (Toolbox)
* **Polars / Pandas**: *Pandas* will be used for rapid prototyping. However, given the ~14 million submissions in CodeNet, **Polars** will be prioritized in production for its highly optimized performance (multi-threaded execution in Rust), allowing the manipulation of gigabytes of data in memory without saturation.
* **Scikit-Learn**: The standard library for classical Machine Learning. It will be the pillar of V0 for implementing dimensionality reduction and data clustering algorithms.
* **Matplotlib & Seaborn**: Used for generating advanced static scientific visualizations (logarithmic boxplots, correlation matrices, distribution charts), which will then be dynamically integrated into the Streamlit interface.

### 1.4. Tooling & Project Management
* **Conda**: Mandatory use of a virtual environment manager to isolate project dependencies and avoid any conflicts on the laboratory server.
* **Git & GitHub**: Source code versioning and feature tracking through the `OJS-submission-analytics` repository.
* **GitHub Copilot**: Used as an AI programming assistant to accelerate the writing of boilerplate code (especially the configuration of Streamlit layouts and complex Matplotlib parameters).

---

## 2. Iterative Project Architecture (Roadmap)

The project is divided into three incremental phases (V0, V1, V2) to secure functional and scientifically validated deliverables before increasing the algorithmic complexity.

```text
[ Raw CSV Data ] ➔  V0: Classical ML & Metadata   ➔ V0 Dashboard (Deliverable 1)
                                   ↓
[ Raw Source Code ] ➔ V1: Embeddings & Code NLP   ➔ V1 Dashboard (Deliverable 2)
                                   ↓
[ Code Graphs / DL ] ➔ V2: Advanced Deep Learning ➔ Final Predictive Model
```

### 2.1. Version 0: MVP — Descriptive Statistics & Classical Machine Learning
* **Data Scope**: Exclusive analysis of **tabular metadata** (`problem_list.csv` for structural metrics and `pXXXXX.csv` files containing submission history: anonymized user ID, timestamp, language used, judge verdict, CPU time, memory consumed). Raw source code is not read at this stage.
* **Scientific Objectives**: 
    1.  Dynamically and interactively replicate the conclusions of the laboratory's research paper (*An Empirical Study of the Error Characteristics in an Online Judge System*).
    2.  Empirically verify specific phenomena, such as the prevalence and resolution difficulty of *Time Limit Exceeded* (TLE) errors, or the inversion of compilation error (CE) vs. logical error (TLE) proportions based on problem difficulty.
* **Applied Machine Learning Algorithms**:
    * **G1–G6 Classification (Shimizu *et al.* 2025)**: Each user is classified by the maximum difficulty letter they successfully solved (Accepted) in AtCoder Beginner Contests. Letters A–F are assigned by ordinal position within each contest in `problem_list.csv`. This is the primary classification method.
    * **K-Means + PCA (exploratory)**: Used in the initial exploration phase to validate the existence of natural behavioral clusters. Results archived in `playground/00_exploration_score_based_DEPRECATED.ipynb`.
* **Interface Deliverable**: A functional local Streamlit dashboard allowing users to filter error distributions and resolution times by difficulty level (A to F) and proficiency group (G1–G6), on AtCoder data only.

### 2.2. Version 1: Textual Analysis & Introduction to Code Embeddings
* **Data Scope**: Extension of the analysis to the **raw source code files** (`.cpp`, `.py`, `.java`, `.c`) contained in the CodeNet archive.
* **Scientific Objectives**: Go beyond the simple observation of the error (the "WA" or "TLE" verdict) to analyze *what the student actually wrote*. The goal is to map the semantic similarity of the codes submitted for the same problem.
* **Fundamental Techniques**:
    * **Tokenization & Vector Representation**: Transforming computer text into numerical structures intelligible to an AI.
    * **Code Embeddings**: Learning and utilizing dense vector spaces. Exploring and manipulating language models specialized for code (e.g., *CodeBERT*, *GraphCodeBERT*, or IBM tokenizers) to transform an entire script into a numerical vector.
    * **Topological Visualization**: Using non-linear dimensionality reduction techniques (like **t-SNE** or **UMAP**) to display "point clouds" representing students' codes on the dashboard. Two codes applying the same algorithmic logic will be geographically close in the cloud, making it possible to visualize correction patterns or recurring errors.

### 2.3. Version 2: Advanced Deep Learning & Predictive Models
* **Data Scope**: Full integration of IBM CodeNet's advanced data structures (control flow graphs, abstract syntax trees - AST).
* **Scientific Objectives**: Build architectures capable of anticipating the learner's behavior or the program's performance even before its submission to the OJS.
* **Applied Deep Learning Techniques**:
    * **Supervised Neural Networks (Fine Classification)**: Training a multi-layer (MLP) or recurrent network on the embeddings generated in V1 to predict in advance the exact type of error (Classification among CE, WA, RE, TLE) a student will trigger, based on their historical profile and current code.
    * **Learning Trajectory Modeling**: Attempting to predict user "dropout" on the platform by analyzing the stagnation of their code vectors over time.
    * **GNN Exploration (Optional)**: Studying IBM's implementation based on *Graph Neural Networks* (GNN) leveraging the *Message Passing* and *Virtual Nodes* mechanism to analyze the source code's AST, paving the way for automatic bug localization support systems.

---

## 3. Timeline & Projected Schedule (17 weeks)

The following schedule aligns with the standard duration of the research internship (mid-May to mid-September) and is strictly structured to mark milestones and track progression.

* **Phase 1: Setup & Data Engineering (Weeks 1 — 3)**
  * *Main Tasks:* Uncompress and deploy the CodeNet archive (7.8GB compressed) on the laboratory server. Extract metadata and create a data sampling pipeline (using *Polars*). Initialize the Git repository and configure Conda environments.
  * *Deliverables:* Operational computing environment. Data cleaning scripts validated on a 100,000-row sample.

* **Phase 2: V0 Development (Weeks 4 — 7)**
  * *Main Tasks:* Reproduce Shimizu *et al.* (2025) G1–G6 proficiency classification. Analyze error distributions (CE, WA, TLE, RE) by difficulty level and user group (RQ1). Code the Streamlit layout and dynamic filters.
  * *Deliverables:* **V0 Deliverable**: Functional dashboard displaying descriptive statistics and the first user profile clusters. Report validating the conclusions of the initial research paper.

* **Phase 3: V1 Research (Weeks 8 — 11)**
  * *Main Tasks:* Extract raw source code files corresponding to major failures. Get familiar with code NLP models (*CodeBERT*) and extract the first *Embeddings*. Map the code vectors visually using t-SNE.
  * *Deliverables:* **V1 Deliverable**: Dashboard update adding a "Semantic Source Code Analysis" tab. Graphical visualization of the code point clouds.

* **Phase 4: V2 Modeling (Weeks 12 — 14)**
  * *Main Tasks:* Design the neural network architecture using *PyTorch*. Train and optimize hyperparameters for verdict prediction. Optional integration of comparative data from *LeetCode*.
  * *Deliverables:* **V2 Deliverable**: Trained predictive model. Functional "AI Error Prediction" tab integrated into the Streamlit interface.

* **Phase 5: Finalization & Defense (Weeks 15 — 17)**
  * *Main Tasks:* Intensive testing phase, code refactoring, and source code documentation. Final drafting of the research internship report in English/French. Preparation of visual materials for the final defense presentation in front of the professors.
  * *Deliverables:* **Final Research Report**. Fully documented source code on GitHub. Completed presentation slide deck.