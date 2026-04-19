# Semester Project Progress Document  
**RT-FSAS (Real-Time Football Strategy Analysis System)**  
**Course:** Machine Learning — Mid-term / Progress Assignment

---

## 1. Problem Definition

### 1.1 Problem Statement

Professional and semi-professional football analysis increasingly relies on **event data** (passes, carries, pressures, shots, etc.) rather than only broadcast video. Coaches and analysts must interpret **high-dimensional, relational game states** under time pressure. Our project addresses the following problem:

> **How can we turn a stream of in-match events into a compact tactical representation, retrieve historically similar situations, and surface actionable coaching insight in (near) real time?**

Concretely, we aim to build a **multimodal decision-support system** that:

1. **Encodes** the current tactical snapshot (player layout and ball context) as a fixed-size vector suitable for similarity search.  
2. **Retrieves** structurally similar past moments from a large corpus of matches.  
3. **Scores** how typical or suboptimal a chosen action is relative to historical patterns.  
4. **Explains** findings in natural language via a **large language model (LLM)** so non-technical stakeholders can use the output.

### 1.2 Problem Scope

**In scope for the semester submission**

- **Data:** StatsBomb **open** event data (no proprietary tracking); we use **La Liga 2015/16** as the primary corpus.  
- **Representation:** Each eligible event with a spatial location is converted into a **PyTorch Geometric** graph (23 nodes: 22 role-based player estimates + 1 ball node; distance-threshold edges with normalized distance as edge attributes).  
- **Learning:** A **graph neural network (GNN)** encoder maps each graph to a **128-dimensional** embedding; supervised training uses **next-event-type prediction** as a proxy task to learn tactically meaningful structure.  
- **Retrieval (planned):** **FAISS** index over embeddings of historical graphs for *k*-nearest-neighbor retrieval.  
- **Interface (planned):** **Dash** dashboard and **Google Gemini**-based coaching narrative conditioned on retrieval + quality signal.

**Out of scope / limitations (explicit)**

- We do **not** use StatsBomb 360 / player-tracking freeze frames in the current pipeline; off-ball positions are **estimated** from formation templates and ball location, which bounds spatial fidelity.  
- We do **not** claim causal optimality of actions; the **Q-delta** scorer and LLM advice are **heuristic** and evaluation-limited to the open dataset and our ablation design.

### 1.3 Expected Outcomes (Final Submission)

| Outcome | Description |
|--------|-------------|
| **O1** | Reproducible pipeline from raw StatsBomb JSON → cached graph tensors (`.pt`). |
| **O2** | Trained **TacticalGNN** checkpoint and documented training curves (loss / accuracy). |
| **O3** | **FAISS** index and a small **retrieval API** (`TacticalRetriever.retrieve`) callable from the app pipeline. |
| **O4** | **End-to-end pipeline** stub → integrated pipeline: graph → embed → retrieve → score → Gemini advice. |
| **O5** | **Dash** demo with at least pitch visualization, retrieval table, and advice panel; **ablation** notes and a short results write-up for the report. |

---

## 2. Research Gap

### 2.1 What Existing Work Typically Does

Sports analytics products and papers often emphasize:

- **Aggregate statistics** (xG, pass completion maps) without a unified **relational** model of *who* is near *whom* at each instant.  
- **Sequence models** (RNNs, Transformers) on event *lists*, which treat the pitch as a bag of events and may underuse **explicit spatial structure** and **team geometry**.  
- **LLM-only** chat interfaces that lack **grounded retrieval** from the same distribution as the match being analyzed, leading to plausible but **unverifiable** tactical claims.

### 2.2 The Gap

There is room for a system that **combines**:

1. **Structured geometric encoding** (graphs) of a tactical snapshot,  
2. **Fast similarity search** over a large historical library of those snapshots, and  
3. **Controlled natural-language explanation** where the LLM is constrained by **retrieved evidence** and a **scalar quality signal** (our planned Q-scorer).

### 2.3 Why the Gap Matters

- **Coaching relevance:** Similarity in **embedding space** (if trained well) can surface “what usually happens next” in comparable shapes, which supports preparation and in-game pattern recognition.  
- **Scientific rigor:** Coupling **retrieval** with **explicit metrics** (e.g., retrieval hit rate, next-event accuracy, ablation vs. a naive baseline) makes claims easier to **evaluate** than pure LLM outputs.  
- **Engineering practicality:** Open data removes licensing barriers for a semester-scale project while still exercising full **ML + IR + LLM** integration.

---

## 3. Proposed Approach and Implementation

### 3.1 Methodology Overview

Our methodology is a **retrieve-and-rag** style architecture specialized for football:

1. **Graph construction** — For each StatsBomb event with a `location`, we build a graph: nodes carry normalized pitch coordinates, team indicator, and coarse role one-hots; edges connect nodes within a spatial **radius**; edge features encode **normalized distance** [1], [2].  
2. **GNN encoder** — A **three-layer GATv2** stack [2] with **global mean pooling** maps variable internal node states to a **128-D** graph embedding (`TacticalGNN.encode`). Edge attributes are included (`edge_dim = 1`) so attention can condition on proximity.  
3. **Supervised pretext task** — Labels come from the **next event’s** StatsBomb `type.name`, mapped to 11 classes (10 frequent types + “other”). This **next-event prediction** task encourages the encoder to capture dynamics, not only static layout.  
4. **Training loop** — Mini-batch training with **AdamW**, **cross-entropy** loss, train/validation split, **early stopping**, and checkpoints (`gnn_best.pt`) plus serialized history (`gnn_history.pt`) for plotting.  
5. **Planned retrieval** — Encode all (or a large subset of) historical graphs and store vectors in **FAISS** [4] for sub-millisecond *k*-NN at inference.  
6. **Planned coaching layer** — **Gemini** generates advice from structured inputs: current state summary, **q_delta**, and top-*k* retrieved metadata (interface contracts in `setup/interface_contracts.md`).

### 3.2 Dataset Currently in Use

| Aspect | Detail |
|--------|--------|
| **Source** | StatsBomb Open Data [3] |
| **Competition / season** | **La Liga, 2015/16** (competition id path `matches/11/27.json` in the open-data layout) |
| **Modality** | Event JSON (passes, carries, pressures, etc.); **no** 360 tracking in our build |
| **Artifacts produced** | `graphs/la_liga_2015_16_train.pt` (subset of matches for faster GNN training), `graphs/la_liga_2015_16_full.pt` (all processed graphs for index building) |
| **Approximate scale** | Tens of thousands of graphs per full pass over all matches with locations (exact counts depend on run; e.g. on the order of **~50k+** train+val graphs when using the scripted train split) |

### 3.3 Implementation Status (Concrete Artifacts)

Implemented or in progress in the repository:

- **`src/data/graph_builder.py`** — Event → PyG `Data` (`x`, `edge_index`, `edge_attr`, `y`).  
- **`src/data/dataset.py`** — Batch build and `torch.save` of train/full lists.  
- **`src/models/gnn_encoder.py`** — `TacticalGNN` with `encode()` and classifier head for training.  
- **`src/training/train_gnn.py`** — Training loop, validation metrics, checkpointing, history.  
- **`notebooks/02_graph_construction.ipynb`** — Visual sanity check of node placement on the pitch.  
- **`notebooks/03_gnn_training.ipynb`** — Loads `gnn_history.pt` and plots **loss** and **accuracy** curves.

### 3.4 Results Obtained So Far

- **Pipeline validity:** Graph tensors build successfully from real La Liga JSON; training script runs on CPU/GPU with batched PyG `DataLoader`.  
- **Training stability:** An initial integration issue (`GATv2Conv` + `edge_attr` without `edge_dim`) was resolved by setting **`edge_dim=1`**, after which forward passes and optimization proceed normally.  
- **Quantitative metrics:** Final numbers (best validation accuracy, convergence epoch) should be taken from your local **`checkpoints/gnn_history.pt`** and the training log after a full run; the notebook plots provide the **evidence** required for the course (monotonic improvement in loss / sensible accuracy vs. random baseline on 11 classes ≈ 9.1%).

*Recommendation for the submitted PDF:* paste **one figure** from `03_gnn_training.ipynb` (loss + accuracy) and **one sentence** with best val accuracy and epoch.

### 3.5 Way Forward to Final Submission

| Step | Action | Purpose |
|------|--------|---------|
| 1 | **Freeze** hyperparameters (lr, batch size, patience) and run training to completion | Reproducible best checkpoint |
| 2 | Implement **`src/retrieval/build_index.py`** | Encode `la_liga_2015_16_full.pt` → FAISS index |
| 3 | Implement **`src/retrieval/retriever.py`** | Clean API for Track B |
| 4 | Implement **`src/models/q_scorer.py`** | Scalar **q_delta** for LLM conditioning |
| 5 | Wire **`src/llm/pipeline.py`** + **`gemini_coach.py`** | End-to-end demo |
| 6 | **`dashboard/app.py`** + **`src/evaluation/ablation.py`** | Usable UI + comparative metrics |
| 7 | **Optional data expansion** | Second competition (e.g., another league/season) if time permits — see Section 4 |

**Methodological improvements (if schedule allows)**

- Richer **node features** (actual lineups when available, minute, score difference).  
- **Temporal context** (short event history as additional graph or sequence module).  
- **Calibration** of retrieval (evaluate whether neighbors share similar *next-event* distributions).

---

## 4. Preliminary Roadmap

### 4.1 Datasets for the Final Draft

| # | Dataset | Role in final draft | Status |
|---|---------|------------------------|--------|
| **D1** | StatsBomb **La Liga 2015/16** (open) | **Primary** — training encoder, building FAISS, main evaluation | **In use** |
| **D2** | *(Optional)* Second open competition (e.g., another StatsBomb competition folder) | **Generalization** — small transfer or retrieval sanity check | **Planned / TBD** |

For the **minimum viable final submission**, **one** primary open dataset (D1) is sufficient; a **second** dataset (D2) strengthens the “robustness” story but is not strictly required if time is constrained.

### 4.2 Estimated Completion (Percentage)

This is a **honest, milestone-based** estimate aligned with `PROJECT_TASKS.md`:

| Area | Weight | Done | Notes |
|------|--------|------|--------|
| Environment, data ingest, graph build, dataset export | 25% | **~100%** | Scripts + notebook demo |
| GNN model + training + training plots | 25% | **~85%** | Code done; finalize best metrics & write-up |
| FAISS index + retriever API | 15% | **~0%** | Next major Track A block |
| Q-scorer + LLM + pipeline + dashboard + ablation | 35% | **~0–15%** | Mostly Track B / integration |

**Overall approximate completion:** **~45–50%** toward a fully integrated semester deliverable (as of this document).  
If only Track A through training is considered, that slice is **~70–75%** complete.

### 4.3 Timeline for Remaining Work (Indicative)

Adjust dates to your actual **final submission** deadline; below assumes **~4–5 weeks** remain.

| Week | Focus | Deliverable |
|------|--------|-------------|
| **W1** | Train GNN to completion; export curves; document best checkpoint | `gnn_best.pt`, figures for report |
| **W2** | `build_index.py` + FAISS persistence; `retriever.py` | Working *k*-NN from live embedding |
| **W3** | `q_scorer.py`; stub→real integration in `pipeline.py` | Scalar signal + pipeline call graph |
| **W4** | Gemini coach + Dash UI | Demo video / screenshots |
| **W5** | Ablation script, polish, report | `evaluation_results.md` + final PDF |

### 4.4 References (IEEE Style)

[1] P. W. Battaglia et al., “Relational inductive biases, deep learning, and graph networks,” *arXiv preprint* arXiv:1806.01261, 2018.  

[2] S. Brody, U. Alon, and E. Yahav, “How attentive are graph attention networks?” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.  

[3] StatsBomb, “StatsBomb open data,” GitHub repository. [Online]. Available: https://github.com/statsbomb/open-data  

[4] J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with GPUs,” *IEEE Trans. Big Data*, vol. 7, no. 3, pp. 535–547, Jul. 2021.  

[5] P. Veličković et al., “Graph attention networks,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.  

[6] T. Brown et al., “Language models are few-shot learners,” in *Adv. Neural Inf. Process. Syst.*, vol. 33, 2020, pp. 1877–1901. *(Context: LLM conditioning / RAG-style use — cite if you describe Gemini in detail.)*  

[7] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in *Adv. Neural Inf. Process. Syst.*, vol. 33, 2020, pp. 9459–9474.  

---

*Document generated to match the current RT-FSAS repository layout and task plan. Update Section 3.4 with your exact validation accuracy and loss figures after your latest training run.*
