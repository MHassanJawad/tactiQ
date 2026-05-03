# RT-FSAS: Real-Time Football Strategy Analysis System  
**A Multimodal Graph Retrieval and LLM-Assisted Coaching Framework**

**Authors:** Hassan Jawad, [Teammate Name]  
**Course:** Machine Learning  
**Institution:** NUST  
**Date:** [Insert Submission Date]

---

## Abstract
This project presents RT-FSAS, a real-time football strategy analysis system that combines graph neural networks (GNNs), vector retrieval, and large language models (LLMs) to generate tactical coaching guidance from event-stream data. Using StatsBomb Open Data (La Liga 2015/16), we convert match events into graph snapshots, train a GATv2-based encoder to learn tactical embeddings, retrieve historically similar situations through FAISS, and generate coaching advice conditioned on a decision-quality signal (`q_delta`) and retrieved context. The encoder is trained via next-event prediction and achieved validation accuracy of approximately 28-29% across 11 classes, significantly above chance (~9.1%). Retrieval ablation on 50 sampled cases produced high internal consistency (Top-1 match rate: 0.78, Top-k contain rate: 1.0), while qualitative advice generation was operational end-to-end. We discuss architectural choices, constraints, sources of error, and future improvements including leakage-aware evaluation, stronger Q-scorer training, and dashboard deployment.

**Keywords:** Football analytics, graph neural networks, retrieval-augmented generation, FAISS, tactical decision support, multimodal AI

---

## I. Introduction
Modern football analysis increasingly relies on event data (passes, carries, duels, pressures, shots) rather than raw video alone. However, tactical interpretation in real-time remains difficult due to high-dimensional spatial interactions and limited decision time during play. This project addresses the question:

> How can we encode live tactical structure, retrieve historically similar situations, and generate actionable coaching advice in near real time?

RT-FSAS is designed as a practical end-to-end pipeline:
1. Convert event snapshots into graph structures;
2. Learn a compact tactical embedding space via GNN;
3. Retrieve similar historical contexts from FAISS;
4. Score action quality (suboptimality signal);
5. Generate grounded coaching advice with Gemini.

The goal is not only model performance but also a complete deployable analytical workflow suitable for presentation/demo.

---

## II. Problem Statement and Motivation
Most lightweight semester projects either:
- use static tabular statistics (limited tactical context), or
- use pure LLM narrative (fluent but weakly grounded).

Our motivation is to bridge these extremes by combining:
- **structured geometric modeling** (graph-based state representation),
- **fast retrieval grounding** (nearest tactical precedents), and
- **human-readable explanation** (LLM advice).

This is particularly relevant for coaching support where explainability and evidence grounding are as important as raw prediction accuracy.

---

## III. Dataset and Preprocessing
### A. Data Source
- **StatsBomb Open Data** (public)
- Primary corpus: **La Liga 2015/16**

### B. Event-to-Graph Representation
Each eligible event is transformed into a PyTorch Geometric graph:
- 23 nodes total:
  - 22 role-estimated players (11 per side),
  - 1 ball node.
- Node features include normalized coordinates, team indicator, and coarse role flags.
- Edges are created using distance thresholding; edge attribute is normalized distance.

Due to absence of full tracking/freeze-frame data in the selected source usage, off-ball positions are approximated from formation-role priors and ball context.

### C. Labels
Supervised target = **next event type**, mapped to 11 classes (10 common + Other).

---

## IV. System Architecture
### A. Graph Encoder (Track A)
- Model: `TacticalGNN` (3-layer `GATv2Conv`)
- Output embedding size: 128
- Training objective: next-event classification

### B. Retrieval Memory (Track A)
- Historical graph embeddings are indexed using **FAISS IndexFlatIP**
- L2-normalized embeddings support cosine-similarity-like retrieval

### C. Decision Quality Scorer (Track A)
- `QScorer`: 3-layer MLP over tactical embedding
- Outputs per-action values and computes:
  - `q_delta = Q(actual_action) - max(Q)`
  - More negative => stronger suboptimality signal

### D. LLM Coaching Layer (Track B)
- `GeminiCoach` consumes:
  - current game state,
  - `q_delta`,
  - top-k retrieved situations
- Produces tactical coaching guidance
- Includes fallback deterministic advice when API key/model call unavailable

### E. Master Pipeline (Track B)
`RTFSASPipeline` executes:
`live_event -> graph -> embedding -> retrieval -> q_delta -> advice`

---

## V. Implementation Summary (What Was Built)
### Completed Components
- `src/data/graph_builder.py`
- `src/data/dataset.py`
- `src/models/gnn_encoder.py`
- `src/training/train_gnn.py` (extended with class weights, scheduler, label smoothing)
- `src/retrieval/build_index.py`
- `src/retrieval/retriever.py`
- `src/models/q_scorer.py`
- `src/llm/gemini_coach.py`
- `src/llm/pipeline.py`
- `src/evaluation/ablation.py`
- `report/evaluation_results.md`

### Not Yet Fully Finalized
- production dashboard UI (`dashboard/app.py`)
- fully trained/calibrated QScorer on football-specific reward objective

---

## VI. Experimental Setup
### A. GNN Training
- Samples:
  - Train: 46,020
  - Validation: 5,113
- Objective: 11-class next-event prediction
- Representative best run:
  - Best epoch (by val loss): 8
  - Train loss: 1.7345
  - Val loss: 1.7500
  - Train acc: 0.2888
  - Val acc: 0.2855

### B. Retrieval and Ablation
Using `report/ablation_metrics.json`:
- Samples evaluated: 50
- Top-1 next-event match rate: 0.78
- Top-k contains-event rate: 1.0
- Avg Top-1 similarity: ~1.0
- Advice metrics:
  - Full avg words: 59.36
  - Dumb baseline avg words: 59
  - Full specificity: 6.1
  - Dumb specificity: 6.0

---

## VII. Results and Discussion
### A. What Worked Well
1. **End-to-end integration is functional**: data -> model -> retrieval -> advice.
2. **GNN performance above chance**: 28-29% vs random ~9.1% (11 classes).
3. **Fast retrieval grounding** via FAISS works with trained embedding space.
4. **Pipeline robustness**: fallback advice path prevents hard failure when LLM credentials are missing.

### B. Why Accuracy Is Not Very High
The modest classification accuracy is expected due to:
1. **Hard prediction task**: next event from one snapshot is inherently uncertain.
2. **Imperfect spatial fidelity**: estimated off-ball positions (no full tracking).
3. **Class imbalance** in event types.
4. **Limited context window**: current formulation mostly uses a single state rather than temporal sequence.
5. **Model capacity vs compute constraints**: CPU-friendly training settings constrain exploration.

Despite this, results are sufficient for meaningful retrieval-based tactical prototyping.

### C. Important Evaluation Caveat
Retrieval metrics are likely optimistic because evaluation samples and indexed corpus are from the same distribution and may contain near-duplicates/self-neighbors. Future holdout protocols are required for stronger claims.

---

## VIII. Figure Placeholders (What to Add)
Use these placeholders directly in your final IEEE document (Word/LaTeX/PDF):

**[Figure 1 Placeholder: System Architecture Diagram]**  
Suggested content: block diagram showing Graph Builder -> TacticalGNN -> FAISS Retriever -> QScorer -> GeminiCoach -> Dashboard.

**[Figure 2 Placeholder: Graph Construction Visualization]**  
Suggested content: screenshot/plot from `notebooks/02_graph_construction.ipynb` showing 23-node pitch mapping.

**[Figure 3 Placeholder: Training Curves]**  
Suggested content: train/val loss and train/val accuracy plot from `checkpoints/gnn_history_*.pt` or `notebooks/03_gnn_training.ipynb`.

**[Figure 4 Placeholder: Retrieval Example Table]**  
Suggested content: top-5 retrieved situations for one query (minute, event type, next event, similarity).

**[Figure 5 Placeholder: Pipeline Output Example]**  
Suggested content: one full pipeline sample showing input event + q_delta + generated advice text.

**[Figure 6 Placeholder: Ablation Metrics Chart]**  
Suggested content: bar chart comparing full vs dumb baseline specificity/word metrics; include retrieval rates.

---

## IX. Conclusion
RT-FSAS demonstrates a practical multimodal football analytics pipeline that integrates graph representation learning, retrieval grounding, and natural-language tactical guidance. While classification accuracy remains moderate, the model learns structure above chance and supports working retrieval-driven analysis. The current system is suitable as a strong semester project prototype with clear expansion paths for production quality.

---

## X. Future Work
1. **Leakage-aware retrieval evaluation**
   - Exclude self-match and same-match near duplicates
   - Use match-wise holdout splits
2. **Train QScorer with better supervision**
   - reward proxies (expected possession value, xThreat/xG-like targets)
3. **Temporal modeling**
   - add short event history (sequence module + graph)
4. **Feature enrichment**
   - scoreline, game state, lineup-specific features, pressure zones
5. **Dashboard finalization**
   - real-time visual frontend for coaching staff workflow
6. **Cross-competition robustness**
   - evaluate transfer/generalization beyond one season

---

## XI. Reproducibility Checklist
- [x] Training script and checkpoints
- [x] FAISS index builder and retriever
- [x] End-to-end pipeline module
- [x] Ablation script + JSON output
- [ ] Dashboard production polish
- [ ] Final PDF in strict IEEE formatting

---

## XII. Deliverables Mapping
1. **Final Report (IEEE format)**  
   - Use this file as technical content draft and transfer to IEEE template.
2. **Plagiarism + AI report**  
   - Attach institution-required similarity report and AI-usage declaration.
3. **Final Presentation**  
   - Build slides from Sections IV, VI, VII, VIII, X.
4. **Code (GitHub link)**  
   - Include repository URL and commit hash used for results.

---

## References (IEEE Style, editable)
[1] P. W. Battaglia *et al.*, “Relational inductive biases, deep learning, and graph networks,” *arXiv preprint* arXiv:1806.01261, 2018.  
[2] S. Brody, U. Alon, and E. Yahav, “How attentive are graph attention networks?” in *Proc. ICLR*, 2022.  
[3] StatsBomb, “StatsBomb Open Data,” GitHub repository. [Online]. Available: https://github.com/statsbomb/open-data  
[4] J. Johnson, M. Douze, and H. Jegou, “Billion-scale similarity search with GPUs,” *IEEE Trans. Big Data*, vol. 7, no. 3, pp. 535-547, 2021.  
[5] P. Lewis *et al.*, “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020.

