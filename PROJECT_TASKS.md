# RT-FSAS Project Plan & Task Breakdown
**Two-Member Parallel Workflow**

This document serves as the master checklist for the RT-FSAS ML project. The work is split evenly into two independent tracks so that both members can work efficiently in parallel with zero blockers.

---

## 🤝 Phase 1: Shared Setup (Completed ✅)
*Both members should have identical environments.*
- [x] Create project folder structure
- [x] Set up Python virtual environment (`rt-fsas-env`) and install `requirements.txt`
- [x] Verify API packages (`statsbombpy`, PyTorch, PyG, Gemini, FAISS, etc.)
- [x] Explore `StatsBomb` open data and decide to use positional fallback since `three-sixty` freeze-frames are missing.
- [x] Define specific Interface Contracts (see `setup/interface_contracts.md`) so Track A and Track B know how to connect.

---

## 🅰️ Track A: Core ML & Data Infrastructure
*Assigned to: Member A*  
*Role: You build the neural networks, parse the football data, create the graph representation, train the models, and build the FAISS vector retrieval system. You hand over the ML artifacts to Member B.*

### Phase 2A: Graph Construction
- [x] **2A.1 Graph Builder (`src/data/graph_builder.py`)**: Convert a StatsBomb JSON event into a PyTorch Geometric 23-node mathematical graph. *(DONE!)*
- [ ] **2A.2 Dataset Script (`src/data/dataset.py`)**: Extract thousands of passes/shots from the 38 La Liga matches and run them through the GraphBuilder to save bulk `.pt` (PyTorch) files to disk.
- [ ] **2A.3 Visualization Demo (`notebooks/02_graph_construction.ipynb`)**: Prove visually that the nodes map accurately on a plotted pitch.

### Phase 3A: GNN Encoder Training
- [ ] **3A.1 GNN Architecture (`src/models/gnn_encoder.py`)**: Build a 3-layer `GATv2Conv` network that squashes a 23-node graph into a single 128-dimensional latent vector (tactical summary).
- [ ] **3A.2 Training Loop (`src/training/train_gnn.py`)**: Train the network to predict the next event type to force it to learn tactical patterns.
- [ ] **3A.3 Training Notebook (`notebooks/03_gnn_training.ipynb`)**: Plot Loss/Accuracy curves to prove the model is actually learning and not randomly guessing.

### Phase 4A: FAISS Vector Memory
- [ ] **4A.1 Build Index (`src/retrieval/build_index.py`)**: Pass 50,000+ historical graphs through your trained GNN to get 50,000 vectors, and slide them into a FAISS nearest-neighbor index.
- [ ] **4A.2 Query API (`src/retrieval/retriever.py`)**: Build the `TacticalRetriever.retrieve()` function that Member B will call to ask "What happened historically in situations similar to this one?"

### Phase 5A: Decision Quality Scorer
- [ ] **5A.1 Q-Scorer (`src/models/q_scorer.py`)**: A fast 3-Layer MLP that takes a tactical vector + an action and outputs a `q_delta` (how bad/suboptimal an observed pass or shot was compared to the "best" average historical decision).

---

## 🅱️ Track B: LLM Engineering & App End-User Pipeline
*Assigned to: Member B*  
*Role: You own the Generative AI (Google Gemini), the prompting logic, the web interface UI, and the final stitching together of all code. You don't need to wait for Member A to finish; you just use "fake" outputs mimicking their functions to build your system early.*

### Phase 5B: Gemini AI RAG Coach
- [ ] **5B.1 LLM Wrapper (`src/llm/gemini_coach.py`)**: Create the class that calls `google-generativeai`. Write the huge, complex master prompt that accepts the `q_delta` and `retrieved_situations` to output "Coaching Advice".
- [ ] **5B.2 Prompt Engineering**: Actively tweak your prompt using "fake data" arrays to ensure Gemini stops formatting like a robot and starts talking like an actual 1st-Team EPL manager analyzing live play.

### Phase 5C: The Master Pipeline
- [ ] **5C.1 Pipeline (`src/llm/pipeline.py`)**: Write `RTFSASPipeline`. This is the brain! It takes a generic live game -> calls Member A's Graph builder -> calls Member A's FAISS Retriever -> calls Member A's Q-Scorer -> hands it all off to your `GeminiCoach`.
- [ ] **5C.2 Pipeline Integration**: Once Member A finishes Phase 2/3/4/5A, remove your fake simulated code and plug exactly into their `.pt` payload outputs.

### Phase 6B: Dash Web App & Ablation Evaluator
- [ ] **6B.1 Dashboard UI (`dashboard/app.py`)**: Build a 4-pane visual analytics frontend using the Plotly Dash framework (A Live Pitch Map, A Win-Probability Line Chart, Historical Retrievals Table, and Streaming Gemini Advice box).
- [ ] **6B.2 Validation System (`src/evaluation/ablation.py`)**: Run 50 test games on a full pipeline variant vs a dumb pipeline variant, comparing metrics like "Retrieval Accuracy" and "Prompt Specificity Length" so we have hard science data for our presentation report.
- [ ] **6B.3 Write Report (`report/evaluation_results.md`)**: Synthesize the hard numbers and graph outputs into the final submission file.

---

## 🚨 Final Integration Playbook
On **Day X**, when Member A finishes training their models:
1. Member A ensures `checkpoints/gnn_best.pt` and `index/tactical.index` are pushed.
2. Member B updates `pipeline.py` to point directly to Member A's trained weights rather than stub-code.
3. You run `python dashboard/app.py` and boom—you have a fully multimodal Football AI!
