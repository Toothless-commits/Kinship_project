# 🧬 Deep Kinship AI: Tri-Subject Kinship Verification & Trait Attribution

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FaceNet](https://img.shields.io/badge/Backbone-InceptionResnetV1-blue?style=for-the-badge)](https://github.com/timesler/facenet-pytorch)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end deep learning framework for **Tri-Subject Kinship Verification** $(Father, Mother, Child)$, **1-to-$N$ Biological Parent Retrieval**, and **Landmark-Guided Facial Phenotype Attribution**.

Unlike conventional pairwise kinship verification ($F\text{-}C$ or $M\text{-}C$) that ignores the joint contribution of both parents, this system evaluates the full nuclear triad using **InceptionResnetV1 (VGGFace2)**, **Order-Invariant Symmetric & Cosine Genetic Fusion (646 dims)**, **Test-Time Augmentation (TTA)**, and a **live Parent Gallery Manager** with multi-photo parent prototypes.

---

## 🌟 Key Features

- **Symmetric Tri-Subject Modeling**: Evaluates $(Father, Mother, Child)$ simultaneously using parent-order-invariant Mendelian ($\min/\max$) and mid-parent hypersphere fusion. Swapping Father and Mother inputs yields the exact same prediction.
- **Compact Anti-Overfitting Architecture**: Combines 6 parameter-free 512-d cosine similarity features with a shared 128-d `LayerNorm` + `GELU` bottleneck (646 fused dimensions total) for strong generalization on family-disjoint splits.
- **Open-Wedge Triad Mining**: Extracts authentic $(Parent_1, Parent_2, Child)$ open wedges from FIW relationship graphs instead of noisy 3-cliques (such as sibling triangles).
- **Test-Time Augmentation (TTA)**: Averages $L_2$-normalized embeddings across original and horizontally flipped face crops during verification, evaluation, and gallery indexing.
- **Landmark-Guided Phenotype Attribution**: Uses MTCNN 5-point facial landmarks to dynamically isolate and compare:
  - 👁️ **Eyes** (ocular midpoint & inter-ocular span)
  - 👃 **Nose** (nasal tip & bridge)
  - 👄 **Mouth & Lips** (oral commissures)
  - 💇 **Hair & Forehead** (supra-orbital band)
  - ✨ **Skin Complexion** (cheek patches in perceptual $\text{YCbCr}$ space)
- **Live Parent Gallery Manager & 1-to-$N$ Search**: Enroll custom parent couples directly in the browser with **1–3 photos per parent** (averaged into normalized identity prototypes), filter searches between **Custom Gallery** and **FIW Dataset**, and rank candidates in a single vectorized GPU pass.

---

## 📐 Architecture & Mathematical Formulation

~~~text
Father Image (160x160) ──► [InceptionResnetV1 + TTA] ──► f_F (512-d) ──┐
                                                                       ├──► 6 Cosine Scalars (512-d) ──────────┐
Mother Image (160x160) ──► [InceptionResnetV1 + TTA] ──► f_M (512-d) ──┼──► Shared Bottleneck (512 -> 128-d)   ├──► [Concat: 646-d]
                                                                       │       └─► Symmetric Fusion (640-d) ───┘           │
Child Image  (160x160) ──► [InceptionResnetV1 + TTA] ──► f_C (512-d) ──┘                                                   ▼
                                                                                                                  [Classifier MLP]
                                                                                                                    646 -> 64 -> 1
                                                                                                                           │
                                                                                                                           ▼
                                                                                                               Kinship Probability [0, 1]
~~~

### 1. Parameter-Free 512-d Cosine Geometry

Let $f_F, f_M, f_C \in \mathbb{R}^{512}$ denote $L_2$-normalized TTA face embeddings and $\mu_{512} = \text{normalize}(f_F + f_M)$ be the unit-sphere mid-parent vector. Six scalar features capture direct angular kinship:

- $\max(\cos_{FC}, \cos_{MC})$ and $\min(\cos_{FC}, \cos_{MC})$
- $\cos(\mu_{512}, f_C)$ and $\cos(f_F, f_M)$
- $\frac{\cos_{FC} + \cos_{MC}}{2}$ and $\vert{}\cos_{FC} - \cos_{MC}\vert{}$

### 2. Symmetric 128-d Relational Fusion

Each embedding passes through a shared projection $\phi: \mathbb{R}^{512} \to \mathbb{R}^{128}$ (`Linear -> LayerNorm -> GELU -> Dropout(0.35)` + $L_2$ normalization) to yield $p_F, p_M, p_C$ and $\mu_{128} = \text{normalize}(p_F + p_M)$:

$$\text{Diff}_{\min} = \min(\vert{}p_F - p_C\vert{}, \vert{}p_M - p_C\vert{}), \quad \text{Diff}_{\max} = \max(\vert{}p_F - p_C\vert{}, \vert{}p_M - p_C\vert{})$$

$$\text{Prod}_{\max} = \max(p_F \odot p_C, p_M \odot p_C), \quad \text{Prod}_{\text{avg}} = \frac{1}{2}(p_F \odot p_C + p_M \odot p_C)$$

$$\text{Diff}_{\text{mid}} = \vert{}\mu_{128} - p_C\vert{}$$

Concatenating the 6 cosine scalars with these 5 symmetric 128-d vectors produces $\mathbf{z} \in \mathbb{R}^{646}$, passed through a regularized MLP (`646 -> 64 -> 1` with `LayerNorm`, `GELU`, and `Dropout(0.4)`).

### 3. Phenotype Attribution Scoring

For structural traits (Eyes, Nose, Mouth, Hair), normalized embeddings $e_C, e_F, e_M$ of isolated crops are compared via temperature-scaled cosine similarity ($\tau = 0.08$):

$$P(\text{Father}) = \frac{\exp(\cos(e_C, e_F) / \tau)}{\exp(\cos(e_C, e_F) / \tau) + \exp(\cos(e_C, e_M) / \tau)}, \quad P(\text{Mother}) = 1.0 - P(\text{Father})$$

For **Skin Complexion**, Euclidean distance $d_F, d_M$ in perceptual $\text{YCbCr}$ color space is computed on cheek patches:

$$P_{\text{skin}}(\text{Father}) = \frac{d_F^{-1}}{d_F^{-1} + d_M^{-1}}, \quad P_{\text{skin}}(\text{Mother}) = 1.0 - P_{\text{skin}}(\text{Father})$$

---

## 📊 Benchmark Results (Family-Disjoint FIW Test Split)

Evaluated with Test-Time Augmentation (TTA) on completely unseen families:

| Benchmark | Metric | Score |
| :--- | :--- | :--- |
| **Validation** | Best Validation ROC-AUC | **0.8744** |
| **Binary Verification (1-to-1)** | Test Accuracy | **67.97%** |
| **Binary Verification (1-to-1)** | Test ROC-AUC | **0.7046** |
| **Binary Verification (1-to-1)** | Kin Precision / F1-Score | **0.73 / 0.64** |
| **Parent Retrieval (1-to-24)** | Top-1 Retrieval Accuracy | **77.78% (14/18)** |
| **Parent Retrieval (1-to-24)** | Mean Reciprocal Rank (MRR) | **0.7884** |

---

## 📁 Repository Structure

~~~text
Kinship_project/
├── app.py                      # 3-tab Streamlit web application
├── src/
│   ├── __init__.py             # Package marker
│   ├── model.py                # TriSubjectKinshipNet (symmetric 646-d fusion head)
│   ├── dataset.py              # KinshipDataset, O(1) hard negative sampling & family split
│   ├── train.py                # Two-stage AMP training pipeline (head warmup + top-block tuning)
│   ├── evaluate.py             # TTA binary verification & 1-to-N Rank-K benchmarks
│   ├── search_engine.py        # Vectorized TTA 1-to-N search & custom parent gallery manager
│   ├── trait_analyzer.py       # Landmark-guided anatomical crop & trait attribution
│   ├── align_faces.py          # MTCNN 160x160 face alignment and caching utility
│   └── build_fiw_triplets.py   # Open-wedge triad extractor from FIW relationship graphs
├── .gitignore                  # Excludes venv/, data/, weights/, and cache artifacts
├── .python-version             # Python version specification (3.11.9)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
~~~

*(Local folders `data/`, `weights/`, and `venv/` are intentionally excluded via `.gitignore`.)*

---

## 🚀 Quick Start

### 1. Environment Setup

Python **3.11** (e.g., `3.11.9` via `pyenv`) with an NVIDIA CUDA GPU (recommended) or CPU:

~~~bash
# Clone the repository
git clone https://github.com/Toothless-commits/Kinship_project.git
cd Kinship_project

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Bash / Zsh
# source venv/bin/activate.fish # Fish shell
# venv\Scripts\activate         # Windows

# Install PyTorch (CUDA 12.1) and dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
~~~

### 2. Prepare Local Directories

Create the untracked directories and place your trained checkpoint at `weights/best_model.pt` (or train one using the pipeline below):

~~~bash
mkdir -p data/raw/FIW data/aligned data/custom_gallery weights
~~~

### 3. Launch the Streamlit Web Application

Disable the file watcher on Linux to avoid `inotify` watch-limit errors on large galleries:

~~~bash
streamlit run app.py --server.fileWatcherType none
~~~

Open `http://localhost:8501` in your browser:

- **⚖️ Tab 1 — 1-to-1 Triplet Verification & Traits**: Upload Father, Mother, and Child photos to get TTA-boosted kinship confidence and side-by-side landmark trait inheritance percentages.
- **🔍 Tab 2 — 1-to-N Parent Search**: Upload a child photo, choose **Custom Gallery Only**, **All Galleries**, or **FIW Dataset Only**, and rank the most probable biological parent couples.
- **🖼️ Tab 3 — Parent Gallery Manager**: Enroll custom parent couples by uploading **1–3 photos per parent**. Faces are automatically MTCNN-aligned, saved in `data/custom_gallery/`, averaged into TTA prototypes, and indexed live in VRAM.

---

## 🔄 End-to-End Training & Evaluation Pipeline

Download the **Kaggle FIW (`recognizing-faces-in-the-wild`)** dataset (`train_relationships.csv` and `train-faces/`) into `data/raw/FIW/`, then run:

### Step 1: Extract Open-Wedge Triads

~~~bash
python src/build_fiw_triplets.py \
  --raw_dir data/raw/FIW \
  --out_csv data/triplets.csv \
  --max_combos 25
~~~

### Step 2: Align Faces with MTCNN

~~~bash
python src/align_faces.py \
  --csv data/triplets.csv \
  --out_dir data/aligned \
  --img_size 160
~~~

### Step 3: Two-Stage Mixed-Precision Training

~~~bash
python src/train.py \
  --epochs_stage1 12 \
  --epochs_stage2 10 \
  --batch_size 32 \
  --save_dir weights
~~~

- **Stage 1 (Frozen Backbone)**: Trains the `proj` bottleneck and `classifier` head with `AdamW` ($\text{lr} = 10^{-3}$, weight decay $10^{-2}$) and cosine annealing.
- **Stage 2 (Selective Fine-Tuning)**: Unfreezes only `block8`, `last_linear`, and `last_bn` ($\text{lr} = 10^{-5}$) alongside the head ($\text{lr} = 10^{-4}$) with AMP mixed precision and gradient clipping (`max_norm = 1.0`).
- **Negative Mining**: Strict 80/10/10 family-disjoint split with 1:1 positive/negative balance (`50%` wrong couple, `25%` wrong father, `25%` wrong mother).

### Step 4: Evaluate Benchmarks

~~~bash
python src/evaluate.py \
  --weights weights/best_model.pt \
  --csv data/triplets.csv \
  --aligned data/aligned \
  --batch_size 32
~~~

### CLI 1-to-N Parent Search

~~~bash
python src/search_engine.py --child path/to/child.jpg --top_k 5 --source All
~~~

---

## 🛠️ Tech Stack

- **Deep Learning**: [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/index.html)
- **Face Backbone & Detection**: [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (`InceptionResnetV1`, `MTCNN`)
- **Data & Metrics**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/), [Pillow](https://python-pillow.org/), [tqdm](https://github.com/tqdm/tqdm)
- **Interface**: [Streamlit](https://streamlit.io/)

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- [Families in the Wild (FIW)](https://web.northeastern.edu/smilelab/fiw/) for large-scale facial kinship benchmarks.
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) by Tim Esler for PyTorch `InceptionResnetV1` and `MTCNN`.
```[cite: 6, 7]
