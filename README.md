# 🧬 Deep Kinship AI: Tri-Subject Kinship Verification & Trait Attribution

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FaceNet](https://img.shields.io/badge/Backbone-InceptionResnetV1-blue?style=for-the-badge)](https://github.com/timesler/facenet-pytorch)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end deep learning framework for **Tri-Subject Kinship Recognition** $(Father, Mother, Child)$ and **Facial Phenotype Attribution**. 

Unlike conventional pairwise kinship verification ($F-C$ or $M-C$) which overlooks the joint genetic contribution of both parents, this system models the full biological triad using **InceptionResnetV1 (VGGFace2)**, **Bilinear Relational Genetic Fusion (2560 dims)**, **MTCNN landmark-guided facial trait decomposition**, and a **vectorized 1-to-$N$ gallery search engine**.

---

## 🌟 Key Features

- **Tri-Subject Joint Modeling**: Evaluates the complete $(Father, Mother, Child)$ triad simultaneously, modeling how combined parental features synthesize in the offspring.
- **Bilinear Genetic Feature Fusion**: Synthesizes 512-dimensional $L_2$-normalized deep facial embeddings through absolute differences, Hadamard products, and mid-parent target deviations into a rich 2560-dimensional representation.
- **Landmark-Guided Phenotype Attribution**: Leverages MTCNN 5-point facial landmarks to dynamically isolate:
  - 👁️ **Eyes** (ocular midpoint & inter-ocular span)
  - 👃 **Nose** (nasal tip & bridge)
  - 👄 **Mouth & Lips** (oral commissures)
  - 💇 **Hair & Forehead** (supra-orbital band)
  - ✨ **Skin Complexion** (malar/cheek tone in perceptual YCbCr space)
- **1-to-$N$ Parent Retrieval Gallery**: Fast, vectorized cosine-and-MLP ranking to retrieve and rank candidate biological parent couples for a queried child image from a gallery.
- **Strict Family-Level Splitting & Hard Negative Mining**: Zero identity/family leakage across train, validation, and test splits, combined with multi-type negative mining (`wrong_father`, `wrong_mother`, `wrong_both`).
- **Interactive Streamlit Web Dashboard**: Dual-mode graphical interface for 1-to-1 Triplet Verification with visual trait inheritance breakdown and 1-to-$N$ Parent Search.

---

## 📐 Mathematical Formulation & Architecture

```
Father Image (160x160) ──► [InceptionResnetV1] ──► feat_F (512-d) ──┐
                                                                     │
Mother Image (160x160) ──► [InceptionResnetV1] ──► feat_M (512-d) ──┼──► [Bilinear Genetic Fusion]
                                                                     │          (2560-d)
Child Image  (160x160) ──► [InceptionResnetV1] ──► feat_C (512-d) ──┘               │
                                                                                    ▼
                                                                           [Classifier MLP]
                                                                        2560 -> 512 -> 128 -> 1
                                                                                    │
                                                                                    ▼
                                                                        Kinship Probability [0, 1]
```

### 1. Bilinear Relational Genetic Fusion

Let $f_F, f_M, f_C \in \mathbb{R}^{512}$ denote $L_2$-normalized face embeddings extracted from the backbone for Father, Mother, and Child respectively. The genetic fusion layer computes:

$$\text{Diff}_{FC} = |f_F - f_C| \in \mathbb{R}^{512}$$

$$\text{Diff}_{MC} = |f_M - f_C| \in \mathbb{R}^{512}$$

$$\text{Prod}_{FC} = f_F \odot f_C \in \mathbb{R}^{512}$$

$$\text{Prod}_{MC} = f_M \odot f_C \in \mathbb{R}^{512}$$

$$\mu_{parent} = \frac{f_F + f_M}{2}, \quad \text{Diff}_{midC} = |\mu_{parent} - f_C| \in \mathbb{R}^{512}$$

The fused vector $\mathbf{z} \in \mathbb{R}^{2560}$ concatenates these components:

$$\mathbf{z} = \Big[ \text{Diff}_{FC} \,\|\, \text{Diff}_{MC} \,\|\, \text{Prod}_{FC} \,\|\, \text{Prod}_{MC} \,\|\, \text{Diff}_{midC} \Big]$$

### 2. Multi-Layer Classification Head

$$\mathbf{h}_1 = \text{Dropout}_{0.4}\Big(\text{ReLU}\big(\text{BatchNorm1d}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1)\big)\Big), \quad \mathbf{W}_1 \in \mathbb{R}^{512 \times 2560}$$

$$\mathbf{h}_2 = \text{Dropout}_{0.3}\Big(\text{ReLU}\big(\text{BatchNorm1d}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)\big)\Big), \quad \mathbf{W}_2 \in \mathbb{R}^{128 \times 512}$$

$$\hat{y} = \sigma(\mathbf{W}_3 \mathbf{h}_2 + b_3), \quad \mathbf{W}_3 \in \mathbb{R}^{1 \times 128}$$

### 3. Phenotype Attribution Scoring

For structural traits (Eyes, Nose, Mouth, Hair), normalized embeddings $e_C, e_F, e_M$ of isolated crops are compared via cosine similarity scaled by temperature $\tau = 0.08$:

$$P(\text{Father}) = \frac{\exp(\cos(e_C, e_F) / \tau)}{\exp(\cos(e_C, e_F) / \tau) + \exp(\cos(e_C, e_M) / \tau)}$$

$$P(\text{Mother}) = 1.0 - P(\text{Father})$$

For **Skin Complexion**, Euclidean distance $d_F, d_M$ in perceptual $\text{YCbCr}$ color space is computed on cheek patches:

$$P_{skin}(\text{Father}) = \frac{d_F^{-1}}{d_F^{-1} + d_M^{-1}}, \quad P_{skin}(\text{Mother}) = 1.0 - P_{skin}(\text{Father})$$

---

## 📁 Repository Structure

```
kinship_project/
├── app.py                      # Streamlit interactive web application
├── test_photos/                # Sample test images (child.jpg, father.jpg, mother.jpg)
├── weights/                    # Checkpoints directory
│   ├── best_model.pt           # Best model checkpoint (highest val AUC/Acc)
│   └── latest_model.pt         # Most recent training epoch checkpoint
├── data/
│   ├── raw/                    # Raw dataset directory (e.g., FIW)
│   ├── aligned/                # MTCNN-aligned 160x160 face crops by Family/MID
│   └── triplets.csv            # Extracted biological triads (father, mother, child)
├── src/
│   ├── __init__.py             # Package marker
│   ├── model.py                # TriSubjectKinshipNet with Bilinear Genetic Fusion
│   ├── dataset.py              # KinshipDataset, negative sampling & family-level split
│   ├── train.py                # Two-stage training pipeline (warmup + fine-tuning)
│   ├── evaluate.py             # Binary verification & 1-to-N Rank-K benchmarks
│   ├── search_engine.py        # Vectorized 1-to-N gallery parent candidate search
│   ├── trait_analyzer.py       # Landmark-guided anatomical crop & trait attribution
│   ├── align_faces.py          # MTCNN face alignment and caching utility
│   └── build_fiw_triplets.py   # Triad generator from FIW relationship graphs
├── .python-version             # Python version specification (3.11.9)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

Python **3.11+** with CUDA-enabled GPU (recommended) or CPU:

```bash
# Clone the repository
git clone https://github.com/Toothless-commits/Kinship_project.git
cd Kinship_project

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser:
- **Tab 1: 1-to-1 Triplet Verification & Traits**:
  Upload photos for Father, Mother, and Child (or test with images in `test_photos/`). The system outputs kinship confidence and displays side-by-side cropped landmarks with quantitative inheritance percentages.
- **Tab 2: 1-to-N Parent Search**:
  Upload an image of a child. The search engine scans the cached parental gallery and ranks candidate biological parents with confidence bars and previews.

---

## 💻 Command Line Usage

### 1-to-N Parent Candidate Search
Search and rank the most probable biological parents for any query child image:

```bash
python src/search_engine.py --child test_photos/child.jpg --top_k 5
```

### Evaluate Model Performance
Run binary verification benchmarks (Accuracy, ROC-AUC, Precision, Recall) and Rank-$K$ retrieval (Top-1, Top-5, MRR):

```bash
python src/evaluate.py \
  --weights weights/best_model.pt \
  --csv data/triplets.csv \
  --aligned data/aligned \
  --batch_size 16
```

---

## 🔄 End-to-End Pipeline

### Step 1: Triplet Mining from Families in the Wild (FIW)
Extract valid $(Father, Mother, Child)$ 3-cliques from FIW relationship metadata:

```bash
python src/build_fiw_triplets.py \
  --raw_dir data/raw/FIW \
  --out_csv data/triplets.csv
```

### Step 2: MTCNN Face Alignment & Normalization
Crop and align all unique family member faces into standardized $160 \times 160$ px images:

```bash
python src/align_faces.py \
  --csv data/triplets.csv \
  --out_dir data/aligned \
  --img_size 160
```

### Step 3: Two-Stage Model Training
Train the network using a progressive fine-tuning strategy:

```bash
python src/train.py \
  --epochs_stage1 10 \
  --epochs_stage2 15 \
  --batch_size 16 \
  --save_dir weights
```

- **Stage 1 (Classifier Warmup)**: InceptionResnetV1 backbone is frozen; only the 2560-d MLP classifier head is trained with AdamW ($\text{lr} = 10^{-3}$).
- **Stage 2 (End-to-End Fine-Tuning)**: Backbone is unfrozen with differential learning rates:
  - Backbone: $\text{lr} = 10^{-5}$ (preserves facial geometry)
  - Classifier: $\text{lr} = 10^{-4}$
  - Cosine annealing scheduler smoothly decays learning rates across epochs.

---

## 📊 Dataset & Negative Mining Strategy

To ensure genuine generalization, data partitioning occurs strictly at the **Family ID** level:
- **80% Train / 10% Validation / 10% Test**: No family seen during training ever appears in validation or testing.
- **Balanced Positive/Negative Pairs ($1:1$)**: For every true biological triad, an un-related negative triad is dynamically constructed:
  - `wrong_father`: Correct mother and child, impostor father selected from another family.
  - `wrong_mother`: Correct father and child, impostor mother selected from another family.
  - `wrong_both`: Both parents replaced with candidates from an unrelated family.

---

## 🛠️ Tech Stack & Dependencies

- **Deep Learning Core**: [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/index.html)
- **Face Recognition Backbone**: [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (`InceptionResnetV1`, `MTCNN`)
- **Data & Evaluation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/), [Pillow](https://python-pillow.org/), [tqdm](https://github.com/tqdm/tqdm)
- **Web App**: [Streamlit](https://streamlit.io/)

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- [Families in the Wild (FIW)](https://web.northeastern.edu/smilelab/fiw/) for large-scale facial kinship benchmarks.
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) by Tim Esler for PyTorch InceptionResnetV1 and MTCNN implementations.
