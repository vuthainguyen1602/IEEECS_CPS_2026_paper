# IEEECS CPS 2026 — Hybrid Ensemble IDS for Edge Computing

Research repository for a two-stage Network Intrusion Detection System (IDS) designed for edge and IoT deployments. The approach combines a high-accuracy teacher ensemble (XGBoost, LightGBM, Random Forest with SMOTEENN) with knowledge distillation into a lightweight decision-tree student model, evaluated on the [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset.

## Highlights

- **Teacher model**: Stacking ensemble with SMOTEENN for handling extreme class imbalance
- **Student model**: Knowledge distillation into a compact decision tree for low-latency edge inference
- **Results**: Teacher accuracy >99.8%; distilled student retains >99% accuracy with significantly lower inference cost
- **Reproducible pipeline**: End-to-end scripts from data preparation through LaTeX table population

## Repository Structure

```
IEEECS_CPS_2026/
├── VNICT2026_Template_LaTeX.tex   # VNICT 2026 conference manuscript (LaTeX)
├── sections/                      # Shared LaTeX sections (abstract, methodology, results, ...)
├── references.bib
│
└── paper_1/                       # Full experimental implementation
    ├── main.tex                   # IEEE journal-style manuscript
    ├── code/                      # Python training & evaluation scripts
    │   ├── data_preparation.py
    │   ├── hybrid_ids_cicids2017.py
    │   ├── knowledge_distillation.py
    │   ├── populate_latex_tables.py
    │   ├── replot_figures.py
    │   ├── config.py
    │   └── shared_utils.py
    ├── results/                   # Metrics, tables, and cached outputs
    ├── Run_on_Colab.ipynb         # Google Colab notebook
    ├── requirements.txt
    └── REPRODUCTION.md            # Step-by-step reproduction guide
```

## Quick Start

### 1. Install dependencies

```bash
cd paper_1
pip install -r requirements.txt
```

### 2. Prepare the dataset

Download the CICIDS2017 CSV files and place them in `paper_1/data/`, then run:

```bash
python code/data_preparation.py
```

### 3. Train and evaluate

```bash
# Train the teacher ensemble
python code/hybrid_ids_cicids2017.py

# Distill to the student model
python code/knowledge_distillation.py

# Regenerate figures and update LaTeX tables
python code/replot_figures.py
python code/populate_latex_tables.py
```

For the full pipeline (including manuscript compilation), see [`paper_1/REPRODUCTION.md`](paper_1/REPRODUCTION.md).

### Google Colab

An interactive notebook is available at [`paper_1/Run_on_Colab.ipynb`](paper_1/Run_on_Colab.ipynb).

## Build the Manuscript

**VNICT 2026 conference paper** (root):

```bash
xelatex VNICT2026_Template_LaTeX.tex
bibtex VNICT2026_Template_LaTeX
xelatex VNICT2026_Template_LaTeX.tex
xelatex VNICT2026_Template_LaTeX.tex
```

> Requires XeLaTeX with Vietnamese language support (`fontspec`, `polyglossia`).

**IEEE journal paper** (`paper_1/`):

```bash
cd paper_1
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Requirements

| Component | Details |
|-----------|---------|
| Python | 3.8+ |
| Key packages | `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`, `matplotlib`, `seaborn`, `pyarrow`, `joblib` |
| Dataset | [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (not included; place in `paper_1/data/`) |
| LaTeX | TeX Live or MiKTeX with `IEEEtran` class |

## Author

**Thái Nguyễn Vũ**  
Bộ môn Công nghệ Thông tin, Cơ sở tại TP. Hồ Chí Minh  
Trường Đại học Giao thông Vận tải  
nvthai@utc2.edu.vn

## License

This repository is provided for academic and research purposes. Please cite appropriately if you use this work.
