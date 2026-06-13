# Reproduction Guide: Hybrid Ensemble IDS for CICIDS2017

This guide outlines the step-by-step execution sequence to reproduce the experimental results and automatically update the LaTeX manuscript.

## 0. Prerequisites
Place the CICIDS2017 CSV files in `data/raw/` (or set `CICIDS2017_RAW_DIR` to another folder).
Install dependencies:
```bash
pip install -r requirements.txt
```

## 1. Data Preparation
Merge raw CSV files, clean records, and split into train/test Parquet files:
```bash
python code/data_preparation.py
```

Optional:
```bash
python code/data_preparation.py --input-dir /path/to/cicids2017/csvs
```

## 2. Model Training (Teacher)
Train the Hybrid Ensemble (Stacking), perform ablation studies, and run K-Fold Cross-Validation:
```bash
python code/hybrid_ids_cicids2017.py
```
*   **Outputs**: `results/all_results.json`, `results/models/ensemble_model.joblib`.

## 3. Knowledge Distillation (Student)
Compress the massive ensemble into a lightweight Student model (Decision Tree) for edge deployment:
```bash
python code/knowledge_distillation.py
```
*   **Outputs**: `results/kd_results.json`, `results/models/student_model.joblib`.

## 4. Visualization & Documentation
Generate finalized figures and inject metrics directly into the LaTeX tables:
```bash
# Update Confusion Matrices and Feature Importance plots
python code/replot_figures.py

# Inject all metrics into sections/results.tex
python code/populate_latex_tables.py
```

## 5. Compile Manuscript
Generate the final PDF:
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---
**Core Files in `code/`**:
- `data_preparation.py`: Merge and clean raw CICIDS2017 CSV files.
- `data_utils.py`: Pandas preprocessing helpers for data preparation.
- `hybrid_ids_cicids2017.py`: The main research engine.
- `knowledge_distillation.py`: Implementation of model compression.
- `populate_latex_tables.py`: Automated documentation bridge.

## Google Colab

1. Upload the repo to Google Drive.
2. Place CICIDS2017 CSV files in `paper_1/data/raw/` or a separate Drive folder.
3. Open `Run_on_Colab.ipynb` in Colab, mount Drive, and run each step.

Example when CSV files are stored at `MyDrive/ids-2017/`:

```bash
python code/data_preparation.py --input-dir "/content/drive/MyDrive/ids-2017"
```

Output Parquet files are saved under `paper_1/data/` on Drive and are used by the training scripts in the following steps.
