"""
Configuration Module for Hybrid Ensemble IDS Experiments.

Centralizes all paths, hyperparameters, and experimental settings
to ensure reproducibility and maintainability.

References:
    - Nguyen et al., "Hybrid Ensemble Machine Learning for Network
      Intrusion Detection in IoT Environments," 2026.
"""

import os
import random
import numpy as np

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.environ.get("CICIDS2017_RAW_DIR", os.path.join(DATA_DIR, "raw"))
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test_data.parquet")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")

LATEX_DIR = os.path.join(PROJECT_ROOT, "sections")

# Reproducibility
RANDOM_SEED = 42

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Experimental Flags
DEBUG_MODE = False            # If True, sample 10% of data for fast verification
SAMPLE_FRACTION = 1.0         # Fraction of training data to use (1.0 = full)

# Cross-Validation Settings
KFOLD_K = 5                   # Number of folds for Stratified K-Fold CV
CV_SAMPLE_FRACTION = 0.2      # Fraction of fold training data to use
CV_USE_FAST_SAMPLER = True    # If True, use RandomUnderSampler instead of SMOTEENN

# Model Hyperparameters

# XGBoost
XGB_PARAMS = dict(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    tree_method='hist', n_jobs=-1, use_label_encoder=False,
    eval_metric='logloss', random_state=RANDOM_SEED, verbosity=1
)

# LightGBM
LGBM_PARAMS = dict(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    n_jobs=-1, random_state=RANDOM_SEED, verbose=1
)

# Random Forest
RF_PARAMS = dict(
    n_estimators=200, max_depth=20, min_samples_split=5,
    min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_SEED, verbose=0
)

# Stacking Meta-Learner
STACKING_CV = 10               # Internal CV folds for StackingClassifier

# Knowledge Distillation — Student Model
STUDENT_MAX_DEPTH = 12
STUDENT_MIN_SAMPLES_SPLIT = 10

# Visualization
PLOT_DPI = 300
PLOT_STYLE = {
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
}
