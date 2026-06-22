"""
Hybrid Ensemble Machine Learning for Network Intrusion Detection.

This module implements the full experimental pipeline described in the paper:
  - Data loading and preprocessing (Section III-A)
  - Class balancing via SMOTEENN (Section III-B)
  - Hybrid Stacking Ensemble (XGBoost + LightGBM + RF) (Section III-C)
  - Baseline comparison, ablation study, and K-Fold CV (Section IV)

Usage:
    python hybrid_ids_cicids2017.py             # Run full experiment
    python hybrid_ids_cicids2017.py --replot <path>  # Regenerate plots
"""

import os
import gc
import json
import copy
import time
import argparse
import warnings
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.combine import SMOTEENN

from config import (
    RANDOM_SEED, set_global_seed,
    DEBUG_MODE, SAMPLE_FRACTION,
    KFOLD_K, CV_SAMPLE_FRACTION, CV_USE_FAST_SAMPLER,
    FIGURES_DIR as OUTPUT_DIR, RESULTS_DIR, MODELS_DIR,
    TRAIN_PATH, TEST_PATH,
    XGB_PARAMS, LGBM_PARAMS, RF_PARAMS, STACKING_CV,
    PLOT_DPI,
)
from populate_latex_tables import populate_latex

warnings.filterwarnings('ignore')
set_global_seed(RANDOM_SEED)

def log_message(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def load_and_preprocess(train_filepath, test_filepath):
    """Load CICIDS2017 dataset and apply feature scaling.

    Corresponds to Section III-A of the paper. Loads pre-split
    Parquet files, extracts numeric features, imputes NaN with 0,
    and applies RobustScaler.

    Parameters
    ----------
    train_filepath : str
        Path to the training Parquet file.
    test_filepath : str
        Path to the test Parquet file.

    Returns
    -------
    X_train_scaled : ndarray
        Scaled training features.
    X_test_scaled : ndarray
        Scaled test features.
    y_train : Series
        Training labels.
    y_test : Series
        Test labels.
    num_features : int
        Number of features.
    feature_names : list[str]
        Ordered list of feature column names.
    """
    log_message("Loading Train and Test Parquet files...")
    train_df = pd.read_parquet(train_filepath)
    test_df = pd.read_parquet(test_filepath)

    if DEBUG_MODE:
        log_message("DEBUG_MODE enabled: Sampling 10% of data for fast verification.")
        train_df = train_df.sample(frac=0.1, random_state=RANDOM_SEED)
        test_df = test_df.sample(frac=0.1, random_state=RANDOM_SEED)
    elif SAMPLE_FRACTION < 1.0:
        log_message(f"Sampling {SAMPLE_FRACTION*100}% of training data for faster experimentation.")
        train_df = train_df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED)

    log_message(f"Train Dataset Shape: {train_df.shape}")
    log_message(f"Test Dataset Shape: {test_df.shape}")

    target_col = 'label_binary'
    exclude_cols = ["label", "label_binary", "source_ip", "destination_ip",
                    "flow_id", "timestamp", "protocol"]
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    log_message(f"Extracting {len(feature_cols)} numeric features...")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Report class balance so the "imbalance" claim is grounded in the
    # actual binary distribution (BENIGN vs ATTACK), not a multiclass figure.
    train_counts = y_train.value_counts()
    minority_label = train_counts.idxmin()
    minority_ratio = train_counts.min() / train_counts.sum() * 100
    log_message(f"Binary class distribution (train):\n{train_counts}")
    log_message(f"Minority class = {minority_label} "
                f"({minority_ratio:.4f}% of training samples)")

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    del train_df, test_df
    gc.collect()

    log_message("Scaling Features (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.shape[1], feature_cols


def balance_training_data(X_train, y_train, sample_frac=1.0):
    """Apply SMOTEENN resampling with disk caching.

    Corresponds to Section III-B. Uses SMOTE for oversampling the
    minority class followed by Edited Nearest Neighbours (ENN) for
    cleaning noisy samples near class boundaries.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.
    y_train : Series or ndarray
        Training labels.
    sample_frac : float, optional
        Fraction identifier for cache file naming (default 1.0).

    Returns
    -------
    X_resampled : ndarray
        Balanced feature matrix.
    y_resampled : ndarray
        Balanced labels.
    """
    from config import CACHE_DIR
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"balanced_train_frac_{sample_frac}.parquet")

    if os.path.exists(cache_file):
        log_message(f"Loading balanced training data from cache: {cache_file}")
        cached_df = pd.read_parquet(cache_file)
        X_resampled = cached_df.drop(columns=['target_label']).values
        y_resampled = cached_df['target_label'].values
        log_message(f"Resampled Training Class Distribution (CACHED):\n{pd.Series(y_resampled).value_counts()}")
        return X_resampled, y_resampled

    log_message("Original Training Class Distribution:")
    log_message(f"\n{y_train.value_counts()}")
    
    log_message("Step 1/2: Pre-downsampling majority class using RandomUnderSampler...")
    # Downsample majority class to be equal to minority class to drastically speed up SMOTEENN
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=RANDOM_SEED)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    
    log_message(f"Distribution after RandomUnderSampler:\n{pd.Series(y_rus).value_counts()}")
    
    log_message("Step 2/2: Applying ENN to clean decision boundaries (Parallel Mode: n_jobs=-1)...")
    log_message("Note: This will now run much faster due to the reduced dataset size.")

    start_time = time.time()
    enn = EditedNearestNeighbours(n_jobs=-1)
    X_resampled, y_resampled = enn.fit_resample(X_rus, y_rus)

    elapsed = time.time() - start_time
    log_message(f"ENN Completed in {elapsed:.2f} seconds.")
    log_message(f"Resampled Training Class Distribution:\n{pd.Series(y_resampled).value_counts()}")

    log_message(f"Saving resampled data to cache: {cache_file}")
    resampled_df = pd.DataFrame(X_resampled)
    resampled_df['target_label'] = y_resampled
    resampled_df.to_parquet(cache_file, index=False)

    gc.collect()
    return X_resampled, y_resampled


def build_base_models(input_dim=None):
    """Initialize the three base classifiers for the ensemble.

    Corresponds to Section III-C. All hyperparameters are loaded
    from ``config.py`` for reproducibility.

    Parameters
    ----------
    input_dim : int, optional
        Number of input features (unused, kept for API compatibility).

    Returns
    -------
    list[tuple[str, estimator]]
        Named estimators for StackingClassifier.
    """
    log_message("Defining Base Models (XGBoost + LightGBM + Random Forest)...")

    xgb_clf = XGBClassifier(**XGB_PARAMS)
    lgb_clf = LGBMClassifier(**LGBM_PARAMS)
    rf_clf = RandomForestClassifier(**RF_PARAMS)

    return [('xgb', xgb_clf), ('lgbm', lgb_clf), ('rf', rf_clf)]


def build_ensemble(base_models):
    """Construct the Hybrid Stacking Ensemble.

    Corresponds to Section III-C. Uses LogisticRegression as the
    meta-learner with ``class_weight='balanced'`` and probability-based
    stacking.

    Parameters
    ----------
    base_models : list[tuple[str, estimator]]
        Named base estimators from ``build_base_models()``.

    Returns
    -------
    StackingClassifier
        Configured stacking ensemble.
    """
    log_message("Constructing Hybrid Ensemble (Stacking Classifier)...")
    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(class_weight='balanced', n_jobs=-1),
        cv=STACKING_CV, n_jobs=-1, verbose=1,
        stack_method='predict_proba'
    )
    return ensemble


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all evaluation metrics and return as dict.

    Because CICIDS2017 (binary) is highly imbalanced, we report both the
    positive-class (ATTACK) metrics AND macro-averaged metrics, which give
    equal weight to the BENIGN and ATTACK classes and are far more
    informative under imbalance than weighted/accuracy figures.
    Per-class Precision/Recall/F1 are also returned for full transparency.
    """
    acc = accuracy_score(y_true, y_pred)
    # Positive-class (ATTACK = 1) metrics — primary detection metrics
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Macro-averaged metrics — treat both classes equally
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class breakdown (0 = BENIGN, 1 = ATTACK)
    prec_pc = precision_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    rec_pc = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'FPR': fpr,
        'Precision-Macro': prec_macro, 'Recall-Macro': rec_macro,
        'F1-Macro': f1_macro,
        'Precision-BENIGN': float(prec_pc[0]), 'Recall-BENIGN': float(rec_pc[0]),
        'F1-BENIGN': float(f1_pc[0]),
        'Precision-ATTACK': float(prec_pc[1]), 'Recall-ATTACK': float(rec_pc[1]),
        'F1-ATTACK': float(f1_pc[1]),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
    }

    if y_prob is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
        metrics['AUC-PR'] = average_precision_score(y_true, y_prob)

    return metrics

def print_metrics(name, metrics):
    """Pretty-print metrics for a model."""
    print("-" * 30)
    print(f"  {name}")
    print("-" * 30)
    for key in ['Accuracy', 'Precision', 'Recall', 'F1-Score',
                'F1-Macro', 'Precision-Macro', 'Recall-Macro',
                'FPR', 'AUC-ROC', 'AUC-PR']:
        if key in metrics:
            fmt = '.6f' if key == 'FPR' else '.4f'
            print(f"  {key:15s}: {metrics[key]:{fmt}}")
    if 'F1-BENIGN' in metrics:
        print(f"  Per-class F1   : BENIGN={metrics['F1-BENIGN']:.4f}, "
              f"ATTACK={metrics['F1-ATTACK']:.4f}")
    print(f"  Confusion Matrix: TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, TP={metrics['TP']}")
    print("-" * 30)


def evaluate_baselines(base_models, X_train, y_train, X_test, y_test):
    """
    Train and evaluate each base model individually.
    Returns a dict of {model_name: metrics}.
    """
    log_message("BASELINE COMPARISON: Evaluating individual models...")

    results = {}
    trained_models = {}

    for name, model in base_models:
        log_message(f"Evaluating baseline: {name}...")
        start = time.time()
        model_clone = copy.deepcopy(model)
        model_clone.fit(X_train, y_train)
        elapsed = time.time() - start
        log_message(f"  {name} trained in {elapsed:.2f}s")

        y_pred = model_clone.predict(X_test)
        try:
            y_prob = model_clone.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['Train Time (s)'] = round(elapsed, 2)
        results[name] = metrics
        trained_models[name] = model_clone
        print_metrics(f"Baseline: {name}", metrics)

    return results, trained_models


def plot_roc_and_pr_curves(all_results, X_test, y_test, trained_models, ensemble_model):
    """Generate ROC and PR curves with both full-view and zoomed-in panels."""
    log_message("Plotting ROC and Precision-Recall Curves...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax_roc_full, ax_pr_full = axes[0]
    ax_roc_zoom, ax_pr_zoom = axes[1]

    models_to_plot = {**trained_models, 'Proposed Hybrid': ensemble_model}
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    
    curve_data = {}
    for i, (name, model) in enumerate(models_to_plot.items()):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            log_message(f"Could not get probabilities for {name}: {e}")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap_val = average_precision_score(y_test, y_prob)
        
        color = colors[i % len(colors)]
        lw = 3 if 'Hybrid' in name else 1.8
        ls = '-' if 'Hybrid' in name else '--'
        alpha = 1.0 if 'Hybrid' in name else 0.85
        
        curve_data[name] = {
            'fpr': fpr, 'tpr': tpr, 'auc': auc_val,
            'prec': prec, 'rec': rec, 'ap': ap_val,
            'color': color, 'lw': lw, 'ls': ls, 'alpha': alpha
        }

    for name, d in curve_data.items():
        roc_label = f"{name} (AUC={d['auc']:.4f})"
        pr_label = f"{name} (AP={d['ap']:.4f})"
        
        for ax_roc in [ax_roc_full, ax_roc_zoom]:
            ax_roc.plot(d['fpr'], d['tpr'], color=d['color'], label=roc_label,
                       lw=d['lw'], ls=d['ls'], alpha=d['alpha'])
        for ax_pr in [ax_pr_full, ax_pr_zoom]:
            ax_pr.plot(d['rec'], d['prec'], color=d['color'], label=pr_label,
                      lw=d['lw'], ls=d['ls'], alpha=d['alpha'])

    ax_roc_full.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax_roc_full.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc_full.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc_full.set_title('ROC Curve (Full View)', fontsize=14, fontweight='bold')
    ax_roc_full.legend(loc='lower right', fontsize=9)
    ax_roc_full.grid(True, alpha=0.3)

    ax_roc_zoom.set_xlim(-0.002, 0.06)
    ax_roc_zoom.set_ylim(0.93, 1.005)
    ax_roc_zoom.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc_zoom.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc_zoom.set_title('ROC Curve (Zoomed: Top-Left Corner)', fontsize=14, fontweight='bold')
    ax_roc_zoom.legend(loc='lower right', fontsize=9)
    ax_roc_zoom.grid(True, alpha=0.3)

    ax_pr_full.set_xlabel('Recall', fontsize=12)
    ax_pr_full.set_ylabel('Precision', fontsize=12)
    ax_pr_full.set_title('Precision-Recall Curve (Full View)', fontsize=14, fontweight='bold')
    ax_pr_full.legend(loc='lower left', fontsize=9)
    ax_pr_full.grid(True, alpha=0.3)

    ax_pr_zoom.set_xlim(0.93, 1.005)
    ax_pr_zoom.set_ylim(0.93, 1.005)
    ax_pr_zoom.set_xlabel('Recall', fontsize=12)
    ax_pr_zoom.set_ylabel('Precision', fontsize=12)
    ax_pr_zoom.set_title('Precision-Recall Curve (Zoomed: Top-Right Corner)', fontsize=14, fontweight='bold')
    ax_pr_zoom.legend(loc='lower left', fontsize=9)
    ax_pr_zoom.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "roc_pr_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved ROC & PR Curves -> {filepath}")



def plot_feature_importance(trained_models, feature_names):
    """Extract and plot top-20 feature importance from tree-based models."""
    log_message("Generating Feature Importance Analysis...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_keys = [k for k in ['xgb', 'lgbm', 'rf'] if k in trained_models]
    num_models = len(model_keys)
    
    rows = 2 if num_models > 2 else 1
    cols = 2 if num_models > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 12))
    axes = axes.flatten() if num_models > 1 else [axes]

    color_map = {'xgb': '#2196F3', 'lgbm': '#FF9800', 'rf': '#4CAF50'}
    title_map = {'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'rf': 'Random Forest'}

    for idx, name in enumerate(model_keys):
        model = trained_models[name]
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20

        ax = axes[idx]
        bars = ax.barh(range(len(indices)), importances[indices], color=color_map[name])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top-20 Features ({title_map[name]})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center', fontsize=8)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved Feature Importance -> {filepath}")


def plot_confusion_matrices(all_results):
    """Plot confusion matrix heatmaps with key metrics annotated below each matrix in a 2x2 grid."""
    log_message("Generating Confusion Matrix Heatmaps...")
    
    models = {
        **all_results.get('baseline_comparison', {}), 
        'Proposed Hybrid': all_results.get('ensemble', {})
    }
    
    num_models = len(models)
    
    rows = 2 if num_models > 2 else 1
    cols = 2 if num_models > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    axes = axes.flatten() if num_models > 1 else [axes]
    
    for i, (name, metrics) in enumerate(models.items()):
        cm = [[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        axes[i].set_title(f"{name}", fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Predicted Label', fontsize=11)
        axes[i].set_ylabel('True Label', fontsize=11)
        axes[i].set_xticklabels(['BENIGN', 'ATTACK'])
        axes[i].set_yticklabels(['BENIGN', 'ATTACK'])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()
    log_message("Saved Confusion Matrices")


def plot_comparison_metrics(all_results):
    """Plot bar charts comparing all models with value annotations on each bar."""
    log_message("Generating Model Comparison Charts...")
    
    models_data = []
    for name, m in all_results.get('baseline_comparison', {}).items():
        row = {'Model': name}
        row.update({k: m[k] for k in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'] if k in m})
        models_data.append(row)
    
    ens = all_results.get('ensemble', {})
    row = {'Model': 'Hybrid Ensemble'}
    row.update({k: ens[k] for k in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'] if k in ens})
    models_data.append(row)
    
    df = pd.DataFrame(models_data)
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Value')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    barplot = sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model', palette='viridis', ax=ax)
    
    min_val = df_melted['Value'].min()
    ax.set_ylim(max(0, min_val - 0.05), 1.01) if min_val > 0.8 else ax.set_ylim(0, 1.05)
    
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=7,
                        fontweight='bold', rotation=45)
    
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score (0-1)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved Model Comparison -> {filepath}")

def plot_cv_stability(all_results):
    """Plot CV results with error bars representing std."""
    if 'cross_validation' not in all_results or not all_results['cross_validation']:
        return
    
    log_message("Generating CV Stability Plot...")
    cv = all_results['cross_validation']
    metrics = list(cv.keys())
    means = [cv[m]['mean'] for m in metrics]
    stds = [cv[m]['std'] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=8, color='#4CAF50', alpha=0.8, edgecolor='black')
    plt.ylim(max(0, min(means) - 0.05), 1.05)
    plt.title('K-Fold Cross-Validation Stability (Mean ± Std)', fontsize=15, fontweight='bold')
    plt.ylabel('Mean Score', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "cv_stability.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved CV Stability Plot -> {filepath}")

def plot_ablation_comparison(all_results):
    """Plot horizontal comparison of ablation experiments with value annotations."""
    if 'ablation_study' not in all_results or not all_results['ablation_study']:
        return
        
    log_message("Generating Ablation Study Comparison...")
    ablation = all_results['ablation_study']
    data = []
    for name, m in ablation.items():
        data.append({'Experiment': name, 'F1-Score': m['F1-Score'], 'Accuracy': m['Accuracy']})
    
    df = pd.DataFrame(data).sort_values(by='F1-Score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='barh', x='Experiment', y=['F1-Score', 'Accuracy'], ax=ax, color=['#E91E63', '#2196F3'])
    
    min_val = min(df['F1-Score'].min(), df['Accuracy'].min())
    ax.set_xlim(max(0, min_val - 0.05), 1.02) if min_val > 0.8 else ax.set_xlim(0, 1.05)
    ax.set_title('Ablation Study: Component Impact', fontsize=15, fontweight='bold')
    ax.set_xlabel('Score', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='lower right')
    
    for container in ax.containers:
        for bar in container:
            width = bar.get_width()
            if width > 0:
                ax.text(width + 0.002, bar.get_y() + bar.get_height()/2.,
                        f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "ablation_impact.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved Ablation Study Plot -> {filepath}")


def run_ablation_study(X_train_raw, y_train_raw, X_train_bal, y_train_bal, X_test, y_test, num_features):
    """
    Ablation Study: Measure the contribution of each component.
    Experiments:
    1. Ensemble WITHOUT ENN (raw imbalanced data)
    2. Soft Voting Baseline (WITH ENN)
    3. Stacking Baseline (XGBoost + LightGBM only, WITH ENN)
    4. Full Proposed (Stacking + ENN + all 3 models) — from main pipeline
    """
    log_message("ABLATION STUDY: Measuring component contributions...")

    ablation_results = {}

    # --- Experiment 1: Ensemble WITHOUT ENN ---
    log_message("[Ablation 1/4] Training Stacking WITHOUT ENN (raw imbalanced data)...")

    t1_start = time.time()
    base_1 = build_base_models(num_features)
    abl1_ensemble = StackingClassifier(
        estimators=base_1,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1),
        cv=STACKING_CV,
        n_jobs=-1,
        stack_method='predict_proba'
    )
    abl1_ensemble.fit(X_train_raw, y_train_raw)
    t1 = time.time() - t1_start
    
    y_pred = abl1_ensemble.predict(X_test)
    y_prob = abl1_ensemble.predict_proba(X_test)[:, 1]
    ablation_results['Stacking (No ENN)'] = compute_metrics(y_test, y_pred, y_prob)
    ablation_results['Stacking (No ENN)']['Train Time (s)'] = round(t1, 2)
    del abl1_ensemble; gc.collect()

    # --- Experiment 2: Soft Voting instead of Stacking ---
    log_message("[Ablation 2/4] Training Soft Voting WITH ENN...")
    base_2 = build_base_models(num_features)
    voting = VotingClassifier(estimators=base_2, voting='soft', n_jobs=-1)
    start = time.time()
    voting.fit(X_train_bal, y_train_bal)
    t2 = time.time() - start
    y_pred = voting.predict(X_test)
    try:
        y_prob = voting.predict_proba(X_test)[:, 1]
    except:
        y_prob = None
    ablation_results['Soft Voting (ENN)'] = compute_metrics(y_test, y_pred, y_prob)
    ablation_results['Soft Voting (ENN)']['Train Time (s)'] = round(t2, 2)
    del voting; gc.collect()

    # --- Experiment 3: Stacking with only XGB + LGBM (no RF) ---
    log_message("[Ablation 3/4] Training Stacking (XGB + LGBM only, no RF)...")
    xgb_clf = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        tree_method='hist', n_jobs=-1, use_label_encoder=False,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    lgb_clf = LGBMClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        n_jobs=-1, random_state=42, verbose=-1
    )
    base_3 = [('xgb', xgb_clf), ('lgbm', lgb_clf)]
    ens_3 = StackingClassifier(
        estimators=base_3,
        final_estimator=LogisticRegression(class_weight='balanced', n_jobs=-1),
        cv=STACKING_CV, n_jobs=-1, verbose=0,
        stack_method='predict_proba'
    )
    start = time.time()
    ens_3.fit(X_train_bal, y_train_bal)
    t3 = time.time() - start
    y_pred = ens_3.predict(X_test)
    try:
        y_prob = ens_3.predict_proba(X_test)[:, 1]
    except:
        y_prob = None
    ablation_results['Stacking (XGB+LGBM only)'] = compute_metrics(y_test, y_pred, y_prob)
    ablation_results['Stacking (XGB+LGBM only)']['Train Time (s)'] = round(t3, 2)
    del ens_3; gc.collect()

    log_message("[Ablation 4/4] Full Proposed model already evaluated in main pipeline.")

    log_message("\n--- ABLATION STUDY RESULTS ---")
    for exp_name, metrics in ablation_results.items():
        print_metrics(f"Ablation: {exp_name}", metrics)

    return ablation_results


def run_kfold_cv(X_data, y_data, num_features, k=5):
    """
    Run Stratified K-Fold Cross-Validation on the full proposed pipeline.
    Returns mean ± std for each metric.
    """
    if k < 2:
        log_message(f"Skipping K-Fold Cross-Validation (k={k} is less than 2).")
        return {}, []

    log_message("-" * 30)
    log_message(f"K-FOLD CROSS-VALIDATION (k={k}): Validating stability...")
    log_message("-" * 30)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        log_message(f"  [Fold {fold_idx + 1}/{k}] Processing...")
        
        if CV_SAMPLE_FRACTION < 1.0:
            np.random.seed(42)
            n_samples = int(len(train_idx) * CV_SAMPLE_FRACTION)
            train_idx_sampled = np.random.choice(train_idx, n_samples, replace=False)
            X_fold_train = X_data[train_idx_sampled]
            y_fold_train = y_data.iloc[train_idx_sampled]
        else:
            X_fold_train = X_data[train_idx]
            y_fold_train = y_data.iloc[train_idx]
            
        X_fold_val = X_data[val_idx]
        y_fold_val = y_data.iloc[val_idx]

        if CV_USE_FAST_SAMPLER:
            sampler = RandomUnderSampler(random_state=42)
            log_message(f"  [Fold {fold_idx + 1}/{k}] Resampling with RandomUnderSampler (Fast)...")
        else:
            sampler = SMOTEENN(random_state=42)
            log_message(f"  [Fold {fold_idx + 1}/{k}] Resampling with SMOTEENN (Slow - kNN cleanup)...")
            
        X_bal, y_bal = sampler.fit_resample(X_fold_train, y_fold_train)

        base_models = build_base_models(num_features)
        ensemble = build_ensemble(base_models)
        ensemble.fit(X_bal, y_bal)

        y_pred = ensemble.predict(X_fold_val)
        try:
            y_prob = ensemble.predict_proba(X_fold_val)[:, 1]
        except:
            y_prob = None

        metrics = compute_metrics(y_fold_val, y_pred, y_prob)
        fold_metrics.append(metrics)
        log_message(f"  [Fold {fold_idx + 1}/{k}] Acc={metrics['Accuracy']:.4f}, F1={metrics['F1-Score']:.4f}, AUC-ROC={metrics.get('AUC-ROC', 'N/A')}")

        del X_bal, y_bal, ensemble
        gc.collect()

    cv_summary = {}
    for key in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR', 'AUC-ROC', 'AUC-PR']:
        values = [m[key] for m in fold_metrics if key in m]
        if values:
            cv_summary[key] = {'mean': np.mean(values), 'std': np.std(values)}

    log_message("\n--- K-FOLD CROSS-VALIDATION SUMMARY ---")
    for key, stats in cv_summary.items():
        print(f"  {key:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return cv_summary, fold_metrics


def save_results(baseline_results, ensemble_metrics, ablation_results, cv_summary):
    """Save all results to JSON and CSV for paper usage."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {
        'baseline_comparison': {},
        'ensemble': {},
        'ablation_study': {},
        'cross_validation': {}
    }

    for name, m in baseline_results.items():
        all_results['baseline_comparison'][name] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}

    all_results['ensemble'] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in ensemble_metrics.items()}

    for name, m in ablation_results.items():
        all_results['ablation_study'][name] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}

    for key, stats in cv_summary.items():
        all_results['cross_validation'][key] = {
            'mean': float(stats['mean']),
            'std': float(stats['std'])
        }

    json_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log_message(f"Saved all results -> {json_path}")

    comparison_rows = []
    for name, m in baseline_results.items():
        row = {'Model': name}
        row.update({k: m[k] for k in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR'] if k in m})
        if 'AUC-ROC' in m:
            row['AUC-ROC'] = m['AUC-ROC']
        comparison_rows.append(row)
    row = {'Model': 'Proposed Hybrid'}
    row.update({k: ensemble_metrics[k] for k in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR'] if k in ensemble_metrics})
    if 'AUC-ROC' in ensemble_metrics:
        row['AUC-ROC'] = ensemble_metrics['AUC-ROC']
    comparison_rows.append(row)
    df = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    log_message(f"Saved comparison table -> {csv_path}")

    return all_results


def main():
    train_path = TRAIN_PATH
    test_path = TEST_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        log_message(f"Parquet files not found in {os.path.dirname(train_path)}. Run data_preparation.py first.")
        return

    total_start = time.time()

    # Step 1: Preprocess
    X_train_scaled, X_test_scaled, y_train, y_test, num_features, feature_names = \
        load_and_preprocess(train_path, test_path)

    # Step 2: Balance Datasets
    X_train_bal, y_train_bal = balance_training_data(X_train_scaled, y_train, sample_frac=SAMPLE_FRACTION)

    # Step 3: Build Base Models
    base_models = build_base_models(num_features)

    # Step 4: Baseline Comparison
    baseline_results, trained_models = evaluate_baselines(
        base_models, X_train_bal, y_train_bal, X_test_scaled, y_test
    )

    base_models_fresh = build_base_models(num_features)
    ensemble_model = build_ensemble(base_models_fresh)

    log_message("Training the Hybrid Ensemble Model...")

    start_train = time.time()
    ensemble_model.fit(X_train_bal, y_train_bal)
    train_elapsed = time.time() - start_train
    log_message(f"Ensemble Training Completed in {train_elapsed:.2f} seconds.")

    # Evaluate ensemble
    log_message("Evaluating Hybrid Ensemble on Test Set...")
    y_pred = ensemble_model.predict(X_test_scaled)
    try:
        y_prob = ensemble_model.predict_proba(X_test_scaled)[:, 1]
    except:
        y_prob = None
    ensemble_metrics = compute_metrics(y_test, y_pred, y_prob)
    ensemble_metrics['Train Time (s)'] = round(train_elapsed, 2)
    print_metrics("PROPOSED HYBRID ENSEMBLE", ensemble_metrics)

    plot_roc_and_pr_curves(baseline_results, X_test_scaled, y_test, trained_models, ensemble_model)

    plot_feature_importance(trained_models, feature_names)

    ablation_results = run_ablation_study(
        X_train_scaled, y_train, X_train_bal, y_train_bal,
        X_test_scaled, y_test, num_features
    )
    # Experiment 4/4: the full proposed model (already trained above)
    ablation_results['Proposed (Stacking + ENN)'] = ensemble_metrics


    cv_summary, fold_metrics = run_kfold_cv(X_train_scaled, y_train, num_features, k=KFOLD_K)

    # Save All Results
    all_results = save_results(baseline_results, ensemble_metrics, ablation_results, cv_summary)

    # Populate LaTeX Tables
    try:
        populate_latex()
        log_message("LaTeX tables successfully populated.")
    except Exception as e:
        log_message(f"Warning: Could not populate LaTeX tables: {e}")
    # Generate plots BEFORE saving models to ensure they appear even if saving fails
    plot_confusion_matrices(all_results)
    plot_comparison_metrics(all_results)
    plot_cv_stability(all_results)
    plot_ablation_comparison(all_results)

    # Save Models for later use
    log_message("Saving trained models to disk...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        for name, model in trained_models.items():
            joblib.dump(model, os.path.join(MODELS_DIR, f"{name}_model.joblib"))
        joblib.dump(ensemble_model, os.path.join(MODELS_DIR, "ensemble_model.joblib"))
    except Exception as e:
        log_message(f"Warning: Could not save models to disk: {e}")

    total_elapsed = time.time() - total_start
    log_message(f"\n{'='*60}")
    log_message(f"ALL EXPERIMENTS COMPLETED in {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} min)")
    log_message(f"{'='*60}")
    log_message(f"Figures saved to: {OUTPUT_DIR}")
    log_message(f"Results saved to: {RESULTS_DIR}")

def replot_from_json(json_path):
    """Utility to regenerate plots from a saved JSON results file."""
    if not os.path.exists(json_path):
        log_message(f"Error: {json_path} not found.")
        return
        
    log_message(f"Re-plotting from: {json_path}...")
    with open(json_path, 'r') as f:
        all_results = json.load(f)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrices(all_results)
    plot_comparison_metrics(all_results)
    plot_cv_stability(all_results)
    plot_ablation_comparison(all_results)
    
    # Also populate LaTeX when re-plotting
    try:
        populate_latex()
        log_message("LaTeX tables successfully populated.")
    except Exception as e:
        log_message(f"Warning: Could not populate LaTeX tables: {e}")
        
    log_message("Re-plotting completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Ensemble Machine Learning for IDS")
    parser.add_argument('--replot', type=str, help='Re-plot figures from a specific all_results.json file')
    args = parser.parse_args()
    
    if args.replot:
        replot_from_json(args.replot)
    else:
        main()
