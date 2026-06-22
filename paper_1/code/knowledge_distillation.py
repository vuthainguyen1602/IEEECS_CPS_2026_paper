"""
Knowledge Distillation for Edge/IoT Deployment.

Compresses the Hybrid Stacking Ensemble (Teacher) into a lightweight
Decision Tree (Student) suitable for resource-constrained edge devices.
Corresponds to Section III-D of the paper.

This module implements TRUE soft-label knowledge distillation and provides
the ablations required to justify the distillation step:

  * Student-Soft  : the distilled student. A DecisionTreeRegressor is trained
                    on the Teacher's *soft probabilities* p = P(ATTACK | x)
                    (i.e. the "dark knowledge"), then thresholded at 0.5 for
                    classification. This transfers the Teacher's confidence,
                    not just its argmax decision.
  * Student-Hard  : the previous approach — a DecisionTreeClassifier trained on
                    the Teacher's hard labels (argmax). Kept for comparison so
                    the value of soft targets can be measured directly.
  * Student-Direct: a DecisionTreeClassifier trained directly on the balanced
                    ground-truth labels (NO teacher). This is the key baseline:
                    distillation is only worthwhile if Student-Soft beats it.

Usage:
    python knowledge_distillation.py
"""

import os
import sys
import json
import time

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    RANDOM_SEED, set_global_seed,
    RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
    TRAIN_PATH, TEST_PATH, SAMPLE_FRACTION,
    STUDENT_MAX_DEPTH, STUDENT_MIN_SAMPLES_SPLIT,
    PLOT_DPI,
)
from hybrid_ids_cicids2017 import (
    load_and_preprocess, balance_training_data, compute_metrics,
)

set_global_seed(RANDOM_SEED)


def _measure_latency(predict_fn, X_samples):
    """Measure single-sample (batch_size=1) inference latency.

    Returns total milliseconds per 1000 samples — the real-time edge metric.
    """
    n = len(X_samples)
    t0 = time.perf_counter()
    for i in range(n):
        _ = predict_fn(X_samples[i].reshape(1, -1))
    elapsed = time.perf_counter() - t0
    return (elapsed / n) * 1000 * 1000


def _save_and_size(model, path):
    """Persist a model and return its on-disk size in MB."""
    joblib.dump(model, path)
    return os.path.getsize(path) / (1024 * 1024)


def main():
    """Run the full knowledge distillation pipeline with soft labels.

    Steps:
        1. Load preprocessed data, balanced data, and the Teacher model.
        2. Generate the Teacher's SOFT probabilities on the training data.
        3. Train three students: Soft (distilled), Hard (distilled),
           Direct (ground-truth baseline).
        4. Evaluate all models on the held-out test set (incl. macro-F1).
        5. Save results (JSON) and generate the trade-off plot.
    """
    # Verify Teacher exists
    teacher_path = os.path.join(MODELS_DIR, "ensemble_model.joblib")
    if not os.path.exists(teacher_path):
        teacher_path = os.path.join(MODELS_DIR, "hybrid_model.joblib")
    if not os.path.exists(teacher_path):
        print("ERROR: Teacher model not found. Run hybrid_ids_cicids2017.py first.")
        return

    print("Loading and Preprocessing Data...")
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = \
        load_and_preprocess(TRAIN_PATH, TEST_PATH)

    # Balanced ground-truth data (RUS+ENN), reused from cache, for the
    # Student-Direct baseline so it is trained like the Teacher was.
    print("Preparing balanced ground-truth data for the direct baseline...")
    X_train_bal, y_train_bal = balance_training_data(
        X_train_scaled, y_train, sample_frac=SAMPLE_FRACTION
    )

    print("Loading Teacher Model (Hybrid Ensemble)...")
    teacher_model = joblib.load(teacher_path)

    # --- Soft targets: the Teacher's probability of the ATTACK class ---
    print("Generating Teacher SOFT probabilities on training data...")
    teacher_soft = teacher_model.predict_proba(X_train_scaled)[:, 1]
    teacher_hard = (teacher_soft >= 0.5).astype(int)

    # === Student-Soft: distilled from soft probabilities (dark knowledge) ===
    print(f"Training Student-Soft (DT Regressor on soft targets, "
          f"depth={STUDENT_MAX_DEPTH})...")
    student_soft_reg = DecisionTreeRegressor(
        max_depth=STUDENT_MAX_DEPTH,
        min_samples_split=STUDENT_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_SEED,
    )
    student_soft_reg.fit(X_train_scaled, teacher_soft)

    def soft_proba(X):
        return np.clip(student_soft_reg.predict(X), 0.0, 1.0)

    def soft_predict(X):
        return (soft_proba(X) >= 0.5).astype(int)

    # === Student-Hard: distilled from hard labels (previous approach) ===
    print(f"Training Student-Hard (DT Classifier on hard labels, "
          f"depth={STUDENT_MAX_DEPTH})...")
    student_hard = DecisionTreeClassifier(
        max_depth=STUDENT_MAX_DEPTH,
        min_samples_split=STUDENT_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_SEED,
    )
    student_hard.fit(X_train_scaled, teacher_hard)

    # === Student-Direct: trained on balanced ground truth (NO teacher) ===
    print(f"Training Student-Direct (DT Classifier on ground-truth labels, "
          f"depth={STUDENT_MAX_DEPTH})...")
    student_direct = DecisionTreeClassifier(
        max_depth=STUDENT_MAX_DEPTH,
        min_samples_split=STUDENT_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_SEED,
    )
    student_direct.fit(X_train_bal, y_train_bal)

    # --- Persist students and record sizes ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    soft_size = _save_and_size(student_soft_reg,
                               os.path.join(MODELS_DIR, "student_soft_model.joblib"))
    hard_size = _save_and_size(student_hard,
                               os.path.join(MODELS_DIR, "student_hard_model.joblib"))
    direct_size = _save_and_size(student_direct,
                                 os.path.join(MODELS_DIR, "student_direct_model.joblib"))
    # Backwards-compatible alias: the "main" student is the soft one.
    joblib.dump(student_soft_reg, os.path.join(MODELS_DIR, "student_model.joblib"))
    teacher_size = os.path.getsize(teacher_path) / (1024 * 1024)

    # --- Evaluation on the held-out test set ---
    print("Evaluating Teacher vs Students...")
    y_test_arr = np.asarray(y_test)

    teacher_prob = teacher_model.predict_proba(X_test_scaled)[:, 1]
    teacher_pred = (teacher_prob >= 0.5).astype(int)
    m_teacher = compute_metrics(y_test_arr, teacher_pred, teacher_prob)

    soft_prob_test = soft_proba(X_test_scaled)
    m_soft = compute_metrics(y_test_arr, (soft_prob_test >= 0.5).astype(int), soft_prob_test)

    hard_prob_test = student_hard.predict_proba(X_test_scaled)[:, 1]
    m_hard = compute_metrics(y_test_arr, student_hard.predict(X_test_scaled), hard_prob_test)

    direct_prob_test = student_direct.predict_proba(X_test_scaled)[:, 1]
    m_direct = compute_metrics(y_test_arr, student_direct.predict(X_test_scaled), direct_prob_test)

    # --- Real-time inference latency (batch_size = 1) ---
    print("Measuring real-time inference latency (batch_size=1)...")
    np.random.seed(RANDOM_SEED)
    n_samples_test = min(1000, len(X_test_scaled))
    sample_indices = np.random.choice(len(X_test_scaled), size=n_samples_test, replace=False)
    X_samples = X_test_scaled[sample_indices]

    teacher_ms = _measure_latency(teacher_model.predict, X_samples)
    soft_ms = _measure_latency(soft_predict, X_samples)
    hard_ms = _measure_latency(student_hard.predict, X_samples)
    direct_ms = _measure_latency(student_direct.predict, X_samples)

    # --- Headline (soft student vs teacher) ---
    speedup = teacher_ms / soft_ms
    compression = teacher_size / soft_size
    f1_retention = m_soft['F1-Score'] / m_teacher['F1-Score'] * 100

    def _row(tag, m, ms, size):
        print(f"{tag:24s} Acc={m['Accuracy']:.4f}  F1={m['F1-Score']:.4f}  "
              f"F1-macro={m['F1-Macro']:.4f}  AUC={m.get('AUC-ROC', float('nan')):.4f}  "
              f"{ms:.2f} ms/1k  {size:.4f} MB")

    print("-" * 70)
    print("KNOWLEDGE DISTILLATION RESULTS")
    print("-" * 70)
    _row("Teacher (Stacking)", m_teacher, teacher_ms, teacher_size)
    _row("Student-Soft (distill)", m_soft, soft_ms, soft_size)
    _row("Student-Hard (distill)", m_hard, hard_ms, hard_size)
    _row("Student-Direct (no KD)", m_direct, direct_ms, direct_size)
    print("-" * 70)
    print(f"Soft vs Teacher : {speedup:.0f}x faster, {compression:.0f}x smaller, "
          f"{f1_retention:.2f}% F1 retained")
    print(f"Distillation gain (Soft - Direct): "
          f"dF1={m_soft['F1-Score'] - m_direct['F1-Score']:+.4f}, "
          f"dF1-macro={m_soft['F1-Macro'] - m_direct['F1-Macro']:+.4f}")
    print(f"Soft-target gain (Soft - Hard):    "
          f"dF1={m_soft['F1-Score'] - m_hard['F1-Score']:+.4f}, "
          f"dF1-macro={m_soft['F1-Macro'] - m_hard['F1-Macro']:+.4f}")
    print("-" * 70)

    def _block(m, ms, size):
        return {
            "Accuracy": float(m['Accuracy']),
            "F1-Score": float(m['F1-Score']),
            "F1-Macro": float(m['F1-Macro']),
            "Precision-ATTACK": float(m['Precision-ATTACK']),
            "Recall-ATTACK": float(m['Recall-ATTACK']),
            "AUC-ROC": float(m.get('AUC-ROC', float('nan'))),
            "AUC-PR": float(m.get('AUC-PR', float('nan'))),
            "FPR": float(m['FPR']),
            "Inference_ms_per_1k": float(ms),
            "Size_MB": float(size),
        }

    kd_results = {
        "teacher": _block(m_teacher, teacher_ms, teacher_size),
        "student": _block(m_soft, soft_ms, soft_size),          # main = soft
        "student_soft": _block(m_soft, soft_ms, soft_size),
        "student_hard": _block(m_hard, hard_ms, hard_size),
        "student_direct": _block(m_direct, direct_ms, direct_size),
        "speedup_factor": float(speedup),
        "compression_factor": float(compression),
        "distillation_method": "soft-label (DecisionTreeRegressor on teacher P(attack))",
        "gain_soft_minus_direct_F1": float(m_soft['F1-Score'] - m_direct['F1-Score']),
        "gain_soft_minus_hard_F1": float(m_soft['F1-Score'] - m_hard['F1-Score']),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "kd_results.json"), "w") as f:
        json.dump(kd_results, f, indent=4)

    # --- Plot: F1 vs Inference Speed Trade-off (all students) ---
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 7))
    points = [
        ("Teacher (Stacking)", teacher_ms, m_teacher['F1-Score'], 'red', '*', 320),
        ("Student-Soft (KD)", soft_ms, m_soft['F1-Score'], 'blue', 'o', 200),
        ("Student-Hard (KD)", hard_ms, m_hard['F1-Score'], 'green', 's', 150),
        ("Student-Direct (no KD)", direct_ms, m_direct['F1-Score'], 'gray', '^', 150),
    ]
    for label, x, y, c, mk, s in points:
        plt.scatter([x], [y], color=c, s=s, label=label, marker=mk, zorder=5)

    plt.xscale('log')
    plt.xlabel("Inference Time (ms per 1000 samples) — Log Scale", fontsize=12)
    plt.ylabel("F1-Score (ATTACK)", fontsize=12)
    plt.title("Knowledge Distillation: F1 vs Inference Speed Trade-off", fontsize=14)
    plt.legend(fontsize=11, loc='lower left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.4f}"))

    mid_x = np.sqrt(teacher_ms * soft_ms)
    mid_y = (m_teacher['F1-Score'] + m_soft['F1-Score']) / 2
    plt.annotate("", xy=(soft_ms, m_soft['F1-Score']),
                 xytext=(teacher_ms, m_teacher['F1-Score']),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.text(mid_x, mid_y + 0.001,
             f"{speedup:.0f}x Faster\n{compression:.0f}x Smaller\n"
             f"{f1_retention:.1f}% F1 Retained",
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='lightyellow', edgecolor='gray'))

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "inference_tradeoff.png")
    plt.savefig(fig_path, dpi=PLOT_DPI)
    plt.close()
    print(f"Saved inference trade-off plot to {fig_path}")


if __name__ == "__main__":
    main()
