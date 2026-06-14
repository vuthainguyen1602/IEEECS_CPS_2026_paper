"""
Knowledge Distillation for Edge/IoT Deployment.

Compresses the Hybrid Stacking Ensemble (Teacher) into a lightweight
Decision Tree (Student) suitable for resource-constrained edge devices.
Corresponds to Section III-D of the paper.

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    RANDOM_SEED, set_global_seed,
    RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
    TRAIN_PATH, TEST_PATH,
    STUDENT_MAX_DEPTH, STUDENT_MIN_SAMPLES_SPLIT,
    PLOT_DPI,
)
from hybrid_ids_cicids2017 import load_and_preprocess

set_global_seed(RANDOM_SEED)


def main():
    """Run the full knowledge distillation pipeline.

    Steps:
        1. Load preprocessed data and Teacher model.
        2. Generate Teacher's hard labels on training data.
        3. Train a lightweight Student (Decision Tree) on those labels.
        4. Evaluate both models on the held-out test set.
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

    print("Loading Teacher Model (Hybrid Ensemble)...")
    teacher_model = joblib.load(teacher_path)

    print("Generating Teacher Labels on training data...")
    teacher_hard_labels = teacher_model.predict(X_train_scaled)

    print(f"Training Student Model (Decision Tree depth={STUDENT_MAX_DEPTH}) "
          f"via Knowledge Distillation...")
    student_model = DecisionTreeClassifier(
        max_depth=STUDENT_MAX_DEPTH,
        min_samples_split=STUDENT_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_SEED
    )
    student_model.fit(X_train_scaled, teacher_hard_labels)

    # Save student model
    student_model_path = os.path.join(MODELS_DIR, "student_model.joblib")
    joblib.dump(student_model, student_model_path)

    # Evaluation
    print("Evaluating Teacher vs Student...")

    teacher_size_mb = os.path.getsize(teacher_path) / (1024 * 1024)
    student_size_mb = os.path.getsize(student_model_path) / (1024 * 1024)

    # Measure Accuracy and F1 using bulk prediction
    y_pred_teacher = teacher_model.predict(X_test_scaled)
    teacher_acc = accuracy_score(y_test, y_pred_teacher)
    teacher_f1 = f1_score(y_test, y_pred_teacher)

    y_pred_student = student_model.predict(X_test_scaled)
    student_acc = accuracy_score(y_test, y_pred_student)
    student_f1 = f1_score(y_test, y_pred_student)

    # Measure Real-time Inference Speed (batch_size = 1)
    print("Measuring real-time inference latency (batch_size=1)...")
    np.random.seed(RANDOM_SEED)
    n_samples_test = min(1000, len(X_test_scaled))
    sample_indices = np.random.choice(len(X_test_scaled), size=n_samples_test, replace=False)
    X_samples = X_test_scaled[sample_indices]

    t0_teacher = time.perf_counter()
    for i in range(n_samples_test):
        _ = teacher_model.predict(X_samples[i].reshape(1, -1))
    teacher_time = time.perf_counter() - t0_teacher
    teacher_ms_per_1k = (teacher_time / n_samples_test) * 1000 * 1000  # Total ms per 1000 samples

    t0_student = time.perf_counter()
    for i in range(n_samples_test):
        _ = student_model.predict(X_samples[i].reshape(1, -1))
    student_time = time.perf_counter() - t0_student
    student_ms_per_1k = (student_time / n_samples_test) * 1000 * 1000  # Total ms per 1000 samples

    # Derived metrics
    speedup = teacher_ms_per_1k / student_ms_per_1k
    compression = teacher_size_mb / student_size_mb
    f1_retention = student_f1 / teacher_f1 * 100

    # Report
    print("-" * 50)
    print("KNOWLEDGE DISTILLATION RESULTS")
    print("-" * 50)
    print(f"Teacher (Hybrid Ensemble) - Acc: {teacher_acc:.4f}, F1: {teacher_f1:.4f}")
    print(f"                            Inference: {teacher_ms_per_1k:.2f} ms/1k, "
          f"Size: {teacher_size_mb:.2f} MB")
    print(f"Student (Decision Tree)   - Acc: {student_acc:.4f}, F1: {student_f1:.4f}")
    print(f"                            Inference: {student_ms_per_1k:.2f} ms/1k, "
          f"Size: {student_size_mb:.4f} MB")
    print("-" * 50)
    print(f"Speedup:     {speedup:.0f}x faster inference")
    print(f"Compression: {compression:.0f}x smaller footprint")
    print(f"F1 Retained: {f1_retention:.2f}% of Teacher's F1")
    print("-" * 50)

    # Save JSON
    kd_results = {
        "teacher": {
            "Accuracy": float(teacher_acc),
            "F1-Score": float(teacher_f1),
            "Inference_ms_per_1k": float(teacher_ms_per_1k),
            "Size_MB": float(teacher_size_mb)
        },
        "student": {
            "Accuracy": float(student_acc),
            "F1-Score": float(student_f1),
            "Inference_ms_per_1k": float(student_ms_per_1k),
            "Size_MB": float(student_size_mb)
        },
        "speedup_factor": float(speedup),
        "compression_factor": float(compression)
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "kd_results.json"), "w") as f:
        json.dump(kd_results, f, indent=4)

    # Plot: Accuracy vs Inference Speed Trade-off
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 7))

    plt.scatter([teacher_ms_per_1k], [teacher_f1], color='red', s=300,
                label='Teacher (Hybrid Ensemble)', marker='*', zorder=5)
    plt.scatter([student_ms_per_1k], [student_f1], color='blue', s=200,
                label='Student (Decision Tree)', marker='o', zorder=5)

    plt.xscale('log')
    plt.xlabel("Inference Time (ms per 1000 samples) — Log Scale", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.title("Knowledge Distillation: Accuracy vs Inference Speed Trade-off",
              fontsize=14)
    plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.4f}"))

    # Arrow from Teacher to Student
    mid_x = np.sqrt(teacher_ms_per_1k * student_ms_per_1k)
    mid_y = (teacher_f1 + student_f1) / 2
    plt.annotate("", xy=(student_ms_per_1k, student_f1),
                 xytext=(teacher_ms_per_1k, teacher_f1),
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
