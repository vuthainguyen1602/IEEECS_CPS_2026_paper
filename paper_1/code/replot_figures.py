import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TRAIN_PATH

def replot_confusion_matrices():
    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    if not os.path.exists(results_path):
        print("all_results.json not found! Please wait for main script to finish.")
        return
        
    with open(results_path, 'r') as f:
        all_results = json.load(f)
        
    print("Re-plotting Confusion Matrices...")
    
    models = {
        **all_results.get('baseline_comparison', {}), 
        'Proposed Hybrid': all_results.get('ensemble', {})
    }
    
    num_models = len(models)
    rows = 2 if num_models > 2 else 1
    cols = 2 if num_models > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 12))
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
    output_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Done! 2x2 Confusion matrix saved to: {output_path}")

def replot_feature_importance():
    model_path = os.path.join(MODELS_DIR, "ensemble_model.joblib")
    if not os.path.exists(model_path):
        print(f"{model_path} not found!")
        return
        
    print("Re-plotting Feature Importance...")
    import joblib
    import numpy as np
    
    ensemble = joblib.load(model_path)
    trained_models = ensemble.named_estimators_
    
    # Extract feature names
    import pyarrow.parquet as pq
    train_path = TRAIN_PATH
    if os.path.exists(train_path):
        dataset = pq.ParquetDataset(train_path)
        all_cols = dataset.schema.names
        exclude_cols = ["label", "label_binary", "source_ip", "destination_ip", "flow_id", "timestamp", "protocol"]
        # In hybrid_ids_cicids2017, it selects numeric columns. We assume all remaining are numeric.
        feature_names = [c for c in all_cols if c not in exclude_cols]
    else:
        # Fallback if parquet is missing
        feature_names = [f"Feature_{i}" for i in range(100)]
    
    model_keys = [k for k in ['xgb', 'lgbm', 'rf'] if k in trained_models]
    num_models = len(model_keys)
    
    rows = 2 if num_models > 2 else 1
    cols = 2 if num_models > 1 else 1
    
    # Plotting
    fig, axes = plt.subplots(rows, cols, figsize=(20, 18))
    axes = axes.flatten() if num_models > 1 else [axes]

    color_map = {'xgb': '#2196F3', 'lgbm': '#FF9800', 'rf': '#4CAF50'}
    title_map = {'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'rf': 'Random Forest'}

    for idx, name in enumerate(model_keys):
        model = trained_models[name]
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]

        ax = axes[idx]
        bars = ax.barh(range(len(indices)), importances[indices], color=color_map.get(name, 'gray'))
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=13)
        ax.set_xlabel('Feature Importance', fontsize=15)
        ax.set_title(f'Top-20 Features ({title_map.get(name, name)})', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center', fontsize=11)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Done! 2x2 Feature Importance saved to: {output_path}")

if __name__ == "__main__":
    replot_confusion_matrices()
    replot_feature_importance()
