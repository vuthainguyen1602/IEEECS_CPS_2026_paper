import json
import os
import re

def format_num(val, precision=4):
    """Formats a number with a comma as the decimal separator."""
    if val is None or val == "--":
        return "--"
    return f"{val:.{precision}f}".replace(".", ",")

def populate_latex():
    from config import RESULTS_DIR, PROJECT_ROOT
    LATEX_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "sections")
    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    latex_path = os.path.join(LATEX_DIR, "results.tex")
    
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    if not os.path.exists(latex_path):
        print(f"LaTeX file not found at {latex_path}")
        return

    with open(latex_path, 'r') as f:
        latex_content = f.read()
    
    # Baseline Comparison
    # XGBoost
    xgb = data['baseline_comparison']['xgb']
    latex_content = re.sub(
        r"XGBoost\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+",
        f"XGBoost          & {format_num(xgb['Accuracy'])} & {format_num(xgb['Precision'])} & {format_num(xgb['Recall'])} & {format_num(xgb['F1-Score'])} & {format_num(xgb['AUC-ROC'])}",
        latex_content
    )
    # LightGBM
    lgbm = data['baseline_comparison']['lgbm']
    latex_content = re.sub(
        r"LightGBM\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+",
        f"LightGBM         & {format_num(lgbm['Accuracy'])} & {format_num(lgbm['Precision'])} & {format_num(lgbm['Recall'])} & {format_num(lgbm['F1-Score'])} & {format_num(lgbm['AUC-ROC'])}",
        latex_content
    )
    # Random Forest
    rf = data['baseline_comparison']['rf']
    latex_content = re.sub(
        r"Random Forest\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+",
        f"Random Forest    & {format_num(rf['Accuracy'])} & {format_num(rf['Precision'])} & {format_num(rf['Recall'])} & {format_num(rf['F1-Score'])} & {format_num(rf['AUC-ROC'])}",
        latex_content
    )
    # Hybrid (Ours)
    ens = data['ensemble']
    # Use re.escape for parts that contain backslashes
    pattern = r"\\textbf\{Hybrid \(Ours\)\}\s+&\s+\\textbf\{[0-9\.,-]+\}\s+&\s+\\textbf\{[0-9\.,-]+\}\s+&\s+\\textbf\{[0-9\.,-]+\}\s+&\s+\\textbf\{[0-9\.,-]+\}\s+&\s+\\textbf\{[0-9\.,-]+\}"
    replacement = f"\\\\textbf{{Hybrid (Ours)}} & \\\\textbf{{{format_num(ens['Accuracy'])}}} & \\\\textbf{{{format_num(ens['Precision'])}}} & \\\\textbf{{{format_num(ens['Recall'])}}} & \\\\textbf{{{format_num(ens['F1-Score'])}}} & \\\\textbf{{{format_num(ens['AUC-ROC'])}}}"
    latex_content = re.sub(pattern, replacement, latex_content)
    
    # Ablation Study
    ablation = data['ablation_study']
    # Stacking (No SMOTEENN)
    a1 = ablation.get('Stacking (No SMOTEENN)', {'Accuracy': '--', 'F1-Score': '--', 'FPR': '--', 'AUC-ROC': '--'})
    latex_content = re.sub(
        r"Stacking \(No SM\)\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+",
        f"Stacking (No SM)              & {format_num(a1['Accuracy'])} & {format_num(a1['F1-Score'])} & {format_num(a1['FPR'], 6)} & {format_num(a1['AUC-ROC'])}",
        latex_content
    )
    # Soft Voting (with SMOTEENN)
    a2 = ablation.get('Soft Voting (SMOTEENN)', {'Accuracy': '--', 'F1-Score': '--', 'FPR': '--', 'AUC-ROC': '--'})
    latex_content = re.sub(
        r"Soft Voting \(w/ SM\)\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+",
        f"Soft Voting (w/ SM)           & {format_num(a2['Accuracy'])} & {format_num(a2['F1-Score'])} & {format_num(a2['FPR'], 6)} & {format_num(a2['AUC-ROC'])}",
        latex_content
    )
    # Stacking (XGB+LGBM, no RF)
    a3 = ablation.get('Stacking (XGB+LGBM only)', {'Accuracy': '--', 'F1-Score': '--', 'FPR': '--', 'AUC-ROC': '--'})
    latex_content = re.sub(
        r"Stacking \(w/o RF\)\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+",
        f"Stacking (w/o RF)             & {format_num(a3['Accuracy'])} & {format_num(a3['F1-Score'])} & {format_num(a3['FPR'], 6)} & {format_num(a3['AUC-ROC'])}",
        latex_content
    )
    # Hybrid (Full)
    a4 = ablation.get('Proposed (Stacking + SMOTEENN)', ens)
    pattern = r"\\textbf\{Hybrid \(Full\)\}\s+&\s+\\textbf\{[0-9\.,-]+\}\s+&\s+\\textbf\{[0-9\.,-]+\}\s+&\s+\\textbf\{[0-9\.,-]+\}\s+&\s+\\textbf\{[0-9\.,-]+\}"
    replacement = f"\\\\textbf{{Hybrid (Full)}}        & \\\\textbf{{{format_num(a4['Accuracy'])}}} & \\\\textbf{{{format_num(a4['F1-Score'])}}} & \\\\textbf{{{format_num(a4['FPR'], 6)}}} & \\\\textbf{{{format_num(a4['AUC-ROC'])}}}"
    latex_content = re.sub(pattern, replacement, latex_content)

    # Cross-Validation
    cv = data['cross_validation']
    for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR', 'AUC-ROC', 'AUC-PR']:
        m = cv.get(metric_name)
        if m:
            # Escape the $ signs for regex
            pattern = re.escape(metric_name) + r"\s+&\s+\$[0-9\.,-]+\s+\\pm\s+[0-9\.,-]+\$"
            replacement = f"{metric_name:12s} & ${format_num(m['mean'])} \\\\pm {format_num(m['std'], 6)}$"
            latex_content = re.sub(pattern, replacement, latex_content)

    # --- Table Literature Comparison ---
    # Proposed (Ours)   & Hybrid Ensemble & 99.85 & 99.66 \\
    pattern = r"\\textbf\{Proposed \(Ours\)\}\s+&\s+\\textbf\{Hybrid Ensemble\}\s+&\s+\\textbf\{[0-9\.,-]+\}\s+&\s+\\textbf\{[0-9\.,-]+\}"
    replacement = f"\\\\textbf{{Proposed (Ours)}}     & \\\\textbf{{Hybrid Ensemble}} & \\\\textbf{{{format_num(ens['Accuracy']*100, 2)}}} & \\\\textbf{{{format_num(ens['F1-Score']*100, 2)}}}"
    latex_content = re.sub(pattern, replacement, latex_content)

    # Knowledge Distillation
    kd_path = os.path.join(RESULTS_DIR, "kd_results.json")
    if os.path.exists(kd_path):
        with open(kd_path, 'r') as f:
            kd_data = json.load(f)
        
        teacher = kd_data['teacher']
        student = kd_data['student']
        
        latex_content = re.sub(
            r"Teacher \(Ens\.\)\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+",
            f"Teacher (Ens.)     & {format_num(teacher['Accuracy'])} & {format_num(teacher['F1-Score'])} & {format_num(teacher['Inference_ms_per_1k'], 2)} & {format_num(teacher.get('Size_MB', 0), 2)}",
            latex_content
        )
        # Also handle -- placeholders
        latex_content = re.sub(
            r"Teacher \(Ens\.\)\s+&\s+--\s+&\s+--\s+&\s+--\s+&\s+--",
            f"Teacher (Ens.)     & {format_num(teacher['Accuracy'])} & {format_num(teacher['F1-Score'])} & {format_num(teacher['Inference_ms_per_1k'], 2)} & {format_num(teacher.get('Size_MB', 0), 2)}",
            latex_content
        )
        latex_content = re.sub(
            r"Student \(DT\)\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+\s+&\s+[0-9\.,-]+",
            f"Student (DT)       & {format_num(student['Accuracy'])} & {format_num(student['F1-Score'])} & {format_num(student['Inference_ms_per_1k'], 2)} & {format_num(student.get('Size_MB', 0), 4)}",
            latex_content
        )
        latex_content = re.sub(
            r"Student \(DT\)\s+&\s+--\s+&\s+--\s+&\s+--\s+&\s+--",
            f"Student (DT)       & {format_num(student['Accuracy'])} & {format_num(student['F1-Score'])} & {format_num(student['Inference_ms_per_1k'], 2)} & {format_num(student.get('Size_MB', 0), 4)}",
            latex_content
        )

    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"Successfully populated {latex_path}")

if __name__ == "__main__":
    populate_latex()
