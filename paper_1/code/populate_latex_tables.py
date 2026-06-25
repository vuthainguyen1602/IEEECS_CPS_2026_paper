"""
Populate the LaTeX result tables in ``sections/results.tex`` from the JSON
artifacts produced by the experiment pipeline.

It fills exactly the two tables that currently exist in results.tex:

  * Table ``tab:overall_results`` — columns: Acc. | F1 | AUC | Trễ (ms/1k) | Size (MB)
        rows: XGBoost, LightGBM, Random Forest, Stacking (Teacher), Student (DT)
  * Table ``tab:literature_comparison`` — columns: Acc. (%) | F1 (%)
        proposed rows: "Stacking Teacher" and "Student DT (Chưng cất)"

Sources:
  * results/all_results.json  -> baseline_comparison{xgb,lgbm,rf}, ensemble
  * results/kd_results.json   -> teacher (latency/size), student (soft, distilled)

The replacements are done with regex + replacement *functions* (not strings),
so no backslash-escaping gymnastics are required.
"""

import json
import os
import re


def format_num(val, precision=4):
    """Format a number with a comma as the decimal separator (vi-VN)."""
    if val is None or val == "--":
        return "--"
    if isinstance(val, float) and val != val:  # NaN
        return "--"
    return f"{val:.{precision}f}".replace(".", ",")


# A numeric cell may be a vi-VN number (digits, comma/dot, minus) or a literal "--".
_CELL = r"(?:[0-9.,\-]+|--)"


def _sub_once(content, pattern, build_fn, label):
    """Apply a single regex substitution, warning if nothing matched."""
    new_content, n = re.subn(pattern, build_fn, content, count=1)
    if n == 0:
        print(f"  [warn] pattern not found / unchanged: {label}")
    return new_content


def populate_latex():
    from config import RESULTS_DIR, PROJECT_ROOT
    latex_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "sections")
    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    kd_path = os.path.join(RESULTS_DIR, "kd_results.json")
    latex_path = os.path.join(latex_dir, "results.tex")

    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return
    if not os.path.exists(latex_path):
        print(f"LaTeX file not found at {latex_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    kd = None
    if os.path.exists(kd_path):
        with open(kd_path, "r", encoding="utf-8") as f:
            kd = json.load(f)

    with open(latex_path, "r", encoding="utf-8") as f:
        latex = f.read()

    base = data.get("baseline_comparison", {})
    ens = data.get("ensemble", {})

    # ===================================================================
    # Table 1: tab:overall_results — Acc. & F1 & AUC & Trễ (ms/1k) & Size (MB)
    # ===================================================================

    def _plain_row(name, m):
        """XGBoost/LightGBM/Random Forest rows (no bold), latency/size = --."""
        nonlocal latex
        # name & cell & cell & cell & cell & cell
        pattern = (re.escape(name) + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL
                   + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL)

        def build(_):
            return (f"{name:16s} & {format_num(m.get('Accuracy'))} "
                    f"& {format_num(m.get('F1-Score'))} "
                    f"& {format_num(m.get('AUC-PR'))} & -- & --")

        latex = _sub_once(latex, pattern, build, f"Table1 row {name}")

    if "xgb" in base:
        _plain_row("XGBoost", base["xgb"])
    if "lgbm" in base:
        _plain_row("LightGBM", base["lgbm"])
    if "rf" in base:
        _plain_row("Random Forest", base["rf"])

    # Stacking (Teacher): Acc/F1/AUC bold; latency & size plain (from kd teacher)
    if ens:
        teacher_lat = kd["teacher"]["Inference_ms_per_1k"] if kd else "--"
        teacher_size = kd["teacher"]["Size_MB"] if kd else "--"
        pattern = (r"\\textbf\{Stacking \(Teacher\)\}\s*&\s*\\textbf\{" + _CELL
                   + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL
                   + r"\}\s*&\s*" + _CELL + r"\s*&\s*" + _CELL)

        def build_teacher(_):
            return (r"\textbf{Stacking (Teacher)} "
                    f"& \\textbf{{{format_num(ens.get('Accuracy'))}}} "
                    f"& \\textbf{{{format_num(ens.get('F1-Score'))}}} "
                    f"& \\textbf{{{format_num(ens.get('AUC-PR'))}}} "
                    f"& {format_num(teacher_lat, 2)} & {format_num(teacher_size, 2)}")

        latex = _sub_once(latex, pattern, build_teacher, "Table1 Stacking (Teacher)")

    # Student (DT): all five cells bold, from kd student (soft-distilled)
    if kd and "student" in kd:
        st = kd["student"]
        pattern = (r"\\textbf\{Student \(DT\)\}\s*&\s*\\textbf\{" + _CELL
                   + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL
                   + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}")

        def build_student(_):
            return (r"\textbf{Student (DT)} "
                    f"& \\textbf{{{format_num(st.get('Accuracy'))}}} "
                    f"& \\textbf{{{format_num(st.get('F1-Score'))}}} "
                    f"& \\textbf{{{format_num(st.get('AUC-PR'))}}} "
                    f"& \\textbf{{{format_num(st.get('Inference_ms_per_1k'), 2)}}} "
                    f"& \\textbf{{{format_num(st.get('Size_MB'), 4)}}}")

        latex = _sub_once(latex, pattern, build_student, "Table1 Student (DT)")

    # ===================================================================
    # Table 2: tab:literature_comparison — Acc. (%) & F1 (%)
    # ===================================================================
    if ens:
        pattern = (r"(\\textbf\{Stacking Teacher\}\s*&\s*\\textbf\{Hybrid Ensemble\}\s*&\s*)"
                   r"\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}")

        def build_lit_teacher(m):
            return (m.group(1)
                    + f"\\textbf{{{format_num(ens.get('Accuracy', 0) * 100, 2)}}} "
                    f"& \\textbf{{{format_num(ens.get('F1-Score', 0) * 100, 2)}}}")

        latex = _sub_once(latex, pattern, build_lit_teacher, "Table2 Stacking Teacher")

    if kd and "student" in kd:
        st = kd["student"]
        pattern = (r"(\\textbf\{Student DT \(Chưng cất\)\}\s*&\s*\\textbf\{Lightweight DT\}\s*&\s*)"
                   r"\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}")

        def build_lit_student(m):
            return (m.group(1)
                    + f"\\textbf{{{format_num(st.get('Accuracy', 0) * 100, 2)}}} "
                    f"& \\textbf{{{format_num(st.get('F1-Score', 0) * 100, 2)}}}")

        latex = _sub_once(latex, pattern, build_lit_student, "Table2 Student DT")

    # ===================================================================
    # Table 3: tab:ablation — Cấu hình & Acc & F1 & FPR & AUC
    # ===================================================================
    abl = data.get("ablation_study", {})

    def _abl_row(label, key):
        nonlocal latex
        m = abl.get(key)
        if not m:
            return
        pattern = (re.escape(label) + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL
                   + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL)

        def build(_):
            return (f"{label} & {format_num(m.get('Accuracy'))} "
                    f"& {format_num(m.get('F1-Score'))} "
                    f"& {format_num(m.get('FPR'), 6)} "
                    f"& {format_num(m.get('AUC-PR'))}")

        latex = _sub_once(latex, pattern, build, f"Ablation {key}")

    _abl_row("Stacking, không ENN", "Stacking (No ENN)")
    _abl_row("Soft Voting + ENN", "Soft Voting (ENN)")
    _abl_row("Stacking bỏ RF + ENN", "Stacking (XGB+LGBM only)")

    if "Proposed (Stacking + ENN)" in abl:
        mp = abl["Proposed (Stacking + ENN)"]
        lbl = "Đề xuất (đủ 3 mô hình + ENN)"
        pattern = (r"\\textbf\{" + re.escape(lbl) + r"\}\s*&\s*\\textbf\{" + _CELL
                   + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL
                   + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}")

        def build_abl_prop(_):
            return (r"\textbf{" + lbl + "} "
                    f"& \\textbf{{{format_num(mp.get('Accuracy'))}}} "
                    f"& \\textbf{{{format_num(mp.get('F1-Score'))}}} "
                    f"& \\textbf{{{format_num(mp.get('FPR'), 6)}}} "
                    f"& \\textbf{{{format_num(mp.get('AUC-PR'))}}}")

        latex = _sub_once(latex, pattern, build_abl_prop, "Ablation Proposed")

    # ===================================================================
    # Table 4: tab:student_variants — Acc & F1 & F1-macro & Trễ (ms/1k)
    # ===================================================================
    def _student_row(label, key, bold=False):
        nonlocal latex
        if not kd or key not in kd:
            return
        m = kd[key]
        cells = [format_num(m.get('Accuracy')), format_num(m.get('F1-Score')),
                 format_num(m.get('F1-Macro')), format_num(m.get('FPR'), 6),
                 format_num(m.get('Inference_ms_per_1k'), 2)]
        if bold:
            pattern = (r"\\textbf\{" + re.escape(label) + r"\}\s*&\s*\\textbf\{" + _CELL
                       + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL
                       + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL
                       + r"\}")

            def build(_):
                return (r"\textbf{" + label + "} & "
                        + " & ".join(f"\\textbf{{{c}}}" for c in cells))
        else:
            pattern = (re.escape(label) + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL
                       + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL)

            def build(_):
                return f"{label} & " + " & ".join(cells)

        latex = _sub_once(latex, pattern, build, f"Student {key}")

    _student_row("Teacher (Stacking)", "teacher")
    _student_row("Student-Soft (chưng cất)", "student_soft", bold=True)
    _student_row("Student-Hard (nhãn cứng)", "student_hard")
    _student_row("Student-Direct (không KD)", "student_direct")

    # ===================================================================
    # Table 5: tab:perclass — Hiệu năng theo từng lớp của Teacher
    # ===================================================================
    if ens:
        def _pc_row(label, pkey, rkey, fkey, bold=False):
            nonlocal latex
            vals = [format_num(ens.get(pkey)), format_num(ens.get(rkey)),
                    format_num(ens.get(fkey))]
            if bold:
                pattern = (r"\\textbf\{" + re.escape(label) + r"\}\s*&\s*\\textbf\{" + _CELL
                           + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}\s*&\s*\\textbf\{" + _CELL + r"\}")

                def build(_):
                    return (r"\textbf{" + label + "} & "
                            + " & ".join(f"\\textbf{{{v}}}" for v in vals))
            else:
                pattern = (re.escape(label) + r"\s*&\s*" + _CELL + r"\s*&\s*" + _CELL
                           + r"\s*&\s*" + _CELL)

                def build(_):
                    return f"{label} & " + " & ".join(vals)

            latex = _sub_once(latex, pattern, build, f"PerClass {label}")

        _pc_row("BENIGN (bình thường)", "Precision-BENIGN", "Recall-BENIGN", "F1-BENIGN")
        _pc_row("ATTACK (tấn công)", "Precision-ATTACK", "Recall-ATTACK", "F1-ATTACK")
        _pc_row("Trung bình macro", "Precision-Macro", "Recall-Macro", "F1-Macro", bold=True)

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Successfully populated {latex_path}")


if __name__ == "__main__":
    populate_latex()
