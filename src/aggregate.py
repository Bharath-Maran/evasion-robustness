"""Summarize results/*.json into results/summary.md and the LaTeX tables of the project update."""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

import pipeline as P

NAMES = {"tfidf_lr": "TF-IDF+LR", "roberta-base": "RoBERTa", "bert-base-uncased": "BERT"}
LABELS = {"hatexplain": "HateXplain", "dyna_exp": "DynaHate-Exp", "dyna_imp": "DynaHate-Imp"}
STRATEGIES = ("leet_light", "leet_heavy", "homoglyph", "spacing", "repeat", "algospeak", "mix")


def load():
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(P.RESULTS, "*__s*.json")) if "benign_check" not in p]
    groups = defaultdict(list)
    for r in runs:
        groups[r["model"], r["task"], r["augment"]].append(r)
    return groups


def stat(rows, get):
    vals = [get(r) for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def cell(rows, get, digits=3):
    mean, std = stat(rows, get)
    return f"{mean:.{digits}f}" + (f" ({std:.{digits}f})" if len(rows) > 1 else "")


def tex_table(path, header, rows, spec):
    lines = [r"\begin{tabular}{@{}" + spec + r"@{}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main(tex_dir):
    groups = load()
    models = ("tfidf_lr", "roberta-base", "bert-base-uncased")
    out, clean, recall, attack, mitigation = [], [], [], [], []
    for task in P.TASKS:
        for model in models:
            rows = groups.get((model, task, False))
            if not rows:
                continue
            base = rows[0]
            clean.append([LABELS[task], NAMES[model], str(len(rows)), cell(rows, lambda r: r["clean_full_test"]["f1"]),
                          cell(rows, lambda r: r["clean_full_test"]["macro_f1"]), cell(rows, lambda r: r["clean"]["recall"]),
                          cell(rows, lambda r: r["clean"]["precision"])])
            row = [LABELS[task], NAMES[model], cell(rows, lambda r: r["clean"]["recall"], 2)]
            row += [cell(rows, lambda r: r["organic"][f"{s}@0.2"]["recall"], 2).split(" ")[0] for s in STRATEGIES]
            recall.append(row)
            for r in rows:
                if "attack" in r:
                    a = r["attack"]
                    attack.append([LABELS[task], NAMES[model], f"{r['clean']['recall']:.2f}",
                                   f"{r['organic']['mix@0.1']['recall']:.2f}", f"{a['0.1']['recall']:.2f}",
                                   f"{r['organic']['mix@0.2']['recall']:.2f}", f"{a['0.2']['recall']:.2f}",
                                   f"{r['organic']['mix@0.3']['recall']:.2f}", f"{a['0.3']['recall']:.2f}",
                                   f"{a['0.3']['attack_success']:.2f}", f"{a['avg_queries']:.0f}"])
            out.append(f"{task} {model} n={len(rows)} clean F1 {cell(rows, lambda r: r['clean']['f1'])} "
                       f"benign changed by normalizer {base['benign_changed_by_normalizer']:.3f}")
        rows = groups.get(("roberta-base", task, False))
        aug = groups.get(("roberta-base", task, True))
        if rows and aug:
            r0 = next(r for r in rows if r["seed"] == 0)
            a0 = aug[0]
            c0 = json.load(open(os.path.join(P.RESULTS, f"benign_check__{task}__s0.json")))
            c1 = json.load(open(os.path.join(P.RESULTS, f"benign_check__{task}__s0__aug.json")))
            mix = "mix@0.2"
            mitigation.append([LABELS[task], f"{r0['clean']['recall']:.2f}", f"{r0['organic'][mix]['recall']:.2f}",
                               f"{r0['organic'][mix]['normalized']['recall']:.2f}", f"{a0['organic'][mix]['recall']:.2f}",
                               f"{c0['fpr_clean']:.2f}", f"{c0['strategies']['mix']['fpr']:.2f}",
                               f"{c0['strategies']['mix']['fpr_normalized']:.2f}", f"{c1['strategies']['mix']['fpr']:.2f}"])
    os.makedirs(tex_dir, exist_ok=True)
    tex_table(f"{tex_dir}/clean.tex", ["Task", "Model", "$n$", "$F_1$", "Macro $F_1$", "Recall$^*$", "Prec.$^*$"], clean, "llrrrrr")
    tex_table(f"{tex_dir}/recall.tex", ["Task", "Model", "Clean", "Leet-L", "Leet-H", "Homog.", "Space", "Repeat", "Algo", "Mix"], recall, "ll" + "r" * 8)
    tex_table(f"{tex_dir}/attack.tex", ["Task", "Model", "Clean", "Org .1", "ML .1", "Org .2", "ML .2", "Org .3", "ML .3", "ASR", "Qry"], attack, "ll" + "r" * 9)
    tex_table(f"{tex_dir}/mitigation.tex", ["Task", "Clean", "Mix", "M1", "M2", "Clean", "Mix", "M1", "M2"], mitigation, "l" + "r" * 8)
    body = ["# Results summary", "", *out, ""]
    for title, table in (("Clean", clean), ("Recall at budget 0.2", recall), ("Attack, seed 0", attack), ("Mitigations, RoBERTa seed 0", mitigation)):
        body += [f"## {title}", ""] + [" | ".join(r) for r in table] + [""]
    open(os.path.join(P.RESULTS, "summary.md"), "w").write("\n".join(body))
    print("\n".join(body))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(P.ROOT, "results", "tables"))
