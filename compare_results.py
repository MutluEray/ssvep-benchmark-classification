"""Merge benchmark runs into one table, paired statistics and a portfolio figure.

Usage
  python3 compare_results.py results/benchmark --out results/final

Reads per_subject.csv from one or more run directories. For every window shared by two
methods, runs a paired Wilcoxon signed-rank test across subjects. Use --methods to choose
which methods appear (default: fbcca, ecca, trca).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ORDER = ["fbcca", "ecca", "trca"]
LABELS = {"fbcca": "FBCCA (no training)", "ecca": "Extended CCA", "trca": "Ensemble TRCA"}


def load(dirs):
    data = {}                                  # (method, window) -> {subject: (acc, itr)}
    for d in dirs:
        with open(Path(d) / "per_subject.csv") as f:
            for r in csv.DictReader(f):
                key = (r["method"], round(float(r["window_s"]), 2))
                data.setdefault(key, {})[int(r["subject"])] = (float(r["accuracy"]), float(r["itr"]))
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/final"))
    p.add_argument("--reference", default="trca", help="method the others are tested against")
    p.add_argument("--methods", default=",".join(ORDER), help="methods to include, in plot order")
    args = p.parse_args()
    data = load(args.runs)
    wanted = [m.strip() for m in args.methods.split(",")]
    data = {k: v for k, v in data.items() if k[0] in wanted}
    methods = [m for m in wanted if any(k[0] == m for k in data)]
    windows = sorted({k[1] for k in data})
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- summary table
    rows = []
    print(f"{'window':>7}  " + "  ".join(f"{m:>18}" for m in methods))
    for w in windows:
        line = [f"{w:>6.2f}s"]
        for m in methods:
            d = data.get((m, w))
            if not d:
                line.append(f"{'-':>18}")
                continue
            acc = np.array([v[0] for v in d.values()]); itr = np.array([v[1] for v in d.values()])
            sem = acc.std(ddof=1) / np.sqrt(len(acc))
            rows.append(dict(method=m, window_s=w, n=len(acc), acc_mean=acc.mean(), acc_sem=sem,
                             itr_mean=itr.mean(), itr_sem=itr.std(ddof=1) / np.sqrt(len(itr))))
            line.append(f"{100 * acc.mean():6.1f}±{100 * sem:4.1f} ({itr.mean():5.0f})")
        print("  ".join(line))
    print("cells: accuracy % ± SEM (ITR bits/min)")
    with open(args.out / "summary_all.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)

    # ---- paired tests against the reference method
    ref = args.reference
    tests = []
    print(f"\nPaired Wilcoxon signed-rank vs {ref} (per-subject accuracy)")
    for w in windows:
        if (ref, w) not in data:
            continue
        for m in methods:
            if m == ref or (m, w) not in data:
                continue
            subs = sorted(set(data[(ref, w)]) & set(data[(m, w)]))
            a = np.array([data[(m, w)][s][0] for s in subs]); b = np.array([data[(ref, w)][s][0] for s in subs])
            diff = a - b
            pval = wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
            wins = int(np.sum(diff > 0)); losses = int(np.sum(diff < 0))
            tests.append(dict(window_s=w, method=m, reference=ref, n=len(subs), mean_diff=diff.mean(),
                              wins=wins, losses=losses, p_value=pval))
            print(f"  {w:.2f}s  {m:<11} - {ref}: {100 * diff.mean():+6.2f} pts  "
                  f"(better in {wins}/{len(subs)}, worse in {losses}/{len(subs)})  p = {pval:.3g}")
    if tests:
        with open(args.out / "paired_tests.csv", "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(tests[0].keys())); w_.writeheader(); w_.writerows(tests)
    print("  (uncorrected p-values; apply Bonferroni/Holm across the comparisons you report)")

    plot(rows, methods, args.out / "accuracy_itr_all_methods.png")
    print(f"\nSaved {args.out}/summary_all.csv, paired_tests.csv, accuracy_itr_all_methods.png")


def plot(rows, methods, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"fbcca": ("#9aa5b1", "o", ":"), "ecca": ("#4f8a8b", "s", "-"), "trca": ("#1f4e79", "D", "-")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in methods:
        sel = sorted((r for r in rows if r["method"] == m), key=lambda r: r["window_s"])
        if not sel:
            continue
        c, mk, ls = styles.get(m, ("k", "o", "-"))
        w = [r["window_s"] for r in sel]
        axes[0].errorbar(w, [100 * r["acc_mean"] for r in sel], yerr=[100 * r["acc_sem"] for r in sel],
                         color=c, marker=mk, ls=ls, capsize=3, label=LABELS.get(m, m))
        axes[1].errorbar(w, [r["itr_mean"] for r in sel], yerr=[r["itr_sem"] for r in sel],
                         color=c, marker=mk, ls=ls, capsize=3, label=LABELS.get(m, m))
    axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(0, 100)
    axes[1].set_ylabel("ITR (bits/min)")
    for ax in axes:
        ax.set_xlabel("Data length (s)"); ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_title("Accuracy, 35 subjects, leave-one-block-out (mean ± SEM)", fontsize=10)
    axes[1].set_title("ITR (selection time = data length + 0.5 s)", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
