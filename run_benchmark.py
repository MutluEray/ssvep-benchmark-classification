"""Within-subject leave-one-block-out benchmark: accuracy and ITR vs data length.

This is the standard protocol on the Tsinghua benchmark (train on 5 blocks of a subject,
test on the 6th, rotate). Methods:
  fbcca  - training-free FBCCA (Chen et al., 2015, JNE)
  ecca   - extended CCA with individual templates (Chen et al., 2015, PNAS)
  trca   - ensemble TRCA (Nakanishi et al., 2018)

Examples
  python3 run_benchmark.py --n-subjects 3 --test-blocks 1     # quick check
  python3 run_benchmark.py                                    # all methods, 0.3-2.0 s, all 35 subjects
  python3 run_benchmark.py --methods trca --windows 0.5,1.0
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import baselines
import calibrated
import config
import data
import evaluate
import filterbank as fb
import preprocessing as pp


def parse_args():
    p = argparse.ArgumentParser(description="Leave-one-block-out SSVEP benchmark")
    p.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    p.add_argument("--methods", default="fbcca,ecca,trca")
    p.add_argument("--windows", default=",".join(str(w) for w in config.BENCHMARK["windows_s"]),
                   help="data lengths in seconds after the 0.14 s latency")
    p.add_argument("--n-subjects", type=int, default=None)
    p.add_argument("--test-blocks", default="1,2,3,4,5,6", help="1-based blocks to test (all = 6-fold)")
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "benchmark")
    return p.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    windows = [float(w) for w in args.windows.split(",")]
    test_blocks = [int(b) - 1 for b in args.test_blocks.split(",")]
    subjects = data.subject_list(args.n_subjects)
    data.check_data_dir(args.data_dir, subjects)
    stim_freqs, _ = data.load_freq_phase(args.data_dir)
    idx = data.channel_indices(config.SSVEP_CHANNELS)
    targets, blocks = data.trial_labels()
    n_bands = config.BENCHMARK["n_subbands"]
    weights = fb.band_weights(n_bands)
    q_refs = {w: baselines.harmonic_references(stim_freqs, int(round(w * config.FS)), config.FS,
                                               config.BENCHMARK["n_harmonics"]) for w in windows}
    unknown = set(methods) - {"fbcca", "ecca", "trca"}
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(sorted(unknown))}. Choose from fbcca, ecca, trca.")
    rows = []                                    # one row per subject x window x method

    for s in subjects:
        x = pp.demean(data.load_subject(args.data_dir, s)[:, idx, :])
        xb_full = fb.filter_bank(x, config.FS, n_bands)                        # (240, B, 9, 1500)
        for w in windows:
            xw = fb.window(xb_full, config.FS, w)
            line = [f"S{s:02d} {w:.2f}s"]
            for m in methods:
                correct = []
                if m == "fbcca":
                    pred = np.argmax(baselines.fbcca_scores(xw[:, :n_bands], q_refs[w], weights), axis=1)
                    mask = np.isin(blocks, test_blocks)
                    correct = pred[mask] == targets[mask]
                else:
                    for tb in test_blocks:
                        tr, te = blocks != tb, blocks == tb
                        model = (calibrated.ECCA(q_refs[w], weights) if m == "ecca"
                                 else calibrated.EnsembleTRCA(weights))
                        model.fit(xw[tr, :n_bands], targets[tr])
                        correct = np.concatenate([correct, model.predict(xw[te, :n_bands]) == targets[te]])
                acc = float(np.mean(correct))
                rows.append(dict(subject=s, window_s=w, method=m, accuracy=acc,
                                 itr=evaluate.itr_bits_per_min(config.N_TARGETS, acc, w + config.GAZE_SHIFT_S)))
                line.append(f"{m} {100 * acc:5.1f}%")
            print("  ".join(line))

    # ---- save per-subject rows and the summary
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_subject.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["subject", "window_s", "method", "accuracy", "itr"])
        wr.writeheader()
        wr.writerows(rows)
    summary = []
    for m in dict.fromkeys(r["method"] for r in rows):
        for w in windows:
            sel = [r for r in rows if r["method"] == m and r["window_s"] == w]
            if not sel:
                continue
            a = np.array([r["accuracy"] for r in sel]); i = np.array([r["itr"] for r in sel])
            summary.append(dict(method=m, window_s=w, acc_mean=a.mean(), acc_sem=a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0,
                                itr_mean=i.mean(), itr_sem=i.std(ddof=1) / np.sqrt(len(i)) if len(i) > 1 else 0.0,
                                n_subjects=len(a)))
    with open(args.out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        for r in summary:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n{'method':<12}{'window':>8}{'acc %':>10}{'± sem':>8}{'ITR':>9}")
    for r in summary:
        print(f"{r['method']:<12}{r['window_s']:>7.2f}s{100 * r['acc_mean']:>10.2f}{100 * r['acc_sem']:>8.2f}{r['itr_mean']:>9.1f}")
    plot_curves(summary, args.out / "accuracy_itr_vs_window.png")
    print(f"\nSaved {args.out / 'summary.csv'}, per_subject.csv and the figure.")


def plot_curves(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in dict.fromkeys(r["method"] for r in summary):
        sel = [r for r in summary if r["method"] == m]
        w = [r["window_s"] for r in sel]
        axes[0].errorbar(w, [100 * r["acc_mean"] for r in sel], yerr=[100 * r["acc_sem"] for r in sel],
                         marker="o", capsize=3, label=m)
        axes[1].errorbar(w, [r["itr_mean"] for r in sel], yerr=[r["itr_sem"] for r in sel],
                         marker="o", capsize=3, label=m)
    axes[0].set_ylabel("Accuracy (%)"); axes[1].set_ylabel("ITR (bits/min)")
    for ax in axes:
        ax.set_xlabel("Data length (s)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("Leave-one-block-out accuracy (mean ± SEM)")
    axes[1].set_title("ITR (selection time = data length + 0.5 s)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
