"""Presentation figures (regenerated from the pipeline instead of reused from the report)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from evaluate import micro_roc_pr

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


def confusion_matrix_plot(cm, stim_freqs, title, path):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ticks = np.arange(0, len(stim_freqs), 4)
    labels = [f"{stim_freqs[i]:.1f}" for i in ticks]
    ax.set_xticks(ticks, labels, rotation=90); ax.set_yticks(ticks, labels)
    ax.set_xlabel("Predicted target (Hz)"); ax.set_ylabel("True target (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised rate")
    _save(fig, path)


def roc_plot(results, path, title="ROC (micro-average, one-vs-rest)"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, res in results.items():
        if res.get("scores") is None:
            continue
        rp = micro_roc_pr(res["y_true"], res["scores"])
        ax.plot(rp["fpr"], rp["tpr"], label=f"{name} (AUC = {rp['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def pr_plot(results, path, title="Precision-recall (micro-average)"):
    from sklearn.metrics import precision_recall_curve
    from sklearn.preprocessing import label_binarize
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, res in results.items():
        if res.get("scores") is None:
            continue
        Y = label_binarize(res["y_true"], classes=np.arange(config.N_TARGETS))
        p, r, _ = precision_recall_curve(Y.ravel(), res["scores"].ravel())
        ap = micro_roc_pr(res["y_true"], res["scores"])["ap"]
        ax.plot(r, p, label=f"{name} (AP = {ap:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def feature_importance_plot(model, feature_names, title, path):
    est = getattr(model, "best_estimator_", model)
    if not hasattr(est, "feature_importances_"):
        return
    imp = est.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(range(len(imp)), imp[order], color="#3b6ea5")
    ax.set_xticks(range(len(imp)), [feature_names[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("Impurity-based importance"); ax.set_title(title)
    _save(fig, path)


def accuracy_bar_plot(rows, path, title):
    rows = sorted(rows, key=lambda r: r["accuracy"])
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(rows) + 1.2))
    names = [r["model"] for r in rows]
    accs = [100 * r["accuracy"] for r in rows]
    colors = ["#9aa5b1" if r.get("training_free") else "#3b6ea5" for r in rows]
    ax.barh(names, accs, color=colors)
    for i, a in enumerate(accs):
        ax.text(a + 0.5, i, f"{a:.1f}%", va="center", fontsize=9)
    ax.axvline(100 / config.N_TARGETS, color="k", ls=":", lw=1, label="Chance (2.5%)")
    ax.set_xlim(0, 105); ax.set_xlabel("Accuracy (%)"); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def example_spectra_plot(x9, fs, stim_freq, weights, path):
    """Amplitude spectrum of one trial after mean / weighted / median channel combination."""
    combos = {"Mean": x9.mean(axis=0),
              "Weighted mean": np.average(x9, axis=0, weights=np.asarray(weights) / np.sum(weights)),
              "Median": np.median(x9, axis=0)}
    freqs = np.fft.rfftfreq(x9.shape[-1], d=1.0 / fs)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, w in combos.items():
        ax.plot(freqs, np.abs(np.fft.rfft(w)), lw=1, label=name)
    for h in (1, 2, 3):
        ax.axvline(h * stim_freq, color="k", ls=":", lw=0.8)
    ax.set_xlim(0, 50); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Amplitude")
    ax.set_title(f"Single trial, {stim_freq:.1f} Hz target (dotted: fundamental and harmonics)")
    ax.legend(fontsize=8)
    _save(fig, path)
