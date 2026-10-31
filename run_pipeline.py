"""Entry point: build features, train/evaluate classifiers, write metrics and figures.

Examples
  Reproduce the report (Table 3.2 and the 82.5 % 10-fold result, pipeline A):
    python3 run_pipeline.py --pipeline A --protocol holdout
    python3 run_pipeline.py --pipeline A --protocol kfold --models et
  Re-analysis:
    python3 run_pipeline.py --pipeline A --protocol subject --channels corrected
    python3 run_pipeline.py --pipeline B --protocol block
    python3 run_pipeline.py --pipeline B --protocol subject
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import baselines
import config
import data
import evaluate
import models
import pipelines
import plots
import preprocessing as pp

DEFAULT_MODELS = {"A": "svm,dt,rf,gb,et", "B": "svm,dt,rf,gb,et"}


def parse_args():
    p = argparse.ArgumentParser(description="SSVEP benchmark: feature pipelines + classifier comparison")
    p.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    p.add_argument("--pipeline", choices=["A", "B"], default="A",
                   help="A = original multi-block features (project.ipynb); B = single-trial sub-band features (project_3.ipynb)")
    p.add_argument("--protocol", choices=["holdout", "kfold", "subject", "block"], default="holdout")
    p.add_argument("--models", default=None, help="comma list from: svm,dt,rf,gb,et,xgb,nn")
    p.add_argument("--channels", choices=["original", "corrected"], default="original",
                   help="pipeline A only: 'original' reproduces the off-by-one channel indices")
    p.add_argument("--n-subjects", type=int, default=None, help="use subjects 1..N (quick runs)")
    p.add_argument("--group-folds", type=int, default=config.GROUP_FOLDS)
    p.add_argument("--tune", action="store_true", help="grid-search Extra Trees inside each training fold")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    subjects = data.subject_list(args.n_subjects)
    data.check_data_dir(args.data_dir, subjects)
    stim_freqs, _ = data.load_freq_phase(args.data_dir)
    use_cache = not args.no_cache

    if args.pipeline == "A":
        D = pipelines.build_pipeline_a(args.data_dir, subjects, args.channels, use_cache)
        feature_names = [f"Block {i + 1}" for i in range(config.N_BLOCKS)]
        blocks = None
    else:
        D = pipelines.build_pipeline_b(args.data_dir, subjects, use_cache)
        feature_names = [f"{lo}-{hi} Hz" for lo, hi in config.PIPELINE_B["subbands"]]
        blocks = D["blocks"]

    tag = f"{args.pipeline}_{args.protocol}"
    if args.pipeline == "A" and args.channels == "corrected":
        tag += "_corrected-channels"
    if args.tune:
        tag += "_tuned"
    if args.n_subjects:
        tag += f"_n{args.n_subjects}"
    out_dir = args.out / tag
    fig_dir = out_dir / "figures"

    y, groups = D["y"], D["groups"]
    splits = evaluate.get_splits(args.protocol, y, groups, blocks, args.group_folds)
    print(f"\nPipeline {args.pipeline} | protocol {args.protocol} | {len(y)} samples | {len(splits)} split(s)")

    results, rows = {}, []
    for name in (args.models or DEFAULT_MODELS[args.pipeline]).split(","):
        name = name.strip()
        try:
            model, key = models.get_model(name, args.pipeline, args.tune)
        except ImportError as e:
            print(f"Skipping {name}: {e}")
            continue
        label = models.DISPLAY[name]
        print(f"\n== {label} ==")
        res = evaluate.run_model(model, D[key], y, splits, name=label)
        results[label] = res
        rows.append(evaluate.summarize(label, res, args.pipeline, args.protocol))

    # Training-free baselines (no fitting, so evaluated on all samples)
    if args.pipeline == "A":
        free = {"PSDA 6-block vote (no training)": baselines.psda_block_vote(D["X"], stim_freqs)}
    else:
        free = {"PSDA rule (no training)": D["psda_pred"], "CCA (no training)": D["cca_pred"],
                "FBCCA (no training)": D["fbcca_pred"]}
    for label, pred in free.items():
        res = evaluate.training_free_result(y, pred)
        results[label] = res
        row = evaluate.summarize(label, res, args.pipeline, "training-free")
        row["training_free"] = True
        rows.append(row)

    # ---- metrics table
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "pipeline", "protocol", "accuracy", "fold_mean", "fold_std", "n_test",
              "itr_bits_min", "roc_auc_micro", "ap_micro"]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n{'Model':<34}{'Acc %':>8}{'Fold mean±sd':>16}{'ITR':>9}{'AUC':>7}")
    for r in rows:
        auc_s = f"{r['roc_auc_micro']:.3f}" if "roc_auc_micro" in r else "-"
        print(f"{r['model']:<34}{100 * r['accuracy']:>8.2f}"
              f"{100 * r['fold_mean']:>9.2f}±{100 * r['fold_std']:<5.2f}{r['itr_bits_min']:>9.1f}{auc_s:>7}")
    print(f"\nITR assumes {evaluate.selection_time(args.pipeline):.1f} s per selection. Metrics: {out_dir / 'metrics.csv'}")

    if args.no_figures:
        return

    # ---- figures
    trained = {k: v for k, v in results.items() if v.get("scores") is not None}
    title_sfx = f"pipeline {args.pipeline}, {args.protocol}"
    plots.accuracy_bar_plot(rows, fig_dir / "accuracy.png", f"Accuracy ({title_sfx})")
    if trained:
        plots.roc_plot(trained, fig_dir / "roc.png")
        plots.pr_plot(trained, fig_dir / "precision_recall.png")
        best = max(trained, key=lambda k: np.mean(trained[k]["y_true"] == trained[k]["y_pred"]))
        cm = evaluate.confusion(trained[best]["y_true"], trained[best]["y_pred"])
        plots.confusion_matrix_plot(cm, stim_freqs, f"{best} ({title_sfx})", fig_dir / "confusion_best.png")
        for label in ("Extra Trees", "Gradient Boosting"):
            if label in trained:
                plots.feature_importance_plot(trained[label]["model"], feature_names,
                                              f"{label} feature importance",
                                              fig_dir / f"importance_{label.lower().replace(' ', '_')}.png")
    if "FBCCA (no training)" in results:
        cm = evaluate.confusion(y, results["FBCCA (no training)"]["y_pred"])
        plots.confusion_matrix_plot(cm, stim_freqs, "FBCCA (no training)", fig_dir / "confusion_fbcca.png")

    # Illustrative spectrum: subject 1, target 1 (8 Hz), block 1, pipeline-B filtering
    idx = data.channel_indices(config.SSVEP_CHANNELS)
    x = pp.demean(data.load_subject(args.data_dir, 1)[:1, idx, :])
    x = pp.clip(pp.fir_bandpass(x, config.FS, *config.PIPELINE_B["band"]), config.FS, config.PIPELINE_B["window_s"])
    plots.example_spectra_plot(x[0], config.FS, stim_freqs[0], config.PIPELINE_B["weights"],
                               fig_dir / "example_spectrum.png")


if __name__ == "__main__":
    main()
