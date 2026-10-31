"""Evaluation protocols, metrics and information transfer rate.

Protocols
  holdout : random 80/20 split, random_state=42            (original, used for Table 3.2)
  kfold   : random 10-fold CV, shuffle, random_state=42     (original, used for the 82.52 %)
  subject : GroupKFold over subjects -> tested on unseen subjects        (corrected)
  block   : leave-one-block-out, single-trial pipeline B only             (corrected)
"""
import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, auc, average_precision_score, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import label_binarize

import config


def get_splits(protocol, y, groups=None, blocks=None, n_group_folds=config.GROUP_FOLDS, seed=config.SEED):
    idx = np.arange(len(y))
    if protocol == "holdout":
        tr, te = train_test_split(idx, test_size=config.TEST_SIZE, random_state=seed)
        return [(tr, te)]
    if protocol == "kfold":
        return list(KFold(n_splits=config.KFOLD_SPLITS, shuffle=True, random_state=seed).split(idx))
    if protocol == "subject":
        n = min(n_group_folds, len(np.unique(groups)))
        return list(GroupKFold(n_splits=n).split(idx, y, groups))
    if protocol == "block":
        if blocks is None:
            raise ValueError("Leave-one-block-out needs single-trial data: use --pipeline B")
        return list(LeaveOneGroupOut().split(idx, y, blocks))
    raise ValueError(f"Unknown protocol '{protocol}'")


def _scores(model, X, n_classes):
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
    else:
        raw = model.decision_function(X)
    classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])), dtype=int)
    full = np.full((X.shape[0], n_classes), raw.min() - 1.0)
    full[:, classes] = raw
    return full


def run_model(model, X, y, splits, n_classes=config.N_TARGETS, name=""):
    """Fit/predict over the given splits; returns out-of-fold predictions and the last fitted model."""
    y_true, y_pred, scores, fold_acc = [], [], [], []
    fitted = None
    for k, (tr, te) in enumerate(splits, 1):
        t0 = time.time()
        fitted = clone(model) if hasattr(model, "get_params") else model.__class__()
        fitted.fit(X[tr], y[tr])
        pred = np.asarray(fitted.predict(X[te])).astype(int)
        fold_acc.append(accuracy_score(y[te], pred))
        y_true.append(y[te]); y_pred.append(pred); scores.append(_scores(fitted, X[te], n_classes))
        print(f"  {name} fold {k}/{len(splits)}: acc={fold_acc[-1]:.4f} ({time.time() - t0:.1f}s)")
    return dict(y_true=np.concatenate(y_true), y_pred=np.concatenate(y_pred),
                scores=np.concatenate(scores), fold_acc=np.array(fold_acc), model=fitted)


def micro_roc_pr(y_true, scores, n_classes=config.N_TARGETS):
    Y = label_binarize(y_true, classes=np.arange(n_classes))
    fpr, tpr, _ = roc_curve(Y.ravel(), scores.ravel())
    return dict(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr),
                ap=average_precision_score(Y, scores, average="micro"))


def itr_bits_per_min(n_classes, p, seconds_per_selection):
    """Wolpaw ITR. p is accuracy in [0, 1]."""
    if p <= 1.0 / n_classes:
        return 0.0
    bits = np.log2(n_classes) + p * np.log2(p)
    if p < 1.0:
        bits += (1 - p) * np.log2((1 - p) / (n_classes - 1))
    return bits * 60.0 / seconds_per_selection


def selection_time(pipeline):
    """Seconds of data (+ gaze shift) needed for ONE decision in each pipeline."""
    if pipeline == "A":   # 6 blocks x 4 s window, each preceded by a 0.5 s cue
        w = config.PIPELINE_A["window_s"]
        return config.N_BLOCKS * ((w[1] - w[0]) + config.GAZE_SHIFT_S)
    w = config.PIPELINE_B["window_s"]
    return (w[1] - w[0]) + config.GAZE_SHIFT_S


def summarize(label, res, pipeline, protocol):
    acc = accuracy_score(res["y_true"], res["y_pred"])
    row = dict(model=label, pipeline=pipeline, protocol=protocol,
               accuracy=acc, fold_mean=res["fold_acc"].mean(), fold_std=res["fold_acc"].std(),
               n_test=len(res["y_true"]),
               itr_bits_min=itr_bits_per_min(config.N_TARGETS, acc, selection_time(pipeline)))
    if res.get("scores") is not None:
        rp = micro_roc_pr(res["y_true"], res["scores"])
        row.update(roc_auc_micro=rp["roc_auc"], ap_micro=rp["ap"])
    return row


def training_free_result(y_true, y_pred):
    """Package fixed predictions (baselines) in the same format as run_model."""
    acc = accuracy_score(y_true, y_pred)
    return dict(y_true=y_true, y_pred=y_pred, scores=None, fold_acc=np.array([acc]), model=None)


def confusion(y_true, y_pred, n_classes=config.N_TARGETS):
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
