"""Calibration-based SSVEP detectors (use the subject's own training blocks).

  * eCCA  - extended CCA with individual templates (Chen et al., 2015, PNAS):
            rho_k = sum_i sign(r_i) r_i^2 over four correlations, summed over a filter bank.
  * eTRCA - ensemble task-related component analysis (Nakanishi et al., 2018, IEEE TBME).

Inputs are filter-bank, windowed trials: (n_trials, n_bands, n_channels, n_times).
"""
import numpy as np
from scipy.linalg import eigh


def _center_t(x):
    """(..., C, T) -> centered (..., T, C)."""
    xt = np.swapaxes(x, -1, -2)
    return xt - xt.mean(axis=-2, keepdims=True)


def _pearson_rows(a, b):
    """Pearson correlation between matching rows of a and b (..., T)."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return (a * b).sum(-1) / np.where(den == 0, 1.0, den)


def _first_x_weight(qx, rx, qy):
    """X-side canonical weight(s) for X (qx, rx) against one or many Y bases qy.

    qx: (T, C), rx: (C, C), qy: (K, T, R)  ->  wx: (C, K), r: (K,)
    """
    m = np.einsum("tc,ktr->kcr", qx, qy)
    u, s, _ = np.linalg.svd(m)
    wx = np.linalg.solve(rx, u[:, :, 0].T)
    return wx, s[:, 0]


class ECCA:
    def __init__(self, q_refs, band_weights):
        self.q_refs = q_refs            # (K, T, 2Nh), orthonormal reference bases
        self.w = band_weights

    def fit(self, X, y):
        n_bands = X.shape[1]
        self.classes_ = np.unique(y)
        self.tmpl_, self.qt_, self.rt_, self.w_ty_ = [], [], [], []
        for b in range(n_bands):
            tmpl = np.stack([X[y == k, b].mean(axis=0) for k in self.classes_])   # (K, C, T)
            tt = _center_t(tmpl)                                                   # (K, T, C)
            qt, rt = np.linalg.qr(tt)
            # X-side weights of CCA(template_k, Y_k), one per class
            m = np.einsum("ktc,ktr->kcr", qt, self.q_refs)
            u = np.linalg.svd(m)[0][:, :, 0]                                       # (K, C)
            w_ty = np.linalg.solve(rt, u[..., None])[..., 0]                       # (K, C)
            self.tmpl_.append(tt); self.qt_.append(qt); self.rt_.append(rt); self.w_ty_.append(w_ty)
        return self

    def decision_function(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for b, wb in enumerate(self.w):
            tt, qt, w_ty = self.tmpl_[b], self.qt_[b], self.w_ty_[b]
            for i in range(X.shape[0]):
                xt = _center_t(X[i, b])                                            # (T, C)
                qx, rx = np.linalg.qr(xt)
                w_xy, r1 = _first_x_weight(qx, rx, self.q_refs)                    # (C, K), (K,)
                w_xt, _ = _first_x_weight(qx, rx, qt)                              # CCA(X, template_k)
                r2 = self._proj_corr(xt, tt, w_xy)
                r3 = self._proj_corr(xt, tt, w_xt)
                r4 = self._proj_corr(xt, tt, w_ty.T)
                rho = sum(np.sign(r) * r ** 2 for r in (r1, r2, r3, r4))
                scores[i] += wb * rho
        return scores

    @staticmethod
    def _proj_corr(xt, tt, w):
        """corr(X w_k, template_k w_k) for every class k. xt: (T, C), tt: (K, T, C), w: (C, K)."""
        return _pearson_rows((xt @ w).T, np.einsum("ktc,ck->kt", tt, w))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def _trca_filter(trials):
    """Leading TRCA spatial filter for trials (n_trials, C, T)."""
    x = trials - trials.mean(axis=-1, keepdims=True)
    s_sum = x.sum(axis=0)
    q = np.einsum("nct,ndt->cd", x, x)
    s = s_sum @ s_sum.T - q
    _, vecs = eigh(s, q)
    return vecs[:, -1]


class EnsembleTRCA:
    def __init__(self, band_weights):
        self.w = band_weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.W_, self.tmpl_proj_ = [], []
        for b in range(X.shape[1]):
            W = np.stack([_trca_filter(X[y == k, b]) for k in self.classes_], axis=1)   # (C, K)
            tmpl = np.stack([X[y == k, b].mean(axis=0) for k in self.classes_])       # (K, C, T)
            tp = np.einsum("kct,cj->ktj", tmpl, W).reshape(len(self.classes_), -1)
            tp -= tp.mean(axis=1, keepdims=True)
            tp /= np.linalg.norm(tp, axis=1, keepdims=True)
            self.W_.append(W); self.tmpl_proj_.append(tp)
        return self

    def decision_function(self, X):
        scores = 0.0
        for b, wb in enumerate(self.w):
            xp = np.einsum("nct,cj->ntj", X[:, b], self.W_[b]).reshape(X.shape[0], -1)
            xp -= xp.mean(axis=1, keepdims=True)
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            scores = scores + wb * (xp @ self.tmpl_proj_[b].T)
        return scores

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
