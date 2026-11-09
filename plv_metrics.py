"""PLV edge metrics; undefined correlations stay NaN."""
import numpy as np

def corr_rows(a, b, eps=1e-8):
    a, b = a-a.mean(-1, keepdims=True), b-b.mean(-1, keepdims=True)
    den = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    return np.divide((a*b).sum(-1), den, out=np.full(den.shape, np.nan), where=den>eps)


def fisher_mean(r):
    r = np.asarray(r)
    return float(np.tanh(np.arctanh(np.clip(r[np.isfinite(r)], -1+1e-7, 1-1e-7)).mean())) if np.isfinite(r).any() else float('nan')


def metrics(y, pred, template):
    """Inputs [windows, edges]; correlations with zero variance remain undefined."""
    error = pred-y
    mse = float(np.mean(error**2))
    base = float(np.mean((template-y)**2))
    spatial = corr_rows(y, pred)
    temporal = corr_rows(y.T, pred.T)
    out = dict(mae=float(np.abs(error).mean()), mse=mse, template_mse=base,
               delta_mse=base-mse, skill=1-mse/base if base>1e-12 else np.nan,
               spatial_r=fisher_mean(spatial), temporal_r=fisher_mean(temporal),
               spatial_valid=float(np.isfinite(spatial).mean()),
               temporal_valid=float(np.isfinite(temporal).mean()))
    for q in (.25, .5):
        k = int(np.ceil(y.shape[-1]*q))
        # Stable sorting resolves ties using ascending original edge index.
        gi = np.argsort(-y, axis=-1, kind='stable')[:, :k]
        hi = np.argsort(-pred, axis=-1, kind='stable')[:, :k]
        out[f'f1_{q}'] = float(np.mean([len(np.intersect1d(g,h))/k for g,h in zip(gi,hi)]))
    return out

