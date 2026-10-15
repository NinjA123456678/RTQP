from __future__ import annotations

import numpy as np


def binary_auc(labels, scores):
    """Tie-aware AUROC and trapezoidal PR-AUC; no score thresholds fitted on test data."""
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=np.float64)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return {"auroc": None, "auprc": None, "reason": "both classes are required"}
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(sorted_labels)[ends].astype(np.float64)
    fp = (ends + 1) - tp
    tpr, fpr = np.r_[0, tp / positives], np.r_[0, fp / negatives]
    precision, recall = np.r_[1, tp / (tp + fp)], np.r_[0, tp / positives]
    return {"auroc": float(np.trapz(tpr, fpr)), "auprc": float(np.trapz(precision, recall))}


def summarize_injection(prepared, window):
    historical = sorted({t - delta for t in prepared["times"]["test"] for delta in range(1, window + 1)})
    selected = prepared["mask"][historical]
    labels = prepared["injected"][historical][selected]
    scores = prepared["features"]["scores"][historical][selected]
    confidence = prepared["features"]["confidence"][historical][selected]
    result = {"time_slices": historical, "unique_observed_inputs": int(selected.sum()),
              "injected_count": int(labels.sum()), "std_ddof": 0, **binary_auc(labels, scores)}
    for name, subset in (("unperturbed", ~labels), ("injected", labels)):
        result[name] = {"count": int(subset.sum()),
                        "score_mean": float(scores[subset].mean()) if subset.any() else None,
                        "score_std": float(scores[subset].std()) if subset.any() else None,
                        "confidence_mean": float(confidence[subset].mean()) if subset.any() else None,
                        "confidence_std": float(confidence[subset].std()) if subset.any() else None}
    return result
