"""Classification, threshold selection and temporal localisation metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def safe_auc(y_true, scores):
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true, scores):
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def best_threshold(y_true, scores, metric="balanced_accuracy"):
    """Threshold maximising `metric` ("balanced_accuracy" or "f1") on (y_true, scores).

    Balanced accuracy suits the clip decision (both error types matter equally).
    F1 suits the per-step segment threshold: fake steps are under 10 % of the
    frames, and balanced accuracy there picks a very low threshold that produces
    many spurious short segments.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    positives = sorted_y.sum()
    negatives = len(sorted_y) - positives
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    tpr = tp / max(positives, 1)
    tnr = 1.0 - fp / max(negatives, 1)
    if metric == "f1":
        balanced = 2 * tp / np.maximum(2 * tp + fp + (positives - tp), 1)
    else:
        balanced = (tpr + tnr) / 2.0
    # only consider cut points between distinct scores
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    balanced = np.where(distinct, balanced, -1.0)
    best = int(np.argmax(balanced))
    nxt = sorted_scores[best + 1] if best + 1 < len(sorted_scores) else sorted_scores[best] - 1e-6
    return float((sorted_scores[best] + nxt) / 2.0)


def classification_report(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= threshold).astype(int)
    if len(y_true) == 0:
        return {}
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": safe_auc(y_true, scores),
        "average_precision": safe_ap(y_true, scores),
        "threshold": float(threshold),
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


# ----------------------------------------------------------------------------
# Temporal localisation
# ----------------------------------------------------------------------------
def timeline_from_windows(starts, ends, probs, duration, resolution=0.05, quantile=0.15):
    """Per-window scores -> dense curve on a grid.

    A window score is P(the window contains a manipulation). If time t really
    is manipulated then *every* window covering t should fire, so the curve at
    t is a low quantile (roughly the second-lowest of the ~8 covering windows)
    of the scores of all windows that contain t. This keeps
    the peak height of a confidently detected segment and gives boundaries
    that are sharp to within the labelling tolerance (min_overlap_seconds).
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    n = max(1, int(np.ceil(duration / resolution)))
    times = (np.arange(n) + 0.5) * resolution
    curve = np.zeros(n)
    if len(starts) == 0:
        return times, curve
    order = np.argsort(starts)
    starts, ends, probs = starts[order], ends[order], probs[order]
    window = float(np.median(ends - starts))
    for i, t in enumerate(times):
        # windows with start <= t < end; starts are sorted and windows have equal length
        hi = np.searchsorted(starts, t, side="right")
        lo = np.searchsorted(starts, t - window, side="right")
        covering = probs[lo:hi]
        covering = covering[(starts[lo:hi] <= t) & (ends[lo:hi] > t)]
        covering = covering[~np.isnan(covering)]
        if covering.size == 0:
            # grid cell before the first / after the last window centre: nearest window
            nearest = np.argmin(np.abs((starts + ends) / 2.0 - t))
            curve[i] = 0.0 if np.isnan(probs[nearest]) else probs[nearest]
        else:
            curve[i] = np.quantile(covering, quantile)
    return times, curve


def segments_from_curve(times, curve, threshold, min_duration=0.1, resolution=None):
    """Contiguous runs above threshold -> [(start, end, score)]."""
    if len(times) == 0:
        return []
    if resolution is None:
        resolution = float(times[1] - times[0]) if len(times) > 1 else 0.05
    above = curve >= threshold
    segments = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        if (not flag or i == len(above) - 1) and start is not None:
            stop = i if not flag else i + 1
            s = times[start] - resolution / 2.0
            e = times[stop - 1] + resolution / 2.0
            if e - s >= min_duration:
                segments.append((float(max(0.0, s)), float(e), float(curve[start:stop].max())))
            start = None
    return segments


def segment_proposals(times, curve, resolution, min_duration=0.1,
                      thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), nms_iou=0.5):
    """Threshold-free proposals: segments at many thresholds, scored by their
    mean curve value, merged with non-maximum suppression."""
    candidates = []
    for threshold in thresholds:
        for s, e, _ in segments_from_curve(times, curve, threshold, min_duration, resolution):
            inside = (times >= s) & (times < e)
            score = float(curve[inside].mean()) if inside.any() else float(threshold)
            candidates.append((s, e, score))
    candidates.sort(key=lambda item: -item[2])
    kept = []
    for cand in candidates:
        if all(segment_iou(cand, k) < nms_iou for k in kept):
            kept.append(cand)
    return kept


def segment_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def average_precision_at_iou(predictions, ground_truth, iou_threshold):
    """predictions: {video: [(s, e, score)]}, ground_truth: {video: [(s, e)]}."""
    total_gt = sum(len(v) for v in ground_truth.values())
    if total_gt == 0:
        return float("nan")
    flat = [(score, video, (s, e)) for video, segs in predictions.items() for s, e, score in segs]
    flat.sort(key=lambda item: -item[0])
    matched = {video: np.zeros(len(segs), dtype=bool) for video, segs in ground_truth.items()}
    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))
    for i, (_, video, seg) in enumerate(flat):
        gts = ground_truth.get(video, [])
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[video][j]:
                continue
            iou = segment_iou(seg, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched[video][best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1
    if len(flat) == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    # interpolated AP (area under the precision envelope)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * envelope))


def frame_level_scores(curve_times, curve, periods):
    """Binary ground truth on the curve grid for frame-level AUC."""
    truth = np.zeros(len(curve_times), dtype=int)
    for s, e in periods:
        truth[(curve_times >= s) & (curve_times < e)] = 1
    return truth
