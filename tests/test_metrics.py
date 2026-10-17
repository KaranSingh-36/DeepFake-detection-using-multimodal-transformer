import numpy as np

from fusiontrace.metrics import (
    average_precision_at_iou,
    best_threshold,
    segments_from_curve,
    timeline_from_windows,
)


def test_best_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = best_threshold(y, s)
    assert 0.3 < thr < 0.7


def test_timeline_and_segments_recover_a_fake_period():
    starts = np.arange(0, 6.0 - 1.6 + 1e-6, 0.2)
    ends = starts + 1.6
    # a model that fires whenever the window overlaps the fake period [2, 3] by >= 0.2 s
    overlap = np.minimum(ends, 3.0) - np.maximum(starts, 2.0)
    probs = np.where(overlap >= 0.2, 0.9, 0.05)
    times, curve = timeline_from_windows(starts, ends, probs, 6.0, 0.05)
    segments = segments_from_curve(times, curve, 0.5, 0.2, 0.05)
    assert len(segments) == 1
    s, e, score = segments[0]
    assert 1.7 < s < 2.3 and 2.7 < e < 3.3 and score > 0.8


def test_average_precision_at_iou():
    gt = {"a": [(1.0, 2.0)], "b": [(0.0, 1.0)]}
    perfect = {"a": [(1.0, 2.0, 0.9)], "b": [(0.0, 1.0, 0.8)]}
    assert average_precision_at_iou(perfect, gt, 0.5) == 1.0
    half = {"a": [(1.0, 2.0, 0.9)], "b": [(5.0, 6.0, 0.95)]}
    ap = average_precision_at_iou(half, gt, 0.5)
    assert 0.2 < ap < 0.6
    assert average_precision_at_iou({"a": [], "b": []}, gt, 0.5) == 0.0
