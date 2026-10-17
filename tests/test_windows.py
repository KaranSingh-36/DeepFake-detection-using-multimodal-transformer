import numpy as np

from fusiontrace.windows import (
    VideoMeta,
    enumerate_window_starts,
    parse_periods,
    window_class,
    window_labels,
    window_state,
)


def meta(video=True, audio=False, periods=((2.0, 2.7),), duration=6.0):
    return VideoMeta("x.mp4", "train", duration, video, audio, list(periods), "y.mp4")


def test_window_state_rules():
    periods = [(2.0, 2.7)]
    assert window_state(0.0, 1.6, periods) == "negative"
    assert window_state(1.0, 2.6, periods) == "positive"          # 0.6 s overlap
    assert window_state(2.6, 4.2, periods) == "ambiguous"         # 0.1 s < min(0.2, 0.35)
    assert window_state(1.9, 3.5, periods) == "positive"          # fully inside
    short = [(2.0, 2.2)]                                          # needed = min(0.2, 0.1) = 0.1
    assert window_state(1.9, 2.1, short) == "positive"


def test_modality_labels_follow_file_flags():
    audio_only = meta(video=False, audio=True)
    labels = window_labels(audio_only, 1.5, 3.1)
    assert labels == {"video": 0, "audio": 1, "any": 1, "ambiguous": False}
    assert window_class(labels) == 2

    both = meta(video=True, audio=True)
    assert window_class(window_labels(both, 1.5, 3.1)) == 3

    clean_window = window_labels(both, 3.5, 5.1)
    assert clean_window["any"] == 0 and not clean_window["ambiguous"]


def test_enumerate_window_starts_covers_clip_end():
    starts = enumerate_window_starts(4.0, 1.6, 0.4)
    assert starts[0] == 0.0
    assert np.isclose(starts[-1] + 1.6, 4.0, atol=1e-3)
    assert np.all(np.diff(starts) > 0)
    assert len(enumerate_window_starts(1.0, 1.6, 0.4)) == 1
    starts = enumerate_window_starts(4.1, 1.6, 0.5)
    assert np.isclose(starts[-1], 2.5)


def test_parse_periods_handles_strings_and_nan():
    assert parse_periods("[[2.5, 3.224]]") == [(2.5, 3.224)]
    assert parse_periods("[]") == []
    assert parse_periods(float("nan")) == []
    assert parse_periods([[1, 2], [3, 3]]) == [(1.0, 2.0)]
