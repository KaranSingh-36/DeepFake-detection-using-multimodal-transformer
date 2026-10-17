import numpy as np

from fusiontrace.faces import crop_and_resize, default_mouth_box, stable_mouth_box


def test_mouth_box_falls_back_to_lower_centre_without_a_face():
    frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(5)]
    box = stable_mouth_box(frames)
    assert box == default_mouth_box(224, 224) == (45, 112, 179, 224)
    crop = crop_and_resize(frames, box, 224)
    assert crop.shape == (5, 224, 224, 3)


def test_mouth_box_stays_inside_the_frame_and_is_square():
    frames = [np.zeros((120, 200, 3), dtype=np.uint8)]
    x0, y0, x1, y1 = stable_mouth_box(frames)
    assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 120
