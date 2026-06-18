from vision_server.features import flatten_landmarks
from conftest import make_landmark, make_landmarks


def test_flatten_landmarks_returns_63_values():
    landmarks = [make_landmark(x=i * 0.01, y=i * 0.02, z=i * 0.03) for i in range(21)]
    result = flatten_landmarks(landmarks)
    assert len(result) == 63


def test_flatten_landmarks_order():
    landmarks = [make_landmark(x=1.0, y=2.0, z=3.0)] + [
        make_landmark() for _ in range(20)
    ]
    result = flatten_landmarks(landmarks)
    assert result[:3] == [1.0, 2.0, 3.0]


def test_flatten_landmarks_matches_manual_loop():
    landmarks = make_landmarks()
    manual = []
    for lm in landmarks:
        manual.extend([lm.x, lm.y, lm.z])
    assert flatten_landmarks(landmarks) == manual
