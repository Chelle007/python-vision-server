from types import SimpleNamespace

import pytest

from vision_server.features import flatten_landmarks
from vision_server.gestures.hand.geometry import get_hand_rotation
from vision_server.hand_roles import HandRoles
from vision_server.udp import default_payload

from conftest import make_landmarks


def _lock(left="L", right="R"):
    return SimpleNamespace(left=left, right=right)


def test_defaults_to_right_action_hand():
    roles = HandRoles()
    assert roles.action_hand == "right"
    assert roles.move_hand == "left"
    assert roles.mirror_action_hand is False


def test_swap_flips_both_roles():
    roles = HandRoles()
    assert roles.swap() is True
    assert roles.action_hand == "left"
    assert roles.move_hand == "right"
    assert roles.source == "keyboard"


def test_set_action_hand_is_idempotent():
    roles = HandRoles()
    # Unity resends the setting every packet; a repeat must not report a change.
    assert roles.set_action_hand("right", source="unity") is False
    assert roles.set_action_hand("left", source="unity") is True


def test_set_action_hand_rejects_garbage():
    roles = HandRoles()
    with pytest.raises(ValueError):
        roles.set_action_hand("middle")
    assert roles.action_hand == "right"


def test_split_returns_move_then_action():
    roles = HandRoles()
    assert roles.split(_lock()) == ("L", "R")
    roles.swap()
    assert roles.split(_lock()) == ("R", "L")


def test_mirror_only_applies_to_a_left_action_hand():
    roles = HandRoles()
    assert roles.mirror_action_hand is False
    roles.swap()
    assert roles.mirror_action_hand is True


def test_mirror_can_be_disabled():
    roles = HandRoles("left", mirror_left=False)
    assert roles.mirror_action_hand is False


def test_apply_to_payload_reports_mapping():
    roles = HandRoles()
    roles.swap(source="unity")
    data = default_payload()
    roles.apply_to_payload(data)
    assert data["action_hand"] == "left"
    assert data["move_hand"] == "right"
    assert data["hand_roles_source"] == "unity"


def test_flatten_landmarks_mirror_reflects_x_only():
    landmarks = make_landmarks({0: {"x": 0.2, "y": 0.3, "z": 0.4}})

    plain = flatten_landmarks(landmarks)
    mirrored = flatten_landmarks(landmarks, mirror=True)

    assert len(mirrored) == len(plain) == 63
    assert mirrored[0] == pytest.approx(0.8)
    assert mirrored[1] == pytest.approx(0.3)
    assert mirrored[2] == pytest.approx(0.4)


def test_mirrored_left_hand_matches_the_real_right_hand():
    """A left hand mirrored in x must produce the vector its reflection would."""
    left = make_landmarks({0: {"x": 0.2, "y": 0.4}, 5: {"x": 0.35, "y": 0.6}})
    reflected_right = make_landmarks(
        {0: {"x": 0.8, "y": 0.4}, 5: {"x": 0.65, "y": 0.6}}
    )

    assert flatten_landmarks(left, mirror=True) == pytest.approx(
        flatten_landmarks(reflected_right)
    )


def test_hand_rotation_mirror_matches_the_reflected_hand():
    left = make_landmarks(
        {
            0: {"x": 0.30, "y": 0.70, "z": 0.00},
            5: {"x": 0.40, "y": 0.50, "z": 0.02},
            9: {"x": 0.35, "y": 0.45, "z": 0.00},
            17: {"x": 0.22, "y": 0.52, "z": -0.03},
        }
    )
    reflected_right = make_landmarks(
        {
            0: {"x": 0.70, "y": 0.70, "z": 0.00},
            5: {"x": 0.60, "y": 0.50, "z": 0.02},
            9: {"x": 0.65, "y": 0.45, "z": 0.00},
            17: {"x": 0.78, "y": 0.52, "z": -0.03},
        }
    )

    assert get_hand_rotation(left, mirror=True) == pytest.approx(
        get_hand_rotation(reflected_right)
    )
