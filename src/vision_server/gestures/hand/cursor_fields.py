"""Action-hand cursor fields sent to Unity (palm, index tip).

Positions stay in real screen space and are never mirrored — which physical
hand holds the ACTION role does not change where the cursor should point.
"""

from .geometry import get_index_tip_position, get_palm_position

INVALID_COORD = -1.0


def reset_last_point(last_point: list[float]) -> None:
    last_point[0] = INVALID_COORD
    last_point[1] = INVALID_COORD


def apply_cursor_fields(
    data: dict,
    landmarks,
    gestures: dict[str, bool],
    last_point: list[float],
) -> None:
    palm_x, palm_y = get_palm_position(landmarks)
    index_up = gestures["index_up"]
    is_fist = gestures["fist"]

    if index_up and not is_fist:
        last_point[0] = palm_x
        last_point[1] = palm_y
        data["palmX"] = round(palm_x, 3)
        data["palmY"] = round(palm_y, 3)

        tip_x, tip_y = get_index_tip_position(landmarks)
        data["indexTipX"] = round(tip_x, 3)
        data["indexTipY"] = round(tip_y, 3)
    elif is_fist and last_point[0] >= 0.0:
        data["palmX"] = round(last_point[0], 3)
        data["palmY"] = round(last_point[1], 3)
        data["indexTipX"] = INVALID_COORD
        data["indexTipY"] = INVALID_COORD
    else:
        data["palmX"] = round(palm_x, 3)
        data["palmY"] = round(palm_y, 3)
        data["indexTipX"] = INVALID_COORD
        data["indexTipY"] = INVALID_COORD
