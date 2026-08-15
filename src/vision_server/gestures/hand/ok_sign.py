from . import rule
from .classify import classify_hand


@rule("ok_sign")
def is_ok_sign(landmarks) -> bool:
    return classify_hand(landmarks) == "ok_sign"
