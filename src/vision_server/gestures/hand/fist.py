from . import rule
from .classify import classify_hand


@rule("fist")
def is_fist(landmarks) -> bool:
    return classify_hand(landmarks) == "fist"
