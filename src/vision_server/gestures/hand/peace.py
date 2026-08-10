from . import rule
from .classify import classify_hand


@rule("peace")
def is_peace_sign(landmarks) -> bool:
    return classify_hand(landmarks) == "peace"
