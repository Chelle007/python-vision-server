from . import rule
from .classify import classify_hand


@rule("rock_sign")
def is_rock_sign(landmarks) -> bool:
    return classify_hand(landmarks) == "rock_sign"
