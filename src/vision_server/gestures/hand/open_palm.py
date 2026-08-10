from . import rule
from .classify import classify_hand


@rule("open_palm")
def is_open_palm(landmarks) -> bool:
    return classify_hand(landmarks) == "open_palm"
