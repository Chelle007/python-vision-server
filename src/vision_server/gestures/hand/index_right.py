from . import rule
from .classify import classify_hand


@rule("index_right")
def is_index_right(landmarks) -> bool:
    return classify_hand(landmarks) == "index_right"
