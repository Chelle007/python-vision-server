from . import rule
from .classify import classify_hand


@rule("index_down")
def is_index_down(landmarks) -> bool:
    return classify_hand(landmarks) == "index_down"
