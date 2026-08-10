from . import rule
from .classify import classify_hand


@rule("index_left")
def is_index_left(landmarks) -> bool:
    return classify_hand(landmarks) == "index_left"
