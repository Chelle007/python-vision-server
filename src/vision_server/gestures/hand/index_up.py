from . import rule
from .classify import classify_hand


@rule("index_up")
def is_index_up(landmarks) -> bool:
    return classify_hand(landmarks) == "index_up"
