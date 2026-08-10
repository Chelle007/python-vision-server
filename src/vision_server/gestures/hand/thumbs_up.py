from . import rule
from .classify import classify_hand


@rule("thumbs_up")
def is_thumbs_up(landmarks) -> bool:
    return classify_hand(landmarks) == "thumbs_up"
