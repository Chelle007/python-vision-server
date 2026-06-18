from . import rule


@rule("fist")
def is_fist(landmarks) -> bool:
    finger_tips = [8, 12, 16, 20]
    finger_bases = [5, 9, 13, 17]

    folded_count = 0

    for tip, base in zip(finger_tips, finger_bases):
        if landmarks[tip].y > landmarks[base].y:
            folded_count += 1

    return folded_count >= 3
