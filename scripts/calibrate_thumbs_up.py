"""Measure YOUR thumbs-up against the fist it is carved out of.

The thumbs-up thresholds in config.py were calibrated against the recorded
corpus in data/, which contains plenty of fists and no thumbs-up at all. That
fixes the false-positive rate and leaves the MISS rate completely unmeasured —
if your thumb sits lower than assumed, the gesture simply never fires and there
is nothing in the payload to explain why.

This script closes that gap. It records both poses from the webcam, prints where
each one actually lands, and recommends thresholds with real separation between
them.

    python scripts/calibrate_thumbs_up.py

Hold each pose as asked; SPACE starts each phase, Q quits. Run it from the repo
root — the camera config is imported, so a wrong working directory changes what
you are measuring.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_server.camera import LatestFrameCamera  # noqa: E402
from vision_server.config import (  # noqa: E402
    MAX_NUM_HANDS,
    THUMB_UP_CLEARANCE,
    THUMB_UP_MIN_PALM,
    THUMB_UP_REACH,
)
from vision_server.gestures.hand.fingers import (  # noqa: E402
    FINGERS,
    finger_states,
    hand_frame,
    thumb_clearance,
    thumb_reach,
)
from vision_server.tracking import collect_hands, create_hands  # noqa: E402

SAMPLES_PER_PHASE = 120
FIST_PATTERN = (False, False, False, False)

PHASES = [
    ("thumbs_up", "THUMBS UP - fist with the thumb standing straight up"),
    ("fist", "PLAIN FIST - thumb folded over your fingers, as if walking"),
]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; avoids a numpy import for a handful of samples."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def collect(cap, hands, prompt: str) -> list[tuple[float, float]]:
    """Gather (clearance, reach) for frames whose four fingers read as curled.

    Only the fist pattern is sampled, because that is the only pattern where
    the thumb decides anything — a thumbs-up whose fingers are not fully curled
    is a different failure, and would quietly skew the numbers if counted here.
    """
    samples: list[tuple[float, float]] = []
    started = False

    while cap.isOpened() and len(samples) < SAMPLES_PER_PHASE:
        success, frame = cap.read()
        if not success or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        skipped = ""

        if started:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracked = collect_hands(hands.process(rgb))
            if not tracked:
                skipped = "no hand"
            else:
                landmarks = tracked[0].landmarks
                geom = hand_frame(landmarks)
                if geom is None:
                    skipped = "hand pointing at camera"
                elif geom[1] < THUMB_UP_MIN_PALM:
                    skipped = f"palm too foreshortened ({geom[1]:.3f})"
                else:
                    states = finger_states(landmarks, geom)
                    if tuple(states[n] for n in FINGERS) != FIST_PATTERN:
                        skipped = "fingers not fully curled"
                    else:
                        samples.append((
                            thumb_clearance(landmarks, geom),
                            thumb_reach(landmarks, geom),
                        ))

        header = prompt if started else f"{prompt}  --  SPACE to start"
        cv2.putText(frame, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.putText(frame, f"{len(samples)}/{SAMPLES_PER_PHASE}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if skipped:
            cv2.putText(frame, f"skipped: {skipped}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow("Thumbs-up calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return []
        if key == ord(" "):
            started = True

    return samples


def main() -> int:
    cap = LatestFrameCamera()
    hands = create_hands(max_num_hands=MAX_NUM_HANDS)
    results = {}

    try:
        for name, prompt in PHASES:
            print(f"\n=== {prompt}")
            samples = collect(cap, hands, prompt)
            if not samples:
                print("Aborted.")
                return 1
            results[name] = samples
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    up, fist = results["thumbs_up"], results["fist"]
    print(f"\n{'pose':10s} {'n':>4s}  {'clear p5':>9s} {'p50':>6s}  "
          f"{'reach p5':>9s} {'p50':>6s}")
    for name, samples in results.items():
        clear = [c for c, _ in samples]
        reach = [r for _, r in samples]
        print(f"{name:10s} {len(samples):4d}  {percentile(clear, 5):9.2f} "
              f"{percentile(clear, 50):6.2f}  {percentile(reach, 5):9.2f} "
              f"{percentile(reach, 50):6.2f}")

    def fires(sample):
        clear, reach = sample
        return clear >= THUMB_UP_CLEARANCE and reach >= THUMB_UP_REACH

    print(f"\ncurrent thresholds: clearance >= {THUMB_UP_CLEARANCE:.2f}, "
          f"reach >= {THUMB_UP_REACH:.2f}")
    missed = sum(1 for s in up if not fires(s))
    fired = sum(1 for s in fist if fires(s))
    print(f"  would MISS {100*missed/len(up):.1f}% of your thumbs-up frames")
    print(f"  would FIRE on {100*fired/len(fist):.1f}% of your fist frames")
    if missed:
        by_clear = sum(1 for c, _ in up if c < THUMB_UP_CLEARANCE)
        by_reach = sum(1 for _, r in up if r < THUMB_UP_REACH)
        print(f"    of the misses: {by_clear} fail clearance, {by_reach} fail reach")

    # Sit between the two clouds: below the weak tail of the real gesture, above
    # the bulk of the fists. p5 of the positives is the recall side of that gap.
    print()
    for term, name, up_vals, fist_vals in (
        ("THUMB_UP_CLEARANCE", "clearance", [c for c, _ in up], [c for c, _ in fist]),
        ("THUMB_UP_REACH", "reach", [r for _, r in up], [r for _, r in fist]),
    ):
        low, high = percentile(up_vals, 5), percentile(fist_vals, 95)
        if low > high:
            print(f"  {name:9s}: thumbs-up p5={low:.2f} vs fist p95={high:.2f}"
                  f"  -> set {term} = {(low + high) / 2:.2f}")
        else:
            print(f"  {name:9s}: thumbs-up p5={low:.2f} vs fist p95={high:.2f}"
                  f"  -> OVERLAP, this term cannot separate your two poses")

    print("\nLower values favour triggering reliably; higher favours never "
          "opening by itself. If a term overlaps, check the fist phase was "
          "recorded with the thumb folded DOWN over the fingers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
