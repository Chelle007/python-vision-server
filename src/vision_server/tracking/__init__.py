from .player_lock import PlayerLock, collect_faces, collect_hands

__all__ = [
    "create_hands",
    "create_face_mesh",
    "PlayerLock",
    "collect_faces",
    "collect_hands",
]


def __getattr__(name):
    if name == "create_hands":
        from .hands import create_hands

        return create_hands
    if name == "create_face_mesh":
        from .face import create_face_mesh

        return create_face_mesh
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
