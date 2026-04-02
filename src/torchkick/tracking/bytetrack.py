"""
Backward-compatibility shim — imports SoccerTracker as ByteTracker.

Use ``from torchkick.tracking.mot_tracker import SoccerTracker`` for new code.
"""

from torchkick.tracking.mot_tracker import SoccerTracker as ByteTracker  # noqa: F401

__all__ = ["ByteTracker"]
