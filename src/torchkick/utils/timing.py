"""
Timing utilities for performance profiling.

This module provides decorators and functions to measure execution time
of functions and accumulate statistics across multiple calls.

Example:
    >>> from torchkick.utils import timed, print_timing_stats
    >>> 
    >>> @timed
    ... def process_frame(frame):
    ...     # processing logic
    ...     pass
    >>> 
    >>> for frame in frames:
    ...     process_frame(frame)
    >>> 
    >>> print_timing_stats()  # Shows avg, max, total time per function
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar

# Global timing statistics storage
DEBUG_TIMING: bool = True
_timing_stats: dict[str, dict[str, float]] = {}

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """
    Decorator to measure and accumulate function execution time.

    Tracks call count, total time, and maximum time per call. Statistics
    can be retrieved via `print_timing_stats()` or `_timing_stats` dict.

    Args:
        func: The function to wrap with timing instrumentation.

    Returns:
        Wrapped function that records timing on each call.

    Example:
        >>> @timed
        ... def slow_function():
        ...     time.sleep(0.1)
        >>> slow_function()
        >>> print(_timing_stats["slow_function"]["count"])
        1
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not DEBUG_TIMING:
            return func(*args, **kwargs)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        name = func.__name__
        if name not in _timing_stats:
            _timing_stats[name] = {"count": 0, "total": 0.0, "max": 0.0}

        _timing_stats[name]["count"] += 1
        _timing_stats[name]["total"] += elapsed_ms
        _timing_stats[name]["max"] = max(_timing_stats[name]["max"], elapsed_ms)

        return result

    return wrapper  # type: ignore[return-value]


def print_timing_stats() -> None:
    """
    Print accumulated timing statistics to stdout.

    Displays stats sorted by total time (descending), showing average,
    maximum, call count, and total time for each instrumented function.

    Example output:
        === Timing Statistics ===
        process_frame: avg=12.34ms, max=45.67ms, calls=100, total=1234.0ms
        detect_players: avg=8.12ms, max=23.45ms, calls=100, total=812.0ms
    """
    print("\n=== Timing Statistics ===")
    sorted_stats = sorted(_timing_stats.items(), key=lambda x: x[1]["total"], reverse=True)

    for name, stats in sorted_stats:
        count = stats["count"]
        avg = stats["total"] / count if count > 0 else 0
        print(f"{name}: avg={avg:.2f}ms, max={stats['max']:.2f}ms, calls={count}, total={stats['total']:.1f}ms")


def reset_timing_stats() -> None:
    """
    Reset all accumulated timing statistics.

    Clears the global `_timing_stats` dictionary. Useful for starting
    fresh measurements between test runs or processing batches.
    """
    global _timing_stats
    _timing_stats = {}


__all__ = [
    "DEBUG_TIMING",
    "timed",
    "print_timing_stats",
    "reset_timing_stats",
    "_timing_stats",
]
