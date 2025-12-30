"""
Timing utilities for performance profiling.
"""
import time
from functools import wraps

# Global timing statistics
DEBUG_TIMING = True
_timing_stats = {}


def timed(func):
    """Decorator to time function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not DEBUG_TIMING:
            return func(*args, **kwargs)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        name = func.__name__
        if name not in _timing_stats:
            _timing_stats[name] = {'count': 0, 'total': 0, 'max': 0}
        _timing_stats[name]['count'] += 1
        _timing_stats[name]['total'] += elapsed
        _timing_stats[name]['max'] = max(_timing_stats[name]['max'], elapsed)

        return result

    return wrapper


def print_timing_stats():
    """Print accumulated timing statistics."""
    print("\n=== Timing Statistics ===")
    for name, stats in sorted(_timing_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        avg = stats['total'] / stats['count'] if stats['count'] > 0 else 0
        print(
            f"{name}: avg={avg:.2f}ms, max={stats['max']:.2f}ms, calls={stats['count']}, total={stats['total']:.1f}ms"
        )


def reset_timing_stats():
    """Reset all timing statistics."""
    global _timing_stats
    _timing_stats = {}
