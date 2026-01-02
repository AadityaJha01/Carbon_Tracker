"""
Shim wrapper to expose CarbonTracker under src.core namespace.

This keeps imports consistent: `from src.core.tracker import CarbonTracker`.
It delegates to the top-level `tracker.py` implementation which wraps CodeCarbon.
"""
from typing import Optional, Dict
from tracker import CarbonTracker as _CarbonTracker


class CarbonTracker(_CarbonTracker):
    """Alias of top-level CarbonTracker with the same interface.

    Keeping this file ensures code under `src/core` can import a local tracker
    without changing other parts of the codebase.
    """
    pass
