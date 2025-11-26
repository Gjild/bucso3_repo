# src/buc_planner/__init__.py
"""
Dual-Conversion BUC Frequency Planning & Spur-Analysis Tool.

This package provides a planning-grade estimator for LO planning, IF2 filter bank
design, and spur analysis in a dual-conversion block-up converter:

    IF1 -> Mixer1 -> IF2 BPF Bank (parallel, switched) -> Mixer2 -> RF BPF -> RF Out
"""

from .config_models import (
    SystemConfig,
    load_config,
)

from .optimizer import (
    Planner,
    PlannerResult,
)

__all__ = [
    "SystemConfig",
    "load_config",
    "Planner",
    "PlannerResult",
]