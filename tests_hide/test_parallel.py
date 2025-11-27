# tests/test_parallel.py
from __future__ import annotations
from buc_planner.optimizer import Planner

def test_optimizer_parallel_execution(simple_system_config):
    """
    Force parallel execution path in Planner.
    """
    simple_system_config.grids.parallel = True
    planner = Planner(simple_system_config)
    # Just ensure it runs without pickling errors/crashes
    res = planner.run()
    assert res.lo_plans