# tests/test_optimizer_planner.py
from __future__ import annotations

import pytest

from buc_planner.optimizer import Planner
from buc_planner.filters import RFFilter


def test_planner_run_simple_config(simple_system_config):
    """
    Happy-path sanity check:

    * RF BPF CSV is loadable.
    * Planner.run() completes without error.
    * Exactly one LO plan is found for the single RF configuration.
    * IF2 bank filter count is within configured bounds.
    * Spur summary margins (if defined) are non-negative.
    * LO frequencies and signs match the simple test fixture expectations.
    * IF2 bank mapping is consistent.
    """
    cfg = simple_system_config

    # Sanity: RF BPF is loadable
    _ = RFFilter.from_csv(cfg.filters.rf_bpf_csv_path)

    planner = Planner(cfg)
    result = planner.run()

    # One RF configuration → one LO plan
    assert len(result.lo_plans) == 1
    plan = result.lo_plans[0]
    cfg_id = cfg.rf_configurations[0].config_id
    assert plan.config_id == cfg_id

    # In this simple fixture, LO1 and LO2 ranges are singletons,
    # so the chosen LO frequencies and signs are fully determined.
    assert plan.lo1_freq == pytest.approx(5.0e9)
    assert plan.lo2_freq == pytest.approx(4.0e9)
    assert plan.sign_combo.sign1 == +1
    assert plan.sign_combo.sign2 == +1

    # IF2 bank respects constraints
    n_filters = len(result.if2_bank_design.bank.filters)
    assert cfg.filters.if2_constraints.min_filters <= n_filters <= cfg.filters.if2_constraints.max_filters

    # Mapping from config_id -> filter_id is consistent with the bank contents
    assert set(result.if2_bank_design.config_to_filter_id.keys()) == {cfg_id}
    assigned_filter_id = result.if2_bank_design.config_to_filter_id[cfg_id]
    filter_ids = {f.filter_id for f in result.if2_bank_design.bank.filters}
    assert assigned_filter_id in filter_ids

    # Spur summaries present and not violating limits (no negative margins)
    summary = result.summaries[cfg_id]
    if summary.worst_in_band_margin_db is not None:
        assert summary.worst_in_band_margin_db >= 0.0
    if summary.worst_out_band_margin_db is not None:
        assert summary.worst_out_band_margin_db >= 0.0

    # If any spur results exist, they should all belong to this config_id
    spur_cfg_ids = {r.config_id for r in result.spur_results}
    if spur_cfg_ids:
        assert spur_cfg_ids == {cfg_id}


def test_planner_raises_when_no_lo_candidates(simple_system_config):
    """
    If the LO ranges are inconsistent with mixer ranges / RF band such that
    no LO plan candidates exist, Planner.run() should raise a RuntimeError
    with a helpful message.
    """
    cfg = simple_system_config

    # Break LO2 so that it falls outside Mixer2's LO range, ensuring that
    # generate_lo_plan_candidates_for_config() returns no candidates.
    cfg.lo2.freq_range.start = 3.8e9
    cfg.lo2.freq_range.stop = 3.8e9

    planner = Planner(cfg)
    with pytest.raises(RuntimeError, match="No LO plan candidates"):
        planner.run()


def test_planner_prefers_min_if2_filters(simple_system_config):
    """
    With a feasible 1-filter solution available and max_filters > min_filters,
    the planner should stop at the minimal filter count that yields a feasible
    solution (lexicographic objective: minimize IF2 bank size first).
    """
    cfg = simple_system_config

    # Allow more filters than the minimum; the planner should still end up
    # using the minimum number that yields a feasible solution.
    cfg.filters.if2_constraints.max_filters = cfg.filters.if2_constraints.min_filters + 2

    planner = Planner(cfg)
    result = planner.run()

    n_filters = len(result.if2_bank_design.bank.filters)
    assert n_filters == cfg.filters.if2_constraints.min_filters
