# tests/test_spur_engine.py
from __future__ import annotations

import math

import numpy as np
import pytest

from buc_planner.spur_engine import (
    _build_if1_input_bands,
    evaluate_spurs_for_config_and_lo_plan,
    coarse_if2_spur_control_regions_for_lo_plan,
)
from buc_planner.los import generate_lo_plan_candidates_for_config
from buc_planner.filters import IF2Filter, RFFilter


def test_build_if1_input_bands_power_semantics(simple_system_config):
    """
    Check that _build_if1_input_bands implements the documented power
    conventions for fundamental and harmonic segments.

    We use the simple_system_config fixture:

      * IF1 band: 950–1050 MHz  (width = 100 MHz)
      * grid step: 50 MHz  ->  2 segments of 50 MHz for the fundamental
      * 2nd harmonic at -30 dBc integrated
    """
    cfg = simple_system_config
    rf_conf = cfg.rf_configurations[0]

    bands = _build_if1_input_bands(cfg, rf_conf)

    # Separate fundamental and 2nd harmonic bands by name convention
    fund_segs = [b for b in bands if b.name.startswith("IF1_fund_seg")]
    h2_segs = [b for b in bands if "IF1_h2_" in b.name]

    # Fundamental: IF1 ref band is cfg.if1_band
    bw_if1_total = cfg.if1_band.width  # 100 MHz in the fixture
    assert bw_if1_total > 0

    # With a 50 MHz grid step, we expect 2 equal-width segments for the fundamental
    assert len(fund_segs) == 2
    total_linear = 0.0
    for seg in fund_segs:
        bw_seg = seg.f_stop - seg.f_start
        expected_level = 10.0 * math.log10(bw_seg / bw_if1_total)
        assert seg.level_dbc_integrated == pytest.approx(expected_level, abs=1e-6)
        total_linear += 10.0 ** (seg.level_dbc_integrated / 10.0)

    # Sum of segment powers should be 0 dBc (i.e. linear sum ≈ 1)
    assert total_linear == pytest.approx(1.0, rel=1e-6)

    # 2nd harmonic: cfg.if1_harmonics_dbc[2] is the *integrated* level
    # for the full 2nd harmonic band derived from the effective IF1 band.
    if 2 in cfg.if1_harmonics_dbc:
        L_h2 = cfg.if1_harmonics_dbc[2]  # -30 dBc in the fixture
        assert len(h2_segs) > 0

        # Full (unclipped) 2nd harmonic band, based on the effective IF1 band
        if1_eff = rf_conf.if1_subband or cfg.if1_band
        harm_full_start = 2 * if1_eff.start
        harm_full_stop = 2 * if1_eff.stop
        bw_harm_full = harm_full_stop - harm_full_start
        assert bw_harm_full > 0

        total_linear_h2 = 0.0
        for seg in h2_segs:
            bw_seg = seg.f_stop - seg.f_start
            expected_level = L_h2 + 10.0 * math.log10(bw_seg / bw_harm_full)
            assert seg.level_dbc_integrated == pytest.approx(
                expected_level, abs=1e-6
            )
            total_linear_h2 += 10.0 ** (seg.level_dbc_integrated / 10.0)

        # Sum over the *clipped* band should be <= full-band power (i.e. <= -30 dBc)
        # We just check it is close to or below that, but not larger.
        total_dbc_h2 = 10.0 * math.log10(total_linear_h2)
        # Can't mix <= with pytest.approx, so compare to a plain float with tiny slack
        assert total_dbc_h2 <= L_h2 + 1e-6


def test_desired_if2_band_not_treated_as_spur(simple_system_config):
    """
    evaluate_spurs_for_config_and_lo_plan must *not* treat the desired
    IF2 band (the Mixer2 desired product) as a spur.

    The implementation achieves this by:
      * building Mixer2 input bands at IF2 (including a band named
        'IF2_desired_band'),
      * skipping that band when constructing 'M2_fund_from_<band>' spurs.

    This test verifies that no SpurResult is generated that traces back
    to the 'IF2_desired_band' as its input_band_name, while still
    allowing spurs that use the same (m, n) family but originate from
    non-desired IF2 content (e.g. cascaded Mixer1 spurs, isolation).
    """
    cfg = simple_system_config
    rf_conf = cfg.rf_configurations[0]

    # Generate a single LO plan candidate
    lo_plans = generate_lo_plan_candidates_for_config(
        cfg, rf_conf, max_candidates=1
    )
    assert lo_plans, "Expected at least one LO plan candidate"
    lo_plan = lo_plans[0]

    # Build a simple IF2 filter that covers the plan.if2_band, as the rest
    # of the codebase would (planning-grade coverage).
    fc = 0.5 * (lo_plan.if2_band.start + lo_plan.if2_band.stop)
    bw = lo_plan.if2_band.stop - lo_plan.if2_band.start
    if2_filter = IF2Filter(
        filter_id="test_if2",
        fc=fc,
        bw=bw,
        slope_db_per_decade=-60.0,
    )

    # RF filter: reuse the RF BPF CSV from the config
    rf = RFFilter.from_csv(cfg.filters.rf_bpf_csv_path)

    res, summary = evaluate_spurs_for_config_and_lo_plan(
        cfg,
        rf_conf,
        lo_plan,
        if2_filter,
        rf,
        mask_freqs=None,
        mask_levels=None,
    )

    # Ensure we actually got some spur results (so test is not degenerate)
    assert isinstance(res, list)
    assert len(res) > 0

    # The desired IF2 band must not appear as the source of any spur.
    for spur in res:
        assert spur.input_band_name != "IF2_desired_band"

    # We do expect at least one spur from Mixer2 isolation tones
    assert any(
        s.mixer_name == cfg.mixer2.name and s.input_band_name == "isolation"
        for s in res
    )

    # Summary object sanity
    assert summary.config_id == rf_conf.config_id
    # worst margins may be None depending on configuration, but the fields exist
    assert hasattr(summary, "worst_in_band_spur_dbc")
    assert hasattr(summary, "worst_out_band_spur_dbc")


def test_coarse_if2_spur_control_regions_no_spur_table(simple_system_config):
    """
    With the simple_system_config fixture, both mixers have empty spur tables
    and no unspecified floor is configured. In this situation, the coarse
    IF2 spur-control region helper should:

      * return an empty list of IF2SpurControlRegion objects, because there
        are no explicit Mixer1 spur families to consider;
      * return worst_margin as None (no spurs evaluated).

    This checks that coarse_if2_spur_control_regions_for_lo_plan gracefully
    handles the "no spur table" planning mode instead of raising or
    inventing regions.
    """
    from buc_planner.spur_engine import coarse_if2_spur_control_regions_for_lo_plan

    cfg = simple_system_config
    rf_conf = cfg.rf_configurations[0]

    lo_plans = generate_lo_plan_candidates_for_config(
        cfg, rf_conf, max_candidates=1
    )
    assert lo_plans, "Expected at least one LO plan candidate"
    lo_plan = lo_plans[0]

    rf = RFFilter.from_csv(cfg.filters.rf_bpf_csv_path)

    regions, worst_margin = coarse_if2_spur_control_regions_for_lo_plan(
        cfg,
        rf_conf,
        lo_plan,
        rf,
    )

    assert isinstance(regions, list)
    assert regions == []
    assert worst_margin is None
