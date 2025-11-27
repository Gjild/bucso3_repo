# tests/conftest.py
from __future__ import annotations

import pytest

from buc_planner.config_models import (
    Range,
    RfConfiguration,
    SpurTableEntry,
    MixerSpurEnvelopePolicy,
    MixerIsolation,
    MixerRanges,
    MixerConfig,
    LOSynthConfig,
    IF2BankConstraints,
    FilterConfig,
    SpurLimitsConfig,
    GridsAndPerformanceConfig,
    SystemConfig,
)


@pytest.fixture
def rf_bpf_csv(tmp_path):
    """
    Simple RF BPF: 0 dB everywhere.
    Uses a header row that RFFilter.from_csv will ignore.
    """
    p = tmp_path / "rf_bpf.csv"
    p.write_text(
        "freq,att\n"
        "0,0\n"
        "5.0e10,0\n"
    )
    return p


@pytest.fixture
def simple_system_config(rf_bpf_csv) -> SystemConfig:
    """
    Small, self-consistent SystemConfig that allows Planner.run()
    to find a solution quickly and is also useful for unit tests.
    """
    # IF1 band and RF band
    if1_band = Range(start=9.50e8, stop=1.05e9)  # 950–1050 MHz
    rf_band = Range(start=9.5e9, stop=1.05e10)   # 9.5–10.5 GHz

    rf_conf = RfConfiguration(
        config_id="cfg1",
        rf_center=1.0e10,
        rf_bandwidth=1.0e8,  # 100 MHz
        if1_subband=None,
    )

    # Spur envelope – no explicit spur table for this minimal config.
    spur_env = MixerSpurEnvelopePolicy(
        m_max=3,
        n_max=3,
        enforce_envelope_completeness=False,
        unspecified_floor_dbc=None,
    )

    # Mixer ranges
    # Mixer1: IF1 around 1 GHz; IF2 around 6 GHz
    m1_ranges = MixerRanges(
        if_range=Range(start=9.50e8, stop=3.0e9),      # wide enough to see harmonics
        lo_range=Range(start=5.0e9, stop=5.0e9),
        rf_range=Range(start=5.95e9, stop=6.05e9),
    )

    # Mixer2: IF2 around 6 GHz; RF around 10 GHz
    m2_ranges = MixerRanges(
        if_range=Range(start=5.95e9, stop=6.05e9),
        lo_range=Range(start=4.0e9, stop=4.0e9),
        rf_range=Range(start=9.5e9, stop=1.05e10),
    )

    # Mixers with only isolation terms (no explicit spur table)
    mixer1 = MixerConfig(
        name="M1",
        spur_table=[],  # rely on envelope behaviour; no explicit spurs
        spur_envelope=spur_env,
        isolation=MixerIsolation(
            lo_to_rf_dbc=-80.0,
            if_to_rf_dbc=-80.0,
        ),
        ranges=m1_ranges,
        desired_m=1,
        desired_n=1,
    )

    mixer2 = MixerConfig(
        name="M2",
        spur_table=[],
        spur_envelope=spur_env,
        isolation=MixerIsolation(
            lo_to_rf_dbc=-80.0,
            if_to_rf_dbc=-80.0,
        ),
        ranges=m2_ranges,
        desired_m=1,
        desired_n=1,
    )

    # LO synthesizers
    lo1 = LOSynthConfig(
        name="LO1",
        freq_range=Range(start=5.0e9, stop=5.0e9),
        grid_step=1.0e6,
    )

    lo2 = LOSynthConfig(
        name="LO2",
        freq_range=Range(start=4.0e9, stop=4.0e9),
        grid_step=1.0e6,
    )

    # IF2 bank constraints
    if2_constraints = IF2BankConstraints(
        min_filters=1,
        max_filters=1,
        fc_range=Range(start=5.8e9, stop=6.2e9),
        bw_range=Range(start=5.0e7, stop=2.0e8),  # 50–200 MHz
        slope_range=(-80.0, -20.0),
        feasibility_margin_hz=1.0e7,
    )

    filters_cfg = FilterConfig(
        if2_constraints=if2_constraints,
        rf_bpf_csv_path=str(rf_bpf_csv),
    )

    # Spur limits: only in-band is meaningful here, but we also set OOB.
    spur_limits = SpurLimitsConfig(
        in_band_limit_dbc=-30.0,
        out_of_band_limit_dbc=-70.0,
        mask=None,
        out_of_band_range=None,
        mask_eval_mode="center",
    )

    # Grids / performance
    grids = GridsAndPerformanceConfig(
        if1_grid_step_hz=5.0e7,           # 50 MHz
        spur_integration_step_hz=1.0e7,  # 10 MHz
        max_if1_harmonic_order=3,
        min_spur_level_considered_dbc=-120.0,
        max_lo_candidates_per_rf=8,
        max_if2_bank_candidates=8,
        coarse_spur_margin_min_db=-10.0,
        mixer2_if2_focus_margin_hz=0.0,
        parallel=False,                  # keep tests simple (no processes)
        use_numba=False,
        min_lo_candidates_per_rf_after_coarse=1,
    )

    cfg = SystemConfig(
        if1_band=if1_band,
        rf_band=rf_band,
        rf_configurations=[rf_conf],
        non_inverting_mapping_required=True,
        mixer1=mixer1,
        mixer2=mixer2,
        lo1=lo1,
        lo2=lo2,
        filters=filters_cfg,
        spur_limits=spur_limits,
        grids=grids,
        # 2nd IF1 harmonic at −30 dBc integrated
        if1_harmonics_dbc={2: -30.0},
        description="simple test config",
    )
    return cfg
