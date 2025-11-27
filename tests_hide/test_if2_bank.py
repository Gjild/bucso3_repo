# tests/test_config_models.py
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from buc_planner.config_models import (
    Range,
    SystemConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_config_dict(rf_bpf_csv_path: str) -> dict:
    """
    Build a small but self-consistent raw config dict that mirrors the
    `simple_system_config` fixture in conftest, but in plain-JSON/YAML form.

    This helper is used by multiple tests to exercise `load_config`.
    """
    # IF1 band and RF band
    if1_band = {"start": 9.50e8, "stop": 1.05e9}  # 950–1050 MHz
    rf_band = {"start": 9.5e9, "stop": 1.05e10}  # 9.5–10.5 GHz

    rf_conf = {
        "config_id": "cfg1",
        "rf_center": 1.0e10,
        "rf_bandwidth": 1.0e8,  # 100 MHz
        # Explicit null to exercise if1_subband parsing
        "if1_subband": None,
    }

    spur_env = {
        "m_max": 3,
        "n_max": 3,
        "enforce_envelope_completeness": False,
        "unspecified_floor_dbc": None,
    }

    # Mixer ranges
    m1_ranges = {
        "if_range": {"start": 9.50e8, "stop": 3.0e9},
        "lo_range": {"start": 5.0e9, "stop": 5.0e9},
        "rf_range": {"start": 5.95e9, "stop": 6.05e9},
    }
    m2_ranges = {
        "if_range": {"start": 5.95e9, "stop": 6.05e9},
        "lo_range": {"start": 4.0e9, "stop": 4.0e9},
        "rf_range": {"start": 9.5e9, "stop": 1.05e10},
    }

    mixer1 = {
        "name": "M1",
        "spur_table": [],  # no explicit spur families in this base config
        "spur_envelope": spur_env,
        "isolation": {
            "lo_to_rf_dbc": -80.0,
            "if_to_rf_dbc": -80.0,
        },
        "ranges": m1_ranges,
        "desired_m": 1,
        "desired_n": 1,
    }

    mixer2 = {
        "name": "M2",
        "spur_table": [],
        "spur_envelope": spur_env,
        "isolation": {
            "lo_to_rf_dbc": -80.0,
            "if_to_rf_dbc": -80.0,
        },
        "ranges": m2_ranges,
        "desired_m": 1,
        "desired_n": 1,
    }

    lo1 = {
        "name": "LO1",
        "freq_range": {"start": 5.0e9, "stop": 5.0e9},
        "grid_step": 1.0e6,
        # include some LO detail to exercise LOSynthConfig parsing
        "pfd_frequency": 1.0e6,
        "harmonics": [
            {"order": 2, "level_dbc": -30.0},
        ],
        "pll_spurs": [
            {"offset_multiple": 1, "level_dbc": -60.0},
        ],
    }

    lo2 = {
        "name": "LO2",
        "freq_range": {"start": 4.0e9, "stop": 4.0e9},
        "grid_step": 1.0e6,
        "pfd_frequency": None,
        "harmonics": [],
        "pll_spurs": [],
    }

    if2_constraints = {
        "min_filters": 1,
        "max_filters": 2,
        "fc_range": {"start": 5.8e9, "stop": 6.2e9},
        "bw_range": {"start": 5.0e7, "stop": 2.0e8},
        "slope_range": [-80.0, -20.0],
        "feasibility_margin_hz": 1.0e7,
    }

    filters_cfg = {
        "if2_constraints": if2_constraints,
        "rf_bpf_csv_path": rf_bpf_csv_path,
    }

    spur_limits = {
        "in_band_limit_dbc": -30.0,
        "out_of_band_limit_dbc": -70.0,
        "mask": {
            "csv_path": None,
            "apply_in_band": True,
            "apply_out_of_band": True,
        },
        "out_of_band_range": {
            "start": 0.0,
            "stop": 5.0e10,
        },
        "mask_eval_mode": "center",
    }

    grids = {
        "if1_grid_step_hz": 5.0e7,
        "spur_integration_step_hz": 1.0e7,
        "max_if1_harmonic_order": 3,
        # optional fields (we override defaults here so we can assert them)
        "min_spur_level_considered_dbc": -100.0,
        "max_lo_candidates_per_rf": 8,
        "max_if2_bank_candidates": 8,
        "coarse_spur_margin_min_db": -10.0,
        "mixer2_if2_focus_margin_hz": 0.0,
        "parallel": False,
        "use_numba": False,
        "min_lo_candidates_per_rf_after_coarse": 2,
    }

    raw = {
        "if1_band": if1_band,
        "rf_band": rf_band,
        "rf_configurations": [rf_conf],
        "non_inverting_mapping_required": True,
        "mixer1": mixer1,
        "mixer2": mixer2,
        "lo1": lo1,
        "lo2": lo2,
        "filters": filters_cfg,
        "spur_limits": spur_limits,
        "grids": grids,
        # 2nd IF1 harmonic at −30 dBc integrated, key intentionally numeric
        "if1_harmonics_dbc": {2: -30.0},
        "description": "unit-test config",
    }
    return raw


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------


def test_range_contains_and_intersect_basic():
    r = Range(start=10.0, stop=20.0)

    # Contains is inclusive at both ends
    assert r.contains(10.0)
    assert r.contains(15.0)
    assert r.contains(20.0)
    assert not r.contains(9.999)
    assert not r.contains(20.001)

    other = Range(start=15.0, stop=25.0)
    inter = r.intersect(other)
    assert inter is not None
    assert inter.start == pytest.approx(15.0)
    assert inter.stop == pytest.approx(20.0)
    assert inter.width == pytest.approx(5.0)

    disjoint = Range(start=30.0, stop=40.0)
    assert r.intersect(disjoint) is None


# ---------------------------------------------------------------------------
# load_config – JSON / YAML and field wiring
# ---------------------------------------------------------------------------


def test_load_config_from_json_roundtrip(tmp_path, rf_bpf_csv):
    """
    load_config must be able to parse a JSON config and correctly populate the
    nested dataclasses, including LO synths, mixer ranges, IF2 constraints,
    spur limits, grids, and IF1 harmonics.
    """
    raw = _make_raw_config_dict(str(rf_bpf_csv))

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(raw))

    cfg = load_config(cfg_path)
    assert isinstance(cfg, SystemConfig)

    # Top-level bands
    assert cfg.if1_band.start == pytest.approx(raw["if1_band"]["start"])
    assert cfg.if1_band.stop == pytest.approx(raw["if1_band"]["stop"])
    assert cfg.rf_band.start == pytest.approx(raw["rf_band"]["start"])
    assert cfg.rf_band.stop == pytest.approx(raw["rf_band"]["stop"])

    # RF configuration
    assert len(cfg.rf_configurations) == 1
    rc = cfg.rf_configurations[0]
    assert rc.config_id == "cfg1"
    assert rc.if1_subband is None  # explicit null in raw config

    # Mixer1 wiring
    assert cfg.mixer1.name == "M1"
    assert cfg.mixer1.desired_m == 1
    assert cfg.mixer1.desired_n == 1
    assert cfg.mixer1.ranges.if_range.start == pytest.approx(9.50e8)
    assert cfg.mixer1.ranges.rf_range.stop == pytest.approx(6.05e9)

    # LO1 wiring (including harmonics & PLL spurs)
    assert cfg.lo1.name == "LO1"
    assert cfg.lo1.freq_range.start == pytest.approx(5.0e9)
    assert cfg.lo1.grid_step == pytest.approx(1.0e6)
    assert cfg.lo1.pfd_frequency == pytest.approx(1.0e6)
    assert len(cfg.lo1.harmonics) == 1
    assert cfg.lo1.harmonics[0].order == 2
    assert cfg.lo1.harmonics[0].level_dbc == pytest.approx(-30.0)
    assert len(cfg.lo1.pll_spurs) == 1
    assert cfg.lo1.pll_spurs[0].offset_multiple == 1

    # IF2 constraints
    if2c = cfg.filters.if2_constraints
    assert if2c.min_filters == 1
    assert if2c.max_filters == 2
    assert if2c.fc_range.start == pytest.approx(5.8e9)
    assert if2c.bw_range.stop == pytest.approx(2.0e8)
    assert if2c.slope_range == (-80.0, -20.0)
    assert if2c.feasibility_margin_hz == pytest.approx(1.0e7)

    # Spur limits & mask flags
    assert cfg.spur_limits.in_band_limit_dbc == pytest.approx(-30.0)
    assert cfg.spur_limits.out_of_band_limit_dbc == pytest.approx(-70.0)
    assert cfg.spur_limits.out_of_band_range is not None
    assert cfg.spur_limits.out_of_band_range.stop == pytest.approx(5.0e10)
    # mask is present but csv_path=None
    assert cfg.spur_limits.mask is not None
    assert cfg.spur_limits.mask.csv_path is None
    assert cfg.spur_limits.mask.apply_in_band is True
    assert cfg.spur_limits.mask.apply_out_of_band is True

    # Grids & performance – we overrode the defaults so they should match raw
    g = cfg.grids
    assert g.if1_grid_step_hz == pytest.approx(5.0e7)
    assert g.spur_integration_step_hz == pytest.approx(1.0e7)
    assert g.max_if1_harmonic_order == 3
    assert g.min_spur_level_considered_dbc == pytest.approx(-100.0)
    assert g.max_lo_candidates_per_rf == 8
    assert g.max_if2_bank_candidates == 8
    assert g.coarse_spur_margin_min_db == pytest.approx(-10.0)
    assert g.mixer2_if2_focus_margin_hz == pytest.approx(0.0)
    assert g.parallel is False
    assert g.use_numba is False
    assert g.min_lo_candidates_per_rf_after_coarse == 2

    # IF1 harmonics dict: keys should be ints, values floats
    assert cfg.if1_harmonics_dbc[2] == pytest.approx(-30.0)
    assert set(cfg.if1_harmonics_dbc.keys()) == {2}


def test_load_config_from_yaml_uses_yaml_loader(tmp_path, rf_bpf_csv):
    """
    _load_yaml_or_json must switch to the YAML code path for .yaml/.yml
    files. We just exercise that path and lightly sanity-check the result.
    """
    raw = _make_raw_config_dict(str(rf_bpf_csv))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    cfg = load_config(cfg_path)
    assert isinstance(cfg, SystemConfig)

    # A couple of representative sanity checks
    assert cfg.lo2.name == "LO2"
    assert cfg.lo2.freq_range.start == pytest.approx(4.0e9)
    assert cfg.filters.rf_bpf_csv_path == str(rf_bpf_csv)


def test_load_config_grids_defaults_when_optional_fields_missing(tmp_path, rf_bpf_csv):
    """
    When optional grid/performance fields are omitted from the raw config,
    `load_config` must populate sensible defaults (see r_grids).
    """
    raw = _make_raw_config_dict(str(rf_bpf_csv))
    grids = raw["grids"]

    # Remove all optional fields we know have defaults
    for key in [
        "min_spur_level_considered_dbc",
        "max_lo_candidates_per_rf",
        "max_if2_bank_candidates",
        "coarse_spur_margin_min_db",
        "mixer2_if2_focus_margin_hz",
        "parallel",
        "use_numba",
        "min_lo_candidates_per_rf_after_coarse",
    ]:
        grids.pop(key, None)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(raw))

    cfg = load_config(cfg_path)
    g = cfg.grids

    # Required fields
    assert g.if1_grid_step_hz == pytest.approx(5.0e7)
    assert g.spur_integration_step_hz == pytest.approx(1.0e7)
    assert g.max_if1_harmonic_order == 3

    # Defaults from r_grids()
    assert g.min_spur_level_considered_dbc == pytest.approx(-120.0)
    assert g.max_lo_candidates_per_rf == 50
    assert g.max_if2_bank_candidates == 50
    assert g.coarse_spur_margin_min_db == pytest.approx(-10.0)
    assert g.mixer2_if2_focus_margin_hz == 0.0
    assert g.parallel is True
    assert g.use_numba is False
    assert g.min_lo_candidates_per_rf_after_coarse == 1


# ---------------------------------------------------------------------------
# Sanity-check warnings for mixer spur configuration
# ---------------------------------------------------------------------------


def test_load_config_warns_when_spur_table_contains_desired_family(tmp_path, rf_bpf_csv):
    """
    _sanity_check_mixer_spur_config must emit a RuntimeWarning when the
    spur table explicitly contains the desired (m, n) family, since spur
    levels are supposed to be defined *relative* to that family.
    """
    raw = _make_raw_config_dict(str(rf_bpf_csv))
    m1 = raw["mixer1"]

    # Add a spur table entry that matches desired_m, desired_n = (1, 1)
    m1["spur_table"] = [
        {
            "m": 1,
            "n": 1,
            "level_dbc": -10.0,
            "if_range": m1["ranges"]["if_range"],
            "lo_range": m1["ranges"]["lo_range"],
            "rf_range": m1["ranges"]["rf_range"],
        }
    ]

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(raw))

    with pytest.warns(RuntimeWarning, match="spur_table contains the desired fundamental"):
        cfg = load_config(cfg_path)

    # Config should still load and have the expected desired_m/n
    assert cfg.mixer1.desired_m == 1
    assert cfg.mixer1.desired_n == 1


def test_load_config_warns_when_envelope_floor_covers_desired_without_entry(tmp_path, rf_bpf_csv):
    """
    If the spur envelope covers the desired (m, n) family and an
    unspecified_floor_dbc is provided, but there is *no* explicit
    spur_table entry for that family, _sanity_check_mixer_spur_config
    must warn that the configuration may be misleading.
    """
    raw = _make_raw_config_dict(str(rf_bpf_csv))
    m1 = raw["mixer1"]

    # Ensure spur table does NOT contain the desired family
    m1["spur_table"] = []

    # Envelope already covers |m|<=3,|n|<=3; add a floor so that the
    # sanity check triggers the second warning branch.
    m1["spur_envelope"] = deepcopy(m1["spur_envelope"])
    m1["spur_envelope"]["unspecified_floor_dbc"] = -80.0

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(raw))

    with pytest.warns(RuntimeWarning, match="spur_envelope covers the desired fundamental"):
        cfg = load_config(cfg_path)

    # Again, the config should still load; warning is advisory.
    assert cfg.mixer1.spur_envelope.unspecified_floor_dbc == pytest.approx(-80.0)
