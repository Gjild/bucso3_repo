# tests/test_config_models.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from buc_planner.config_models import (
    Range,
    load_config,
)


def test_range_contains_and_width():
    r = Range(start=1.0, stop=5.0)
    assert r.contains(1.0)
    assert r.contains(3.0)
    assert r.contains(5.0)
    assert not r.contains(0.0)
    assert not r.contains(6.0)
    assert r.width == pytest.approx(4.0)


def test_range_intersect():
    r1 = Range(start=0.0, stop=10.0)
    r2 = Range(start=5.0, stop=15.0)
    r3 = Range(start=20.0, stop=30.0)

    inter = r1.intersect(r2)
    assert inter is not None
    assert inter.start == pytest.approx(5.0)
    assert inter.stop == pytest.approx(10.0)

    assert r1.intersect(r3) is None


def test_load_config_from_yaml(tmp_path, rf_bpf_csv):
    """Smoke test: YAML → SystemConfig with a minimal, valid config."""
    cfg_path = tmp_path / "config.yaml"

    # Use POSIX-style path so YAML doesn't choke on Windows backslashes like "\U"
    rf_bpf_csv_str = rf_bpf_csv.as_posix()

    yaml_text = f"""
if1_band:
  start: 9.5e8
  stop: 1.05e9
rf_band:
  start: 9.5e9
  stop: 1.05e10
rf_configurations:
  - config_id: "cfg1"
    rf_center: 1.0e10
    rf_bandwidth: 1.0e8
    if1_subband: null
non_inverting_mapping_required: true

mixer1:
  name: "M1"
  spur_table: []
  spur_envelope:
    m_max: 3
    n_max: 3
    enforce_envelope_completeness: false
    unspecified_floor_dbc: null
  isolation:
    lo_to_rf_dbc: -80.0
    if_to_rf_dbc: -80.0
  ranges:
    if_range: {{ start: 9.5e8, stop: 3.0e9 }}
    lo_range: {{ start: 5.0e9, stop: 5.0e9 }}
    rf_range: {{ start: 5.95e9, stop: 6.05e9 }}
  desired_m: 1
  desired_n: 1

mixer2:
  name: "M2"
  spur_table: []
  spur_envelope:
    m_max: 3
    n_max: 3
    enforce_envelope_completeness: false
    unspecified_floor_dbc: null
  isolation:
    lo_to_rf_dbc: -80.0
    if_to_rf_dbc: -80.0
  ranges:
    if_range: {{ start: 5.95e9, stop: 6.05e9 }}
    lo_range: {{ start: 4.0e9, stop: 4.0e9 }}
    rf_range: {{ start: 9.5e9, stop: 1.05e10 }}
  desired_m: 1
  desired_n: 1

lo1:
  name: "LO1"
  freq_range: {{ start: 5.0e9, stop: 5.0e9 }}
  grid_step: 1.0e6
  harmonics: []
  pll_spurs: []

lo2:
  name: "LO2"
  freq_range: {{ start: 4.0e9, stop: 4.0e9 }}
  grid_step: 1.0e6
  harmonics: []
  pll_spurs: []

filters:
  if2_constraints:
    min_filters: 1
    max_filters: 1
    fc_range: {{ start: 5.8e9, stop: 6.2e9 }}
    bw_range: {{ start: 5.0e7, stop: 2.0e8 }}
    slope_range: [-80.0, -20.0]
    feasibility_margin_hz: 1.0e7
  rf_bpf_csv_path: "{rf_bpf_csv_str}"

spur_limits:
  in_band_limit_dbc: -30.0
  out_of_band_limit_dbc: -70.0
  mask: null
  out_of_band_range: null
  mask_eval_mode: "center"

grids:
  if1_grid_step_hz: 5.0e7
  spur_integration_step_hz: 1.0e7
  max_if1_harmonic_order: 3
  min_spur_level_considered_dbc: -120.0
  max_lo_candidates_per_rf: 8
  max_if2_bank_candidates: 8
  coarse_spur_margin_min_db: -10.0
  mixer2_if2_focus_margin_hz: 0.0
  parallel: false
  use_numba: false
  min_lo_candidates_per_rf_after_coarse: 1

if1_harmonics_dbc:
  "2": -30.0

description: "yaml-loaded config"
"""
    cfg_path.write_text(yaml_text)

    cfg = load_config(cfg_path)
    assert float(cfg.if1_band.start) == pytest.approx(9.5e8)
    assert float(cfg.if1_band.stop) == pytest.approx(1.05e9)
    
    assert cfg.mixer1.name == "M1"
    assert float(cfg.lo1.freq_range.start) == pytest.approx(5.0e9)
    
    assert cfg.filters.rf_bpf_csv_path == rf_bpf_csv_str
    
    assert cfg.if1_harmonics_dbc[2] == pytest.approx(-30.0)
    assert cfg.spur_limits.mask is None
    assert cfg.spur_limits.mask_eval_mode == "center"


def test_load_config_from_json_with_defaults(tmp_path, rf_bpf_csv):
    """
    JSON → SystemConfig, exercising the JSON branch and defaulting behaviour
    for optional fields (e.g. mask_eval_mode, grid knobs).
    """
    cfg_path = tmp_path / "config.json"

    cfg_dict = {
        "if1_band": {"start": 9.5e8, "stop": 1.05e9},
        "rf_band": {"start": 9.5e9, "stop": 1.05e10},
        "rf_configurations": [
            {
                "config_id": "cfg1",
                "rf_center": 1.0e10,
                "rf_bandwidth": 1.0e8,
                "if1_subband": None,
            }
        ],
        "non_inverting_mapping_required": True,
        "mixer1": {
            "name": "M1",
            "spur_table": [],
            "spur_envelope": {
                "m_max": 3,
                "n_max": 3,
                "enforce_envelope_completeness": False,
                "unspecified_floor_dbc": None,
            },
            "isolation": {"lo_to_rf_dbc": -80.0, "if_to_rf_dbc": -80.0},
            "ranges": {
                "if_range": {"start": 9.5e8, "stop": 3.0e9},
                "lo_range": {"start": 5.0e9, "stop": 5.0e9},
                "rf_range": {"start": 5.95e9, "stop": 6.05e9},
            },
            "desired_m": 1,
            "desired_n": 1,
        },
        "mixer2": {
            "name": "M2",
            "spur_table": [],
            "spur_envelope": {
                "m_max": 3,
                "n_max": 3,
                "enforce_envelope_completeness": False,
                "unspecified_floor_dbc": None,
            },
            "isolation": {"lo_to_rf_dbc": -80.0, "if_to_rf_dbc": -80.0},
            "ranges": {
                "if_range": {"start": 5.95e9, "stop": 6.05e9},
                "lo_range": {"start": 4.0e9, "stop": 4.0e9},
                "rf_range": {"start": 9.5e9, "stop": 1.05e10},
            },
            "desired_m": 1,
            "desired_n": 1,
        },
        "lo1": {
            "name": "LO1",
            "freq_range": {"start": 5.0e9, "stop": 5.0e9},
            "grid_step": 1.0e6,
            "harmonics": [],
            "pll_spurs": [],
        },
        "lo2": {
            "name": "LO2",
            "freq_range": {"start": 4.0e9, "stop": 4.0e9},
            "grid_step": 1.0e6,
            "harmonics": [],
            "pll_spurs": [],
        },
        "filters": {
            "if2_constraints": {
                "min_filters": 1,
                "max_filters": 1,
                "fc_range": {"start": 5.8e9, "stop": 6.2e9},
                "bw_range": {"start": 5.0e7, "stop": 2.0e8},
                "slope_range": [-80.0, -20.0],
                "feasibility_margin_hz": 1.0e7,
            },
            "rf_bpf_csv_path": str(rf_bpf_csv),
        },
        "spur_limits": {
            "in_band_limit_dbc": -30.0,
            "out_of_band_limit_dbc": -70.0,
            # mask and out_of_band_range are optional
            "mask": None,
            "out_of_band_range": None,
            # omit mask_eval_mode to exercise default
        },
        "grids": {
            "if1_grid_step_hz": 5.0e7,
            "spur_integration_step_hz": 1.0e7,
            "max_if1_harmonic_order": 3,
            # optional knobs omitted to exercise defaults:
            #   min_spur_level_considered_dbc
            #   max_lo_candidates_per_rf
            #   max_if2_bank_candidates
            #   coarse_spur_margin_min_db
            #   mixer2_if2_focus_margin_hz
            #   parallel
            #   use_numba
            #   min_lo_candidates_per_rf_after_coarse
        },
        "if1_harmonics_dbc": {"2": -30.0},
        "description": "json-loaded config",
    }

    cfg_path.write_text(json.dumps(cfg_dict))

    cfg = load_config(cfg_path)

    # Basic structural sanity
    assert cfg.if1_band.start == pytest.approx(9.5e8)
    assert cfg.rf_configurations[0].config_id == "cfg1"
    assert cfg.mixer2.name == "M2"
    assert cfg.filters.rf_bpf_csv_path == str(rf_bpf_csv)

    # Defaults for spur_limits
    assert cfg.spur_limits.mask is None
    assert cfg.spur_limits.mask_eval_mode == "center"

    # Defaults for grids
    assert cfg.grids.min_spur_level_considered_dbc == pytest.approx(-120.0)
    assert cfg.grids.max_lo_candidates_per_rf == 50
    assert cfg.grids.max_if2_bank_candidates == 50
    assert cfg.grids.coarse_spur_margin_min_db == pytest.approx(-10.0)
    assert cfg.grids.mixer2_if2_focus_margin_hz == pytest.approx(0.0)
    assert cfg.grids.parallel is True
    assert cfg.grids.use_numba is False
    assert cfg.grids.min_lo_candidates_per_rf_after_coarse == 1


def test_sanity_warning_when_spur_table_includes_desired_family(tmp_path, rf_bpf_csv):
    """
    load_config should emit a RuntimeWarning when a mixer spur_table contains
    an entry for the desired fundamental (m, n) family.
    """
    cfg_path = tmp_path / "config_with_desired_in_spur_table.json"

    base_cfg = {
        "if1_band": {"start": 9.5e8, "stop": 1.05e9},
        "rf_band": {"start": 9.5e9, "stop": 1.05e10},
        "rf_configurations": [
            {
                "config_id": "cfg1",
                "rf_center": 1.0e10,
                "rf_bandwidth": 1.0e8,
                "if1_subband": None,
            }
        ],
        "non_inverting_mapping_required": True,
        "filters": {
            "if2_constraints": {
                "min_filters": 1,
                "max_filters": 1,
                "fc_range": {"start": 5.8e9, "stop": 6.2e9},
                "bw_range": {"start": 5.0e7, "stop": 2.0e8},
                "slope_range": [-80.0, -20.0],
                "feasibility_margin_hz": 1.0e7,
            },
            "rf_bpf_csv_path": str(rf_bpf_csv),
        },
        "spur_limits": {
            "in_band_limit_dbc": -30.0,
            "out_of_band_limit_dbc": -70.0,
            "mask": None,
            "out_of_band_range": None,
        },
        "grids": {
            "if1_grid_step_hz": 5.0e7,
            "spur_integration_step_hz": 1.0e7,
            "max_if1_harmonic_order": 3,
        },
        "if1_harmonics_dbc": {},
        "description": "config with desired spur family entry",
    }

    spur_envelope = {
        "m_max": 3,
        "n_max": 3,
        "enforce_envelope_completeness": False,
        "unspecified_floor_dbc": None,
    }

    # Mixer1 spur_table explicitly contains the desired (m=1, n=1)
    mixer1 = {
        "name": "M1",
        "spur_table": [
            {
                "m": 1,
                "n": 1,
                "level_dbc": -10.0,
                "if_range": {"start": 9.5e8, "stop": 3.0e9},
                "lo_range": {"start": 5.0e9, "stop": 5.0e9},
                "rf_range": {"start": 5.95e9, "stop": 6.05e9},
            }
        ],
        "spur_envelope": spur_envelope,
        "isolation": {"lo_to_rf_dbc": -80.0, "if_to_rf_dbc": -80.0},
        "ranges": {
            "if_range": {"start": 9.5e8, "stop": 3.0e9},
            "lo_range": {"start": 5.0e9, "stop": 5.0e9},
            "rf_range": {"start": 5.95e9, "stop": 6.05e9},
        },
        "desired_m": 1,
        "desired_n": 1,
    }

    # Mixer2 is simple and does not include the desired spur family
    mixer2 = {
        "name": "M2",
        "spur_table": [],
        "spur_envelope": spur_envelope,
        "isolation": {"lo_to_rf_dbc": -80.0, "if_to_rf_dbc": -80.0},
        "ranges": {
            "if_range": {"start": 5.95e9, "stop": 6.05e9},
            "lo_range": {"start": 4.0e9, "stop": 4.0e9},
            "rf_range": {"start": 9.5e9, "stop": 1.05e10},
        },
        "desired_m": 1,
        "desired_n": 1,
    }

    lo1 = {
        "name": "LO1",
        "freq_range": {"start": 5.0e9, "stop": 5.0e9},
        "grid_step": 1.0e6,
        "harmonics": [],
        "pll_spurs": [],
    }

    lo2 = {
        "name": "LO2",
        "freq_range": {"start": 4.0e9, "stop": 4.0e9},
        "grid_step": 1.0e6,
        "harmonics": [],
        "pll_spurs": [],
    }

    cfg_dict = dict(base_cfg)
    cfg_dict["mixer1"] = mixer1
    cfg_dict["mixer2"] = mixer2
    cfg_dict["lo1"] = lo1
    cfg_dict["lo2"] = lo2

    cfg_path.write_text(json.dumps(cfg_dict))

    with pytest.warns(RuntimeWarning) as record:
        cfg = load_config(cfg_path)

    # Still get a valid SystemConfig
    assert cfg.mixer1.name == "M1"

    messages = [str(w.message) for w in record]
    assert any(
        "spur_table contains the desired fundamental" in msg for msg in messages
    )


def test_sanity_warning_when_envelope_covers_desired_with_floor(tmp_path, rf_bpf_csv):
    """
    load_config should emit a RuntimeWarning when the spur envelope covers the
    desired (m, n) and unspecified_floor_dbc is set but there is no explicit
    spur_table entry for that family.
    """
    cfg_path = tmp_path / "config_with_floor_and_no_desired_entry.json"

    spur_envelope_with_floor = {
        "m_max": 3,
        "n_max": 3,
        "enforce_envelope_completeness": False,
        "unspecified_floor_dbc": -90.0,
    }

    base_cfg = {
        "if1_band": {"start": 9.5e8, "stop": 1.05e9},
        "rf_band": {"start": 9.5e9, "stop": 1.05e10},
        "rf_configurations": [
            {
                "config_id": "cfg1",
                "rf_center": 1.0e10,
                "rf_bandwidth": 1.0e8,
                "if1_subband": None,
            }
        ],
        "non_inverting_mapping_required": True,
        "filters": {
            "if2_constraints": {
                "min_filters": 1,
                "max_filters": 1,
                "fc_range": {"start": 5.8e9, "stop": 6.2e9},
                "bw_range": {"start": 5.0e7, "stop": 2.0e8},
                "slope_range": [-80.0, -20.0],
                "feasibility_margin_hz": 1.0e7,
            },
            "rf_bpf_csv_path": str(rf_bpf_csv),
        },
        "spur_limits": {
            "in_band_limit_dbc": -30.0,
            "out_of_band_limit_dbc": -70.0,
            "mask": None,
            "out_of_band_range": None,
        },
        "grids": {
            "if1_grid_step_hz": 5.0e7,
            "spur_integration_step_hz": 1.0e7,
            "max_if1_harmonic_order": 3,
        },
        "if1_harmonics_dbc": {},
        "description": "config with spur envelope floor covering desired",
    }

    mixer1 = {
        "name": "M1",
        "spur_table": [],  # no explicit (m=1, n=1)
        "spur_envelope": spur_envelope_with_floor,
        "isolation": {"lo_to_rf_dbc": -80.0, "if_to_rf_dbc": -80.0},
        "ranges": {
            "if_range": {"start": 9.5e8, "stop": 3.0e9},
            "lo_range": {"start": 5.0e9, "stop": 5.0e9},
            "rf_range": {"start": 5.95e9, "stop": 6.05e9},
        },
        "desired_m": 1,
        "desired_n": 1,
    }

    mixer2 = {
        "name": "M2",
        "spur_table": [],  # likewise no explicit desired family
        "spur_envelope": spur_envelope_with_floor,
        "isolation": {"lo_to_rf_dbc": -80.0, "if_to_rf_dbc": -80.0},
        "ranges": {
            "if_range": {"start": 5.95e9, "stop": 6.05e9},
            "lo_range": {"start": 4.0e9, "stop": 4.0e9},
            "rf_range": {"start": 9.5e9, "stop": 1.05e10},
        },
        "desired_m": 1,
        "desired_n": 1,
    }

    lo1 = {
        "name": "LO1",
        "freq_range": {"start": 5.0e9, "stop": 5.0e9},
        "grid_step": 1.0e6,
        "harmonics": [],
        "pll_spurs": [],
    }

    lo2 = {
        "name": "LO2",
        "freq_range": {"start": 4.0e9, "stop": 4.0e9},
        "grid_step": 1.0e6,
        "harmonics": [],
        "pll_spurs": [],
    }

    cfg_dict = dict(base_cfg)
    cfg_dict["mixer1"] = mixer1
    cfg_dict["mixer2"] = mixer2
    cfg_dict["lo1"] = lo1
    cfg_dict["lo2"] = lo2

    cfg_path.write_text(json.dumps(cfg_dict))

    with pytest.warns(RuntimeWarning) as record:
        cfg = load_config(cfg_path)

    # SystemConfig should still be constructed
    assert cfg.mixer1.spur_envelope.unspecified_floor_dbc == pytest.approx(-90.0)

    messages = [str(w.message) for w in record]
    assert any(
        "spur_envelope covers the desired fundamental" in msg for msg in messages
    )
