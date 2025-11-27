# tests/test_config_models.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from buc_planner.config_models import Range, load_config


def _make_raw_config(rf_bpf_csv_path: Path) -> dict:
    """
    Build a small but self-consistent raw config dict suitable for load_config.

    It is intentionally similar to the simple_system_config fixture, but goes
    through the YAML/JSON loader path so we exercise the parsing logic and the
    spur sanity checks.
    """
    return {
        "if1_band": {"start": 9.50e8, "stop": 1.05e9},
        "rf_band": {"start": 9.5e9, "stop": 1.05e10},
        "rf_configurations": [
            {
                "config_id": "cfg1",
                "rf_center": 1.0e10,
                "rf_bandwidth": 1.0e8,
                # if1_subband omitted -> use global IF1 band
            }
        ],
        "non_inverting_mapping_required": True,
        # Mixer1: spur_table *includes* desired (m=1,n=1) → first sanity warning
        "mixer1": {
            "name": "M1",
            "spur_table": [
                {
                    "m": 1,
                    "n": 1,
                    "level_dbc": -1.0,
                    "if_range": {"start": 9.50e8, "stop": 3.0e9},
                    "lo_range": {"start": 5.0e9, "stop": 5.0e9},
                    "rf_range": {"start": 5.95e9, "stop": 6.05e9},
                    "lo_tone_type": "fundamental",
                }
            ],
            "spur_envelope": {
                "m_max": 3,
                "n_max": 3,
                "enforce_envelope_completeness": False,
                # unspecified_floor_dbc omitted -> None
            },
            "isolation": {
                "lo_to_rf_dbc": -80.0,
                "if_to_rf_dbc": -80.0,
            },
            "ranges": {
                "if_range": {"start": 9.50e8, "stop": 3.0e9},
                "lo_range": {"start": 5.0e9, "stop": 5.0e9},
                "rf_range": {"start": 5.95e9, "stop": 6.05e9},
            },
            "desired_m": 1,
            "desired_n": 1,
        },
        # Mixer2: spur envelope covers desired (m=1,n=1) and has a floor,
        # but no spur_table entry for the desired family → second sanity warning
        "mixer2": {
            "name": "M2",
            "spur_table": [],
            "spur_envelope": {
                "m_max": 3,
                "n_max": 3,
                "enforce_envelope_completeness": False,
                "unspecified_floor_dbc": -90.0,
            },
            "isolation": {
                "lo_to_rf_dbc": -80.0,
                "if_to_rf_dbc": -80.0,
            },
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
            "harmonics": [
                {"order": 2, "level_dbc": -30.0},
            ],
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
            "rf_bpf_csv_path": str(rf_bpf_csv_path),
        },
        "spur_limits": {
            "in_band_limit_dbc": -30.0,
            "out_of_band_limit_dbc": -70.0,
            "mask": None,
            "out_of_band_range": None,
            "mask_eval_mode": "center",
        },
        "grids": {
            "if1_grid_step_hz": 5.0e7,
            "spur_integration_step_hz": 1.0e7,
            "max_if1_harmonic_order": 3,
            "min_spur_level_considered_dbc": -120.0,
            "max_lo_candidates_per_rf": 8,
            "max_if2_bank_candidates": 8,
            "coarse_spur_margin_min_db": -10.0,
            "mixer2_if2_focus_margin_hz": 0.0,
            "parallel": False,
            "use_numba": False,
            "min_lo_candidates_per_rf_after_coarse": 1,
        },
        # IF1 2nd harmonic at −30 dBc integrated
        "if1_harmonics_dbc": {2: -30.0},
        "description": "unit-test config",
    }


def test_range_contains_and_intersect_basic():
    r = Range(start=0.0, stop=10.0)

    # contains is inclusive on both ends
    assert r.contains(0.0)
    assert r.contains(10.0)
    assert not r.contains(-1.0)
    assert not r.contains(10.01)

    # width property
    assert r.width == pytest.approx(10.0)

    # intersect with overlapping range
    r2 = Range(start=5.0, stop=15.0)
    inter = r.intersect(r2)
    assert inter is not None
    assert inter.start == pytest.approx(5.0)
    assert inter.stop == pytest.approx(10.0)
    assert inter.width == pytest.approx(5.0)

    # intersect with disjoint range
    r3 = Range(start=20.0, stop=30.0)
    assert r.intersect(r3) is None


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_load_config_parses_and_emits_sanity_warnings(tmp_path, rf_bpf_csv, suffix):
    """
    load_config must:

    * parse both YAML and JSON configs,
    * emit two RuntimeWarnings from the internal spur sanity checks:
        - spur_table contains the desired fundamental (Mixer1),
        - spur_envelope covers the desired fundamental with a floor but no
          spur_table entry (Mixer2),
    * and correctly populate key fields of SystemConfig.
    """
    raw = _make_raw_config(rf_bpf_csv)
    cfg_path = tmp_path / f"cfg{suffix}"

    if suffix == ".yaml":
        cfg_path.write_text(yaml.safe_dump(raw))
    else:
        cfg_path.write_text(json.dumps(raw))

    with pytest.warns(RuntimeWarning) as record:
        cfg = load_config(cfg_path)

    messages = [str(w.message) for w in record]
    # One warning for Mixer1 having the desired (m,n) in spur_table
    assert any("spur_table contains the desired fundamental" in m for m in messages)
    # One warning for Mixer2 envelope + unspecified_floor_dbc covering desired (m,n)
    assert any("spur_envelope covers the desired fundamental" in m for m in messages)

    # Spot-check a few parsed fields
    assert cfg.if1_band.start == pytest.approx(9.50e8)
    assert cfg.rf_band.stop == pytest.approx(1.05e10)
    assert cfg.rf_configurations[0].config_id == "cfg1"

    assert cfg.mixer1.name == "M1"
    assert cfg.mixer2.spur_envelope.unspecified_floor_dbc == pytest.approx(-90.0)

    assert cfg.lo1.freq_range.start == pytest.approx(5.0e9)
    assert cfg.lo1.grid_step == pytest.approx(1.0e6)

    assert cfg.filters.if2_constraints.fc_range.start == pytest.approx(5.8e9)
    assert cfg.filters.if2_constraints.bw_range.stop == pytest.approx(2.0e8)

    # if1_harmonics_dbc keys must have been converted to ints
    assert cfg.if1_harmonics_dbc[2] == pytest.approx(-30.0)


def test_load_config_defaults_non_inverting_true(tmp_path, rf_bpf_csv):
    """
    If non_inverting_mapping_required is omitted in the raw config,
    load_config must default it to True.
    """
    raw = _make_raw_config(rf_bpf_csv)
    raw.pop("non_inverting_mapping_required", None)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    # Same spur sanity warnings as before; we don't re-assert their content here.
    with pytest.warns(RuntimeWarning):
        cfg = load_config(cfg_path)

    assert cfg.non_inverting_mapping_required is True
