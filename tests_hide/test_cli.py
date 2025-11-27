# tests/test_cli.py
from __future__ import annotations

from pathlib import Path

import pytest

from buc_planner.cli import main as cli_main


def test_cli_smoke(tmp_path, rf_bpf_csv):
    """
    Run the CLI end-to-end on a small YAML config and check that it produces the
    expected output files. This reuses the structure from test_load_config.
    """
    cfg_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"

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
  rf_bpf_csv_path: "{rf_bpf_csv}"

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
"""
    cfg_path.write_text(yaml_text)

    cli_main([str(cfg_path), "--out-dir", str(out_dir), "--no-plots"])

    # Basic artifacts should exist
    assert (out_dir / "lo_plan_policy.jsonl").exists()
    assert (out_dir / "spur_ledger.jsonl").exists()
    assert (out_dir / "if2_bank.json").exists()
    assert (out_dir / "run_metadata.json").exists()

def test_cli_missing_file():
    with pytest.raises(FileNotFoundError):
        cli_main(["non_existent_config.yaml"])