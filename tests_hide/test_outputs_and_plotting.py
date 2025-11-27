# tests/test_outputs_and_plotting.py
from __future__ import annotations

import json

import numpy as np

from buc_planner.if2_bank import design_if2_bank_for_lo_plans
from buc_planner.los import generate_lo_plan_candidates_for_config
from buc_planner.outputs import (
    write_lo_plan_policy,
    write_spur_ledger,
    write_if2_bank_description,
    write_run_metadata,
)
from buc_planner.plotting import plot_if2_bank
from buc_planner.spur_engine import SpurResult, ConfigSpurSummary


def _minimal_spur_results(config_id: str) -> list[SpurResult]:
    """
    Construct a single toy SpurResult primarily for ledger / metadata tests.
    """
    scalar_limit = -50.0
    level = -60.0
    margin = scalar_limit - level

    return [
        SpurResult(
            config_id=config_id,
            mixer_name="M2",
            spur_name="test_spur",
            f_start=1.0,
            f_stop=2.0,
            in_band=True,
            out_of_band=False,
            level_dbc=level,
            margin_db=margin,
            filter_att_if2_db=5.0,
            filter_att_rf_db=10.0,
            origin_m=1,
            origin_n=1,
            lo_tone_name="fundamental",
            input_band_name="IF2_desired_band",
            used_unspecified_floor=False,
            scalar_limit_dbc=scalar_limit,
            mask_limit_dbc=None,
            scalar_margin_db=margin,
            mask_margin_db=None,
        )
    ]


def _minimal_summary(config_id: str) -> ConfigSpurSummary:
    """
    Minimal ConfigSpurSummary matching _minimal_spur_results().
    """
    return ConfigSpurSummary(
        config_id=config_id,
        worst_in_band_spur_dbc=-60.0,
        worst_in_band_margin_db=10.0,
        worst_out_band_spur_dbc=None,
        worst_out_band_margin_db=None,
    )


def test_outputs_and_plotting_end_to_end(simple_system_config, tmp_path):
    """
    End-to-end smoke test for:
      * LO plan policy writer
      * Spur ledger writer
      * IF2 bank description + per-filter CSVs
      * Run metadata writer
      * IF2 bank plotting

    The goal is to verify that all output functions produce structurally
    consistent artefacts that can be consumed later in a tooling chain.
    """
    cfg = simple_system_config
    rf_conf = cfg.rf_configurations[0]

    # --- LO candidates & IF2 bank design ---------------------------------
    lo_plans = generate_lo_plan_candidates_for_config(cfg, rf_conf)
    assert lo_plans, "Expected at least one LO plan candidate for the simple config"

    bank_design = design_if2_bank_for_lo_plans(
        lo_plans=lo_plans,
        constraints=cfg.filters.if2_constraints,
        spur_control_regions=None,
    )

    # --- LO plan policy JSONL --------------------------------------------
    lo_policy_path = tmp_path / "lo_policy.jsonl"
    summaries = {rf_conf.config_id: _minimal_summary(rf_conf.config_id)}

    write_lo_plan_policy(
        lo_policy_path,
        cfg,
        lo_plans,
        bank_design,
        summaries,
    )

    lines = lo_policy_path.read_text().strip().splitlines()
    assert lines, "LO plan policy JSONL should contain at least one record"

    rec = json.loads(lines[0])
    assert rec["config_id"] == rf_conf.config_id
    assert rec["if2_filter_id"] == bank_design.config_to_filter_id[rf_conf.config_id]
    assert "spur_summary" in rec
    assert rec["spur_summary"]["worst_in_band_spur_dbc"] == -60.0

    # --- Spur ledger JSONL -----------------------------------------------
    ledger_path = tmp_path / "spur_ledger.jsonl"
    spur_results = _minimal_spur_results(rf_conf.config_id)

    write_spur_ledger(ledger_path, spur_results, top_n_per_config=10)

    ledger_lines = ledger_path.read_text().strip().splitlines()
    assert ledger_lines, "Spur ledger JSONL should contain at least one record"

    rec2 = json.loads(ledger_lines[0])
    assert rec2["config_id"] == rf_conf.config_id
    assert rec2["spur_name"] == "test_spur"
    assert rec2["level_dbc"] == -60.0
    # level_db_absolute is currently defined as equal to level_dbc
    assert rec2["level_db_absolute"] == rec2["level_dbc"]

    # --- IF2 bank description + per-filter CSVs --------------------------
    if2_json = tmp_path / "if2_bank.json"
    if2_csv_dir = tmp_path / "if2_filters"
    m2_if_range = cfg.mixer2.ranges.if_range

    write_if2_bank_description(
        if2_json,
        if2_csv_dir,
        bank_design,
        freq_min=m2_if_range.start,
        freq_max=m2_if_range.stop,
    )

    blob = json.loads(if2_json.read_text())
    assert "filters" in blob
    assert blob["filters"], "IF2 bank JSON should list at least one filter"
    assert "config_to_filter_id" in blob
    assert if2_csv_dir.is_dir(), "IF2 CSV directory should exist"

    # One CSV per filter with the expected header and at least one sample row
    for f in bank_design.bank.filters:
        csv_path = if2_csv_dir / f"{f.filter_id}.csv"
        assert csv_path.exists(), f"Missing CSV for filter {f.filter_id}"
        text = csv_path.read_text().strip().splitlines()
        assert text[0].strip().lower() == "freq,attenuation_db"
        # Use numpy to check that we can parse the numeric data
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        if data.ndim == 1:
            # Single-row CSV still counts as valid
            assert data.size == 2
        else:
            assert data.shape[1] == 2
            assert data.shape[0] >= 2  # at least a couple of points

    # --- Run metadata ----------------------------------------------------
    meta_path = tmp_path / "run_metadata.json"
    write_run_metadata(meta_path, cfg, spur_results)
    meta = json.loads(meta_path.read_text())

    assert meta["n_rf_configurations"] == 1
    assert meta["spur_floor_usage"]["total_spurs"] == len(spur_results)
    assert meta["spur_floor_usage"]["spurs_with_unspecified_floor"] == 0
    assert meta["mixer1"]["name"] == cfg.mixer1.name

    # --- Plotting – just check that a PNG gets written -------------------
    out_png = tmp_path / "if2_bank.png"
    plot_if2_bank(
        bank=bank_design.bank,
        freq_range=m2_if_range,
        out_path=out_png,
        lo_plans=lo_plans,
    )
    assert out_png.exists(), "IF2 bank plot PNG should have been created"


def test_write_spur_ledger_respects_top_n(tmp_path):
    """
    Ensure write_spur_ledger honours the top_n_per_config argument by only
    emitting the worst (highest level_dbc) N spurs per configuration.
    """
    config_id = "cfgX"
    path = tmp_path / "spur_ledger_topn.jsonl"

    spur1 = SpurResult(
        config_id=config_id,
        mixer_name="M2",
        spur_name="spur_low",
        f_start=1.0,
        f_stop=2.0,
        in_band=True,
        out_of_band=False,
        level_dbc=-80.0,
        margin_db=30.0,
        filter_att_if2_db=0.0,
        filter_att_rf_db=0.0,
        origin_m=1,
        origin_n=1,
        lo_tone_name="fundamental",
        input_band_name="IF2_desired_band",
        used_unspecified_floor=False,
        scalar_limit_dbc=-50.0,
        mask_limit_dbc=None,
        scalar_margin_db=-50.0 - (-80.0),
        mask_margin_db=None,
    )
    spur2 = SpurResult(
        config_id=config_id,
        mixer_name="M2",
        spur_name="spur_high",
        f_start=3.0,
        f_stop=4.0,
        in_band=True,
        out_of_band=False,
        level_dbc=-40.0,
        margin_db=10.0,
        filter_att_if2_db=0.0,
        filter_att_rf_db=0.0,
        origin_m=1,
        origin_n=1,
        lo_tone_name="fundamental",
        input_band_name="IF2_desired_band",
        used_unspecified_floor=False,
        scalar_limit_dbc=-30.0,
        mask_limit_dbc=None,
        scalar_margin_db=-30.0 - (-40.0),
        mask_margin_db=None,
    )

    write_spur_ledger(path, [spur1, spur2], top_n_per_config=1)

    lines = path.read_text().strip().splitlines()
    # Only one line should be present due to top_n_per_config=1
    assert len(lines) == 1

    rec = json.loads(lines[0])
    # The kept spur should be the "worst" (largest level_dbc, i.e. -40 dBc)
    assert rec["spur_name"] == "spur_high"
    assert rec["level_dbc"] == -40.0
