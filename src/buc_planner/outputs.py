# src/buc_planner/outputs.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np

from .config_models import SystemConfig
from .if2_bank import IF2BankDesignResult
from .los import LOPlanCandidate
from .spur_engine import SpurResult, ConfigSpurSummary


def write_lo_plan_policy(
    path: str | Path,
    cfg: SystemConfig,
    lo_plans: List[LOPlanCandidate],
    bank_design: IF2BankDesignResult,
    summaries: Dict[str, ConfigSpurSummary],
) -> None:
    """
    JSONL: one record per RF configuration, describing LO1, LO2, IF2 filter, spur summary,
    and key modelling assumptions.
    """
    path = Path(path)
    rf_by_id = {rc.config_id: rc for rc in cfg.rf_configurations}

    with path.open("w") as f:
        for plan in lo_plans:
            summary = summaries.get(plan.config_id)
            rf_conf = rf_by_id[plan.config_id]

            record = {
                "config_id": plan.config_id,
                "rf_center": rf_conf.rf_center,
                "rf_bandwidth": rf_conf.rf_bandwidth,
                "lo1_freq": plan.lo1_freq,
                "lo1_sign": plan.sign_combo.sign1,
                "lo2_freq": plan.lo2_freq,
                "lo2_sign": plan.sign_combo.sign2,
                "if2_filter_id": bank_design.config_to_filter_id.get(plan.config_id),
                "if2_band": {
                    "start": plan.if2_band.start,
                    "stop": plan.if2_band.stop,
                },
                "rf_band": {
                    "start": plan.rf_band.start,
                    "stop": plan.rf_band.stop,
                },
                "spur_summary": {
                    "worst_in_band_spur_dbc": getattr(summary, "worst_in_band_spur_dbc", None),
                    "worst_in_band_margin_db": getattr(summary, "worst_in_band_margin_db", None),
                    "worst_out_band_spur_dbc": getattr(summary, "worst_out_band_spur_dbc", None),
                    "worst_out_band_margin_db": getattr(summary, "worst_out_band_margin_db", None),
                } if summary else None,
                "spur_table_assumptions": {
                    "mixer1": {
                        "enforce_envelope_completeness": cfg.mixer1.spur_envelope.enforce_envelope_completeness,
                        "unspecified_floor_dbc": cfg.mixer1.spur_envelope.unspecified_floor_dbc,
                    },
                    "mixer2": {
                        "enforce_envelope_completeness": cfg.mixer2.spur_envelope.enforce_envelope_completeness,
                        "unspecified_floor_dbc": cfg.mixer2.spur_envelope.unspecified_floor_dbc,
                    },
                },
                "spur_limits": {
                    "in_band_limit_dbc": cfg.spur_limits.in_band_limit_dbc,
                    "out_of_band_limit_dbc": cfg.spur_limits.out_of_band_limit_dbc,
                    "mask": {
                        "csv_path": cfg.spur_limits.mask.csv_path,
                        "apply_in_band": cfg.spur_limits.mask.apply_in_band,
                        "apply_out_of_band": cfg.spur_limits.mask.apply_out_of_band,
                    } if cfg.spur_limits.mask is not None else None,
                    "out_of_band_range": {
                        "start": cfg.spur_limits.out_of_band_range.start,
                        "stop": cfg.spur_limits.out_of_band_range.stop,
                    } if cfg.spur_limits.out_of_band_range is not None else None,
                    "mask_eval_mode": cfg.spur_limits.mask_eval_mode,
                },
                "heuristics": {
                    "min_if2_filters": cfg.filters.if2_constraints.min_filters,
                    "max_if2_filters": cfg.filters.if2_constraints.max_filters,
                    "max_lo_candidates_per_rf": cfg.grids.max_lo_candidates_per_rf,
                    "max_if2_bank_candidates": cfg.grids.max_if2_bank_candidates,
                    "min_spur_level_considered_dbc": cfg.grids.min_spur_level_considered_dbc,
                    "mixer2_if2_focus_margin_hz": cfg.grids.mixer2_if2_focus_margin_hz,
                },
                "modelling_assumptions": {
                    "lo1_lo2_cross_coupling_modelled": False,
                    "if1_grid_segmented": True,
                    "planning_grade_only": True,
                },
            }
            f.write(json.dumps(record) + "\n")

def write_spur_ledger(
    path: str | Path,
    spur_results: List[SpurResult],
    top_n_per_config: int = 50,
) -> None:
    """
    JSONL spur ledger: top N spurs per configuration by level (worst = largest).
    Includes scalar/mask limit information and in-band/out-of-band flags.
    """
    path = Path(path)
    by_cfg: Dict[str, List[SpurResult]] = {}
    for r in spur_results:
        by_cfg.setdefault(r.config_id, []).append(r)

    with path.open("w") as f:
        for config_id, items in by_cfg.items():
            items_sorted = sorted(items, key=lambda r: r.level_dbc, reverse=True)
            for r in items_sorted[:top_n_per_config]:
                rec = {
                    "config_id": r.config_id,
                    "mixer_name": r.mixer_name,
                    "spur_name": r.spur_name,
                    "f_start": r.f_start,
                    "f_stop": r.f_stop,
                    "in_band": r.in_band,
                    "out_of_band": r.out_of_band,
                    "level_dbc": r.level_dbc,
                    # Absolute level in dB under the current planning convention
                    # that the desired RF integrated power is 0 dBc == 0 dB.
                    # If a future model introduces non-zero absolute desired RF
                    # power, this field should be updated accordingly.
                    "level_db_absolute": r.level_dbc,  # assuming desired RF = 0 dBc
                    "margin_db": r.margin_db,
                    "scalar_limit_dbc": r.scalar_limit_dbc,
                    "mask_limit_dbc": r.mask_limit_dbc,
                    "scalar_margin_db": r.scalar_margin_db,
                    "mask_margin_db": r.mask_margin_db,
                    "filter_att_if2_db": r.filter_att_if2_db,
                    "filter_att_rf_db": r.filter_att_rf_db,
                    "origin_m": r.origin_m,
                    "origin_n": r.origin_n,
                    "lo_tone_name": r.lo_tone_name,
                    "input_band_name": r.input_band_name,
                    "used_unspecified_floor": r.used_unspecified_floor,
                }
                f.write(json.dumps(rec) + "\n")
				
def write_if2_bank_description(
    path_json: str | Path,
    csv_dir: str | Path,
    bank_design: IF2BankDesignResult,
    freq_min: float,
    freq_max: float,
    n_points: int = 2001,
) -> None:
    """
    Write IF2 bank description and per-filter CSVs.

    JSON structure:
      {
        "filters": [
          {"filter_id": ..., "fc": ..., "bw": ..., "slope_db_per_decade": ...},
          ...
        ],
        "config_to_filter_id": { "cfg1": "filter_id", ... }
      }

    CSVs:
      One file per filter in csv_dir, named "<filter_id>.csv",
      with columns: frequency, attenuation_db.
    """
    path_json = Path(path_json)
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    filters_info = []
    for f in bank_design.bank.filters:
        filters_info.append(
            {
                "filter_id": f.filter_id,
                "fc": f.fc,
                "bw": f.bw,
                "slope_db_per_decade": f.slope_db_per_decade,
            }
        )

        freqs = np.linspace(freq_min, freq_max, n_points)
        att = f.attenuation_db(freqs)
        csv_path = csv_dir / f"{f.filter_id}.csv"
        data = np.column_stack([freqs, att])
        np.savetxt(csv_path, data, delimiter=",", header="freq,attenuation_db", comments="")

    blob = {
        "filters": filters_info,
        "config_to_filter_id": bank_design.config_to_filter_id,
    }
    path_json.write_text(json.dumps(blob, indent=2))

def write_run_metadata(
    path: str | Path,
    cfg: SystemConfig,
    spur_results: List[SpurResult],
) -> None:
    """
    Write a small JSON metadata blob describing modelling options,
    spur-floor usage, and high-level complexity knobs.

    This is intended as a quick "header" for the run.
    """
    path = Path(path)

    n_spurs = len(spur_results)
    n_spurs_with_floor = sum(1 for r in spur_results if r.used_unspecified_floor)

    metadata = {
        "description": cfg.description,
        "n_rf_configurations": len(cfg.rf_configurations),
        "if1_band": {"start": cfg.if1_band.start, "stop": cfg.if1_band.stop},
        "rf_band": {"start": cfg.rf_band.start, "stop": cfg.rf_band.stop},
        "non_inverting_mapping_required": cfg.non_inverting_mapping_required,
        "mixer1": {
            "name": cfg.mixer1.name,
            "enforce_envelope_completeness": cfg.mixer1.spur_envelope.enforce_envelope_completeness,
            "unspecified_floor_dbc": cfg.mixer1.spur_envelope.unspecified_floor_dbc,
        },
        "mixer2": {
            "name": cfg.mixer2.name,
            "enforce_envelope_completeness": cfg.mixer2.spur_envelope.enforce_envelope_completeness,
            "unspecified_floor_dbc": cfg.mixer2.spur_envelope.unspecified_floor_dbc,
        },
        "grids": {
            "if1_grid_step_hz": cfg.grids.if1_grid_step_hz,
            "spur_integration_step_hz": cfg.grids.spur_integration_step_hz,
            "max_if1_harmonic_order": cfg.grids.max_if1_harmonic_order,
            "coarse_spur_margin_min_db": cfg.grids.coarse_spur_margin_min_db,
            "mixer2_if2_focus_margin_hz": cfg.grids.mixer2_if2_focus_margin_hz,
            "use_numba": cfg.grids.use_numba,
        },
        "spur_limits": {
            "in_band_limit_dbc": cfg.spur_limits.in_band_limit_dbc,
            "out_of_band_limit_dbc": cfg.spur_limits.out_of_band_limit_dbc,
            "mask_csv": cfg.spur_limits.mask.csv_path
            if cfg.spur_limits.mask is not None
            else None,
            "mask_eval_mode": cfg.spur_limits.mask_eval_mode,
        },
        "spur_floor_usage": {
            "total_spurs": n_spurs,
            "spurs_with_unspecified_floor": n_spurs_with_floor,
        },
        "modelling_notes": {
            "lo1_lo2_cross_coupling_modelled": False,
            "planning_grade_only": True,
            "wideband_spur_approximation": True,
        },
    }

    path.write_text(json.dumps(metadata, indent=2))