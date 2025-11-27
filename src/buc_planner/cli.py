# src/buc_planner/cli.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .config_models import load_config
from .optimizer import Planner
from .outputs import (
    write_lo_plan_policy,
    write_spur_ledger,
    write_if2_bank_description,
    write_run_metadata,
)
from .plotting import plot_if2_bank
from .progress import ProgressReporter


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Dual-Conversion BUC Frequency Planner & Spur Analyzer"
    )
    parser.add_argument("config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="buc_planner_out",
        help="Output directory",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable IF2 bank plot generation",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable textual progress indicators",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    
    progress = None if args.no_progress else ProgressReporter()
    planner = Planner(cfg, progress=progress)
    result = planner.run()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # LO plan policy
    write_lo_plan_policy(
        out_dir / "lo_plan_policy.jsonl",
        cfg,
        result.lo_plans,
        result.if2_bank_design,
        result.summaries,
    )

    # Spur ledger
    write_spur_ledger(
        out_dir / "spur_ledger.jsonl",
        result.spur_results,
        top_n_per_config=50,
    )

    # IF2 bank description & CSVs
    if2_json = out_dir / "if2_bank.json"
    if2_csv_dir = out_dir / "if2_filters"
    # freq range for plotting CSVs (use mixer2 IF range as reference)
    freq_min = cfg.mixer2.ranges.if_range.start
    freq_max = cfg.mixer2.ranges.if_range.stop
    write_if2_bank_description(
        if2_json,
        if2_csv_dir,
        result.if2_bank_design,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    if not args.no_plots:
        plot_if2_bank(
            result.if2_bank_design.bank,
            cfg.mixer2.ranges.if_range,
            out_path=out_dir / "if2_bank.png",
            lo_plans=result.lo_plans,
        )

    # Run metadata
    write_run_metadata(
        out_dir / "run_metadata.json",
        cfg,
        result.spur_results,
    )


if __name__ == "__main__":
    main()
