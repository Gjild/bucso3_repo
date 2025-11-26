# src/buc_planner/los.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .config_models import LOSynthConfig, Range, RfConfiguration, SystemConfig, Freq


@dataclass(frozen=True)
class LOSignCombination:
    """
    sign1, sign2 are ±1 representing mixer senses:
        Mixer1: IF2 = LO1 + sign1 * IF1
        Mixer2: RF  = LO2 + sign2 * IF2

    Overall IF1→RF derivative sign ~ sign1 * sign2.
    """
    sign1: int
    sign2: int

    def is_non_inverting(self) -> bool:
        return self.sign1 * self.sign2 > 0


@dataclass
class LOPlanCandidate:
    config_id: str
    lo1_freq: Freq
    lo2_freq: Freq
    sign_combo: LOSignCombination
    if2_band: Range  # required IF2 band for desired path
    rf_band: Range   # resulting RF band from desired path


def _enumerate_lo_grid(lo_cfg: LOSynthConfig) -> np.ndarray:
    """Enumerate LO grid from freq_range with step size."""
    start = lo_cfg.freq_range.start
    stop = lo_cfg.freq_range.stop
    step = lo_cfg.grid_step
    n_steps = int(np.floor((stop - start) / step)) + 1
    return start + np.arange(n_steps) * step


def derive_if2_band_for_lo1(
    if1_band: Range,
    lo1_freq: Freq,
    sign1: int,
) -> Range:
    """
    IF2 = LO1 + sign1 * IF1, where IF1 in [if1_lo, if1_hi].
    """
    if sign1 > 0:
        if2_lo = lo1_freq + sign1 * if1_band.start
        if2_hi = lo1_freq + sign1 * if1_band.stop
    else:
        # sign1 = -1 -> IF2 = LO1 - IF1, and if1 increases => IF2 decreases
        # so band is [LO1 - IF1_hi, LO1 - IF1_lo]
        if2_lo = lo1_freq - if1_band.stop
        if2_hi = lo1_freq - if1_band.start
    return Range(start=if2_lo, stop=if2_hi)


def derive_rf_band_for_lo2(
    if2_band: Range,
    lo2_freq: Freq,
    sign2: int,
) -> Range:
    if sign2 > 0:
        rf_lo = lo2_freq + sign2 * if2_band.start
        rf_hi = lo2_freq + sign2 * if2_band.stop
    else:
        # RF = LO2 - IF2
        rf_lo = lo2_freq - if2_band.stop
        rf_hi = lo2_freq - if2_band.start
    return Range(start=rf_lo, stop=rf_hi)


def generate_lo_plan_candidates_for_config(
    cfg: SystemConfig,
    rf_conf: RfConfiguration,
    max_candidates: int | None = None,
) -> List[LOPlanCandidate]:
    """
    Enumerate coarse LO1/LO2 candidates for a given RF configuration,
    obeying synthesizer ranges, mixer ranges, and non-inverting mapping.

    Strategy:
      * Build an IF1 grid over the (sub-)band clipped to Mixer1 IF range.
      * For each LO1/sign1:
          - Derive IF2(f_IF1) and check Mixer1 RF range.
          - Ensure full IF1 band lies within Mixer1 IF range (spec 6.4).
          - Intersect IF2 band with Mixer2 IF range.
      * For each LO2/sign2:
          - Derive RF(f_IF1) on the IF1 grid.
          - Enforce strict monotonic IF1->RF mapping (if requested).
          - Ensure RF(f_IF1) covers the RF configuration band and stays
            inside Mixer2 RF & global RF bands over entire IF1 grid.
    """
    # Effective IF1 band for this configuration
    if1_band_cfg = rf_conf.if1_subband or cfg.if1_band

    # Clip IF1 band to Mixer1 IF range; if no overlap, configuration is invalid
    m1_if_range = cfg.mixer1.ranges.if_range
    if1_band = if1_band_cfg.intersect(m1_if_range)
    if if1_band is None or if1_band.width <= 0:
        raise RuntimeError(
            f"RF config '{rf_conf.config_id}' IF1 band lies outside Mixer1 IF range."
        )

    lo1_grid = _enumerate_lo_grid(cfg.lo1)
    lo2_grid = _enumerate_lo_grid(cfg.lo2)

    # IF1 grid for mapping checks
    step = cfg.grids.if1_grid_step_hz
    if step <= 0:
        # Fallback: coarse 32-point grid if misconfigured
        n_pts = 32
        if1_grid = np.linspace(if1_band.start, if1_band.stop, n_pts)
    else:
        n_pts = max(int(np.floor(if1_band.width / step)) + 1, 3)
        if1_grid = np.linspace(if1_band.start, if1_band.stop, n_pts)

    lo_sign_combos = [
        LOSignCombination(+1, +1),
        LOSignCombination(-1, -1),
    ]
    if not cfg.non_inverting_mapping_required:
        lo_sign_combos.extend([
            LOSignCombination(+1, -1),
            LOSignCombination(-1, +1),
        ])

    target_rf_lo = rf_conf.rf_center - 0.5 * rf_conf.rf_bandwidth
    target_rf_hi = rf_conf.rf_center + 0.5 * rf_conf.rf_bandwidth

    m1 = cfg.mixer1
    m2 = cfg.mixer2

    candidates: List[LOPlanCandidate] = []

    for lo1_freq in lo1_grid:
        # Mixer1 LO range
        if not m1.ranges.lo_range.contains(lo1_freq):
            continue

        for sign_combo in lo_sign_combos:
            if cfg.non_inverting_mapping_required and not sign_combo.is_non_inverting():
                continue

            # IF2 mapping from IF1 grid
            if sign_combo.sign1 > 0:
                if2_vals = lo1_freq + sign_combo.sign1 * if1_grid
            else:
                if2_vals = lo1_freq - if1_grid  # LO1 - IF1

            if2_band = Range(start=float(if2_vals.min()), stop=float(if2_vals.max()))

            # Mixer1 RF range is effectively IF2 node; require full IF2 band inside
            if not (m1.ranges.rf_range.contains(if2_band.start) and
                    m1.ranges.rf_range.contains(if2_band.stop)):
                continue

            # Mixer2 IF range: require IF2 band fully inside Mixer2 IF range
            m2_if = m2.ranges.if_range
            if not (m2_if.contains(if2_band.start) and m2_if.contains(if2_band.stop)):
                continue

            for lo2_freq in lo2_grid:
                # Mixer2 LO range
                if not m2.ranges.lo_range.contains(lo2_freq):
                    continue

                # RF mapping for IF1 grid
                if sign_combo.sign2 > 0:
                    rf_vals = lo2_freq + sign_combo.sign2 * if2_vals
                else:
                    rf_vals = lo2_freq - if2_vals  # LO2 - IF2

                rf_band = Range(start=float(rf_vals.min()), stop=float(rf_vals.max()))

                # Strictly monotonic increasing mapping if requested
                if cfg.non_inverting_mapping_required:
                    if not np.all(np.diff(rf_vals) > 0.0):
                        continue

                # RF range coverage for this configuration
                if not (rf_band.start <= target_rf_lo and rf_band.stop >= target_rf_hi):
                    continue

                # Mixer2 RF range and global RF band must contain the entire RF band
                if not (m2.ranges.rf_range.contains(rf_band.start) and
                        m2.ranges.rf_range.contains(rf_band.stop)):
                    continue
                if not (cfg.rf_band.contains(rf_band.start) and
                        cfg.rf_band.contains(rf_band.stop)):
                    continue

                cand = LOPlanCandidate(
                    config_id=rf_conf.config_id,
                    lo1_freq=lo1_freq,
                    lo2_freq=lo2_freq,
                    sign_combo=sign_combo,
                    if2_band=if2_band,
                    rf_band=rf_band,
                )
                candidates.append(cand)
                if max_candidates is not None and len(candidates) >= max_candidates:
                    return candidates

    return candidates