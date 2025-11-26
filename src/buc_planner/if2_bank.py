# src/buc_planner/if2_bank.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Mapping
from collections import defaultdict

import numpy as np
from copy import deepcopy

from .config_models import IF2BankConstraints, Range
from .filters import IF2Filter, IF2Bank, design_single_if2_filter_for_band
from .los import LOPlanCandidate
from .spur_engine import IF2SpurControlRegion 



@dataclass
class IF2BankDesignResult:
    bank: IF2Bank
    config_to_filter_id: Dict[str, str]
	
	
def _merge_ranges(ranges: list[Range]) -> list[Range]:
    """Union of 1D ranges, returning disjoint, sorted intervals."""
    if not ranges:
        return []
    ranges_sorted = sorted(ranges, key=lambda r: r.start)
    merged: list[Range] = []
    cur = ranges_sorted[0]
    for r in ranges_sorted[1:]:
        if r.start <= cur.stop:
            cur = Range(start=cur.start, stop=max(cur.stop, r.stop))
        else:
            merged.append(cur)
            cur = r
    merged.append(cur)
    return merged


def design_if2_bank_for_lo_plans(
    lo_plans: List[LOPlanCandidate],
    constraints: IF2BankConstraints,
    spur_control_regions: Optional[Mapping[str, List[IF2SpurControlRegion]]] = None,
    target_n_filters: Optional[int] = None,
) -> IF2BankDesignResult:
    """
    Heuristic IF2 bank design.

    Inputs:
      * lo_plans:           one LO plan per RF configuration.
      * spur_control_regions:
            optional dict config_id -> list[IF2SpurControlRegion] (IF2 regions
            where we want additional rejection). These come from the coarse
            spur pass in spur_engine and include a required_rejection_db depth.
      * constraints:        IF2 bank constraints.
      * target_n_filters:   desired number of filters (will be clamped
            to [min_filters, max_filters]).

    Strategy:
      1. For each config, build a "design band" composed of:
           - mandatory desired IF2 band (from LO plan),
           - optional spur-control regions (merged only for frequency
             keep-away).
         We keep:
           - desired IF2 band driving passband center/BW.
           - spur-control regions influencing both passband width
             (avoid putting edges on top of them) and minimum required
             out-of-band attenuation (via slope selection).

      2. Create one provisional filter per config:
           - fc ~ center of desired IF2 band, clamped to fc_range.
           - bw ~ desired band width + feasibility_margin_hz, clamped.
           - slope chosen such that, where possible, attenuation at the
             nearest spur-control region meets required_rejection_db,
             subject to slope_range.

      3. Greedily merge filters by closeness of fc until we hit
         target_n_filters (or constraints.min_filters by default).

      4. Map each configuration to the filter whose passband best covers
         its desired IF2 band.
    """
    bands_by_config: Dict[str, Range] = {p.config_id: p.if2_band for p in lo_plans}

    # 1) Collect per-config spur-control regions (full objects)
    cfg_control_regions: Dict[str, List[IF2SpurControlRegion]] = defaultdict(list)
    if spur_control_regions:
        for cfg_id, regs in spur_control_regions.items():
            if not regs:
                continue
            cfg_control_regions[cfg_id].extend(regs)

    filter_defs: Dict[str, IF2Filter] = {}
    for plan in lo_plans:
        config_id = plan.config_id
        desired_band = bands_by_config[config_id]

        regions = cfg_control_regions.get(config_id, [])
        # For BW tightening we only need frequency ranges
        control_ranges_merged: List[Range] = []
        if regions:
            control_ranges_merged = _merge_ranges(
                [Range(r.freq_start, r.freq_stop) for r in regions]
            )

        band_center = 0.5 * (desired_band.start + desired_band.stop)
        desired_bw = desired_band.width

        bw_margin = constraints.feasibility_margin_hz
        bw_min = constraints.bw_range.start
        bw_max = constraints.bw_range.stop

        # Start with desired BW + margin
        target_bw = desired_bw + 2 * bw_margin

        # Shrink BW so that passband edges don't sit unnecessarily
        # close to spur-control regions.
        if control_ranges_merged and target_bw > 0.0:
            half_bw = 0.5 * target_bw
            pass_lo = band_center - half_bw
            pass_hi = band_center + half_bw

            def nearest_control_edge_distance() -> float:
                dmin = float("inf")
                for r in control_ranges_merged:
                    for edge in (r.start, r.stop):
                        dmin = min(
                            dmin,
                            abs(edge - pass_lo),
                            abs(edge - pass_hi),
                        )
                return dmin

            d = nearest_control_edge_distance()
            if d > 0.0 and d < half_bw:
                # Bring edges away from the closest control-region edge,
                # but never shrink below the desired IF2 bandwidth.
                new_half_bw = max(desired_bw * 0.5, d * 0.8)
                target_bw = 2.0 * new_half_bw

        # Clamp BW and center to constraints
        target_bw = float(max(bw_min, min(bw_max, target_bw)))
        fc = float(
            max(constraints.fc_range.start, min(constraints.fc_range.stop, band_center))
        )

        # 2) Slope: use required_rejection_db where possible
        s_min, s_max = constraints.slope_range
        # Start from mid of slope_range as neutral
        slope = (s_min + s_max) * 0.5

        if regions and target_bw > 0.0:
            half_bw = 0.5 * target_bw
            required_slopes: List[float] = []

            for reg in regions:
                # Consider both edges of the region; we want the closest
                # point *outside* the passband.
                for edge in (reg.freq_start, reg.freq_stop):
                    dist = abs(edge - fc)
                    if dist <= half_bw:
                        # Region edge inside passband: we cannot use
                        # out-of-band roll-off to meet this requirement.
                        continue
                    offset_ratio = dist / half_bw
                    if offset_ratio <= 1.0:
                        continue
                    if reg.required_rejection_db <= 0.0:
                        continue
                    # A(f) = S * log10(offset_ratio), S < 0.
                    # We need A(f) <= -required_rejection_db.
                    # => S <= -required / log10(offset_ratio).
                    s_req = -reg.required_rejection_db / np.log10(offset_ratio)
                    required_slopes.append(s_req)

            if required_slopes:
                # Pick the steepest (most negative) requirement
                worst_s_req = min(required_slopes)
                slope = worst_s_req
            else:
                # No geometric point where we can use the skirt to help;
                # bias slightly towards a steeper slope.
                slope = min(slope, s_min)

        # Clamp slope to allowed range
        slope = max(s_min, min(slope, s_max))

        f = IF2Filter(
            filter_id=f"if2_auto_{config_id}",
            fc=fc,
            bw=target_bw,
            slope_db_per_decade=float(slope),
        )
        filter_defs[config_id] = f

    filters: List[IF2Filter] = list(filter_defs.values())

    # 2) Determine desired number of filters
    if target_n_filters is None:
        desired = constraints.min_filters
    else:
        desired = max(constraints.min_filters, min(target_n_filters, constraints.max_filters))
    desired = max(desired, 1)

    # 3) Greedy merge by fc
    while len(filters) > desired:
        min_dist = float("inf")
        merge_i = merge_j = None
        for i in range(len(filters)):
            for j in range(i + 1, len(filters)):
                d = abs(filters[i].fc - filters[j].fc)
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j
        if merge_i is None or merge_j is None:
            break

        f1 = filters[merge_i]
        f2 = filters[merge_j]
        merged_band = Range(
            start=min(f1.fc - f1.bw / 2, f2.fc - f2.bw / 2),
            stop=max(f1.fc + f1.bw / 2, f2.fc + f2.bw / 2),
        )
        merged = design_single_if2_filter_for_band(merged_band, constraints)
        filters[merge_i] = merged
        del filters[merge_j]

    while len(filters) < desired:
        # Use a deep copy to avoid aliasing the same IF2Filter instance.
        filters.append(deepcopy(filters[-1]))

    # 4) Map each config to best filter (coverage of desired IF2 band)
    mapping: Dict[str, str] = {}
    for plan in lo_plans:
        config_id = plan.config_id
        band = bands_by_config[config_id]
        best_id = None
        best_score = float("inf")
        for f in filters:
            f_lo = f.fc - f.bw / 2
            f_hi = f.fc + f.bw / 2
            score = abs(band.start - f_lo) + abs(band.stop - f_hi)
            if score < best_score:
                best_score = score
                best_id = f.filter_id
        mapping[config_id] = best_id

    bank = IF2Bank(filters=filters, config_to_filter_id=mapping)
    return IF2BankDesignResult(bank=bank, config_to_filter_id=mapping)