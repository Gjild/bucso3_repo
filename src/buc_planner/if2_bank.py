# src/buc_planner/if2_bank.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Mapping, Iterable, Tuple, FrozenSet
from collections import defaultdict

import numpy as np
import math
import uuid

from .config_models import IF2BankConstraints, Range
from .filters import IF2Filter, IF2Bank, design_single_if2_filter_for_band  # design_single_ still imported for API compat
from .los import LOPlanCandidate, lo_plan_key
from .spur_engine import IF2SpurControlRegion


@dataclass
class IF2BankDesignResult:
    bank: IF2Bank
    config_to_filter_id: Dict[str, str]


# ---------------------------------------------------------------------------
# Range utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Spur-aware single-filter design hook
# ---------------------------------------------------------------------------

def _design_filter_for_band_with_regions(
    band: Range,
    constraints: IF2BankConstraints,
    regions: list[IF2SpurControlRegion],
    filter_id: str,
) -> IF2Filter:
    """
    Design a single IF2 filter for a desired band, taking into account
    spur-control regions for BW tightening and slope selection.

    This is the factored-out logic from the per-config path so that
    merged filters can be redesigned with the *union* of all relevant
    spur-control regions.
    """
    # 1) Merge regions to get pure frequency ranges for BW tightening
    control_ranges_merged: list[Range] = []
    if regions:
        control_ranges_merged = _merge_ranges(
            [Range(r.freq_start, r.freq_stop) for r in regions]
        )

    band_center = 0.5 * (band.start + band.stop)
    desired_bw = band.width
    bw_margin = constraints.feasibility_margin_hz

    bw_min = constraints.bw_range.start
    bw_max = constraints.bw_range.stop

    # Start from desired BW + margin
    target_bw = desired_bw + 2 * bw_margin

    # --- BW tightening away from control-region edges (existing logic) ---
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

    # Clamp BW and center
    target_bw = float(max(bw_min, min(bw_max, target_bw)))
    fc = float(
        max(constraints.fc_range.start, min(constraints.fc_range.stop, band_center))
    )

    # --- Slope selection from required_rejection_db (existing logic) ---
    s_min, s_max = constraints.slope_range
    slope = (s_min + s_max) * 0.5  # neutral start

    if regions and target_bw > 0.0:
        half_bw = 0.5 * target_bw
        required_slopes: list[float] = []

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
            slope = min(required_slopes)
        else:
            # No geometric point where we can use the skirt to help;
            # bias slightly towards a steeper slope.
            slope = min(slope, s_min)

    slope = max(s_min, min(slope, s_max))

    return IF2Filter(
        filter_id=filter_id,
        fc=fc,
        bw=target_bw,
        slope_db_per_decade=float(slope),
    )


# ---------------------------------------------------------------------------
# Spur penalty heuristic (per filter vs regions)
# ---------------------------------------------------------------------------

def _spur_penalty_for_filter(
    filt: IF2Filter,
    regions: Iterable[IF2SpurControlRegion],
    feasibility_margin_hz: float,
    *,
    hard_invalid_threshold_db: Optional[float] = None,
) -> float:
    """
    Heuristic penalty measuring how 'bad' it is to use this filter with the
    given spur-control regions.

    - Regions are considered only if they intersect an extended band around
      the passband.
    - Regions that intersect the passband get higher weight than those
      outside it.
    - Regions closer to the passband are penalized more strongly than those
      further away in the margin.
    - Wide regions can be scaled up relative to very narrow ones.

    Returns:
        Non-negative penalty, or +inf if hard_invalid_threshold_db is set
        and a region demands too much rejection inside the passband.

    Notes:
        - Degenerate filters (bw <= 0) are treated as invalid and yield +inf.
    """
    regions = list(regions)
    if not regions:
        return 0.0

    # Guard against degenerate filters.
    if filt.bw <= 0.0:
        return float("inf")

    pass_lo = filt.fc - 0.5 * filt.bw
    pass_hi = filt.fc + 0.5 * filt.bw

    # Use the existing feasibility margin as a spur margin,
    # but ensure it is not unreasonably small.
    margin = max(feasibility_margin_hz, 0.1 * filt.bw, 1.0)  # Hz

    ext_lo = pass_lo - margin
    ext_hi = pass_hi + margin

    cost = 0.0

    for r in regions:
        # Normalize region ordering.
        r_lo = min(r.freq_start, r.freq_stop)
        r_hi = max(r.freq_start, r.freq_stop)
        req = max(0.0, r.required_rejection_db)

        # Skip regions that are completely outside the extended band.
        if (r_hi <= ext_lo) or (r_lo >= ext_hi):
            continue

        # Treat *touching* the passband boundary as overlapping.
        overlaps_passband = (r_hi >= pass_lo) and (r_lo <= pass_hi)

        # Optional: if a region with large required rejection sits inside
        # the passband, consider this a hard invalid assignment.
        if overlaps_passband and hard_invalid_threshold_db is not None:
            if req >= hard_invalid_threshold_db:
                return float("inf")

        # Distance-based weighting for regions in the margin.
        if overlaps_passband:
            dist_norm = 0.0
        else:
            # Region intersects the margin but not the passband → it's either
            # below or above it. Distance is to the nearest passband edge.
            if r_hi <= pass_lo:
                edge_dist = pass_lo - r_hi  # region entirely below
            else:
                edge_dist = r_lo - pass_hi  # region entirely above

            # Normalize by margin to keep it roughly in [0, 1, ...]
            dist_norm = edge_dist / max(1.0, margin)

        # Base weight:
        #   - inside passband: strong penalty (2x)
        #   - in margin: decays with distance from passband
        if overlaps_passband:
            base_weight = 2.0
        else:
            base_weight = 1.0 / (1.0 + dist_norm)

        # Optionally scale by normalized width: very wide regions are a bit
        # more problematic than tiny ones.
        width_norm = (r_hi - r_lo) / max(1.0, filt.bw)
        width_scale = max(1.0, width_norm)

        region_weight = base_weight * width_scale

        cost += region_weight * req

    return cost


# ---------------------------------------------------------------------------
# Merge cost estimation with robustness + caching
# ---------------------------------------------------------------------------

class _MergeCostCache:
    """
    Simple cache for merge costs keyed by the frozenset of cfg_ids covered
    by the union filter.
    """
    def __init__(self) -> None:
        self._cache: Dict[FrozenSet[str], float] = {}

    def get(self, key: FrozenSet[str]) -> Optional[float]:
        return self._cache.get(key)

    def set(self, key: FrozenSet[str], value: float) -> None:
        self._cache[key] = value


def _estimate_merge_cost_for_cfg_set(
    merged_cfg_ids: FrozenSet[str],
    *,
    band_by_cfg: Dict[str, Range],
    regions_by_cfg: Dict[str, List[IF2SpurControlRegion]],
    constraints: IF2BankConstraints,
    cache: Optional[_MergeCostCache] = None,
) -> float:
    """
    Estimate the 'spur ugliness' cost of designing a single filter to cover
    all configs in merged_cfg_ids.

    - Uses the same design helper as the real filter design to keep the
      heuristic aligned with reality.
    - Returns +inf if the merged design is infeasible.
    - Optionally caches results keyed by merged_cfg_ids.

    Notes:
        - If merged_cfg_ids is empty or if their union has no spur regions,
          the cost is 0.0 (spur-neutral). Geometric criteria will then drive
          the merge decision.
    """
    if cache is not None:
        cached = cache.get(merged_cfg_ids)
        if cached is not None:
            return cached

    # Aggregate spur regions across all configs in this union.
    merged_regions: List[IF2SpurControlRegion] = []
    for cfg_id in merged_cfg_ids:
        merged_regions.extend(regions_by_cfg.get(cfg_id, []))

    if not merged_regions:
        # No spur regions for this union → spur-neutral.
        cost = 0.0
        if cache is not None:
            cache.set(merged_cfg_ids, cost)
        return cost

    # Compute the union passband from all constituent configs' bands.
    lo_vals: List[float] = []
    hi_vals: List[float] = []
    for cfg_id in merged_cfg_ids:
        band = band_by_cfg[cfg_id]
        lo_vals.append(min(band.start, band.stop))
        hi_vals.append(max(band.start, band.stop))

    merged_band = Range(start=min(lo_vals), stop=max(hi_vals))

    # Attempt to design the hypothetical merged filter. If that fails,
    # treat this merge as effectively impossible (infinite cost).
    try:
        tmp_filter = _design_filter_for_band_with_regions(
            band=merged_band,
            constraints=constraints,
            regions=merged_regions,
            filter_id=f"if2_merge_cost_tmp_{uuid.uuid4().hex}",
        )
    except Exception:
        cost = float("inf")
    else:
        cost = _spur_penalty_for_filter(
            tmp_filter,
            merged_regions,
            constraints.feasibility_margin_hz,
        )

    if cache is not None:
        cache.set(merged_cfg_ids, cost)

    return cost


# ---------------------------------------------------------------------------
# Spur-aware merge loop (building geometric proto filters)
# ---------------------------------------------------------------------------

def spur_aware_merge_filters(
    initial_filters: List[IF2Filter],
    band_by_cfg: Dict[str, Range],
    regions_by_cfg: Dict[str, List[IF2SpurControlRegion]],
    constraints: IF2BankConstraints,
    desired_count: int,
) -> List[IF2Filter]:
    """
    Merge filters until we reach desired_count, using a spur-aware heuristic:

    1. Primary key: spur merge cost estimated via _estimate_merge_cost_for_cfg_set.
    2. Tie-breaker: distance between filter center frequencies.

    When there are no spur regions at all, every merge cost is 0.0 and the
    behaviour reduces to the old 'closest fc first' heuristic.

    Invariants / assumptions:
        - Each initial filter must correspond to at least one config, i.e.
          f.cfg_ids is non-empty. An assertion enforces this, and the code
          still has a safe fallback if that invariant is accidentally broken.
        - The merged filters returned here have geometric proto passbands
          (fc, bw) derived from config bands. A later design pass is expected
          to turn them into real filters (with slope) using the same
          _design_filter_for_band_with_regions helper.
    """
    if desired_count <= 0:
        return []

    filters: List[IF2Filter] = list(initial_filters)
    if len(filters) <= desired_count:
        return filters

    # Enforce the "no anonymous filters" invariant.
    for f in filters:
        assert f.cfg_ids, "Each initial filter must have at least one cfg_id"

    merge_cost_cache = _MergeCostCache()
    EPS = 1e-9

    while len(filters) > desired_count:
        best_cost = float("inf")
        best_fc_dist = float("inf")
        best_pair: Optional[Tuple[int, int]] = None

        n = len(filters)
        for i in range(n):
            for j in range(i + 1, n):
                fi = filters[i]
                fj = filters[j]

                merged_cfg_ids = frozenset(fi.cfg_ids | fj.cfg_ids)
                cost = _estimate_merge_cost_for_cfg_set(
                    merged_cfg_ids=merged_cfg_ids,
                    band_by_cfg=band_by_cfg,
                    regions_by_cfg=regions_by_cfg,
                    constraints=constraints,
                    cache=merge_cost_cache,
                )

                fc_dist = abs(fi.fc - fj.fc)

                # Robust float comparison:
                if cost < best_cost - EPS:
                    best_cost = cost
                    best_fc_dist = fc_dist
                    best_pair = (i, j)
                elif abs(cost - best_cost) <= EPS and fc_dist < best_fc_dist - EPS:
                    best_cost = cost
                    best_fc_dist = fc_dist
                    best_pair = (i, j)

        # If we found no pair, or only 'infinitely bad' merges are available,
        # stop merging even if we haven't reached desired_count.
        if best_pair is None or math.isinf(best_cost):
            break

        i, j = best_pair
        fi = filters[i]
        fj = filters[j]

        # Merge passbands geometrically.
        merged_cfg_ids = fi.cfg_ids | fj.cfg_ids

        if merged_cfg_ids:
            # Preferred path: derive passband from the config bands.
            lo_vals: List[float] = []
            hi_vals: List[float] = []
            for cfg_id in merged_cfg_ids:
                band = band_by_cfg[cfg_id]
                lo_vals.append(min(band.start, band.stop))
                hi_vals.append(max(band.start, band.stop))

            pass_lo = min(lo_vals)
            pass_hi = max(hi_vals)
        else:
            # Fallback if the cfg_ids invariant was violated and we ended up
            # merging two filters with empty cfg_ids. Use their current fc/bw
            # geometrically to avoid crashing.
            lo_i = fi.fc - 0.5 * fi.bw
            hi_i = fi.fc + 0.5 * fi.bw
            lo_j = fj.fc - 0.5 * fj.bw
            hi_j = fj.fc + 0.5 * fj.bw
            pass_lo = min(lo_i, lo_j)
            pass_hi = max(hi_i, hi_j)

        fc = 0.5 * (pass_lo + pass_hi)
        bw = pass_hi - pass_lo

        merged_filter = IF2Filter(
            filter_id=f"if2_{uuid.uuid4().hex}",
            fc=fc,
            bw=bw,
            # Slope is a placeholder here; real slope is assigned in the
            # redesign step after merging.
            slope_db_per_decade=fi.slope_db_per_decade,
            cfg_ids=set(merged_cfg_ids),
        )

        # Replace the two original filters with the merged one.
        # Remove the higher index first to keep indices valid.
        for idx in sorted([i, j], reverse=True):
            del filters[idx]
        filters.append(merged_filter)

    return filters


# ---------------------------------------------------------------------------
# Spur-aware mapping (config → filter)
# ---------------------------------------------------------------------------

def _geom_mismatch_penalty(
    cfg_band: Range,
    filt: IF2Filter,
) -> float:
    """
    Simple geometric penalty between config band and filter passband, used as a
    secondary criterion after spur penalty.
    """
    cfg_lo = min(cfg_band.start, cfg_band.stop)
    cfg_hi = max(cfg_band.start, cfg_band.stop)

    pass_lo = filt.fc - 0.5 * filt.bw
    pass_hi = filt.fc + 0.5 * filt.bw

    # Example metric: sum of absolute differences of edges + extra penalty
    # if the config band extends beyond the passband.
    edge_mismatch = abs(cfg_lo - pass_lo) + abs(cfg_hi - pass_hi)

    # Extra penalty for out-of-passband coverage.
    outside_lo = max(0.0, pass_lo - cfg_lo)
    outside_hi = max(0.0, cfg_hi - pass_hi)
    overflow_penalty = outside_lo + outside_hi

    return edge_mismatch + 2.0 * overflow_penalty


def spur_aware_map_configs_to_filters(
    filters: List[IF2Filter],
    band_by_cfg: Dict[str, Range],
    regions_by_cfg: Dict[str, List[IF2SpurControlRegion]],
    constraints: IF2BankConstraints,
) -> Dict[str, str]:
    """
    Map each config to a filter in a spur-aware way.

    For each config:
        1. Compute the spur penalty of using each filter.
        2. Choose the filter with the smallest spur penalty.
        3. Break ties using the geometric mismatch between config band
           and filter passband.
        4. Final tie-break preference: if spur and geometry are essentially
           equal, prefer a filter that already lists this config in cfg_ids.
        5. As a final safeguard, if no filter gets selected (e.g. all penalties
           are +inf), assign the first filter.

    Returns:
        dict[config_id → filter_id]
    """
    cfg_to_filter: Dict[str, str] = {}
    EPS = 1e-9

    if not filters:
        return cfg_to_filter

    filters_by_id: Dict[str, IF2Filter] = {f.filter_id: f for f in filters}

    for cfg_id, cfg_band in band_by_cfg.items():
        cfg_regions = regions_by_cfg.get(cfg_id, [])

        best_filter_id: Optional[str] = None
        best_spur_penalty = float("inf")
        best_geom_penalty = float("inf")

        for f in filters:
            spur_penalty = _spur_penalty_for_filter(
                f,
                cfg_regions,
                constraints.feasibility_margin_hz,
                # Example: treat > 80 dB required *inside* passband as
                # effectively impossible, if desired:
                # hard_invalid_threshold_db=80.0,
            )

            geom_penalty = _geom_mismatch_penalty(cfg_band, f)

            if spur_penalty < best_spur_penalty - EPS:
                best_spur_penalty = spur_penalty
                best_geom_penalty = geom_penalty
                best_filter_id = f.filter_id
            elif abs(spur_penalty - best_spur_penalty) <= EPS and geom_penalty < best_geom_penalty - EPS:
                best_spur_penalty = spur_penalty
                best_geom_penalty = geom_penalty
                best_filter_id = f.filter_id
            else:
                # Optional third-level tie-breaker:
                # if spur and geometry are essentially equal, prefer 'native'
                # filters that already contain this cfg_id in cfg_ids.
                if (
                    abs(spur_penalty - best_spur_penalty) <= EPS
                    and abs(geom_penalty - best_geom_penalty) <= EPS
                    and cfg_id in f.cfg_ids
                    and (
                        best_filter_id is None
                        or cfg_id not in filters_by_id[best_filter_id].cfg_ids
                    )
                ):
                    best_spur_penalty = spur_penalty
                    best_geom_penalty = geom_penalty
                    best_filter_id = f.filter_id

        # Fallback: if every candidate was effectively impossible (spur penalty
        # +inf), still assign something deterministic to avoid crashing.
        if best_filter_id is None:
            best_filter_id = filters[0].filter_id

        cfg_to_filter[cfg_id] = best_filter_id

    return cfg_to_filter


# ---------------------------------------------------------------------------
# Top-level IF2 bank design, now spur-aware
# ---------------------------------------------------------------------------

def design_if2_bank_for_lo_plans(
    lo_plans: List[LOPlanCandidate],
    constraints: IF2BankConstraints,
    # Mapping can be:
    #   - config_id -> [regions]     (legacy behaviour), or
    #   - lo_plan_key(plan) -> [regions] (new, LO-plan-specific behaviour).
    spur_control_regions: Optional[Mapping[object, List[IF2SpurControlRegion]]] = None,
    target_n_filters: Optional[int] = None,
) -> IF2BankDesignResult:
    """
    Spur-aware heuristic IF2 bank design.

    Inputs:
      * lo_plans:           one LO plan per RF configuration.
      * spur_control_regions:
            optional dict config_id -> list[IF2SpurControlRegion] (IF2 regions
            where we want additional rejection), or keyed by lo_plan_key.
            These come from the coarse spur pass in spur_engine and include
            a required_rejection_db depth.
      * constraints:        IF2 bank constraints.
      * target_n_filters:   desired number of filters (clamped to
            [min_filters, max_filters]). The algorithm may return more filters
            than this if all further merges are spur-infeasible.

    Strategy:
      1. For each config, build a design band from LO plans (if2_band) and
         attach its spur-control regions.

      2. Create one provisional filter per config using
         _design_filter_for_band_with_regions.

         These initial filters are *real* filters (fc, bw, slope), with
         cfg_ids={config_id}.

      3. Greedily merge filters using spur_aware_merge_filters:

           - primary key: spur-aware merge cost (estimated by re-running
             the same design helper on the union of bands and regions),
           - tie-breaker: center-frequency distance.

         This produces geometric proto filters (fc, bw, cfg_ids).

      4. Redesign each merged filter as a real IF2Filter using the union
         of IF2 bands and spur-control regions of its cfg_ids.

      5. Map each configuration to a filter using spur_aware_map_configs_to_filters.

    The resulting IF2BankDesignResult is then used by the optimizer and spur
    engine exactly as before.
    """
    # 1) Desired IF2 band per config
    bands_by_config: Dict[str, Range] = {p.config_id: p.if2_band for p in lo_plans}

    # 1b) Collect per-config spur-control regions (full objects)
    cfg_control_regions: Dict[str, List[IF2SpurControlRegion]] = defaultdict(list)

    if spur_control_regions:
        keys = list(spur_control_regions.keys())

        if keys and isinstance(keys[0], str):
            # Legacy: dict[config_id] -> list[regions]
            for cfg_id, regs in spur_control_regions.items():
                if not regs:
                    continue
                cfg_control_regions[cfg_id].extend(regs)
        else:
            # New: dict[lo_plan_key] -> list[regions]. Only pull regions for the
            # LO plans actually used in this bank design.
            for plan in lo_plans:
                key = lo_plan_key(plan)
                regs = spur_control_regions.get(key)
                if regs:
                    cfg_control_regions[plan.config_id].extend(regs)

    # 2) Initial one-per-config filters using the helper
    initial_filters: List[IF2Filter] = []

    for plan in lo_plans:
        config_id = plan.config_id
        desired_band = bands_by_config[config_id]
        regions = cfg_control_regions.get(config_id, [])

        f = _design_filter_for_band_with_regions(
            band=desired_band,
            constraints=constraints,
            regions=regions,
            filter_id=f"if2_auto_{config_id}",
        )
        # Track the originating config(s) on the filter itself
        f.cfg_ids.add(config_id)
        initial_filters.append(f)

    # 3) Determine desired number of filters
    if target_n_filters is None:
        desired = constraints.min_filters
    else:
        desired = max(
            constraints.min_filters,
            min(target_n_filters, constraints.max_filters),
        )
    desired = max(desired, 1)

    # Convenience mapping for merge + mapping
    regions_by_cfg: Dict[str, List[IF2SpurControlRegion]] = dict(cfg_control_regions)

    # 4) Spur-aware merge to produce geometric proto filters
    merged_filters = spur_aware_merge_filters(
        initial_filters=initial_filters,
        band_by_cfg=bands_by_config,
        regions_by_cfg=regions_by_cfg,
        constraints=constraints,
        desired_count=desired,
    )

    # 4b) Redesign each merged filter with union bands/regions so the final
    #      bank consists of fully-shaped filters, not just geometric prototypes.
    final_filters: List[IF2Filter] = []

    for proto in merged_filters:
        cfg_ids = getattr(proto, "cfg_ids", set()) or set()

        if cfg_ids:
            lo_vals: List[float] = []
            hi_vals: List[float] = []
            merged_regions: List[IF2SpurControlRegion] = []

            for cfg_id in cfg_ids:
                band = bands_by_config[cfg_id]
                lo_vals.append(min(band.start, band.stop))
                hi_vals.append(max(band.start, band.stop))
                merged_regions.extend(cfg_control_regions.get(cfg_id, []))

            merged_band = Range(start=min(lo_vals), stop=max(hi_vals))
        else:
            # Fallback: derive band from the proto filter itself.
            pass_lo = proto.fc - 0.5 * proto.bw
            pass_hi = proto.fc + 0.5 * proto.bw
            merged_band = Range(start=pass_lo, stop=pass_hi)
            merged_regions = []

        redesigned = _design_filter_for_band_with_regions(
            band=merged_band,
            constraints=constraints,
            regions=merged_regions,
            filter_id=proto.filter_id,
        )
        redesigned.cfg_ids = set(cfg_ids)
        final_filters.append(redesigned)

    # 5) Spur-aware mapping from configs to filters
    mapping: Dict[str, str] = spur_aware_map_configs_to_filters(
        filters=final_filters,
        band_by_cfg=bands_by_config,
        regions_by_cfg=regions_by_cfg,
        constraints=constraints,
    )

    bank = IF2Bank(filters=final_filters, config_to_filter_id=mapping)
    return IF2BankDesignResult(bank=bank, config_to_filter_id=mapping)
