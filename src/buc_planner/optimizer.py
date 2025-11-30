# src/buc_planner/optimizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from .config_models import SystemConfig, RfConfiguration, Range
from .filters import RFFilter
from .los import (
    generate_lo_plan_candidates_for_config,
    LOPlanCandidate,
    lo_plan_key,
)
from .if2_bank import (
    design_if2_bank_for_lo_plans,
    IF2BankDesignResult,
)
from .spur_engine import (
    evaluate_spurs_for_config_and_lo_plan,
    SpurResult,
    ConfigSpurSummary,
	coarse_if2_spur_control_regions_for_lo_plan,
	IF2SpurControlRegion,
    prepare_mask,
)
from .progress import ProgressReporter, NullProgressReporter

logger = logging.getLogger(__name__)

LOPlanKey = tuple[str, float, float, int, int]

@dataclass
class PlannerResult:
    system_config: SystemConfig
    lo_plans: List[LOPlanCandidate]
    if2_bank_design: IF2BankDesignResult
    spur_results: List[SpurResult]
    summaries: Dict[str, ConfigSpurSummary]

@dataclass
class SearchResult:
    """
    Internal helper result from _search_lo_plans_for_filter_count.

    worst_margin_db:
        The worst (most negative) spur margin across all configs for this
        LO-plan / IF2-bank combination. None means "no margin computed"
        (no applicable limits).
    """
    lo_plans: List[LOPlanCandidate]
    if2_bank_design: IF2BankDesignResult
    spur_results: List[SpurResult]
    summaries: Dict[str, ConfigSpurSummary]
    feasible: bool
    worst_margin_db: Optional[float]

class Planner:
    """
    High-level planning engine.
    """

    def __init__(self, cfg: SystemConfig, progress: ProgressReporter | None = None):
        self.cfg = cfg
        self.rf_filter = RFFilter.from_csv(cfg.filters.rf_bpf_csv_path)
        # Pre-load spur mask (if any) once for the whole run
        self._mask_freqs, self._mask_levels = prepare_mask(cfg)
        # Progress reporter (defaults to a no-op)
        self.progress = progress or NullProgressReporter()

    def _enumerate_lo_plans(self) -> Dict[str, List[LOPlanCandidate]]:
        """
        Enumerate LO candidates for all RF configurations.

        Returns:
            dict[config_id] -> list[LOPlanCandidate]
        """
        cfg = self.cfg
        per_cfg: Dict[str, List[LOPlanCandidate]] = {}

        total_cfgs = len(cfg.rf_configurations)
        self.progress.start("Enumerating LO candidates", total=total_cfgs)

        for rf_conf in cfg.rf_configurations:
            cands = generate_lo_plan_candidates_for_config(
                cfg,
                rf_conf,
                max_candidates=cfg.grids.max_lo_candidates_per_rf,
            )
            if not cands:
                self.progress.end()
                raise RuntimeError(
                    f"No LO plan candidates found for RF config '{rf_conf.config_id}'. "
                    "Check LO ranges, mixer ranges, and RF band."
                )
            per_cfg[rf_conf.config_id] = cands
            self.progress.advance()

        self.progress.end()
        return per_cfg
		
    def _coarse_prune_lo_candidates(
        self,
        per_cfg_candidates: Dict[str, List[LOPlanCandidate]],
    ) -> tuple[Dict[str, List[LOPlanCandidate]], Dict[LOPlanKey, List[IF2SpurControlRegion]]]:
        """
        Use coarse spur analysis to:
          * Reject clearly hopeless LO candidates (margin << 0 dB),
          * Derive IF2 spur-control regions per configuration.

        Returns:
          (pruned_candidates, spur_control_regions)
          where spur_control_regions is config_id -> list[IF2SpurControlRegion]
        """
        cfg = self.cfg
        pruned: Dict[str, List[LOPlanCandidate]] = {}
        # LO-plan-specific spur-control regions
        control_regions_by_plan: Dict[LOPlanKey, List[IF2SpurControlRegion]] = {}

        coarse_min = cfg.grids.coarse_spur_margin_min_db  # negative threshold
        min_survivors_cfg = max(1, cfg.grids.min_lo_candidates_per_rf_after_coarse)
        rf_by_id: Dict[str, RfConfiguration] = {
            rc.config_id: rc for rc in cfg.rf_configurations
        }

        self.progress.start(
            "Coarse spur pruning",
            total=len(per_cfg_candidates),
        )

        for config_id, cand_list in per_cfg_candidates.items():
            rf_conf = rf_by_id[config_id]
            survivors: List[LOPlanCandidate] = []
            # Regions per candidate (for this config only)
            regions_per_cand: Dict[LOPlanKey, List[IF2SpurControlRegion]] = {}
            cfg_worst_margin: Optional[float] = None

            for cand in cand_list:
                regions, worst_margin = coarse_if2_spur_control_regions_for_lo_plan(
                    cfg,
                    rf_conf,
                    cand,
                    self.rf_filter,
                )
                key = lo_plan_key(cand)
                regions_per_cand[key] = regions

                if worst_margin is not None:
                    if cfg_worst_margin is None or worst_margin < cfg_worst_margin:
                        cfg_worst_margin = worst_margin

                if worst_margin is not None and worst_margin < coarse_min:
                    # Drop this candidate at the coarse stage
                    continue

                survivors.append(cand)

            if not survivors:
                k = min(len(cand_list), min_survivors_cfg)
                logger.warning(
                    "Config '%s': all %d LO candidates failed coarse spur "
                    "margin threshold (worst margin %.1f dB); keeping the "
                    "first %d candidates for detailed analysis.",
                    config_id,
                    len(cand_list),
                    cfg_worst_margin if cfg_worst_margin is not None else float("nan"),
                    k,
                )
                survivors = cand_list[:k]
            elif len(survivors) < min_survivors_cfg and len(cand_list) > len(survivors):
                needed = min_survivors_cfg - len(survivors)
                extra = [c for c in cand_list if c not in survivors][:needed]
                if extra:
                    logger.info(
                        "Config '%s': coarse pruning left %d survivors (<%d); "
                        "adding %d borderline candidates back.",
                        config_id,
                        len(survivors),
                        min_survivors_cfg,
                        len(extra),
                    )
                    survivors.extend(extra)

            pruned[config_id] = survivors

            # Attach spur-control regions only for the surviving candidates
            for cand in survivors:
                key = lo_plan_key(cand)
                control_regions_by_plan[key] = regions_per_cand.get(key, [])

            self.progress.advance()

        self.progress.end()
        return pruned, control_regions_by_plan  

    def _search_lo_plans_for_filter_count(
        self,
        per_cfg_candidates: Dict[str, List[LOPlanCandidate]],
        control_regions_by_plan: Dict[LOPlanKey, List[IF2SpurControlRegion]],
        n_filters: int,
    ) -> Optional[SearchResult]:
        """
        For a fixed IF2 filter count (n_filters), search over LO-plan combinations
        and find a solution that:

          * If at least one feasible combination exists (all spur margins >= 0 dB):
                - minimize (n_distinct_LO1, n_distinct_LO2) lexicographically.

          * If NO feasible combination exists:
                - return the "least bad" combination, i.e. the one whose global
                  worst spur margin is closest to 0 dB (largest margin, even if
                  negative). Ties are broken by LO reuse (fewer LO retunes).

        Search is bounded by:
            * max_lo_candidates_per_rf   – limits candidates per RF config
            * max_if2_bank_candidates    – limits total (LO-plan, IF2-bank) combos
                                           evaluated at this filter count.
        """
        cfg = self.cfg
        rf_confs = cfg.rf_configurations
        ordered_cfg_ids = [rc.config_id for rc in rf_confs]

        # Limit candidates per config to keep combinatorics sane
        max_per_cfg = max(1, min(4, cfg.grids.max_lo_candidates_per_rf))
        trimmed: Dict[str, List[LOPlanCandidate]] = {}
        for cid in ordered_cfg_ids:
            cands = per_cfg_candidates.get(cid, [])
            trimmed[cid] = cands[:max_per_cfg]

        max_combos = max(1, cfg.grids.max_if2_bank_candidates)

        self.progress.start(
            f"LO-plan search for n_filters={n_filters}",
            total=max_combos,
        )

        # Best feasible solution (all margins >= 0 dB)
        best_feasible: Optional[SearchResult] = None
        best_feasible_score: Optional[tuple[int, int]] = None  # (n_LO1, n_LO2)
        best_feasible_margin: Optional[float] = None           # worst margin for that solution

        # Best *infeasible* solution (no margin >= 0 for at least one config),
        # chosen by largest worst_margin_db (closest to 0), tie-broken by LO reuse.
        best_infeasible: Optional[SearchResult] = None
        best_infeasible_margin: Optional[float] = None          # worst (negative) margin
        best_infeasible_score: Optional[tuple[int, int]] = None # (n_LO1, n_LO2)

        combos_tested = 0

        def dfs(
            idx: int,
            current: List[LOPlanCandidate],
            used_lo1: set[float],
            used_lo2: set[float],
        ) -> None:
            nonlocal best_feasible, best_feasible_score, best_feasible_margin
            nonlocal best_infeasible, best_infeasible_margin, best_infeasible_score
            nonlocal combos_tested

            if combos_tested >= max_combos:
                return

            # Reached a full assignment: one LO plan per configuration
            if idx == len(ordered_cfg_ids):
                combos_tested += 1
                self.progress.advance()  # update progress bar

                # Design IF2 bank for this LO-plan set
                bank_design = design_if2_bank_for_lo_plans(
                    current,
                    cfg.filters.if2_constraints,
                    spur_control_regions=control_regions_by_plan,
                    target_n_filters=n_filters,
                )

                # Evaluate spurs for this LO-plan + IF2 bank
                spur_results, summaries = self._evaluate_for_bank_design(
                    current,
                    bank_design,
                )

                # Compute global worst margin across all configs
                global_worst_margin: Optional[float] = None
                for summary in summaries.values():
                    for m in (
                        summary.worst_in_band_margin_db,
                        summary.worst_out_band_margin_db,
                    ):
                        if m is None:
                            continue
                        if global_worst_margin is None or m < global_worst_margin:
                            global_worst_margin = m

                # Number of distinct LO frequencies used
                n_lo1 = len(used_lo1)
                n_lo2 = len(used_lo2)
                score = (n_lo1, n_lo2)

                feasible = (
                    global_worst_margin is None
                    or global_worst_margin >= 0.0
                )

                res = SearchResult(
                    lo_plans=list(current),
                    if2_bank_design=bank_design,
                    spur_results=spur_results,
                    summaries=summaries,
                    feasible=feasible,
                    worst_margin_db=global_worst_margin,
                )

                if feasible:
                    # Lexicographic minimization of LO retunes among feasible solutions
                    if best_feasible_score is None or score < best_feasible_score:
                        best_feasible_score = score
                        best_feasible_margin = global_worst_margin
                        best_feasible = res
                else:
                    # For infeasible, we prefer *less negative* (closer to 0) margin.
                    if global_worst_margin is not None:
                        if (
                            best_infeasible_margin is None
                            or global_worst_margin > best_infeasible_margin
                            or (
                                global_worst_margin == best_infeasible_margin
                                and (
                                    best_infeasible_score is None
                                    or score < best_infeasible_score
                                )
                            )
                        ):
                            best_infeasible_margin = global_worst_margin
                            best_infeasible_score = score
                            best_infeasible = res

                return

            cfg_id = ordered_cfg_ids[idx]
            cands = trimmed.get(cfg_id, [])
            if not cands:
                return  # no candidates for this config; nothing to do

            for cand in cands:
                new_used_lo1 = set(used_lo1)
                new_used_lo2 = set(used_lo2)
                new_used_lo1.add(cand.lo1_freq)
                new_used_lo2.add(cand.lo2_freq)

                # If we already have a feasible solution, use LO reuse score
                # to prune branches that cannot beat it.
                if best_feasible_score is not None:
                    partial_score = (len(new_used_lo1), len(new_used_lo2))
                    if partial_score > best_feasible_score:
                        continue

                current.append(cand)
                dfs(idx + 1, current, new_used_lo1, new_used_lo2)
                current.pop()

                if combos_tested >= max_combos:
                    break

        dfs(0, [], set(), set())

        self.progress.end()

        if best_feasible is not None:
            return best_feasible
        if best_infeasible is not None:
            return best_infeasible

        # Nothing evaluated successfully at all (e.g. zero candidates)
        return None


    def _evaluate_for_bank_design(
        self,
        lo_plans: List[LOPlanCandidate],
        bank_design: IF2BankDesignResult,
    ) -> tuple[List[SpurResult], Dict[str, ConfigSpurSummary]]:
        """
        Run detailed spur analysis for all configurations for a given IF2 bank design.
        """
        cfg = self.cfg
        spur_results: List[SpurResult] = []
        summaries: Dict[str, ConfigSpurSummary] = {}

        rf_by_id: Dict[str, RfConfiguration] = {rc.config_id: rc for rc in cfg.rf_configurations}

        self.progress.start(
            "Detailed spur evaluation",
            total=len(lo_plans),
        )

        if cfg.grids.parallel and len(lo_plans) > 1:
            with ProcessPoolExecutor() as ex:
                futures = {}
                for plan in lo_plans:
                    rf_conf = rf_by_id[plan.config_id]
                    if2_filter = next(
                        f for f in bank_design.bank.filters
                        if f.filter_id == bank_design.config_to_filter_id[plan.config_id]
                    )
                    fut = ex.submit(
                        evaluate_spurs_for_config_and_lo_plan,
                        cfg,
                        rf_conf,
                        plan,
                        if2_filter,
                        self.rf_filter,
                        self._mask_freqs,
                        self._mask_levels,
                    )
                    futures[fut] = plan.config_id

                for fut in as_completed(futures):
                    config_id = futures[fut]
                    res, summary = fut.result()
                    spur_results.extend(res)
                    summaries[config_id] = summary
                    self.progress.advance()
        else:
            for plan in lo_plans:
                rf_conf = rf_by_id[plan.config_id]
                if2_filter = next(
                    f for f in bank_design.bank.filters
                    if f.filter_id == bank_design.config_to_filter_id[plan.config_id]
                )
                res, summary = evaluate_spurs_for_config_and_lo_plan(
                    cfg,
                    rf_conf,
                    plan,
                    if2_filter,
                    self.rf_filter,
                    self._mask_freqs,
                    self._mask_levels,
                )
                spur_results.extend(res)
                summaries[plan.config_id] = summary
                self.progress.advance()

        self.progress.end()
        return spur_results, summaries


    def run(self) -> PlannerResult:
        """
        Main planning pipeline:

        1. Enumerate LO candidates per RF configuration.
        2. Coarse prune LO candidates using fast spur estimates; derive
           IF2 spur-control regions.
        3. Sweep IF2 filter count from min_filters to max_filters:
             * for each filter count, search LO-plan combinations
               (bounded search) to:
                 - design IF2 bank,
                 - run detailed spur analysis,
                 - select a feasible solution that minimizes
                   (n_distinct_LO1, n_distinct_LO2).
        4. Return the first (min-filter-count) feasible solution.
        """
        cfg = self.cfg

        # 1) Enumerate raw LO candidates per configuration
        per_cfg_candidates = self._enumerate_lo_plans()

        # 2) Coarse pruning + IF2 spur-control regions
        # NOTE: _coarse_prune_lo_candidates may currently return None
        # (placeholder implementation). In that case, fall back to
        # "no pruning" and empty spur-control regions.
        coarse_result = self._coarse_prune_lo_candidates(per_cfg_candidates)
        
        if coarse_result is None:
            # Fallback: no coarse pruning; keep all candidates and no
            # spur-control regions.
            pruned_candidates = per_cfg_candidates
            control_regions_by_plan: Dict[LOPlanKey, List[IF2SpurControlRegion]] = {}
        else:
            pruned_candidates, control_regions_by_plan = coarse_result

        min_f = cfg.filters.if2_constraints.min_filters
        max_f = cfg.filters.if2_constraints.max_filters

        best_feasible_result: Optional[SearchResult] = None
        best_infeasible_result: Optional[SearchResult] = None

        # 3) Sweep IF2 filter count (lexicographic: min filters first)
        for n_filters in range(min_f, max_f + 1):
            search_res = self._search_lo_plans_for_filter_count(
                pruned_candidates,
                control_regions_by_plan,
                n_filters,
            )
            if search_res is None:
                continue

            if search_res.feasible:
                # First n_filters with any feasible solution is the lexicographic
                # optimum in terms of IF2 filter count; within this n_filters,
                # LO retune counts have already been minimized in _search_lo_plans_for_filter_count.
                best_feasible_result = search_res
                break
            else:
                # Track best infeasible across all filter counts
                if best_infeasible_result is None:
                    best_infeasible_result = search_res
                else:
                    def _margin(res: SearchResult) -> float:
                        # Larger margin is "better" (less violation).
                        # None is treated as very negative here.
                        return res.worst_margin_db if res.worst_margin_db is not None else float("-inf")

                    if _margin(search_res) > _margin(best_infeasible_result):
                        best_infeasible_result = search_res

        final_res = best_feasible_result or best_infeasible_result

        if final_res is None:
            # This should be very rare: no combinations evaluated at all.
            raise RuntimeError(
                "No IF2 bank / LO plan combinations could be evaluated. "
                "Check that LO ranges, mixer ranges, and RF band definitions "
                "allow at least one mapping."
            )

        if not final_res.feasible:
            logger.warning(
                "No feasible IF2 bank / LO plan combination found within "
                "[min_filters, max_filters] that satisfies spur limits for all "
                "configurations; returning least-bad combination with worst "
                "spur margin %.1f dB.",
                final_res.worst_margin_db if final_res.worst_margin_db is not None else float("nan"),
            )

        return PlannerResult(
            system_config=cfg,
            lo_plans=final_res.lo_plans,
            if2_bank_design=final_res.if2_bank_design,
            spur_results=final_res.spur_results,
            summaries=final_res.summaries,
        )
