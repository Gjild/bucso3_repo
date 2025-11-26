# src/buc_planner/optimizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging  # NEW

from .config_models import SystemConfig, RfConfiguration, Range
from .filters import RFFilter
from .los import (
    generate_lo_plan_candidates_for_config,
    LOPlanCandidate,
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
)

logger = logging.getLogger(__name__)


@dataclass
class PlannerResult:
    system_config: SystemConfig
    lo_plans: List[LOPlanCandidate]
    if2_bank_design: IF2BankDesignResult
    spur_results: List[SpurResult]
    summaries: Dict[str, ConfigSpurSummary]


class Planner:
    """
    High-level planning engine.
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.rf_filter = RFFilter.from_csv(cfg.filters.rf_bpf_csv_path)

    def _enumerate_lo_plans(self) -> Dict[str, List[LOPlanCandidate]]:
        """
        Enumerate LO candidates for all RF configurations.

        Returns:
            dict[config_id] -> list[LOPlanCandidate]
        """
        cfg = self.cfg
        per_cfg: Dict[str, List[LOPlanCandidate]] = {}
        for rf_conf in cfg.rf_configurations:
            cands = generate_lo_plan_candidates_for_config(
                cfg,
                rf_conf,
                max_candidates=cfg.grids.max_lo_candidates_per_rf,
            )
            if not cands:
                raise RuntimeError(
                    f"No LO plan candidates found for RF config '{rf_conf.config_id}'. "
                    "Check LO ranges, mixer ranges, and RF band."
                )
            per_cfg[rf_conf.config_id] = cands
        return per_cfg
		
    def _coarse_prune_lo_candidates(
        self,
        per_cfg_candidates: Dict[str, List[LOPlanCandidate]],
    ) -> tuple[Dict[str, List[LOPlanCandidate]], Dict[str, List[IF2SpurControlRegion]]]:
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
        control_regions_by_cfg: Dict[str, List[IF2SpurControlRegion]] = {}

        coarse_min = cfg.grids.coarse_spur_margin_min_db  # negative threshold
        rf_by_id: Dict[str, RfConfiguration] = {
            rc.config_id: rc for rc in cfg.rf_configurations
        }

        for config_id, cand_list in per_cfg_candidates.items():
            rf_conf = rf_by_id[config_id]
            survivors: List[LOPlanCandidate] = []
            all_regions: List[IF2SpurControlRegion] = []
            cfg_worst_margin: Optional[float] = None

            for cand in cand_list:
                regions, worst_margin = coarse_if2_spur_control_regions_for_lo_plan(
                    cfg,
                    rf_conf,
                    cand,
                    self.rf_filter,
                )

                if worst_margin is not None:
                    if cfg_worst_margin is None or worst_margin < cfg_worst_margin:
                        cfg_worst_margin = worst_margin

                # If worst_margin is "very negative" (e.g. < -10 dB),
                # we consider this LO plan hopeless and drop it.
                if worst_margin is not None and worst_margin < coarse_min:
                    continue

                survivors.append(cand)
                all_regions.extend(regions)

            if not survivors:
                # Fallback: keep at least one candidate so planner can fail
                # with a clearer downstream error, but log that coarse
                # pruning rejected all candidates.
                logger.warning(
                    "Config '%s': all %d LO candidates failed coarse spur "
                    "margin threshold (worst margin %.1f dB); keeping the "
                    "first candidate for detailed analysis.",
                    config_id,
                    len(cand_list),
                    cfg_worst_margin if cfg_worst_margin is not None else float("nan"),
                )
                survivors = cand_list[:1]

            pruned[config_id] = survivors
            control_regions_by_cfg[config_id] = all_regions

        return pruned, control_regions_by_cfg
    

    def _search_lo_plans_for_filter_count(
        self,
        per_cfg_candidates: Dict[str, List[LOPlanCandidate]],
        control_regions_by_cfg: Dict[str, List[IF2SpurControlRegion]],
        n_filters: int,
    ) -> Optional[tuple[List[LOPlanCandidate], IF2BankDesignResult, List[SpurResult], Dict[str, ConfigSpurSummary]]]:
        """
        For a fixed IF2 filter count (n_filters), search over LO-plan combinations
        and find a feasible solution that minimizes (n_distinct_LO1, n_distinct_LO2)
        in lexicographic order.

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

        best_lo_plans: Optional[List[LOPlanCandidate]] = None
        best_bank_design: Optional[IF2BankDesignResult] = None
        best_spur_results: List[SpurResult] = []
        best_summaries: Dict[str, ConfigSpurSummary] = {}
        best_score: Optional[tuple[int, int]] = None  # (n_distinct_LO1, n_distinct_LO2)

        combos_tested = 0

        def dfs(
            idx: int,
            current: List[LOPlanCandidate],
            used_lo1: set[float],
            used_lo2: set[float],
        ) -> None:
            nonlocal best_lo_plans, best_bank_design, best_spur_results, best_summaries
            nonlocal best_score, combos_tested

            if combos_tested >= max_combos:
                return

            # Reached a full assignment: one LO plan per configuration
            if idx == len(ordered_cfg_ids):
                combos_tested += 1

                # Design IF2 bank for this LO-plan set
                bank_design = design_if2_bank_for_lo_plans(
                    current,
                    cfg.filters.if2_constraints,
                    spur_control_regions=control_regions_by_cfg,
                    target_n_filters=n_filters,
                )

                # Evaluate spurs for this LO-plan + IF2 bank
                spur_results, summaries = self._evaluate_for_bank_design(
                    current,
                    bank_design,
                )

                # Check feasibility: all margins (if defined) must be >= 0 dB
                violating: list[str] = []
                for cid, summary in summaries.items():
                    if (
                        summary.worst_in_band_margin_db is not None
                        and summary.worst_in_band_margin_db < 0.0
                    ):
                        violating.append(cid)
                    if (
                        summary.worst_out_band_margin_db is not None
                        and summary.worst_out_band_margin_db < 0.0
                    ):
                        violating.append(cid)

                if violating:
                    return  # infeasible combination

                # Feasible: evaluate LO reuse score
                n_lo1 = len(used_lo1)
                n_lo2 = len(used_lo2)
                score = (n_lo1, n_lo2)
                if best_score is None or score < best_score:
                    best_score = score
                    best_lo_plans = list(current)
                    best_bank_design = bank_design
                    best_spur_results = spur_results
                    best_summaries = summaries
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
                if best_score is not None:
                    partial_score = (len(new_used_lo1), len(new_used_lo2))
                    if partial_score > best_score:
                        continue

                current.append(cand)
                dfs(idx + 1, current, new_used_lo1, new_used_lo2)
                current.pop()

                if combos_tested >= max_combos:
                    break

        dfs(0, [], set(), set())

        if best_lo_plans is None or best_bank_design is None:
            return None

        return best_lo_plans, best_bank_design, best_spur_results, best_summaries



    def _select_single_lo_plan_per_config(
        self,
        per_cfg_candidates: Dict[str, List[LOPlanCandidate]],
    ) -> List[LOPlanCandidate]:
        """
        Select one LO plan per configuration.

        Heuristic (improved):
          * Use up to K best candidates per configuration (K derived from
            max_lo_candidates_per_rf).
          * Explore combinations in a depth-first manner, with pruning
            by partial LO reuse score.
          * Objective: minimize (n_distinct_LO1, n_distinct_LO2) in
            lexicographic order, then a small tie-breaker based on
            average |RF_center - config.rf_center|.
        """
        cfg = self.cfg
        rf_confs = cfg.rf_configurations
        max_per_cfg = max(1, min(4, cfg.grids.max_lo_candidates_per_rf))
        max_global_combos = 256  # hard safety cap

        # Order configs as given in config file
        ordered_cfg_ids = [rc.config_id for rc in rf_confs]

        # Limit number of candidates per config
        trimmed: Dict[str, List[LOPlanCandidate]] = {}
        for cid in ordered_cfg_ids:
            cands = per_cfg_candidates[cid]
            trimmed[cid] = cands[:max_per_cfg]

        best_plan: List[LOPlanCandidate] | None = None
        best_score: tuple[int, int, float] | None = None
        explored = 0

        def dfs(idx: int,
                current: List[LOPlanCandidate],
                used_lo1: set[float],
                used_lo2: set[float]) -> None:
            nonlocal best_plan, best_score, explored
            if explored >= max_global_combos:
                return
            if idx == len(ordered_cfg_ids):
                explored += 1
                n_lo1 = len(used_lo1)
                n_lo2 = len(used_lo2)

                # tie-breaker: average RF center error
                err_sum = 0.0
                for plan in current:
                    rc = next(rc for rc in rf_confs if rc.config_id == plan.config_id)
                    rf_center_plan = 0.5 * (plan.rf_band.start + plan.rf_band.stop)
                    err_sum += abs(rf_center_plan - rc.rf_center)
                avg_err = err_sum / max(len(current), 1)

                score = (n_lo1, n_lo2, avg_err)
                if best_score is None or score < best_score:
                    best_score = score
                    best_plan = list(current)
                return

            cfg_id = ordered_cfg_ids[idx]
            for cand in trimmed[cfg_id]:
                new_used_lo1 = set(used_lo1)
                new_used_lo2 = set(used_lo2)
                new_used_lo1.add(cand.lo1_freq)
                new_used_lo2.add(cand.lo2_freq)

                # Simple partial lower bound: we can't do better than current
                # number of distinct LOs even if remaining configs reuse them.
                partial_score = (len(new_used_lo1), len(new_used_lo2), 0.0)
                if best_score is not None and partial_score > best_score:
                    continue

                current.append(cand)
                dfs(idx + 1, current, new_used_lo1, new_used_lo2)
                current.pop()

        dfs(0, [], set(), set())

        # Fallback to greedy if search was somehow blocked
        if best_plan is None:
            used_lo1: set[float] = set()
            used_lo2: set[float] = set()
            best_plan = []
            for rc in rf_confs:
                cands = trimmed[rc.config_id]
                best_cand = None
                best_score_local = None
                for cand in cands:
                    new_lo1 = 0 if cand.lo1_freq in used_lo1 else 1
                    new_lo2 = 0 if cand.lo2_freq in used_lo2 else 1
                    score = (new_lo1 + new_lo2, new_lo1, new_lo2, cand.lo1_freq, cand.lo2_freq)
                    if best_score_local is None or score < best_score_local:
                        best_score_local = score
                        best_cand = cand
                best_plan.append(best_cand)
                used_lo1.add(best_cand.lo1_freq)
                used_lo2.add(best_cand.lo2_freq)

        return best_plan

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
                    )
                    futures[fut] = plan.config_id

                for fut in as_completed(futures):
                    config_id = futures[fut]
                    res, summary = fut.result()
                    spur_results.extend(res)
                    summaries[config_id] = summary
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
                )
                spur_results.extend(res)
                summaries[plan.config_id] = summary

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
        pruned_candidates, control_regions_by_cfg = self._coarse_prune_lo_candidates(
            per_cfg_candidates
        )

        min_f = cfg.filters.if2_constraints.min_filters
        max_f = cfg.filters.if2_constraints.max_filters

        best_lo_plans: Optional[List[LOPlanCandidate]] = None
        best_bank_design: Optional[IF2BankDesignResult] = None
        best_spur_results: List[SpurResult] = []
        best_summaries: Dict[str, ConfigSpurSummary] = {}

        # 3) Sweep IF2 filter count (lexicographic: min filters first)
        for n_filters in range(min_f, max_f + 1):
            search_res = self._search_lo_plans_for_filter_count(
                pruned_candidates,
                control_regions_by_cfg,
                n_filters,
            )
            if search_res is None:
                continue

            lo_plans, bank_design, spur_results, summaries = search_res
            best_lo_plans = lo_plans
            best_bank_design = bank_design
            best_spur_results = spur_results
            best_summaries = summaries
            # First n_filters with any feasible solution is the lexicographic
            # optimum in terms of IF2 filter count; within this n_filters,
            # LO retune counts have already been minimized.
            break

        if best_bank_design is None or best_lo_plans is None:
            raise RuntimeError(
                "No feasible IF2 bank / LO plan combination found within "
                "[min_filters, max_filters] that satisfies spur limits for all configurations."
            )

        return PlannerResult(
            system_config=cfg,
            lo_plans=best_lo_plans,
            if2_bank_design=best_bank_design,
            spur_results=best_spur_results,
            summaries=best_summaries,
        )
