# src/buc_planner/spur_engine.py
from __future__ import annotations

try:
    from numba import njit
except Exception:  # noqa: BLE001
    njit = None


from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable

import numpy as np

from .config_models import (
    SystemConfig,
    RfConfiguration,
    Range,
    dB,
    dBc,
    Freq,
)
from .filters import IF2Filter, RFFilter
from .mixers import (
    MixerWidebandSpurBand,
    MixerInputBand,
    resolve_spur_families_for_tones,
    LOTone,
    generate_wideband_spur_band,
)
from .los import LOPlanCandidate

import logging
logger = logging.getLogger(__name__)


@dataclass
class SpurResult:
    """
    Single spur (wideband or narrowband) at the RF output.

    All levels are expressed in dBc relative to the desired RF integrated power.
    The current planning-grade model assumes that the desired RF integrated
    power is 0 dBc (i.e. same reference as IF1), so level_dbc is also an
    absolute dB value under that convention. If a non-zero desired RF gain
    G_chain,des is introduced in the future, the mapping to absolute dB should
    be adjusted in the output layer.
    """
    config_id: str
    mixer_name: str
    spur_name: str
    f_start: Freq
    f_stop: Freq
    in_band: bool
    out_of_band: bool
    level_dbc: dBc
    margin_db: Optional[dB]
    filter_att_if2_db: dB
    filter_att_rf_db: dB
    origin_m: int
    origin_n: int
    lo_tone_name: str
    input_band_name: str
    used_unspecified_floor: bool = False
    # Limit bookkeeping
    scalar_limit_dbc: Optional[dBc] = None
    mask_limit_dbc: Optional[dBc] = None
    scalar_margin_db: Optional[dB] = None
    mask_margin_db: Optional[dB] = None
	
	
@dataclass
class IF2SpurControlRegion:
    """
    Coarse region at the Mixer2 IF (IF2) node where we need
    additional attenuation from the IF2 BPF bank.

    required_rejection_db is "extra attenuation needed at IF2"
    so that the spur would just meet the scalar spur limit,
    based on a simple no-filter coarse model.
    """
    config_id: str
    kind: str               # "in_band" or "out_of_band"
    freq_start: Freq
    freq_stop: Freq
    required_rejection_db: dB
    spur_name: str
    mixer_name: str


@dataclass
class ConfigSpurSummary:
    config_id: str
    worst_in_band_spur_dbc: Optional[dBc]
    worst_in_band_margin_db: Optional[dB]
    worst_out_band_spur_dbc: Optional[dBc]
    worst_out_band_margin_db: Optional[dB]


def _segment_range(r: Range, step_hz: float) -> List[Range]:
    """
    Split a Range into contiguous sub-ranges with max width ≈ step_hz.

    If step_hz <= 0 or larger than the range width, return [r] unchanged.
    """
    if step_hz <= 0 or r.width <= 0 or step_hz >= r.width:
        return [r]

    n_seg = int(np.ceil(r.width / step_hz))
    edges = np.linspace(r.start, r.stop, n_seg + 1)
    segments: List[Range] = []
    for i in range(n_seg):
        segments.append(Range(start=float(edges[i]), stop=float(edges[i + 1])))
    return segments


def _build_if1_input_bands(
    cfg: SystemConfig,
    rf_conf: RfConfiguration,
) -> List[MixerInputBand]:
    """
    Fundamental + harmonics, clipped by Mixer1 IF range, then segmented along
    an IF1 frequency grid (spec 9.5).

    Power semantics (aligned with spec):

      * Fundamental:
          - The IF1 reference power (0 dBc) is defined as the integrated power
            over the full configured IF1 band cfg.if1_band.
          - We model flat PSD over this band.
          - When we segment the (possibly smaller) effective IF1 band for this
            configuration, each segment is assigned integrated power
                0 dBc + 10*log10(BW_seg / BW_IF1_total)
            so that the sum of all segments over the full IF1 band would be
            0 dBc.

      * Harmonic k:
          - cfg.if1_harmonics_dbc[k] is interpreted as the integrated power
            over the *entire* harmonic band BEFORE clipping:
                [k * IF1_eff_start, k * IF1_eff_stop],
            where IF1_eff_start/stop come from the effective IF1 band for this
            configuration (rf_conf.if1_subband or cfg.if1_band).
            This is slightly different from the numeric example in the spec
            (which uses the global 950–2450 MHz band), but is an equivalent
            planning-grade model tied to the actually used IF1 band.
          - We again assume flat PSD over that harmonic band.
          - After clipping by Mixer1 IF range and segmenting, each segment
            gets integrated power:
                L_harm,k + 10*log10(BW_seg / BW_harm_full)
            where BW_harm_full is the width of the un-clipped harmonic band.
          - Portions of the harmonic band outside Mixer1 IF range simply do
            not generate spurs.

    Uses:
      * cfg.grids.if1_grid_step_hz as segment width.
      * cfg.grids.max_if1_harmonic_order to truncate harmonics.
      * cfg.if1_harmonics_dbc for harmonic levels.
    """
    bands: List[MixerInputBand] = []

    # Effective IF1 band for this configuration (may be a sub-band)
    if1_band_cfg = rf_conf.if1_subband or cfg.if1_band

    # Clip by Mixer1 IF range
    m1_if_range = cfg.mixer1.ranges.if_range
    base_band = if1_band_cfg.intersect(m1_if_range)
    if base_band is None or base_band.width <= 0:
        return bands

    step = cfg.grids.if1_grid_step_hz
    base_segments = _segment_range(base_band, step)

    # --- Fundamental segments (0 dBc integrated over full cfg.if1_band) ---
    if1_ref_band = cfg.if1_band
    bw_if1_total = max(if1_ref_band.width, 1e-30)  # protect log10

    for idx, seg in enumerate(base_segments):
        bw_seg = max(seg.width, 1e-30)
        # segment integrated power relative to IF1 reference
        level_seg_dbc = 10.0 * np.log10(bw_seg / bw_if1_total)
        bands.append(
            MixerInputBand(
                name=f"IF1_fund_seg{idx}",
                f_start=seg.start,
                f_stop=seg.stop,
                level_dbc_integrated=level_seg_dbc,
            )
        )

    # --- Harmonics ---
    max_h = cfg.grids.max_if1_harmonic_order

    for order, lvl in cfg.if1_harmonics_dbc.items():
        if order <= 1 or order > max_h:
            continue

        # Full harmonic band BEFORE clipping, based on the effective IF1 band
        harm_full = Range(
            start=order * if1_band_cfg.start,
            stop=order * if1_band_cfg.stop,
        )
        bw_harm_full = max(harm_full.width, 1e-30)

        # Clip by Mixer1 IF range (spec: clipping happens before spur generation)
        harm_clipped = harm_full.intersect(m1_if_range)
        if harm_clipped is None or harm_clipped.width <= 0:
            continue

        harm_step = step * order if step > 0 else 0.0
        harm_segments = _segment_range(harm_clipped, harm_step)

        for idx, seg in enumerate(harm_segments):
            bw_seg = max(seg.width, 1e-30)
            # Integrated power of this segment relative to IF1 fundamental:
            # L_harm,k (whole band) + 10*log10(BW_seg / BW_harm_full)
            level_seg_dbc = lvl + 10.0 * np.log10(bw_seg / bw_harm_full)
            bands.append(
                MixerInputBand(
                    name=f"IF1_h{order}_seg{idx}",
                    f_start=seg.start,
                    f_stop=seg.stop,
                    level_dbc_integrated=level_seg_dbc,
                )
            )

    return bands


def _build_lo_tones_for_synth(
    lo_freq: Freq,
    synth_cfg,
) -> List[LOTone]:
    tones: List[LOTone] = [LOTone(name="fundamental", freq=lo_freq, level_dbc=0.0)]
    for h in synth_cfg.harmonics:
        tones.append(
            LOTone(
                name=f"harmonic_{h.order}",
                freq=h.order * lo_freq,
                level_dbc=h.level_dbc,
            )
        )
    if synth_cfg.pfd_frequency and synth_cfg.pll_spurs:
        for spur in synth_cfg.pll_spurs:
            df = spur.offset_multiple * synth_cfg.pfd_frequency
            tones.append(
                LOTone(
                    name=f"pll_spur_+{spur.offset_multiple}",
                    freq=lo_freq + df,
                    level_dbc=spur.level_dbc,
                )
            )
            tones.append(
                LOTone(
                    name=f"pll_spur_-{spur.offset_multiple}",
                    freq=lo_freq - df,
                    level_dbc=spur.level_dbc,
                )
            )
    return tones
	
	
def coarse_if2_spur_control_regions_for_lo_plan(
    cfg: SystemConfig,
    rf_conf: RfConfiguration,
    lo_plan: LOPlanCandidate,
    rf_filter: RFFilter,
) -> tuple[list[IF2SpurControlRegion], Optional[dB]]:
    """
    Coarse pass used for:
      * deriving IF2 spur-control regions, and
      * early pruning of hopeless LO plans.

    Model:
      * Only Mixer1 wideband spurs.
      * Ignore IF2 filter completely.
      * Map surviving Mixer1 spurs through Mixer2 fundamental path
        into RF to classify in-band / out-of-band.
      * Compare spur level to scalar limits (mask ignored for speed).
      * For any spur above its limit (negative margin), record an
        IF2 region with required_rejection_db at IF2 equal to -margin.
      * Return:
          (regions, worst_margin_dB)
        where worst_margin_dB is the minimum scalar margin across
        all considered spurs (negative => violation).

    NOTE:
      Frequency-dependent spur masks (if configured) are intentionally
      ignored in this coarse pass for performance. Masks are applied only
      in the detailed spur evaluation.
    """
    grids = cfg.grids
    regions: list[IF2SpurControlRegion] = []
    worst_margin: Optional[dB] = None

    in_band_limit = cfg.spur_limits.in_band_limit_dbc
    oob_limit = cfg.spur_limits.out_of_band_limit_dbc
    oob_range = cfg.spur_limits.out_of_band_range
    rf_band_global = cfg.rf_band

    # Per-configuration in-band RF channel
    rf_inband = Range(
        start=rf_conf.rf_center - 0.5 * rf_conf.rf_bandwidth,
        stop=rf_conf.rf_center + 0.5 * rf_conf.rf_bandwidth,
    )

    # 1) Mixer1 input bands (same helper as detailed path)
    m1_bands = _build_if1_input_bands(cfg, rf_conf)
    if not m1_bands:
        return regions, None

    # 2) Mixer1 spur families (all LO1 tones)
    m1_lo_tones = _build_lo_tones_for_synth(lo_plan.lo1_freq, cfg.lo1)
    m1_spur_specs = resolve_spur_families_for_tones(cfg.mixer1, m1_lo_tones)

    m2_if_range = cfg.mixer2.ranges.if_range

    # Helper: classify center -> in/out band (no mask)
    def classify_rf_center(f_center: float) -> tuple[bool, bool]:
        in_band = rf_inband.contains(f_center)
        if oob_range is None:
            out_band = not in_band
        else:
            out_band = oob_range.contains(f_center) and not in_band
        return in_band, out_band

    # 3) For each Mixer1 spur, see what happens if it enters Mixer2 as IF
    sign2 = lo_plan.sign_combo.sign2
    lo2_freq = lo_plan.lo2_freq

    for in_band in m1_bands:
        for spec in m1_spur_specs:
            spur = generate_wideband_spur_band(
                input_band=in_band,
                spur_spec=spec,
                min_level_considered_dbc=grids.min_spur_level_considered_dbc,
            )
            if spur is None:
                continue

            # Spur at Mixer1 RF node, now see intersection with Mixer2 IF range
            spur_if_range = Range(start=spur.f_start, stop=spur.f_stop)
            inter = spur_if_range.intersect(m2_if_range)
            if inter is None or inter.width <= 0:
                continue

            # Use intersection as "IF2 spur band"
            if2_spur_band = inter
            if if2_spur_band.width <= 0:
                continue

            # Map IF2 spur band through Mixer2 fundamental path
            if sign2 > 0:
                rf_lo = lo2_freq + sign2 * if2_spur_band.start
                rf_hi = lo2_freq + sign2 * if2_spur_band.stop
            else:
                rf_lo = lo2_freq - if2_spur_band.stop
                rf_hi = lo2_freq - if2_spur_band.start

            rf_center = 0.5 * (rf_lo + rf_hi)
            in_band, out_band = classify_rf_center(rf_center)
            if not in_band and not out_band:
                continue

            # Spur level at RF *before* any IF2/RF filter:
            level_before = spur.spur_level_rel_if1_dbc

            # RF BPF coarse attenuation at center (cheap)
            a_rf = float(rf_filter.attenuation_db(np.array([rf_center]))[0])
            level_after_rf = level_before - a_rf

            # Pick scalar limit
            scalar_limit: Optional[dBc] = None
            if in_band and in_band_limit is not None:
                scalar_limit = in_band_limit
            elif out_band and oob_limit is not None:
                scalar_limit = oob_limit
            if scalar_limit is None:
                continue

            margin = scalar_limit - level_after_rf  # positive = OK, negative = violation

            if worst_margin is None or margin < worst_margin:
                worst_margin = margin

            # Only record spur-control regions for actual violations
            if margin < 0.0:
                regions.append(
                    IF2SpurControlRegion(
                        config_id=rf_conf.config_id,
                        kind="in_band" if in_band else "out_of_band",
                        freq_start=if2_spur_band.start,
                        freq_stop=if2_spur_band.stop,
                        required_rejection_db=-margin,  # extra attenuation needed
                        spur_name=spur.name,
                        mixer_name=spur.mixer_name,
                    )
                )

    return regions, worst_margin


if njit is not None:
    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _integrate_spur_linear_numba(
        spur_linear_before: float,
        total_att_db: np.ndarray,
    ) -> float:
        gains_linear = 10.0 ** (-total_att_db / 10.0)
        avg_gain = gains_linear.mean()
        return spur_linear_before * avg_gain
else:
    _integrate_spur_linear_numba = None
	
def _integrate_spur_band_through_filters(
    spur_band: MixerWidebandSpurBand,
    if2_filter: IF2Filter,
    rf_filter: RFFilter,
    cfg: SystemConfig,
    apply_if2_filter: bool = True,
) -> Tuple[dBc, dB, dB]:
    """
    Integrate a wideband spur over its band with IF2 and RF filter attenuations.

    Modelling details:

      * The spur is treated as having flat PSD across [f_start, f_stop].
      * The pre-filter *integrated* spur level is spur_band.spur_level_rel_if1_dbc
        (relative to the IF1 reference integrated power).
      * We sample attenuation A_total(f) in dB on a grid across the spur band,
        convert to linear gain^2 G2(f) = 10^(-A_total/10), and then use the
        *average* G2 over the band:
            P_after = P_before * mean(G2(f))

        This is equivalent to performing an explicit ∑ PSD * G2(f) * Δf
        integration under the flat-PSD assumption, but is cheaper to compute.

    Desired RF integrated power is taken as 0 dBc, so the returned spur level
    is directly in dBc relative to the desired RF product.
    """
    f_start, f_stop = spur_band.f_start, spur_band.f_stop
    if f_stop <= f_start:
        return -300.0, 0.0, 0.0

    span = f_stop - f_start
    df = cfg.grids.spur_integration_step_hz

    # Defensive: if df is <= 0 or larger than the band, fall back
    # to a single-step integration over the entire band.
    if df <= 0.0 or df >= span:
        n_steps = 1
    else:
        n_steps = max(int(np.ceil(span / df)), 1)

    freqs = f_start + (np.arange(n_steps) + 0.5) * (f_stop - f_start) / n_steps

    if apply_if2_filter:
        a_if2_db = if2_filter.attenuation_db(freqs)
    else:
        a_if2_db = np.zeros_like(freqs)
    a_rf_db = rf_filter.attenuation_db(freqs)
    total_att_db = a_if2_db + a_rf_db

    spur_linear_before = 10.0 ** (spur_band.spur_level_rel_if1_dbc / 10.0)

    if cfg.grids.use_numba and _integrate_spur_linear_numba is not None:
        spur_linear_after = _integrate_spur_linear_numba(
            float(spur_linear_before),
            total_att_db.astype(np.float64),
        )
    else:
        gains_linear = 10.0 ** (-total_att_db / 10.0)
        avg_gain = gains_linear.mean()
        spur_linear_after = spur_linear_before * avg_gain

    spur_level_after_dbc = 10.0 * np.log10(spur_linear_after + 1e-30)

    if2_att_db_equiv = float(a_if2_db.mean())
    rf_att_db_equiv = float(a_rf_db.mean())
    return spur_level_after_dbc, if2_att_db_equiv, rf_att_db_equiv


def classify_spur_band_in_out(
    spur_band: MixerWidebandSpurBand,
    rf_band: Range,
    oob_range: Optional[Range],
) -> Tuple[bool, bool]:
    center = spur_band.center_freq
    in_band = rf_band.contains(center)
    if oob_range is None:
        out_band = not in_band
    else:
        out_band = oob_range.contains(center) and not in_band
    return in_band, out_band


def evaluate_spurs_for_config_and_lo_plan(
    cfg: SystemConfig,
    rf_conf: RfConfiguration,
    lo_plan: LOPlanCandidate,
    if2_filter: IF2Filter,
    rf_filter: RFFilter,
) -> Tuple[List[SpurResult], ConfigSpurSummary]:
    """
    Full spur evaluation for a single RF configuration and LO plan.

    Wideband spurs:
      * Mixer1: IF1 fundamentals + harmonics → IF2, then through IF2 & into Mixer2.
      * Mixer2: spurs from IF2 desired band + cascaded Mixer1 spurs +
                Mixer1 isolation contributions.

    Narrowband spurs:
      * Mixer2 LO→RF and IF→RF isolation, treated as tones (including LO harmonics/PLLs).
    """
    results: List[SpurResult] = []
    grids = cfg.grids

    # Spur mask (if provided)
    mask_cfg = cfg.spur_limits.mask
    mask_freqs: Optional[np.ndarray] = None
    mask_levels: Optional[np.ndarray] = None
    if mask_cfg and mask_cfg.csv_path:
        data = np.genfromtxt(mask_cfg.csv_path, delimiter=",", comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.size == 0 or data.shape[1] < 2:
            raise ValueError(
                f"Spur mask CSV '{mask_cfg.csv_path}' must have at least two columns (freq, level_dbc)."
            )

        # Treat leading NaN row as an un-commented header
        if data.shape[0] > 1 and (np.isnan(data[0, 0]) or np.isnan(data[0, 1])):
            data = data[1:, :]
            if data.size == 0 or data.shape[1] < 2:
                raise ValueError(
                    f"Spur mask CSV '{mask_cfg.csv_path}' has only a header and no data rows."
                )

        mask_freqs = data[:, 0].astype(float)
        mask_levels = data[:, 1].astype(float)
        idx = np.argsort(mask_freqs)
        mask_freqs = mask_freqs[idx]
        mask_levels = mask_levels[idx]

    def interpolate_mask_level_center(freq: float) -> Optional[float]:
        if mask_freqs is None or mask_levels is None:
            return None
        lo_f = mask_freqs[0]
        hi_f = mask_freqs[-1]
        f_clamped = float(np.clip(freq, lo_f, hi_f))
        return float(np.interp(f_clamped, mask_freqs, mask_levels))

    def interpolate_mask_level_worst_case(f_start: float, f_stop: float) -> Optional[float]:
        if mask_freqs is None or mask_levels is None:
            return None
        lo_f = mask_freqs[0]
        hi_f = mask_freqs[-1]
        # Sample along the spur band
        n = 16
        freqs = np.linspace(f_start, f_stop, n)
        freqs = np.clip(freqs, lo_f, hi_f)
        levels = np.interp(freqs, mask_freqs, mask_levels)
        return float(levels.min())

    mask_eval_mode = cfg.spur_limits.mask_eval_mode.lower()
    if mask_eval_mode not in ("center", "worst_case"):
        mask_eval_mode = "center"

    # 1) Mixer1 input bands (IF1 fundamental + harmonics, clipped by Mixer1 IF)
    m1_bands = _build_if1_input_bands(cfg, rf_conf)
    m1_lo_tones = _build_lo_tones_for_synth(lo_plan.lo1_freq, cfg.lo1)
    m1_spur_specs = resolve_spur_families_for_tones(cfg.mixer1, m1_lo_tones)

    m1_spurs: List[MixerWidebandSpurBand] = []
    for in_band in m1_bands:
        for spec in m1_spur_specs:
            spur = generate_wideband_spur_band(
                input_band=in_band,
                spur_spec=spec,
                min_level_considered_dbc=grids.min_spur_level_considered_dbc,
            )
            if spur is not None:
                m1_spurs.append(spur)

    # 2) Mixer2 input bands
    if2_band = lo_plan.if2_band
    m2_if_range = cfg.mixer2.ranges.if_range

    # Optional focus region: desired IF2 ± margin
    focus_margin = grids.mixer2_if2_focus_margin_hz
    if focus_margin > 0.0:
        focus_band = Range(
            start=if2_band.start - focus_margin,
            stop=if2_band.stop + focus_margin,
        )
    else:
        focus_band = None

    def focus_clip(r: Range) -> Optional[Range]:
        if focus_band is None:
            return r
        return r.intersect(focus_band)

    # Desired IF2 band as mixer2 input (reference 0 dBc)
    base_if2 = if2_band.intersect(m2_if_range)
    if base_if2 is None or base_if2.width <= 0:
        m2_input_bands: List[MixerInputBand] = []
    else:
        base_if2 = focus_clip(base_if2) or base_if2
        m2_input_bands = [
            MixerInputBand(
                name="IF2_desired_band",
                f_start=base_if2.start,
                f_stop=base_if2.stop,
                level_dbc_integrated=0.0,
            )
        ]

    # Cascaded Mixer1 spurs that survive IF2 and land in Mixer2 IF range
    for spur in m1_spurs:
        spur_range = Range(start=spur.f_start, stop=spur.f_stop)
        inter = spur_range.intersect(m2_if_range)
        if inter is None or inter.width <= 0:
            continue
        inter = focus_clip(inter) or inter
        if inter.width <= 0:
            continue

        df = grids.spur_integration_step_hz
        n_steps = max(int(np.ceil(inter.width / df)), 1)
        freqs = inter.start + (np.arange(n_steps) + 0.5) * inter.width / n_steps
        a_if2 = if2_filter.attenuation_db(freqs)
        a_if2_avg = float(a_if2.mean())

        new_level = spur.spur_level_rel_if1_dbc - a_if2_avg
        if new_level < grids.min_spur_level_considered_dbc:
            continue

        m2_input_bands.append(
            MixerInputBand(
                name=f"M1spur[{spur.name}]_into_M2",
                f_start=inter.start,
                f_stop=inter.stop,
                level_dbc_integrated=new_level,
            )
        )

    # 2a) Mixer1 isolation contributions into Mixer2
    # LO1 -> IF2 leakage: each LO1 tone leaks, sees IF2 filter, then is an IF2 tone into Mixer2
    if cfg.mixer1.isolation.lo_to_rf_dbc is not None:
        for tone in m1_lo_tones:
            f0 = tone.freq
            if not m2_if_range.contains(f0):
                continue
            a_if2 = float(if2_filter.attenuation_db(np.array([f0]))[0])
            level_after_if2 = cfg.mixer1.isolation.lo_to_rf_dbc + tone.level_dbc - a_if2
            if level_after_if2 < grids.min_spur_level_considered_dbc:
                continue
            bw = max(grids.spur_integration_step_hz, 1.0)
            band = Range(start=f0 - 0.5 * bw, stop=f0 + 0.5 * bw)
            band = focus_clip(band) or band
            if band.width <= 0:
                continue
            m2_input_bands.append(
                MixerInputBand(
                    name=f"M1_LO_leak_{tone.name}",
                    f_start=band.start,
                    f_stop=band.stop,
                    level_dbc_integrated=level_after_if2,
                )
            )

    # IF1 -> IF2 leakage from Mixer1: approximate as a wideband leak of the IF1 band
    if cfg.mixer1.isolation.if_to_rf_dbc is not None:
        if1_band = rf_conf.if1_subband or cfg.if1_band
        leak_band = if1_band.intersect(m2_if_range)
        if leak_band is not None and leak_band.width > 0:
            leak_band = focus_clip(leak_band) or leak_band
            if leak_band.width > 0:
                df = grids.spur_integration_step_hz
                n_steps = max(int(np.ceil(leak_band.width / df)), 1)
                freqs = leak_band.start + (np.arange(n_steps) + 0.5) * leak_band.width / n_steps
                a_if2 = if2_filter.attenuation_db(freqs)
                a_if2_avg = float(a_if2.mean())
                level_after_if2 = cfg.mixer1.isolation.if_to_rf_dbc - a_if2_avg
                if level_after_if2 >= grids.min_spur_level_considered_dbc:
                    m2_input_bands.append(
                        MixerInputBand(
                            name="M1_IF_leak_wideband",
                            f_start=leak_band.start,
                            f_stop=leak_band.stop,
                            level_dbc_integrated=level_after_if2,
                        )
                    )

    # 3) Mixer2 spur families
    m2_lo_tones = _build_lo_tones_for_synth(lo_plan.lo2_freq, cfg.lo2)
    m2_spur_specs = resolve_spur_families_for_tones(cfg.mixer2, m2_lo_tones)

    m2_spurs: List[MixerWidebandSpurBand] = []
    for in_band in m2_input_bands:
        for spec in m2_spur_specs:
            spur = generate_wideband_spur_band(
                input_band=in_band,
                spur_spec=spec,
                min_level_considered_dbc=grids.min_spur_level_considered_dbc,
            )
            if spur is not None:
                m2_spurs.append(spur)

    # In-band is the RF *configuration* channel; out-of-band is defined
    # by spur_limits.out_of_band_range (if provided).
    rf_inband = Range(
        start=rf_conf.rf_center - 0.5 * rf_conf.rf_bandwidth,
        stop=rf_conf.rf_center + 0.5 * rf_conf.rf_bandwidth,
    )
    oob_range = cfg.spur_limits.out_of_band_range

    in_band_limit = cfg.spur_limits.in_band_limit_dbc
    oob_limit = cfg.spur_limits.out_of_band_limit_dbc

    worst_in_dbc: Optional[dBc] = None
    worst_in_margin: Optional[dB] = None
    worst_oob_dbc: Optional[dBc] = None
    worst_oob_margin: Optional[dB] = None

    def update_worst(
        level_dbc: dBc,
        in_band: bool,
        out_band: bool,
        margin: Optional[dB],
    ):
        nonlocal worst_in_dbc, worst_in_margin, worst_oob_dbc, worst_oob_margin
        if margin is None:
            return

        # We care about the spur with the *smallest* margin (most negative),
        # not the largest absolute spur level.
        if in_band:
            if worst_in_margin is None or margin < worst_in_margin:
                worst_in_margin = margin
                worst_in_dbc = level_dbc
        elif out_band:
            if worst_oob_margin is None or margin < worst_oob_margin:
                worst_oob_margin = margin
                worst_oob_dbc = level_dbc

    def process_spur(
        spur: MixerWidebandSpurBand,
        mixer_name: str,
        apply_if2_filter: bool,
    ):
        nonlocal results

        in_band, out_band = classify_spur_band_in_out(spur, rf_inband, oob_range)
        if not in_band and not out_band:
            return

        lvl_after_dbc, a_if2, a_rf = _integrate_spur_band_through_filters(
            spur, if2_filter, rf_filter, cfg, apply_if2_filter=apply_if2_filter
        )
        level_dbc = lvl_after_dbc

        f_center = spur.center_freq
        # Scalar limits
        scalar_limit: Optional[dBc] = None
        if in_band and in_band_limit is not None:
            scalar_limit = in_band_limit
        elif out_band and oob_limit is not None:
            scalar_limit = oob_limit

        # Mask limits
        mask_limit: Optional[dBc] = None
        if mask_cfg is not None:
            if in_band and mask_cfg.apply_in_band:
                if mask_eval_mode == "center":
                    mask_limit = interpolate_mask_level_center(f_center)
                else:
                    mask_limit = interpolate_mask_level_worst_case(spur.f_start, spur.f_stop)
            elif out_band and mask_cfg.apply_out_of_band:
                if mask_eval_mode == "center":
                    mask_limit = interpolate_mask_level_center(f_center)
                else:
                    mask_limit = interpolate_mask_level_worst_case(spur.f_start, spur.f_stop)

        effective_limit: Optional[dBc] = None
        if scalar_limit is not None and mask_limit is not None:
            effective_limit = min(scalar_limit, mask_limit)
        elif scalar_limit is not None:
            effective_limit = scalar_limit
        elif mask_limit is not None:
            effective_limit = mask_limit

        scalar_margin = scalar_limit - level_dbc if scalar_limit is not None else None
        mask_margin = mask_limit - level_dbc if mask_limit is not None else None
        margin = effective_limit - level_dbc if effective_limit is not None else None

        update_worst(level_dbc, in_band, out_band, margin)

        res = SpurResult(
            config_id=rf_conf.config_id,
            mixer_name=mixer_name,
            spur_name=spur.name,
            f_start=spur.f_start,
            f_stop=spur.f_stop,
            in_band=in_band,
            out_of_band=out_band,
            level_dbc=level_dbc,
            margin_db=margin,
            filter_att_if2_db=a_if2,
            filter_att_rf_db=a_rf,
            origin_m=spur.m,
            origin_n=spur.n,
            lo_tone_name=spur.lo_tone_name,
            input_band_name=spur.input_band_name,
            used_unspecified_floor=spur.used_unspecified_floor,
            scalar_limit_dbc=scalar_limit,
            mask_limit_dbc=mask_limit,
            scalar_margin_db=scalar_margin,
            mask_margin_db=mask_margin,
        )
        results.append(res)

    # Only Mixer2 spurs reach RF band as wideband products in this model
    for spur in m2_spurs:
        process_spur(spur, mixer_name=cfg.mixer2.name, apply_if2_filter=False)

    # 4) Narrowband isolation spurs for Mixer2 (LO->RF, IF->RF)
    def process_narrowband_spur(
        name: str,
        freq: Freq,
        level_before_dbc: dBc,
    ):
        nonlocal results

        # Attenuation: only RF filter (isolation spur is at Mixer2 RF port)
        a_if2 = 0.0
        a_rf = float(rf_filter.attenuation_db(np.array([freq]))[0])
        level_after = level_before_dbc - a_rf

        in_band = rf_inband.contains(freq)
        if oob_range is None:
            out_band = not in_band
        else:
            out_band = oob_range.contains(freq) and not in_band
        if not in_band and not out_band:
            return

        # Limits
        scalar_limit: Optional[dBc] = None
        if in_band and in_band_limit is not None:
            scalar_limit = in_band_limit
        elif out_band and oob_limit is not None:
            scalar_limit = oob_limit

        mask_limit: Optional[dBc] = None
        if mask_cfg is not None:
            if in_band and mask_cfg.apply_in_band:
                if mask_eval_mode == "center":
                    mask_limit = interpolate_mask_level_center(freq)
                else:
                    mask_limit = interpolate_mask_level_worst_case(freq, freq)
            elif out_band and mask_cfg.apply_out_of_band:
                if mask_eval_mode == "center":
                    mask_limit = interpolate_mask_level_center(freq)
                else:
                    mask_limit = interpolate_mask_level_worst_case(freq, freq)

        effective_limit: Optional[dBc] = None
        if scalar_limit is not None and mask_limit is not None:
            effective_limit = min(scalar_limit, mask_limit)
        elif scalar_limit is not None:
            effective_limit = scalar_limit
        elif mask_limit is not None:
            effective_limit = mask_limit

        scalar_margin = scalar_limit - level_after if scalar_limit is not None else None
        mask_margin = mask_limit - level_after if mask_limit is not None else None
        margin = effective_limit - level_after if effective_limit is not None else None

        update_worst(level_after, in_band, out_band, margin)

        res = SpurResult(
            config_id=rf_conf.config_id,
            mixer_name=cfg.mixer2.name,
            spur_name=name,
            f_start=freq,
            f_stop=freq,
            in_band=in_band,
            out_of_band=out_band,
            level_dbc=level_after,
            margin_db=margin,
            filter_att_if2_db=a_if2,
            filter_att_rf_db=a_rf,
            origin_m=0,
            origin_n=0,
            lo_tone_name=name,
            input_band_name="isolation",
            used_unspecified_floor=False,
            scalar_limit_dbc=scalar_limit,
            mask_limit_dbc=mask_limit,
            scalar_margin_db=scalar_margin,
            mask_margin_db=mask_margin,
        )
        results.append(res)

    # Mixer2 LO->RF isolation for all LO2 tones
    if cfg.mixer2.isolation.lo_to_rf_dbc is not None:
        for tone in m2_lo_tones:
            level_before = cfg.mixer2.isolation.lo_to_rf_dbc + tone.level_dbc
            process_narrowband_spur(
                name=f"Mixer2_LO_leak_{tone.name}",
                freq=tone.freq,
                level_before_dbc=level_before,
            )

    # Mixer2 IF->RF isolation (tone at IF2 band center)
    if cfg.mixer2.isolation.if_to_rf_dbc is not None:
        if2_center = 0.5 * (if2_band.start + if2_band.stop)
        process_narrowband_spur(
            name="Mixer2_IF_leak_center",
            freq=if2_center,
            level_before_dbc=cfg.mixer2.isolation.if_to_rf_dbc,
        )

    summary = ConfigSpurSummary(
        config_id=rf_conf.config_id,
        worst_in_band_spur_dbc=worst_in_dbc,
        worst_in_band_margin_db=worst_in_margin,
        worst_out_band_spur_dbc=worst_oob_dbc,
        worst_out_band_margin_db=worst_oob_margin,
    )
    return results, summary