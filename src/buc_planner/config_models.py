# src/buc_planner/config_models.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Literal, Union

import yaml  # planning-grade: assume PyYAML available


Freq = float  # Hz (or consistent unit across config)
dB = float
dBc = float


@dataclass
class Range:
    """Closed interval [start, stop]. Units consistent with Freq."""
    start: Freq
    stop: Freq

    def contains(self, f: Freq) -> bool:
        return self.start <= f <= self.stop

    def intersect(self, other: "Range") -> Optional["Range"]:
        lo = max(self.start, other.start)
        hi = min(self.stop, other.stop)
        if lo <= hi:
            return Range(lo, hi)
        return None

    @property
    def width(self) -> float:
        return self.stop - self.start


@dataclass
class RfConfiguration:
    """Single RF configuration (e.g. RF channel / band)."""
    config_id: str
    rf_center: Freq
    rf_bandwidth: Freq
    # optional IF1 sub-band; if None, use global IF1 band
    if1_subband: Optional[Range] = None


@dataclass
class SpurTableEntry:
    """
    Spur table entry for a mixer: f_spur = m * f_LO ± n * f_IF.

    All levels are relative to the desired mixer product (fundamental) at that mixer.
    """
    m: int
    n: int
    level_dbc: dBc
    if_range: Range
    lo_range: Range
    rf_range: Range
    lo_tone_type: Optional[str] = None  # "fundamental", "harmonic_k", "pll_spur_N", etc.


@dataclass
class MixerSpurEnvelopePolicy:
    m_max: int
    n_max: int
    enforce_envelope_completeness: bool = False
    unspecified_floor_dbc: Optional[dBc] = None  # if None => ignore unspecified


@dataclass
class MixerIsolation:
    lo_to_rf_dbc: dBc
    if_to_rf_dbc: dBc


@dataclass
class MixerRanges:
    """
    Valid operating ranges for a mixer.

    Note:
      * For Mixer1, rf_range represents the *output* of Mixer1, i.e. the IF2 node
        (not the final RF band).
      * For Mixer2, rf_range is the true RF output range.
    """
    if_range: Range
    lo_range: Range
    rf_range: Range


@dataclass
class MixerConfig:
    name: str
    spur_table: List[SpurTableEntry]
    spur_envelope: MixerSpurEnvelopePolicy
    isolation: MixerIsolation
    ranges: MixerRanges
    # desired fundamental mapping: e.g. (m=+1,n=1) for LO+IF or (m=-1,n=1) for LO-IF.
    desired_m: int = 1
    desired_n: int = 1


@dataclass
class LOHarmonic:
    order: int
    level_dbc: dBc  # relative to fundamental LO tone


@dataclass
class PLLSpur:
    offset_multiple: int  # N in ±N * f_PFD
    level_dbc: dBc       # relative to LO fundamental


@dataclass
class LOSynthConfig:
    name: str
    freq_range: Range
    grid_step: Freq          # LO frequency step (Hz)
    pfd_frequency: Optional[Freq] = None
    harmonics: List[LOHarmonic] = field(default_factory=list)
    pll_spurs: List[PLLSpur] = field(default_factory=list)


@dataclass
class IF2BankConstraints:
    min_filters: int
    max_filters: int
    fc_range: Range
    bw_range: Range
    slope_range: Tuple[dB, dB]  # (S_min, S_max) in dB/decade (negative)
    # margin for early feasibility checks (Hz)
    feasibility_margin_hz: Freq = 0.0


@dataclass
class RFMaskConfig:
    csv_path: Optional[str] = None
    apply_in_band: bool = True
    apply_out_of_band: bool = True


@dataclass
class SpurLimitsConfig:
    in_band_limit_dbc: Optional[dBc] = None
    out_of_band_limit_dbc: Optional[dBc] = None
    mask: Optional[RFMaskConfig] = None
    out_of_band_range: Optional[Range] = None
    # How to evaluate mask vs wideband spur:
    #   "center"      -> mask at spur center frequency (current behaviour)
    #   "worst_case"  -> minimum allowed level over whole spur band
    mask_eval_mode: str = "center"  # "center" or "worst_case"


@dataclass
class GridsAndPerformanceConfig:
    if1_grid_step_hz: Freq
    spur_integration_step_hz: Freq
    max_if1_harmonic_order: int
    min_spur_level_considered_dbc: dBc = -120.0
    max_lo_candidates_per_rf: int = 50
    max_if2_bank_candidates: int = 50
    coarse_spur_margin_min_db: dB = -10.0
    # Optional restriction of Mixer2 spur generation to desired IF2 ± margin
    mixer2_if2_focus_margin_hz: Freq = 0.0
    parallel: bool = True
    use_numba: bool = False  # allows toggling JIT



@dataclass
class FilterConfig:
    if2_constraints: IF2BankConstraints
    rf_bpf_csv_path: str


@dataclass
class SystemConfig:
    """
    Top-level configuration object for a planning run.
    """
    if1_band: Range
    rf_band: Range
    rf_configurations: List[RfConfiguration]
    non_inverting_mapping_required: bool
    mixer1: MixerConfig
    mixer2: MixerConfig
    lo1: LOSynthConfig
    lo2: LOSynthConfig
    filters: FilterConfig
    spur_limits: SpurLimitsConfig
    grids: GridsAndPerformanceConfig

    # IF1 harmonics amplitudes: order -> level_dbc (integrated)
    if1_harmonics_dbc: Dict[int, dBc] = field(default_factory=dict)

    # Metadata / description
    description: Optional[str] = None


def _load_yaml_or_json(path: Path) -> dict:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    else:
        return json.loads(text)


def load_config(path: Union[str, Path]) -> SystemConfig:
    """
    Load a SystemConfig from a JSON or YAML file.
    """
    path = Path(path)
    raw = _load_yaml_or_json(path)

    def r_rng(d) -> Range:
        return Range(start=d["start"], stop=d["stop"])

    def r_rf_config(d) -> RfConfiguration:
        if1_sb = d.get("if1_subband")
        return RfConfiguration(
            config_id=str(d["config_id"]),
            rf_center=d["rf_center"],
            rf_bandwidth=d["rf_bandwidth"],
            if1_subband=r_rng(if1_sb) if if1_sb is not None else None,
        )

    def r_spur_entry(d) -> SpurTableEntry:
        return SpurTableEntry(
            m=d["m"],
            n=d["n"],
            level_dbc=d["level_dbc"],
            if_range=r_rng(d["if_range"]),
            lo_range=r_rng(d["lo_range"]),
            rf_range=r_rng(d["rf_range"]),
            lo_tone_type=d.get("lo_tone_type"),
        )

    def r_spur_envelope(d) -> MixerSpurEnvelopePolicy:
        return MixerSpurEnvelopePolicy(
            m_max=d["m_max"],
            n_max=d["n_max"],
            enforce_envelope_completeness=d.get("enforce_envelope_completeness", False),
            unspecified_floor_dbc=d.get("unspecified_floor_dbc"),
        )

    def r_mixer_ranges(d) -> MixerRanges:
        return MixerRanges(
            if_range=r_rng(d["if_range"]),
            lo_range=r_rng(d["lo_range"]),
            rf_range=r_rng(d["rf_range"]),
        )

    def r_mixer(d) -> MixerConfig:
        return MixerConfig(
            name=d["name"],
            spur_table=[r_spur_entry(st) for st in d.get("spur_table", [])],
            spur_envelope=r_spur_envelope(d["spur_envelope"]),
            isolation=MixerIsolation(
                lo_to_rf_dbc=d["isolation"]["lo_to_rf_dbc"],
                if_to_rf_dbc=d["isolation"]["if_to_rf_dbc"],
            ),
            ranges=r_mixer_ranges(d["ranges"]),
            desired_m=d.get("desired_m", 1),
            desired_n=d.get("desired_n", 1),
        )

    def r_lo(d) -> LOSynthConfig:
        return LOSynthConfig(
            name=d["name"],
            freq_range=r_rng(d["freq_range"]),
            grid_step=d["grid_step"],
            pfd_frequency=d.get("pfd_frequency"),
            harmonics=[
                LOHarmonic(order=h["order"], level_dbc=h["level_dbc"])
                for h in d.get("harmonics", [])
            ],
            pll_spurs=[
                PLLSpur(offset_multiple=s["offset_multiple"], level_dbc=s["level_dbc"])
                for s in d.get("pll_spurs", [])
            ],
        )

    def r_if2_constraints(d) -> IF2BankConstraints:
        return IF2BankConstraints(
            min_filters=d["min_filters"],
            max_filters=d["max_filters"],
            fc_range=r_rng(d["fc_range"]),
            bw_range=r_rng(d["bw_range"]),
            slope_range=(d["slope_range"][0], d["slope_range"][1]),
            feasibility_margin_hz=d.get("feasibility_margin_hz", 0.0),
        )

    def r_filters(d) -> FilterConfig:
        return FilterConfig(
            if2_constraints=r_if2_constraints(d["if2_constraints"]),
            rf_bpf_csv_path=d["rf_bpf_csv_path"],
        )

    def r_spur_limits(d) -> SpurLimitsConfig:
        mask_cfg = d.get("mask")
        mask = None
        if mask_cfg:
            mask = RFMaskConfig(
                csv_path=mask_cfg.get("csv_path"),
                apply_in_band=mask_cfg.get("apply_in_band", True),
                apply_out_of_band=mask_cfg.get("apply_out_of_band", True),
            )
        oob_range = d.get("out_of_band_range")
        return SpurLimitsConfig(
            in_band_limit_dbc=d.get("in_band_limit_dbc"),
            out_of_band_limit_dbc=d.get("out_of_band_limit_dbc"),
            mask=mask,
            out_of_band_range=r_rng(oob_range) if oob_range else None,
            mask_eval_mode=d.get("mask_eval_mode", "center"),
        )

    def r_grids(d) -> GridsAndPerformanceConfig:
        return GridsAndPerformanceConfig(
            if1_grid_step_hz=d["if1_grid_step_hz"],
            spur_integration_step_hz=d["spur_integration_step_hz"],
            max_if1_harmonic_order=d["max_if1_harmonic_order"],
            min_spur_level_considered_dbc=d.get("min_spur_level_considered_dbc", -120.0),
            max_lo_candidates_per_rf=d.get("max_lo_candidates_per_rf", 50),
            max_if2_bank_candidates=d.get("max_if2_bank_candidates", 50),
            coarse_spur_margin_min_db=d.get("coarse_spur_margin_min_db", -10.0),
            mixer2_if2_focus_margin_hz=d.get("mixer2_if2_focus_margin_hz", 0.0),
            parallel=d.get("parallel", True),
            use_numba=d.get("use_numba", False),
        )

    return SystemConfig(
        if1_band=r_rng(raw["if1_band"]),
        rf_band=r_rng(raw["rf_band"]),
        rf_configurations=[r_rf_config(rc) for rc in raw["rf_configurations"]],
        non_inverting_mapping_required=raw.get("non_inverting_mapping_required", True),
        mixer1=r_mixer(raw["mixer1"]),
        mixer2=r_mixer(raw["mixer2"]),
        lo1=r_lo(raw["lo1"]),
        lo2=r_lo(raw["lo2"]),
        filters=r_filters(raw["filters"]),
        spur_limits=r_spur_limits(raw["spur_limits"]),
        grids=r_grids(raw["grids"]),
        if1_harmonics_dbc={int(k): float(v) for k, v in raw.get("if1_harmonics_dbc", {}).items()},
        description=raw.get("description"),
    )