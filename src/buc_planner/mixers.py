# src/buc_planner/mixers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

import logging

logger = logging.getLogger(__name__)

from .config_models import (
    MixerConfig,
    SpurTableEntry,
    Freq,
    dBc,
    Range,
)


@dataclass
class LOTone:
    """
    Represents a specific LO tone at the mixer LO port: fundamental, harmonic, or PLL spur.
    """
    name: str
    freq: Freq
    level_dbc: dBc  # relative to LO fundamental


@dataclass
class MixerInputBand:
    """
    Wideband input used to generate spurs: e.g., IF1 fundamental band, IF1 harmonic,
    or IF2 wideband at Mixer2 input.
    """
    name: str
    f_start: Freq
    f_stop: Freq
    level_dbc_integrated: dBc  # integrated power vs IF1 fundamental (0 dBc)

    @property
    def center(self) -> Freq:
        return 0.5 * (self.f_start + self.f_stop)

    @property
    def bandwidth(self) -> Freq:
        return self.f_stop - self.f_start


@dataclass
class SpurFamilySpec:
    """Resolved spur-family spec combining a table entry and specific LO tone."""
    entry: SpurTableEntry
    lo_tone: LOTone
    effective_level_dbc: dBc  # L_spur_table + L_lo_tone if applicable
    mixer_name: str
    used_unspecified_floor: bool = False


@dataclass
class MixerWidebandSpurBand:
    """
    Represents wideband spur after a single mixing stage, before filters.
    """
    name: str
    mixer_name: str
    m: int
    n: int
    lo_tone_name: str
    input_band_name: str
    center_freq: Freq
    f_start: Freq
    f_stop: Freq
    spur_level_rel_if1_dbc: dBc  # relative to IF1 fundamental integrated power
    used_unspecified_floor: bool = False
    
    @property
    def bandwidth(self) -> Freq:
        """Convenience helper: width of the spur band."""
        return self.f_stop - self.f_start


def resolve_spur_families_for_tones(
    mixer: MixerConfig,
    lo_tones: List[LOTone],
) -> List[SpurFamilySpec]:
    """
    Combine spur table with a set of LO tones, enforcing the (m,n) envelope and
    SpurTableEntry validity ranges.

    * LO validity is enforced via entry.lo_range.
    * IF and RF validity are enforced later in generate_wideband_spur_band.
    """
    specs: List[SpurFamilySpec] = []

    # 1) Build effective spur table, injecting floor entries if requested
    table_entries: List[SpurTableEntry] = list(mixer.spur_table)
    floor_pairs: Set[Tuple[int, int]] = set()

    env = mixer.spur_envelope
    desired_pair = (mixer.desired_m, mixer.desired_n)
    existing_pairs = {(e.m, e.n) for e in table_entries}
    missing: List[Tuple[int, int]] = []
    for m in range(-env.m_max, env.m_max + 1):
        for n in range(-env.n_max, env.n_max + 1):
            if n == 0:
                continue  # n != 0 per spec
            # Do not require the desired path to be present as a spur family
            if (m, n) == desired_pair:
                continue
            if (m, n) in existing_pairs:
                continue
            missing.append((m, n))

    if missing:
        if env.enforce_envelope_completeness and env.unspecified_floor_dbc is None:
            logger.error(
                "Mixer '%s': spur envelope |m|<=%d, |n|<=%d has %d unspecified (m,n) "
                "pairs and no unspecified_floor_dbc configured.",
                mixer.name,
                env.m_max,
                env.n_max,
                len(missing),
            )
            raise ValueError(
                f"Mixer '{mixer.name}': spur envelope |m|<={env.m_max}, |n|<={env.n_max} "
                f"requires entries for all families; {len(missing)} (m,n) pairs are "
                f"unspecified and no unspecified_floor_dbc is configured."
            )
        if env.unspecified_floor_dbc is not None:
            logger.warning(
                "Mixer '%s': applying unspecified_floor_dbc=%.1f dBc to %d missing spur "
                "families within envelope |m|<=%d, |n|<=%d.",
                mixer.name,
                env.unspecified_floor_dbc,
                len(missing),
                env.m_max,
                env.n_max,
            )
            for m_val, n_val in missing:
                table_entries.append(
                    SpurTableEntry(
                        m=m_val,
                        n=n_val,
                        level_dbc=env.unspecified_floor_dbc,
                        if_range=mixer.ranges.if_range,
                        lo_range=mixer.ranges.lo_range,
                        rf_range=mixer.ranges.rf_range,
                        lo_tone_type="fundamental",
                    )
                )
                floor_pairs.add((m_val, n_val))
        elif not env.enforce_envelope_completeness:
            # Spec option (1): warn & ignore unspecified spur families
            logger.warning(
                "Mixer '%s': spur envelope |m|<=%d, |n|<=%d has %d unspecified spur "
                "families which will be ignored.",
                mixer.name,
                env.m_max,
                env.n_max,
                len(missing),
            )

    # 2) Resolve against LO tones (enforcing LO range)
    for entry in table_entries:
        if (entry.m, entry.n) == desired_pair:
            continue
        lt = entry.lo_tone_type or "fundamental"
        used_floor = (entry.m, entry.n) in floor_pairs

        def add_spec_for_tone(tone: LOTone, effective_level: dBc) -> None:
            # Enforce LO validity range
            if not entry.lo_range.contains(tone.freq):
                return
            specs.append(
                SpurFamilySpec(
                    entry=entry,
                    lo_tone=tone,
                    effective_level_dbc=effective_level,
                    mixer_name=mixer.name,
                    used_unspecified_floor=used_floor,
                )
            )

        if lt == "fundamental":
            for tone in lo_tones:
                eff_level = entry.level_dbc + tone.level_dbc
                add_spec_for_tone(tone, eff_level)
        elif lt.startswith("harmonic_"):
            for tone in lo_tones:
                if tone.name == lt:
                    add_spec_for_tone(tone, entry.level_dbc)
        elif lt.startswith("pll_spur_"):
            tag = lt[len("pll_spur_") :]
            for tone in lo_tones:
                if not tone.name.startswith("pll_spur_"):
                    continue
                if not tag:
                    add_spec_for_tone(tone, entry.level_dbc)
                else:
                    if tone.name.endswith(tag):
                        add_spec_for_tone(tone, entry.level_dbc)
        else:
            # Unknown lo_tone_type: try exact name match; otherwise treat as "fundamental"
            matched = False
            for tone in lo_tones:
                if tone.name == lt:
                    add_spec_for_tone(tone, entry.level_dbc)
                    matched = True
            if not matched:
                for tone in lo_tones:
                    eff_level = entry.level_dbc + tone.level_dbc
                    add_spec_for_tone(tone, eff_level)

    return specs


def generate_wideband_spur_band(
    input_band: MixerInputBand,
    spur_spec: SpurFamilySpec,
    min_level_considered_dbc: dBc,
) -> Optional[MixerWidebandSpurBand]:
    """
    Generate a single wideband spur band for a given input band and spur family spec.

    Enforces SpurTableEntry.if_range and rf_range by clipping the input and output
    bands; if there is no overlap, the spur is discarded.

    Bandwidth: |n| * BW_input_eff
    Center frequency: approximately m * f_LO ± n * f_input_center (on the clipped IF band).
    """
    m = spur_spec.entry.m
    n = spur_spec.entry.n
    if n == 0:
        return None

    if spur_spec.effective_level_dbc < min_level_considered_dbc:
        return None

    # Clip input band by IF validity range
    in_range = Range(start=input_band.f_start, stop=input_band.f_stop)
    clipped_if = in_range.intersect(spur_spec.entry.if_range)
    if clipped_if is None or clipped_if.width <= 0:
        return None

    f_lo = spur_spec.lo_tone.freq
    f_in_center = 0.5 * (clipped_if.start + clipped_if.stop)
    bw_in = clipped_if.width
    bw_spur = abs(n) * bw_in

    # Approximate center; sign is implicit in (m, n)
    f_center = m * f_lo + n * f_in_center
    f_start = f_center - 0.5 * bw_spur
    f_stop = f_center + 0.5 * bw_spur

    # Clip by RF validity range
    spur_range = Range(start=f_start, stop=f_stop)
    clipped_rf = spur_range.intersect(spur_spec.entry.rf_range)
    if clipped_rf is None or clipped_rf.width <= 0:
        return None

    f_start = clipped_rf.start
    f_stop = clipped_rf.stop
    f_center = 0.5 * (f_start + f_stop)

    # Spur power relative to IF1 fundamental
    spur_level_rel_if1 = input_band.level_dbc_integrated + spur_spec.effective_level_dbc

    name = f"{spur_spec.entry.m},{spur_spec.entry.n}::{input_band.name}@{spur_spec.lo_tone.name}"
    return MixerWidebandSpurBand(
        name=name,
        mixer_name=spur_spec.mixer_name,
        m=m,
        n=n,
        lo_tone_name=spur_spec.lo_tone.name,
        input_band_name=input_band.name,
        center_freq=f_center,
        f_start=f_start,
        f_stop=f_stop,
        spur_level_rel_if1_dbc=spur_level_rel_if1,
        used_unspecified_floor=spur_spec.used_unspecified_floor,
    )