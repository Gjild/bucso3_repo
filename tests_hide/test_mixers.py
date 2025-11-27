# tests/test_mixers.py
from __future__ import annotations

import pytest

from buc_planner.config_models import (
    MixerConfig,
    SpurTableEntry,
    MixerSpurEnvelopePolicy,
    MixerIsolation,
    MixerRanges,
    Range,
)
from buc_planner.mixers import (
    LOTone,
    MixerInputBand,
    SpurFamilySpec,
    resolve_spur_families_for_tones,
    generate_wideband_spur_band,
)


def _make_basic_mixer(name: str) -> MixerConfig:
    return MixerConfig(
        name=name,
        spur_table=[],
        spur_envelope=MixerSpurEnvelopePolicy(
            m_max=1,
            n_max=1,
            enforce_envelope_completeness=False,
            unspecified_floor_dbc=-90.0,
        ),
        isolation=MixerIsolation(lo_to_rf_dbc=-80.0, if_to_rf_dbc=-80.0),
        ranges=MixerRanges(
            if_range=Range(0.0, 10.0),
            lo_range=Range(0.0, 10.0),
            rf_range=Range(0.0, 10.0),
        ),
        desired_m=1,
        desired_n=1,
    )


def test_resolve_spur_families_injects_floor():
    """
    When unspecified_floor_dbc is set and enforce_envelope_completeness is False,
    resolve_spur_families_for_tones should synthesize entries for all (m, n)
    within the envelope (except n == 0 and the desired family) and mark them as
    coming from the unspecified floor.
    """
    mixer = _make_basic_mixer("M")

    tones = [LOTone(name="fundamental", freq=1.0, level_dbc=0.0)]
    specs = resolve_spur_families_for_tones(mixer, tones)

    # Envelope |m|<=1, |n|<=1 (excluding n==0 and desired (1,1)) gives:
    # (-1,-1), (-1,1), (0,-1), (0,1), (1,-1) -> 5 families, 1 tone => 5 specs.
    assert len(specs) == 5

    for s in specs:
        assert s.used_unspecified_floor
        # Level is table level (-90) + LO tone level (0)
        assert s.effective_level_dbc == pytest.approx(
            mixer.spur_envelope.unspecified_floor_dbc
        )
        # Desired family must not be treated as spur
        assert not (s.entry.m == mixer.desired_m and s.entry.n == mixer.desired_n)


def test_resolve_spur_families_lo_tone_type_harmonic():
    """Specific lo_tone_type should bind only to the matching LO tone and not be scaled."""
    mixer = MixerConfig(
        name="M",
        spur_table=[
            SpurTableEntry(
                m=2,
                n=1,
                level_dbc=-50.0,
                if_range=Range(0.0, 10.0),
                lo_range=Range(0.0, 10.0),
                rf_range=Range(0.0, 10.0),
                lo_tone_type="harmonic_2",
            )
        ],
        spur_envelope=MixerSpurEnvelopePolicy(
            m_max=2,
            n_max=2,
            enforce_envelope_completeness=False,
            unspecified_floor_dbc=None,
        ),
        isolation=MixerIsolation(lo_to_rf_dbc=-80.0, if_to_rf_dbc=-80.0),
        ranges=MixerRanges(
            if_range=Range(0.0, 10.0),
            lo_range=Range(0.0, 10.0),
            rf_range=Range(0.0, 10.0),
        ),
        desired_m=1,
        desired_n=1,
    )

    tones = [
        LOTone(name="fundamental", freq=1.0, level_dbc=0.0),
        LOTone(name="harmonic_2", freq=2.0, level_dbc=-20.0),
    ]

    specs = resolve_spur_families_for_tones(mixer, tones)
    assert len(specs) == 1
    spec = specs[0]
    # For specific tone type, level_dbc is not scaled by tone.level_dbc
    assert spec.effective_level_dbc == pytest.approx(-50.0)
    assert spec.lo_tone.name == "harmonic_2"


def test_resolve_spur_families_fundamental_scaling_and_lo_range():
    """
    For lo_tone_type 'fundamental' (or default), all LO tones within the entry's
    LO range should be used, and the spur level should be scaled by each tone's
    own level_dbc.
    """
    mixer = MixerConfig(
        name="M",
        spur_table=[
            SpurTableEntry(
                m=1,
                n=1,
                level_dbc=-40.0,
                if_range=Range(0.0, 10.0),
                lo_range=Range(4.0, 6.0),  # only tones in [4, 6] are valid
                rf_range=Range(0.0, 20.0),
                lo_tone_type="fundamental",
            )
        ],
        spur_envelope=MixerSpurEnvelopePolicy(
            m_max=1,
            n_max=1,
            enforce_envelope_completeness=False,
            unspecified_floor_dbc=None,
        ),
        isolation=MixerIsolation(lo_to_rf_dbc=-80.0, if_to_rf_dbc=-80.0),
        ranges=MixerRanges(
            if_range=Range(0.0, 20.0),
            lo_range=Range(0.0, 20.0),
            rf_range=Range(0.0, 20.0),
        ),
        desired_m=99,  # make sure this entry is not treated as the desired path
        desired_n=99,
    )

    tones = [
        LOTone(name="t0", freq=3.5, level_dbc=-10.0),  # outside LO range -> ignored
        LOTone(name="t1", freq=5.0, level_dbc=0.0),
        LOTone(name="t2", freq=5.5, level_dbc=-20.0),
    ]

    specs = resolve_spur_families_for_tones(mixer, tones)
    # Only t1 and t2 are within lo_range [4, 6]
    assert len(specs) == 2

    # Collect by tone name for easier assertions
    by_tone = {s.lo_tone.name: s for s in specs}
    assert set(by_tone.keys()) == {"t1", "t2"}

    # effective_level = entry.level_dbc + tone.level_dbc
    assert by_tone["t1"].effective_level_dbc == pytest.approx(-40.0 + 0.0)
    assert by_tone["t2"].effective_level_dbc == pytest.approx(-40.0 - 20.0)


def test_resolve_spur_families_enforce_envelope_completeness_raises():
    """
    If enforce_envelope_completeness is True and no unspecified_floor_dbc is
    configured, missing (m, n) families inside the envelope must cause a
    ValueError.
    """
    mixer = MixerConfig(
        name="M",
        spur_table=[],  # no explicit spur entries
        spur_envelope=MixerSpurEnvelopePolicy(
            m_max=1,
            n_max=1,
            enforce_envelope_completeness=True,
            unspecified_floor_dbc=None,
        ),
        isolation=MixerIsolation(lo_to_rf_dbc=-80.0, if_to_rf_dbc=-80.0),
        ranges=MixerRanges(
            if_range=Range(0.0, 10.0),
            lo_range=Range(0.0, 10.0),
            rf_range=Range(0.0, 10.0),
        ),
        desired_m=1,
        desired_n=1,
    )

    tones: list[LOTone] = [LOTone(name="fundamental", freq=1.0, level_dbc=0.0)]

    with pytest.raises(ValueError):
        resolve_spur_families_for_tones(mixer, tones)


def test_generate_wideband_spur_band_geometry():
    entry = SpurTableEntry(
        m=1,
        n=2,
        level_dbc=-40.0,
        if_range=Range(0.0, 10.0),
        lo_range=Range(0.0, 10.0),
        rf_range=Range(0.0, 30.0),
        lo_tone_type="fundamental",
    )
    tone = LOTone(name="fundamental", freq=5.0, level_dbc=0.0)
    spec = SpurFamilySpec(
        entry=entry,
        lo_tone=tone,
        effective_level_dbc=-40.0,
        mixer_name="M",
    )

    inp = MixerInputBand(
        name="IF1_seg",
        f_start=1.0,
        f_stop=3.0,
        level_dbc_integrated=-3.0,
    )

    spur = generate_wideband_spur_band(
        input_band=inp,
        spur_spec=spec,
        min_level_considered_dbc=-120.0,
    )
    assert spur is not None

    # Input center = 2.0, BW=2.0; spur BW=|n|*BW=4.0
    assert spur.bandwidth == pytest.approx(4.0)
    expected_center = entry.m * tone.freq + entry.n * 2.0
    assert spur.center_freq == pytest.approx(expected_center)

    # Level relative to IF1 is input level + spur effective
    assert spur.spur_level_rel_if1_dbc == pytest.approx(-3.0 + (-40.0))


def test_generate_wideband_spur_band_min_level_filtering():
    """
    generate_wideband_spur_band should early-exit if the spur_spec.effective_level_dbc
    is below min_spur_level_considered_dbc, regardless of the input band level.
    """
    entry = SpurTableEntry(
        m=1,
        n=1,
        level_dbc=-130.0,
        if_range=Range(0.0, 10.0),
        lo_range=Range(0.0, 10.0),
        rf_range=Range(0.0, 20.0),
        lo_tone_type="fundamental",
    )
    tone = LOTone(name="fundamental", freq=5.0, level_dbc=0.0)
    spec = SpurFamilySpec(
        entry=entry,
        lo_tone=tone,
        effective_level_dbc=-130.0,
        mixer_name="M",
    )

    inp = MixerInputBand(
        name="IF1_seg",
        f_start=2.0,
        f_stop=4.0,
        level_dbc_integrated=0.0,
    )

    spur = generate_wideband_spur_band(
        input_band=inp,
        spur_spec=spec,
        min_level_considered_dbc=-120.0,
    )
    # Effective level is below threshold -> spur must be discarded
    assert spur is None


def test_generate_wideband_spur_band_clipping_if_and_rf_ranges():
    """
    The spur band should be clipped by both IF validity range and RF validity
    range, with the center frequency recomputed on the clipped band.
    """
    entry = SpurTableEntry(
        m=1,
        n=1,
        level_dbc=-20.0,
        if_range=Range(0.0, 10.0),   # clips input band
        lo_range=Range(0.0, 30.0),
        rf_range=Range(22.0, 28.0),  # clips output band
        lo_tone_type="fundamental",
    )
    # LO tone at 20
    tone = LOTone(name="fundamental", freq=20.0, level_dbc=0.0)
    spec = SpurFamilySpec(
        entry=entry,
        lo_tone=tone,
        effective_level_dbc=-20.0,
        mixer_name="M",
    )

    # Input band extends below IF range: [-5, 15] -> clipped to [0, 10]
    inp = MixerInputBand(
        name="IF1_seg",
        f_start=-5.0,
        f_stop=15.0,
        level_dbc_integrated=-3.0,
    )

    spur = generate_wideband_spur_band(
        input_band=inp,
        spur_spec=spec,
        min_level_considered_dbc=-120.0,
    )
    assert spur is not None

    # After IF clipping: [0, 10] -> center 5, BW 10
    # Spur band (before RF clipping): center = 20 + 5 = 25, BW = 10 -> [20, 30]
    # RF range clips to [22, 28]
    assert spur.f_start == pytest.approx(22.0)
    assert spur.f_stop == pytest.approx(28.0)
    assert spur.bandwidth == pytest.approx(6.0)
    assert spur.center_freq == pytest.approx(25.0)

    # Power bookkeeping is still input + effective spur level
    assert spur.spur_level_rel_if1_dbc == pytest.approx(-3.0 + (-20.0))


def test_generate_wideband_spur_band_rejects_when_no_if_overlap():
    """
    If the input band does not overlap the SpurTableEntry IF range, the spur
    must be discarded.
    """
    entry = SpurTableEntry(
        m=1,
        n=1,
        level_dbc=-20.0,
        if_range=Range(100.0, 200.0),  # disjoint from input band
        lo_range=Range(0.0, 10.0),
        rf_range=Range(0.0, 50.0),
        lo_tone_type="fundamental",
    )
    tone = LOTone(name="fundamental", freq=5.0, level_dbc=0.0)
    spec = SpurFamilySpec(
        entry=entry,
        lo_tone=tone,
        effective_level_dbc=-20.0,
        mixer_name="M",
    )

    inp = MixerInputBand(
        name="IF1_seg",
        f_start=0.0,
        f_stop=10.0,
        level_dbc_integrated=0.0,
    )

    spur = generate_wideband_spur_band(
        input_band=inp,
        spur_spec=spec,
        min_level_considered_dbc=-120.0,
    )
    assert spur is None