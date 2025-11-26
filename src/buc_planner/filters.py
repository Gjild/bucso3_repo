# src/buc_planner/filters.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from .config_models import Freq, dB, IF2BankConstraints, Range


@dataclass
class IF2Filter:
    """
    Planning-grade IF2 BPF model:
        A(f) = 0 dB inside [fc - BW/2, fc + BW/2]
             = S * log10(|f - fc| / (BW/2))  outside, where S < 0 (dB/decade)
    """
    filter_id: str
    fc: Freq
    bw: Freq
    slope_db_per_decade: dB

    def attenuation_db(self, freq: np.ndarray) -> np.ndarray:
        """Return attenuation A(f) in dB for vector freq."""
        f = np.asarray(freq, dtype=float)

        # Guard against zero / negative BW: treat as extremely narrow band.
        # This avoids division by zero while still keeping behaviour well-defined.
        half_bw = max(self.bw / 2.0, 1e-12)

        # in-band: 0 dB
        in_band = (f >= self.fc - half_bw) & (f <= self.fc + half_bw)
        out_band = ~in_band
        att = np.zeros_like(f, dtype=float)

        if np.any(out_band):
            # normalize offset to band edge
            offset = np.abs(f[out_band] - self.fc) / half_bw
            offset[offset <= 0] = 1e-12
            att[out_band] = self.slope_db_per_decade * np.log10(offset)

        return att


@dataclass
class RFFilter:
    """
    RF BPF defined by CSV: frequency vs attenuation (dB). Attenuation outside CSV
    range is 'hold last value'.
    """
    freqs: np.ndarray
    att_db: np.ndarray

    @classmethod
    def from_csv(cls, path: str | Path) -> "RFFilter":
        path = Path(path)
        data = np.genfromtxt(path, delimiter=",", comments="#", skip_header=0)

        # Handle single-row and sanity-check shape
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.size == 0 or data.shape[1] < 2:
            raise ValueError(
                f"RF BPF CSV '{path}' must have at least two columns (freq, attenuation_db)."
            )

        # If the first row contains NaNs and there are more rows, treat it as a header.
        if data.shape[0] > 1 and (np.isnan(data[0, 0]) or np.isnan(data[0, 1])):
            data = data[1:, :]
            if data.size == 0 or data.shape[1] < 2:
                raise ValueError(
                    f"RF BPF CSV '{path}' has only a header and no data rows."
                )

        freqs = data[:, 0].astype(float)
        att_db = data[:, 1].astype(float)

        # Ensure increasing frequency
        idx = np.argsort(freqs)
        return cls(freqs=freqs[idx], att_db=att_db[idx])
    
    def attenuation_db(self, freq: np.ndarray) -> np.ndarray:
        """
        Interpolated attenuation A(f) in dB for vector freq.

        - Linear interpolation within [min(freqs), max(freqs)]
        - Hold nearest endpoint value outside that range
        """
        f = np.asarray(freq, dtype=float)
        if self.freqs.size == 0:
            # Extremely defensive; should not happen with a valid CSV
            return np.zeros_like(f, dtype=float)

        fmin = self.freqs[0]
        fmax = self.freqs[-1]
        f_clamped = np.clip(f, fmin, fmax)
        return np.interp(f_clamped, self.freqs, self.att_db)



@dataclass
class IF2Bank:
    filters: list[IF2Filter]
    # Optional mapping from RF configuration ID -> filter_id
    config_to_filter_id: dict[str, str] | None = None

    def select_filter_for_config(self, config_id: str) -> IF2Filter:
        """
        Return the IF2Filter assigned to this RF configuration.

        If a config→filter mapping is available (e.g. produced by the optimizer),
        it is used. Otherwise a deterministic round-robin mapping is used as a
        fallback so that the bank remains usable in simple/manual scenarios.
        """
        if not self.filters:
            raise RuntimeError(
                "IF2Bank has no filters configured; cannot select a filter for "
                f"config_id='{config_id}'."
            )

        if self.config_to_filter_id and config_id in self.config_to_filter_id:
            fid = self.config_to_filter_id[config_id]
            for f in self.filters:
                if f.filter_id == fid:
                    return f
            # If the mapping points at a missing filter, fall back below.

        # Fallback: deterministic round-robin based on a stable hash of the ID.
        # We avoid Python's built-in hash(), which is randomized per process.
        idx = (sum(ord(c) for c in config_id) % len(self.filters))
        return self.filters[idx]


def design_single_if2_filter_for_band(
    band: Range,
    constraints: IF2BankConstraints,
    default_slope_db_per_decade: dB = -60.0,
) -> IF2Filter:
    """
    Create an IF2Filter that minimally covers the given IF2 band given constraints.
    This is a building block used by the IF2 bank designer.

    Very simple heuristic:
      * fc = midpoint of band, clamped to fc_range
      * bw = band.width * (1 + small_margin), clamped to bw_range
    """
    band_center = 0.5 * (band.start + band.stop)
    band_bw = band.width

    # apply margin: we allow constraints.feasibility_margin_hz as extra
    bw_margin = constraints.feasibility_margin_hz
    target_bw = band_bw + 2 * bw_margin

    # clamp to ranges
    fc = np.clip(band_center, constraints.fc_range.start, constraints.fc_range.stop)
    bw_min, bw_max = constraints.bw_range.start, constraints.bw_range.stop
    bw = float(np.clip(target_bw, bw_min, bw_max))

    if bw <= 0.0:
        raise ValueError(
            "Computed IF2 filter bandwidth is non-positive. "
            "Check IF2 bw_range and feasibility_margin_hz constraints."
        )

    s_min, s_max = constraints.slope_range
    slope = float(np.clip(default_slope_db_per_decade, s_min, s_max))

    return IF2Filter(
        filter_id=f"if2_auto_{band_center:.0f}",
        fc=fc,
        bw=bw,
        slope_db_per_decade=slope,
    )