# src/buc_planner/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt

from .filters import IF2Bank
from .config_models import Range
from .los import LOPlanCandidate


def plot_if2_bank(
    bank: IF2Bank,
    freq_range: Range,
    out_path: Optional[str | Path] = None,
    lo_plans: Optional[List[LOPlanCandidate]] = None,
) -> None:
    """
    Plot overlaid IF2 filter responses over a given frequency range.

    If lo_plans is provided, desired IF2 bands for each configuration are
    overlaid as translucent spans.
    """
    freqs = np.linspace(freq_range.start, freq_range.stop, 2000)
    plt.figure()
    for f in bank.filters:
        att = f.attenuation_db(freqs)
        plt.plot(freqs, att, label=f.filter_id)

    # Overlay desired IF2 bands
    if lo_plans:
        for i, plan in enumerate(lo_plans):
            band = plan.if2_band
            alpha = 0.08
            plt.axvspan(
                band.start,
                band.stop,
                alpha=alpha,
                linestyle="--",
                label="_if2_band" if i > 0 else "desired IF2 bands",
            )

    plt.xlabel("Frequency")
    plt.ylabel("Attenuation (dB)")
    plt.title("IF2 Filter Bank Responses")
    plt.legend()
    plt.grid(True)
    if out_path:
        out_path = Path(out_path)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()