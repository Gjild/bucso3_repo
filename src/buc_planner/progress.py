# src/buc_planner/progress.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TextIO
import sys
import shutil


@dataclass
class _BarState:
    desc: str
    total: Optional[int]
    value: int = 0


class ProgressReporter:
    """
    Simple, dependency-free textual progress bar for long-running phases.

    - Writes to stderr so it does not pollute data files.
    - Only supports one active bar at a time (sequential phases).
    """

    def __init__(self, stream: Optional[TextIO] = None, enabled: bool = True) -> None:
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self._state: Optional[_BarState] = None

    # Public API --------------------------------------------------------

    def start(self, description: str, total: Optional[int] = None) -> None:
        """Start (or restart) a progress bar for a new phase."""
        if not self.enabled:
            return

        # Close previous bar line, if any
        if self._state is not None:
            self._finish_line()

        self._state = _BarState(desc=description, total=total, value=0)
        self._render()

    def advance(self, n: int = 1) -> None:
        """Advance the current bar by n units."""
        if not self.enabled or self._state is None:
            return
        self._state.value += n
        self._render()

    def end(self) -> None:
        """Finish the current bar (newline) and clear state."""
        if not self.enabled or self._state is None:
            return
        self._finish_line()
        self._state = None

    def message(self, text: str) -> None:
        """
        Print a one-off status message, without breaking the bar.
        """
        if not self.enabled:
            return
        if self._state is not None:
            self._clear_line()
        self.stream.write(text + "\n")
        self.stream.flush()
        if self._state is not None:
            self._render()

    # Internal helpers --------------------------------------------------

    def _render(self) -> None:
        if self._state is None or not self.enabled:
            return

        desc = self._state.desc
        total = self._state.total
        value = self._state.value

        try:
            width = shutil.get_terminal_size(fallback=(80, 20)).columns
        except Exception:
            width = 80

        if total is None or total <= 0:
            text = f"{desc}: {value}"
        else:
            frac = max(0.0, min(1.0, value / float(total)))
            bar_width = max(10, min(40, width - len(desc) - 20))
            filled = int(bar_width * frac)
            bar = "#" * filled + "-" * (bar_width - filled)
            percent = int(frac * 100)
            text = f"{desc} [{bar}] {percent:3d}% ({value}/{total})"

        self._clear_line()
        self.stream.write(text[: width - 1])
        self.stream.flush()

    def _clear_line(self) -> None:
        try:
            width = shutil.get_terminal_size(fallback=(80, 20)).columns
        except Exception:
            width = 80
        self.stream.write("\r" + " " * (width - 1) + "\r")

    def _finish_line(self) -> None:
        # Clear and print a final summary line for the phase
        self._clear_line()
        if self._state is None:
            return
        desc = self._state.desc
        total = self._state.total
        value = self._state.value
        if total is None:
            line = f"{desc}: {value}\n"
        else:
            line = f"{desc}: {value}/{total}\n"
        self.stream.write(line)
        self.stream.flush()


class NullProgressReporter(ProgressReporter):
    """
    Drop-in replacement that does nothing.
    Useful when calling Planner programmatically with no progress.
    """

    def __init__(self) -> None:
        super().__init__(enabled=False)