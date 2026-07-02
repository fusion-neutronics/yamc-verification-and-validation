"""Parse the ``MC_BACKENDS`` environment variable into a run configuration.

This logic was previously copy-pasted into every notebook's config cell. The
tokens are independent: any subset of ``openmc-cpu``, ``yamc-cpu``, ``yamc-gpu``
is valid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

VALID_BACKENDS = ("openmc-cpu", "yamc-cpu", "yamc-gpu")
DEFAULT_BACKENDS = "openmc-cpu,yamc-cpu,yamc-gpu"


@dataclass(frozen=True)
class Backends:
    """Resolved backend selection.

    Attributes
    ----------
    tokens : list[str]
        The validated ``MC_BACKENDS`` tokens in request order.
    run_openmc : bool
        Whether ``openmc-cpu`` was requested.
    computes : list[str]
        The yamc compute suffixes requested, e.g. ``["cpu", "gpu"]``.
    """

    tokens: list[str]
    run_openmc: bool
    computes: list[str]

    @property
    def codes(self) -> list[str]:
        """Cache/label codes: ``openmc`` first (if present) then ``yamc-<compute>``."""
        out: list[str] = []
        if self.run_openmc:
            out.append("openmc")
        out.extend(f"yamc-{c}" for c in self.computes)
        return out


def resolve(raw: str | None = None) -> Backends:
    """Resolve backends from ``raw`` (defaults to the ``MC_BACKENDS`` env var).

    Raises ``ValueError`` on unknown tokens or an empty selection.
    """
    if raw is None:
        raw = os.environ.get("MC_BACKENDS", DEFAULT_BACKENDS)
    tokens = [b.strip().lower() for b in raw.split(",") if b.strip()]
    bad = [b for b in tokens if b not in VALID_BACKENDS]
    if bad:
        raise ValueError(
            f"MC_BACKENDS contains unknown token(s) {bad}; "
            f"valid tokens are {list(VALID_BACKENDS)}"
        )
    if not tokens:
        raise ValueError("MC_BACKENDS resolved to an empty list")
    run_openmc = "openmc-cpu" in tokens
    computes = [b.split("-", 1)[1] for b in tokens if b.startswith("yamc-")]
    return Backends(tokens=tokens, run_openmc=run_openmc, computes=computes)
