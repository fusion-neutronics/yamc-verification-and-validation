"""Live, in-place terminal dashboard for the verification orchestrator.

The orchestrator maintains a *raw* state (counts, timestamps, observed run
times) and calls :func:`render_lines` / :class:`LiveDisplay` with the current
monotonic clock; all derived values (elapsed, ETA) are computed here so the
state stays simple. The layout mirrors issue #29::

    running with YAMC 0.4.0, built 3 mins ago
    120/300 cached, 80/180 completed ETA 30 mins
    neutron sphere - completed 15 mins
        yamc-cpu - 50/50 completed 10 mins
        yamc-gpu - 20/50 cached 30/50 completed 5 mins
        openmc - 50/50 cached
    photon sphere
        yamc-cpu - 30/50 running Fe ETA 9 mins  <--- in progress
        yamc-gpu - 50/50 todo
        openmc - 50/50 cached

Raw state schema::

    {
      "yamc_version": str, "yamc_built_ago": float | None,
      "tasks": [{
        "label": str, "status": "queued"|"running"|"done",
        "t_start": float | None, "t_end": float | None,
        "codes": {code: {
          "total": int, "cached": int, "computed": int,
          "running_key": str | None, "status": "queued"|"running"|"done",
          "t_start": float | None, "t_end": float | None,
          "obs_times": [float], "seed_avg": float | None,
        }},
      }],
    }
"""
from __future__ import annotations

import sys


def humanize(seconds) -> str:
    if seconds is None:
        return "?"
    s = float(seconds)
    if s < 1:
        return "<1 sec"
    if s < 90:
        n = round(s)
        return f"{n} sec" + ("" if n == 1 else "s")
    mins = s / 60.0
    if mins < 90:
        n = round(mins)
        return f"{n} min" + ("" if n == 1 else "s")
    return f"{mins / 60.0:.1f} hrs"


def _avg(code, global_avg):
    obs = code.get("obs_times") or []
    if obs:
        return sum(obs) / len(obs)
    if code.get("seed_avg"):
        return code["seed_avg"]
    return global_avg


def _global_avg(state):
    times = []
    for t in state["tasks"]:
        for c in t["codes"].values():
            times.extend(c.get("obs_times") or [])
            if not (c.get("obs_times")) and c.get("seed_avg"):
                times.append(c["seed_avg"])
    return (sum(times) / len(times)) if times else None


def _code_eta(code, global_avg):
    remaining = code["total"] - code["cached"] - code["computed"]
    if remaining <= 0:
        return 0.0
    avg = _avg(code, global_avg)
    return remaining * avg if avg else None


def _code_line(code, now, global_avg):
    total, cached, computed = code["total"], code["cached"], code["computed"]
    done = cached + computed
    status = code["status"]

    if status == "queued":
        if total and cached >= total:
            return f"{total}/{total} cached"
        if cached:
            return f"{cached}/{total} cached {total - cached}/{total} todo"
        return f"{total}/{total} todo"

    if status == "running":
        if cached and computed:
            body = f"{cached}/{total} cached {computed}/{total} completed"
        else:
            body = f"{done}/{total}"
        eta = _code_eta(code, global_avg)
        # Name the item currently simulating when one is known (set between an
        # item_start and its completion; cached items are instant and set none).
        running_key = code.get("running_key")
        parts = [body]
        if running_key:
            parts.append(f"running {running_key}")
        if eta is not None:
            parts.append(f"ETA {humanize(eta)}")
        elif not running_key:
            parts.append("running")
        return " ".join(parts) + "  <--- in progress"

    # done
    elapsed = None
    if code.get("t_start") is not None and code.get("t_end") is not None:
        elapsed = code["t_end"] - code["t_start"]
    if cached and computed:
        return f"{cached}/{total} cached {computed}/{total} completed {humanize(elapsed)}"
    if cached and not computed:
        return f"{total}/{total} cached"
    if computed and not cached:
        return f"{computed}/{total} completed {humanize(elapsed)}"
    return f"0/{total} (none ran)"


def _task_header(t, now):
    label = t["label"]
    if t["status"] == "queued":
        return label
    if t["status"] == "running":
        el = (now - t["t_start"]) if t.get("t_start") is not None else None
        return f"{label}  <running {humanize(el)}>"
    el = None
    if t.get("t_start") is not None and t.get("t_end") is not None:
        el = t["t_end"] - t["t_start"]
    return f"{label} - completed {humanize(el)}"


def render_lines(state, now: float) -> list[str]:
    """Render the dashboard to a list of plain-text lines."""
    lines = []
    v = state.get("yamc_version", "?")
    ago = state.get("yamc_built_ago")
    lines.append(f"running with YAMC {v}, built {humanize(ago)} ago" if ago is not None
                 else f"running with YAMC {v}")

    global_avg = _global_avg(state)
    tot = cached = computed = 0
    eta_total = 0.0
    for t in state["tasks"]:
        for c in t["codes"].values():
            tot += c["total"]
            cached += c["cached"]
            computed += c["computed"]
            if c["status"] != "done":
                e = _code_eta(c, global_avg)
                if e:
                    eta_total += e
    to_compute = max(0, tot - cached)
    eta_str = f" ETA {humanize(eta_total)}" if eta_total > 0 else ""
    lines.append(f"{cached}/{tot} cached, {computed}/{to_compute} completed{eta_str}")
    lines.append("")

    for t in state["tasks"]:
        lines.append(_task_header(t, now))
        for code, c in t["codes"].items():
            lines.append(f"    {code} - {_code_line(c, now, global_avg)}")
    return lines


class LiveDisplay:
    """Render the dashboard in place (ANSI) on a TTY, or as snapshots otherwise."""

    def __init__(self, stream=None, tty=None):
        self.stream = stream or sys.stdout
        self._tty = self.stream.isatty() if tty is None else tty
        self._prev = 0

    def update(self, state, now: float) -> None:
        lines = render_lines(state, now)
        out = self.stream
        if self._tty:
            if self._prev:
                out.write(f"\033[{self._prev}A")
            for ln in lines:
                out.write("\033[2K" + ln + "\n")
            self._prev = len(lines)
        else:
            out.write("\n".join(lines) + "\n" + ("-" * 48) + "\n")
        out.flush()
