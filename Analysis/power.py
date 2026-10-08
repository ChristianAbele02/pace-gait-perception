"""Per-participant detection power (sensitivity) for the 2-cps stride oscillation.

The prevalence model in ``population.py`` (Ince, Kay & Schyns 2022) needs the
within-participant sensitivity ``s``: the probability that a true oscillator is flagged
significant by the permutation test. With the default ``sensitivity=1.0`` the prevalence
MAP is a lower bound, as the planning simulations showed. This module estimates ``s`` to
de-bias it (b1) and reports achieved power per participant (b7).

``s`` is a Monte-Carlo estimate using ``phase_analysis.analyse_measure``, the confirmatory
2 & 4 cps Holm test that ``population.py`` also counts, so both use the same criterion.
Simulated observers have ``n_targets`` trials whose hit probability oscillates at 2 cps
with a reference amplitude; ``s`` is the fraction flagged significant. Results are cached in
``derivatives/planning/sensitivity_cache.json``, keyed by bucketed ``n_targets`` and amplitude.

Usage::

    python -m Analysis.power            # print a sensitivity table + cache it
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    from . import config as C
    from . import phase_analysis as pa
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    import phase_analysis as pa

# Reference 2-cps amplitude on the 0..1 hit-rate scale that defines a true oscillation.
# 0.05 is a modest modulation in line with Davidson; override per call for curves.
REF_AMP = 0.05
BASE_RATE = 0.75            # threshold-level base hit rate
SIM_N = 160                 # Monte-Carlo observers per cell
SIM_NPERM = 300             # permutations per simulated test (real analysis: 1000)
RNG = np.random.default_rng(7)

_CACHE_PATH = C.DERIV_DIR / "planning" / "sensitivity_cache.json"


def _bucket(n_targets: int) -> int:
    """Bucket trial counts so the cache is reusable across nearby Ns."""
    if n_targets <= 0:
        return 0
    step = 50 if n_targets < 400 else 100
    return int(round(n_targets / step) * step)


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except (ValueError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _simulate_sensitivity(n_targets: int, amp: float, base: float = BASE_RATE,
                          n_sim: int = SIM_N, nperm: int = SIM_NPERM) -> float:
    """Fraction of simulated 2-cps oscillators flagged significant (= sensitivity)."""
    if n_targets < 12:
        return 0.0
    w = 2 * np.pi * 2.0 / 100.0     # 2 cps on the 0..100 % stride axis
    hits = 0
    for _ in range(n_sim):
        phase = RNG.uniform(0, 100, n_targets)
        phi0 = RNG.uniform(0, 2 * np.pi)
        p = np.clip(base + amp * np.cos(w * phase + phi0), 0.01, 0.99)
        correct = (RNG.uniform(size=n_targets) < p).astype(float)
        res = pa.analyse_measure(phase, correct, "mean", nperm=nperm)
        if res is not None and res["significant"]:
            hits += 1
    return hits / n_sim


def sensitivity(n_targets: int, amp: float = REF_AMP, *, use_cache: bool = True) -> float:
    """Per-participant detection power for a 2-cps oscillation of amplitude ``amp``.

    Cached by bucketed ``n_targets`` and amplitude. This is the ``s`` that de-biases
    the population prevalence estimate.
    """
    # Version tag: bump it when the significance criterion changes. v2 = 2 & 4 cps Holm.
    key = f"v2_n{_bucket(n_targets)}_a{amp:.3f}"
    cache = _load_cache() if use_cache else {}
    if key in cache:
        return float(cache[key])
    s = _simulate_sensitivity(_bucket(n_targets), amp)
    if use_cache:
        cache[key] = s
        _save_cache(cache)
    return s


def mean_sensitivity(n_targets_list, amp: float = REF_AMP) -> float:
    """Mean detection power across participants given each one's trial count."""
    vals = [sensitivity(int(n), amp) for n in n_targets_list if n and n > 0]
    return float(np.mean(vals)) if vals else 1.0


# ===========================================================================
# CLI: print a sensitivity table (and warm the cache)
# ===========================================================================
def main():
    """Measure and cache the per-participant detection sensitivity used to de-bias prevalence."""
    print("Per-participant detection power (sensitivity) for a 2-cps oscillation")
    print(f"  base hit rate {BASE_RATE}, {SIM_N} sims x {SIM_NPERM} perms, "
          f"test = phase_analysis.analyse_measure\n")
    amps = [0.03, 0.05, 0.07]
    ns = [80, 200, 400, 800, 1300, 2000]
    header = "  n_targets | " + " | ".join(f"amp={a:.2f}" for a in amps)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in ns:
        cells = " | ".join(f"{sensitivity(n, a)*100:5.0f}%" for a in amps)
        print(f"  {n:9d} | {cells}")
    print(f"\nCache -> {_CACHE_PATH}")


if __name__ == "__main__":
    main()
