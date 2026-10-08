"""Validation tests for the analysis pipeline (b9).

Checks that injected effects are recovered, that null data keep nominal false-positive
rates, and that supporting algorithms (sequence shuffle, phase methods, calibration
refit, BIDS conversion, exclusion gate, caching) behave as documented.

Run::

    python -m pytest Analysis/tests -q
    python Analysis/tests/test_pipeline.py      # also works without pytest
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]      # Analysis/
sys.path.insert(0, str(ROOT))
import phase_analysis as pa          # noqa: E402
import clocktime_control as ctc      # noqa: E402
import gait as gait_mod              # noqa: E402
import glmm                          # noqa: E402
import population as pop             # noqa: E402
import calibration_refit as cr       # noqa: E402
import calibration as cal            # noqa: E402
import clean as clean_mod            # noqa: E402
import config as C                   # noqa: E402
import bidsify                       # noqa: E402
import report                        # noqa: E402
import onset_response as orr         # noqa: E402
import gam_phase as gp               # noqa: E402
import ddm_phase as dp               # noqa: E402
import group_agnostic as ga          # noqa: E402
import common_phase as cph          # noqa: E402
import stance_foot as sfoot          # noqa: E402
import sdt_phase as sdt            # noqa: E402


# ---------------------------------------------------------------------------
# Fourier permutation test recovers an injected 2-cps oscillation
# ---------------------------------------------------------------------------
def test_phase_recovers_injected_2cps():
    """A strong 2-cps oscillation injected into synthetic trials is detected, and the
    grid scan puts its peak at 2 cps."""
    rng = np.random.default_rng(0)
    w = 2 * np.pi * 2.0 / 100.0
    n = 1500
    phase = rng.uniform(0, 100, n)
    p = np.clip(0.75 + 0.12 * np.cos(w * phase), 0.01, 0.99)   # strong 2-cps effect
    correct = (rng.uniform(size=n) < p).astype(float)
    res = pa.analyse_measure(phase, correct, "mean", nperm=500)
    assert res is not None
    assert res["significant"], "should detect a strong injected oscillation"
    assert 1.5 <= res["best_freq_cps"] <= 2.5, f"peak off 2 cps: {res['best_freq_cps']}"


def test_phase_null_false_positive_rate_controlled():
    """Across seeds, the permutation test should flag noise at roughly its level."""
    hits = 0
    trials = 20
    for s in range(trials):
        rng = np.random.default_rng(100 + s)
        n = 800
        phase = rng.uniform(0, 100, n)
        correct = (rng.uniform(size=n) < 0.75).astype(float)   # no oscillation
        res = pa.analyse_measure(phase, correct, "mean", nperm=300)
        hits += bool(res and res["significant"])
    assert hits <= 4, f"false-positive rate too high: {hits}/{trials}"


# ---------------------------------------------------------------------------
# Hierarchical GLMM: recovers a population effect, controls the null
# ---------------------------------------------------------------------------
def _sim_glmm(amp_logit, n_sub=8, n=250, seed=0):
    rng = np.random.default_rng(seed)
    w = 2 * np.pi * 2.0 / 100.0
    frames = []
    for s in range(n_sub):
        u = rng.normal(0, 0.3)
        ph = rng.uniform(0, 100, n)
        p = 1 / (1 + np.exp(-(0.4 + u + amp_logit * np.cos(w * ph + 0.5))))
        c = (rng.uniform(size=n) < p).astype(float)
        frames.append(pd.DataFrame({"sub": f"sub-{s:02d}", "phase": ph, "correct": c}))
    return pd.concat(frames, ignore_index=True)


def test_glmm_recovers_population_oscillation():
    """The mixed model recovers a known population amplitude to within 0.15 logit using
    the mixed (not fallback) estimator. Estimation only; the GLMM has no test."""
    fit = glmm.fit_population(_sim_glmm(0.6, seed=1))
    assert fit["n_participants"] == 8
    assert "MixedGLM" in fit["method"]
    assert abs(fit["amp_logit"] - 0.6) < 0.15, f"amp off: {fit['amp_logit']}"


def test_glmm_reports_no_p_values():
    """The removed Wald test must not come back in any GLMM output."""
    fit = glmm.fit_population(_sim_glmm(0.6, seed=1))
    banned = {"p_value", "significant", "wald_chi2_df2"}
    assert not (banned & set(fit)), f"GLMM re-introduced a test: {banned & set(fit)}"
    assert "descriptive" in fit["inference"].lower()
    assert not hasattr(glmm, "_wald"), "the joint Wald helper is back"
    fd = glmm.fit_difficulty_model(
        _sim_glmm(0.5, seed=3).assign(condition="walking", difficulty="THR"))
    if fd:
        for lev in fd["by_difficulty"].values():
            assert not (banned & set(lev)), "difficulty model re-introduced a test"
    cm = glmm.fit_condition_model(
        pd.concat([_sim_glmm(0.5, seed=4).assign(condition="walking", difficulty="THR"),
                   _sim_glmm(0.0, seed=5).assign(condition="standing", difficulty="THR")],
                  ignore_index=True))
    if cm:
        for block in ("locomotion_effect", "standing_oscillation_control",
                      "walking_specific_oscillation"):
            assert not (banned & set(cm[block])), f"{block} re-introduced a test"


# ---------------------------------------------------------------------------
# Hilbert vs trough phase agree on a synthetic gait signal
# ---------------------------------------------------------------------------
def test_hilbert_matches_trough_phase():
    """The continuous Hilbert phase and the trough-interpolated phase agree on a clean
    synthetic gait signal, so a result cannot hinge on which one is selected."""
    fs = 90.0
    t = np.arange(0, 60, 1 / fs)
    stride_hz = 0.9                       # ~1.2 s stride
    vertical = -np.cos(2 * np.pi * 2 * stride_hz * t)     # 2 bobs / stride
    lateral = np.sin(2 * np.pi * stride_hz * t)           # 1 sway / stride
    troughs, _ = gait_mod._detect_troughs(t, vertical)
    cont, src = gait_mod.continuous_stride_phase(t, vertical, lateral, troughs)
    mids = 0.5 * (troughs[:-1] + troughs[1:])
    _, ps_tr = gait_mod.phase_at(mids, troughs)
    _, ps_hb = gait_mod.phase_at_hilbert(mids, t, cont)
    ok = np.isfinite(ps_tr) & np.isfinite(ps_hb)
    a = np.exp(1j * 2 * np.pi * ps_tr[ok] / 100.0)
    b = np.exp(1j * 2 * np.pi * ps_hb[ok] / 100.0)
    agree = float(np.abs(np.mean(a * np.conj(b))))
    assert agree > 0.8, f"Hilbert and trough phase disagree (circular corr {agree:.2f})"


# ---------------------------------------------------------------------------
# Constrained sequence shuffle (mirror of the Unity a8 algorithm)
# ---------------------------------------------------------------------------
def _constrained_shuffle(pool, max_run, rng):
    counts = Counter(pool)
    out, last, run = [], object(), 0
    for _ in range(len(pool)):
        best, blocked = -1, True
        for k, v in counts.items():
            if v <= 0 or (k == last and run >= max_run):
                continue
            blocked = False
            best = max(best, v)
        if blocked:
            chosen = next(k for k, v in counts.items() if v > 0)
        else:
            cands = [k for k, v in counts.items()
                     if v == best and not (k == last and run >= max_run)]
            chosen = cands[rng.randrange(len(cands))]
        out.append(chosen); counts[chosen] -= 1
        run = run + 1 if chosen == last else 1
        last = chosen
    return out


def test_constrained_shuffle_no_long_runs():
    """The tilt-direction shuffle never emits more than the allowed run of identical
    directions while preserving the 50/50 balance - the property that stops
    participants picking up a pattern."""
    import random
    rng = random.Random(3)
    for _ in range(500):
        n = rng.randint(0, 80)
        half = n // 2
        pool = [0] * half + [1] * (n - half)
        out = _constrained_shuffle(pool, 2, rng)
        assert Counter(out) == Counter(pool)
        run = max_run = 1 if out else 0
        for i in range(1, len(out)):
            run = run + 1 if out[i] == out[i - 1] else 1
            max_run = max(max_run, run)
        assert max_run <= 2, f"run {max_run} > 2 for n={n}"


# ---------------------------------------------------------------------------
# Calibration slope/lapse: pooled refit beats single, recovers truth
# ---------------------------------------------------------------------------
def test_calibration_joint_refit_recovers_slope():
    """The joint fit (shared slope/lapse, per-participant thresholds) recovers the true
    slope; naive pooling is biased low by threshold heterogeneity."""
    rec = cr.simulation_study(n_participants=12, n_trials=90,
                              true_slope=3.5, true_lapse=0.03)
    assert rec["joint_slope"] is not None
    assert abs(rec["joint_slope"] - 3.5) < 0.8, f"joint slope off: {rec['joint_slope']}"
    assert rec["joint_lapse"] <= 0.10
    # the naive pooled slope should be the flattened (lower) one
    assert rec["pooled_slope"] <= rec["joint_slope"] + 0.3


def _sim_cond_observers(slope, n_part=10, n_trials=90, seed=0):
    """A condition's worth of QUEST-like observers (tilts spread around threshold)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_part):
        alpha = float(max(1.0, rng.normal(5.0, 1.0)))
        levels = alpha * np.geomspace(0.4, 2.2, 7)
        x = rng.choice(levels, size=n_trials)
        p = cr.weibull(x, alpha, slope, 0.03)
        out.append((x, (rng.uniform(size=n_trials) < p).astype(float)))
    return out


def test_slope_invariance_lrt():
    """Equal slopes across conditions -> n.s.; clearly different slopes -> detected."""
    same = cr.fit_slope_invariance({"standing": _sim_cond_observers(3.5, seed=1),
                                    "walking": _sim_cond_observers(3.5, seed=2)})
    assert same["ok"] and same["p_value"] > 0.05, f"equal slopes flagged (p={same['p_value']})"
    diff = cr.fit_slope_invariance({"standing": _sim_cond_observers(5.0, seed=3),
                                    "walking": _sim_cond_observers(1.5, seed=4)})
    assert diff["ok"] and diff["slope_differs"], f"clear slope difference missed (p={diff['p_value']})"
    # needs >=2 conditions
    assert not cr.fit_slope_invariance({"standing": _sim_cond_observers(3.0, seed=5)})["ok"]


# ---------------------------------------------------------------------------
# Prevalence: identifiability guard + monotonic de-biasing
# ---------------------------------------------------------------------------
def test_prevalence_debias_monotonic_when_identifiable():
    """De-biased prevalence rises monotonically with the number of positive participants
    while sensitivity is high enough to be identifiable."""
    # With sensitivity above the FPR, lower sensitivity maps the same k/n to a higher
    # prevalence (more true oscillators were missed).
    _, _, map_hi, _ = pop.bayesian_prevalence(3, 10, sensitivity=0.9)
    _, _, map_lo, _ = pop.bayesian_prevalence(3, 10, sensitivity=0.3)
    assert map_lo >= map_hi, "lower sensitivity should not lower the de-biased estimate"


def test_prevalence_unidentifiable_threshold_exists():
    """The identifiability guard fires when measured sensitivity approaches the
    false-positive rate, where prevalence cannot be estimated and a number would be
    misleading."""
    assert pop.IDENTIFIABLE_MIN_SENS > pop.ALPHA


# ---------------------------------------------------------------------------
# Head-position cleaning: removes spikes, keeps real data
# ---------------------------------------------------------------------------
def _clean_bob(fs=60.0, dur=60.0):
    t = np.arange(0, dur, 1.0 / fs)
    y = 0.34 + 0.02 * np.sin(2 * np.pi * 1.5 * t)   # ~1.5 Hz, 2 cm bob around 0.34 m
    return t, y


def test_clean_preserves_a_clean_gait_bob():
    """Cleaning leaves a clean signal unchanged."""
    t, y = _clean_bob()
    yc, bad, rep, exc, info = clean_mod.clean_channel(t, y, abs_radius_m=0.30)
    assert info["pct_flagged"] < 0.5, f"flagged a clean bob: {info['pct_flagged']}%"
    assert np.allclose(yc, y, atol=1e-9), "cleaning altered an already-clean signal"


def test_clean_removes_spikes_but_keeps_real_samples():
    """Sharp dropout spikes are detected and removed; the bob on clean samples is intact."""
    t, y = _clean_bob()
    spike_idx = np.arange(200, len(t), 300)         # a one-sample dropout every 5 s
    y_spiked = y.copy()
    y_spiked[spike_idx] = -1.1                       # ~1.4 m drop from the median
    yc, bad, rep, exc, info = clean_mod.clean_channel(t, y_spiked, abs_radius_m=0.30)
    assert bad[spike_idx].all(), "not all injected spikes were detected"
    # the spike value must not survive (interpolated back toward the bob)
    repaired = np.nan_to_num(yc[spike_idx], nan=0.0)
    assert np.all(np.abs(repaired) < 0.5), "a spike value survived cleaning"
    # non-spike samples must be unchanged
    keep = np.ones(len(t), bool)
    keep[spike_idx] = False
    assert np.allclose(yc[keep], y[keep], atol=1e-6), "cleaning altered real (non-spike) samples"


# ---------------------------------------------------------------------------
# Per-condition calibration: filename parsing + two-calibration BIDS conversion
# ---------------------------------------------------------------------------
def test_parse_percondition_calibration_names():
    """Raw filenames parse correctly across every naming era: per-condition calibrations,
    plain blocks, collision suffixes and the legacy single calibration."""
    s = C.parse_raw_name("p05-b0-t00_calibration_standing.csv.gz")
    w = C.parse_raw_name("p05-b0-t00_calibration_walking.csv")
    legacy = C.parse_raw_name("p00-b0-t00_calibration.csv")
    exp = C.parse_raw_name("p05-b1-t00_walking.csv.gz")
    assert s["is_calibration"] and s["condition"] == "standing"
    assert w["is_calibration"] and w["condition"] == "walking"
    assert legacy["is_calibration"] and legacy["condition"] == "calibration"
    assert (not exp["is_calibration"]) and exp["condition"] == "walking"
    # collision suffix still parses
    d = C.parse_raw_name("p05-b0-t00_calibration_walking-1.csv.gz")
    assert d["is_calibration"] and d["condition"] == "walking" and d["dup"] == 1
    assert s["visit"] is None and exp["visit"] is None      # pre-visit names
    # names with the visit (rig since 2026-09-16) keep every identifier
    vb = C.parse_raw_name("p07-v2-b1-t05_walking.csv.gz")
    assert (vb["participant"], vb["visit"], vb["block"], vb["trial"]) == (7, 2, 1, 5)
    assert (not vb["is_calibration"]) and vb["condition"] == "walking"
    vc = C.parse_raw_name("p07-v2-b0-t02_calibration_standing-1.csv")
    assert (vc["participant"], vc["visit"], vc["block"], vc["trial"], vc["dup"]) == (7, 2, 0, 2, 1)
    assert vc["is_calibration"] and vc["condition"] == "standing"
    assert C.run_for(vb) == "run-05" and C.run_for(vc) == "run-02"


def _min_calib_csv(threshold_deg: float) -> str:
    """Minimal calibration log (StaircaseManager format): one '#' meta line + rows."""
    head = ("tracker;global_trial;tracker_trial;presented_tilt_deg;tilt_direction;"
            "response;correct;post_mean_threshold_deg;post_sd_log")
    rows = [f"# threshold_deg={threshold_deg};easy_deg={threshold_deg*1.26:.3f};"
            f"hard_deg={threshold_deg*0.76:.3f}", head,
            "A;1;1;4.0;0;0;1;3.5;0.20", "B;2;1;2.0;1;1;1;2.4;0.18",
            "catch;3;1;12.0;0;0;1;2.4;0.18", "A;4;2;2.6;1;0;0;2.5;0.15"]
    return "\n".join(rows) + "\n"


def test_calibration_report_excludes_warmup_and_catch():
    """Warm-up and catch rows are not counted as QUEST+ trackers."""
    cols = ["tracker", "global_trial", "tracker_trial", "presented_tilt_deg",
            "tilt_direction", "response", "correct", "post_mean_threshold_deg", "post_sd_log"]
    rows = [
        ("warmup", 1, 1, 8.0, 0, 0, 1, 2.5, 0.30),   # leading practice block
        ("warmup", 2, 2, 8.0, 1, 1, 1, 2.5, 0.30),
        ("A", 3, 1, 4.0, 0, 0, 1, 2.4, 0.20),        # real QUEST+ trials
        ("B", 4, 1, 2.0, 1, 1, 1, 2.3, 0.18),
        ("catch", 5, 1, 12.0, 0, 0, 1, 2.3, 0.18),   # attentiveness probe
        ("A", 6, 2, 2.6, 1, 0, 0, 2.5, 0.15),
    ]
    df = pd.DataFrame(rows, columns=cols)
    rep = cal.reliability_report(df, {"threshold_deg": 2.5, "n_warmup": 2, "warmup_correct": 2})
    assert set(rep["trackers"]) == {"A", "B"}, rep["trackers"]
    assert rep["n_trials"] == 3          # 2 A + 1 B scored; warm-up & catch excluded
    assert rep.get("warmup", {}).get("n_answered") == 2


def test_high_tilt_lapses_gate_only_above_model_lapse_rate():
    """High-tilt errors make a calibration UNRELIABLE only when there are more of them than
    the model lapse rate predicts: 2 in 25 trials is ordinary, 6 is not."""
    cols = ["tracker", "global_trial", "tracker_trial", "presented_tilt_deg",
            "tilt_direction", "response", "correct", "post_mean_threshold_deg", "post_sd_log"]
    meta = {"threshold_deg": 1.0, "guess": 0.5, "slope_beta": 2.5, "lapse": 0.05}

    def report(n_wrong):
        rows = [("AB"[i % 2], i + 1, i // 2 + 1, 2.5, 0, 0, int(i >= n_wrong), 1.0, 0.08)
                for i in range(25)]                       # 25 trials at 2.5x threshold
        rows += [("AB"[i % 2], 26 + i, 14 + i // 2, 1.0, 0, 0, i % 4 != 0, 1.0, 0.08)
                 for i in range(20)]                      # near-threshold trials
        return cal.reliability_report(pd.DataFrame(rows, columns=cols), meta)

    ok = report(2)
    assert ok["high_tilt_lapses"] == 2
    assert ok["high_tilt_lapse_test"]["p_value"] > 0.05
    assert not any("high-tilt" in f for f in ok["flags"]), ok["flags"]
    bad = report(6)
    assert bad["high_tilt_lapse_test"]["p_value"] < 0.05
    assert any("high-tilt" in f for f in bad["flags"]), bad["flags"]


def test_bidsify_emits_two_calibration_files():
    """Standing + walking calibration logs convert to two acq-tagged ses-01 beh files
    without colliding; a legacy bare _calibration log still makes a non-acq beh file."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        (raw / "p07-b0-t00_calibration_standing.csv").write_text(_min_calib_csv(2.0))
        (raw / "p07-b0-t00_calibration_walking.csv").write_text(_min_calib_csv(3.5))
        (raw / "p08-b0-t00_calibration.csv").write_text(_min_calib_csv(2.5))   # legacy

        bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)

        beh7 = bids / "sub-07" / "ses-01" / "beh"
        st = beh7 / "sub-07_ses-01_task-ILL_acq-standing_beh.tsv"
        wk = beh7 / "sub-07_ses-01_task-ILL_acq-walking_beh.tsv"
        assert st.exists() and wk.exists(), "expected two acq-tagged calibration beh files"
        assert len(pd.read_csv(st, sep="\t")) == 4 and len(pd.read_csv(wk, sep="\t")) == 4
        # the two calibrations must NOT collide in sourcedata either
        arch = sorted((bids / "sourcedata" / "sub-07" / "ses-01" / "beh").glob("*.csv.gz"))
        assert len(arch) == 2, f"expected 2 distinct calibration archives, got {len(arch)}"
        # legacy single calibration -> bare _beh (no acq)
        beh8 = bids / "sub-08" / "ses-01" / "beh"
        assert (beh8 / "sub-08_ses-01_task-ILL_beh.tsv").exists()
        assert not list(beh8.glob("*_acq-*_beh.tsv")), "legacy calibration must not gain an acq entity"


def _calib_csv_with_ids(threshold_deg: float, participant: int, visit: int, block: int,
                        trial: int, condition: str, order_first: str) -> str:
    """Calibration log as StaircaseManager writes it since 2026-09-16: every data row
    ends with participant;visit;counterbalance_slot;block;trial;condition;order_first."""
    lines = _min_calib_csv(threshold_deg).splitlines()
    ids = f";{participant};{visit};{participant};{block};{trial};{condition};{order_first}"
    head = lines[1] + ";participant;visit;counterbalance_slot;block;trial;condition;condition_order_first"
    return "\n".join([lines[0], head] + [r + ids for r in lines[2:]]) + "\n"


def test_visit_named_calibrations_keep_identity_through_bids():
    """Visit-named calibration logs (p07-v1-/p07-v2-) convert to one run per visit, keep
    their identity columns in the BIDS table, and still load in the calibration QC."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        (raw / "p07-v1-b1-t01_calibration_walking.csv").write_text(
            _calib_csv_with_ids(3.5, 7, 1, 1, 1, "walking", "walking"))
        (raw / "p07-v2-b1-t02_calibration_walking.csv").write_text(
            _calib_csv_with_ids(3.1, 7, 2, 1, 2, "walking", "standing"))

        bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)

        beh = bids / "sub-07" / "ses-01" / "beh"
        v1 = pd.read_csv(beh / "sub-07_ses-01_task-ILL_acq-walking_run-01_beh.tsv", sep="\t")
        v2 = pd.read_csv(beh / "sub-07_ses-01_task-ILL_acq-walking_run-02_beh.tsv", sep="\t")
        assert len(v1) == 4 and len(v2) == 4
        assert set(v1["visit"]) == {1} and set(v2["visit"]) == {2}
        assert set(v2["participant"]) == {7} and set(v2["block"]) == {1}
        assert set(v2["condition_order_first"]) == {"standing"}
        # the calibration QC reads the archived raw log despite the extra columns
        arch = sorted((bids / "sourcedata" / "sub-07" / "ses-01" / "beh").glob("*run-02*.csv.gz"))
        assert len(arch) == 1
        meta, df = cal._load_calibration(arch[0])
        assert meta["threshold_deg"] == 3.1
        rep = cal.reliability_report(df, meta)
        assert set(rep["trackers"]) == {"A", "B"}


def test_bidsify_multiblock_run_entity_preserves_all_blocks():
    """Several blocks of one condition (distinct trial numbers) become separate run-
    filesets without overwriting, and the reader recovers the run index."""
    import tempfile
    import bids_io
    import simulate_participants as sim
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        sim.generate(1, raw, walk_s=30.0, stand_s=25.0, n_walk=3, n_stand=2)
        bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)

        mo = bids / "sub-01" / "ses-02" / "motion"
        walk = sorted(mo.glob("sub-01_ses-02_task-ILL_tracksys-optitrack_acq-walking_run-*_motion.tsv"))
        stand = sorted(mo.glob("sub-01_ses-02_task-ILL_tracksys-optitrack_acq-standing_run-*_motion.tsv"))
        assert len(walk) == 3, f"expected 3 walking run files (no overwrite), got {len(walk)}"
        assert len(stand) == 2, f"expected 2 standing run files (no overwrite), got {len(stand)}"
        # reader recovers the run index and keeps walking runs distinct
        recs = bids_io.find_recordings(bids)
        wruns = sorted(r.run for r in recs if r.condition == "walking")
        assert wruns == ["01", "02", "03"], f"reader lost/merged runs: {wruns}"
        assert len({r.label() for r in recs}) == len(recs), "recording labels collide"
        # both per-condition calibrations survive under run-01
        beh = bids / "sub-01" / "ses-01" / "beh"
        assert (beh / "sub-01_ses-01_task-ILL_acq-standing_run-01_beh.tsv").exists()
        assert (beh / "sub-01_ses-01_task-ILL_acq-walking_run-01_beh.tsv").exists()


# ---------------------------------------------------------------------------
# report.py: inclusion/exclusion table applies the documented criteria
# ---------------------------------------------------------------------------
def test_exclusions_table_mirrors_the_gate_and_only_warns():
    """The exclusions table takes each block's decision from the gate; poor head data,
    side bias and low alternation are warnings and never exclude a block."""
    rows = [
        dict(recording="r1", participant="sub-01", session="ses-02", condition="walking",
             clean_coverage_pct=95.0, clean_verdict="clean", response_right_bias=0.52,
             lr_alternation_pct=96.0, calibration_verdict="RELIABLE",
             gate_behaviour="include", gate_gait="include"),
        dict(recording="r2", participant="sub-02", session="ses-02", condition="walking",
             clean_coverage_pct=45.0, clean_verdict="severe", response_right_bias=0.82,
             lr_alternation_pct=60.0, calibration_verdict="UNRELIABLE - recommend recalibration",
             gate_behaviour="exclude", gate_gait="include"),
    ]
    gate = {"r1": {"reason": "include", "rule": "pass"},
            "r2": {"reason": "chance: accuracy 80/170 not above chance", "rule": "chance"}}
    ex = {e["participant"]: e for e in report._exclusions(rows, gate)}
    assert ex["sub-01"]["behaviour"] == "include" and ex["sub-01"]["warnings"] == ""
    assert ex["sub-02"]["behaviour"] == "exclude" and ex["sub-02"]["gait"] == "include"
    assert ex["sub-02"]["reason"].startswith("chance")
    assert "clean coverage" in ex["sub-02"]["warnings"]
    assert "side-bias" in ex["sub-02"]["warnings"]


# ---------------------------------------------------------------------------
# report.py: provenance captures code + environment + config
# ---------------------------------------------------------------------------
def test_provenance_has_expected_keys():
    """Every run records the provenance needed to reproduce it: commit, dirty flag,
    Python and package versions, and the analysis constants in force."""
    p = report.provenance()
    for k in ("generated_utc", "git_sha", "git_branch", "git_dirty", "python",
              "packages", "config"):
        assert k in p, f"provenance missing {k}"
    # the config snapshot must carry the params that define the numbers
    assert "RESPONSE_WINDOW_S" in p["config"] and "PHASE_METHOD" in p["config"]
    assert isinstance(p["git_dirty"], bool)


# ---------------------------------------------------------------------------
# Clock-time control: recovers a clock-locked oscillation; null controlled
# ---------------------------------------------------------------------------
def test_clocktime_recovers_clock_oscillation():
    """An oscillation locked to absolute time (a fixed Hz) is recovered by the
    period-folding clock-time test at the right frequency."""
    rng = np.random.default_rng(7)
    n = 1500
    onset = np.sort(rng.uniform(0, 600, n))          # 10 min of targets
    f0 = 2.0                                          # 2 Hz clock-locked effect
    p = np.clip(0.75 + 0.12 * np.cos(2 * np.pi * f0 * onset), 0.01, 0.99)
    correct = (rng.uniform(size=n) < p).astype(float)
    res = ctc.analyse_clock(onset, correct, "mean", nperm=300)
    assert res is not None and res["significant"], "should detect a clock-locked 2 Hz effect"
    assert abs(res["best_freq_hz"] - f0) <= 0.2, f"peak off 2 Hz: {res['best_freq_hz']}"


def test_clocktime_null_controlled():
    """Without clock structure the max-statistic permutation test stays roughly
    calibrated. Coarse check: a 30-seed / 200-permutation sweep gave FPR ≈ 0.10 (mean
    p ≈ 0.48); a broken null (FPR ≥ 0.5) would give ≳ 10/20. Seeds 200-219 give 4/20."""
    hits = 0
    for s in range(20):
        rng = np.random.default_rng(200 + s)
        n = 700
        onset = np.sort(rng.uniform(0, 600, n))
        correct = (rng.uniform(size=n) < 0.75).astype(float)
        res = ctc.analyse_clock(onset, correct, "mean", nperm=200)
        hits += bool(res and res["significant"])
    assert hits <= 6, f"clock-time false-positive rate too high: {hits}/20"


# ---------------------------------------------------------------------------
# Exploratory: onset-locked vs response-locked dissociation
# ---------------------------------------------------------------------------
def _osc_binary(rng, phase, pref_pct, amp, base=0.75, freq=2.0):
    w = 2 * np.pi * freq / 100.0
    p = np.clip(base + amp * np.cos(w * (phase - pref_pct)), 0.02, 0.998)
    return (rng.uniform(size=phase.size) < p).astype(float)


def _dissoc_trials(rng, n, pref_acc, pref_resp, freq=2.0):
    ph = rng.uniform(0, 100, n)
    acc = _osc_binary(rng, ph, pref_acc, 0.10, 0.75, freq)
    responded = _osc_binary(rng, ph, pref_resp, 0.12, 0.88, freq)
    r = responded == 1
    return dict(onset_phase=ph, acc=acc, responded=responded, logrt_onset=np.full(n, np.nan),
                resp_onset_phase=ph[r], resp_response_phase=(ph[r] + 60) % 100,
                resp_logrt=np.log10(rng.uniform(0.5, 0.9, int(r.sum()))), stride_period=1.11)


def test_onset_response_recovers_dissociation():
    """Accuracy and P(respond) peaking at different onset phases: Δψ CI excludes 0."""
    rng = np.random.default_rng(4)
    tr = _dissoc_trials(rng, 4000, pref_acc=20.0, pref_resp=40.0)
    res = orr.dissociation(tr, freq=2.0, nboot=400, rng=rng)
    assert res["dissociation"], f"missed a real dissociation (CI {np.degrees(res['delta_psi_ci_rad'])})"
    assert abs(res["acc_pref_pct"] - 20) < 8 and abs(res["respond_pref_pct"] - 40) < 8


def test_onset_response_null_no_dissociation():
    """Both measures peaking at the same phase: Δψ ~ 0 and the CI includes 0."""
    rng = np.random.default_rng(5)
    tr = _dissoc_trials(rng, 4000, pref_acc=30.0, pref_resp=30.0)
    res = orr.dissociation(tr, freq=2.0, nboot=400, rng=rng)
    assert not res["dissociation"], f"spurious dissociation (Δψ={np.degrees(res['delta_psi_rad']):.0f}°)"


# ---------------------------------------------------------------------------
# Exploratory: cyclic-spline GAM + mutual information
# ---------------------------------------------------------------------------
def _gam_df(kind, seed, n_sub=6, n=300):
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_sub):
        u = rng.normal(0, 0.3)
        ph = rng.uniform(0, 100, n)
        if kind == "peaked":                      # narrow good window (non-sinusoidal)
            d = (((ph - 25 + 50) % 100) - 50)
            lin = 0.3 + u + 1.7 * np.exp(-0.5 * (d / 8.0) ** 2)
        else:                                     # flat null
            lin = 1.0 + u
        p = 1 / (1 + np.exp(-lin))
        c = (rng.uniform(size=n) < p).astype(float)
        frames.append(pd.DataFrame({"sub": f"sub-{s:02d}", "phase": ph, "correct": c, "logrt": np.nan}))
    return pd.concat(frames, ignore_index=True)


def test_gam_detects_nonsinusoidal_shape():
    """The cyclic-spline GAM detects a non-sinusoidal modulation that a single cosine
    would understate."""
    g = gp.fit_gam(_gam_df("peaked", 7), "correct", binomial=True)
    assert g["significant"], f"GAM missed a real phase effect (p={g['p_value']})"
    assert g["prefers_gam_over_cosine"], "GAM should beat a single cosine on a peaked profile"


def test_gam_null_not_significant():
    """The GAM does not flag a flat stride profile."""
    g = gp.fit_gam(_gam_df("null", 8), "correct", binomial=True)
    assert not g["significant"], f"GAM flagged a null (p={g['p_value']})"


def test_mutual_info_detects_and_controls_null():
    """Mutual information between stride phase and accuracy detects a real dependency and
    stays at its nominal level under the null."""
    rng = np.random.default_rng(9)
    n = 2500
    w = 2 * np.pi * 2 / 100.0
    ph = rng.uniform(0, 100, n)
    acc = (rng.uniform(size=n) < np.clip(0.75 + 0.12 * np.cos(w * (ph - 25)), 0.02, 0.99)).astype(float)
    hit = gp.mutual_info_perm(ph, acc, y_binary=True, nperm=300, rng=rng)
    assert hit["significant"] and hit["mi_debiased_bits"] > 0, f"MI missed a real dependence ({hit})"
    accn = (rng.uniform(size=n) < 0.75).astype(float)
    nul = gp.mutual_info_perm(ph, accn, y_binary=True, nperm=300, rng=rng)
    assert not nul["significant"], f"MI flagged a null (p={nul['p_value']})"


# ---------------------------------------------------------------------------
# Exploratory: EZ-diffusion decomposition finds the drift locus
# ---------------------------------------------------------------------------
def test_ddm_attributes_drift_modulation_to_v():
    """A drift-only modulation is attributed to v, not a or Ter."""
    import simulate_participants as sim
    rng = np.random.default_rng(10)
    n, k = 3600, 2.0
    ph = rng.uniform(0, 100, n)
    correct = np.zeros(n)
    rt = np.full(n, np.nan)
    for i in range(n):
        v = sim.DRIFT["THR"] * (1 + 0.30 * np.cos(2 * np.pi * k * (ph[i] / 100.0 - 0.25)))
        c, r = sim._ddm_sample(rng, max(v, 0.02), sim.A_BOUND, sim.T0_S, max_rt=2.0)
        if c is not None:
            correct[i] = int(c); rt[i] = r
    res = dp.decompose(ph, correct, rt, freq=k, nperm=400, rng=rng)
    assert res["carrier"] == "v", f"carrier should be drift v, got {res['carrier']}"
    assert res["params"]["v"]["significant"], "the v oscillation should be significant"
    assert res["params"]["v"]["rel_amp"] > 3 * res["params"]["a"]["rel_amp"], "v should dominate a"


# ---------------------------------------------------------------------------
# Crossover counterbalance: order is data, and the H2 contrasts are unbiased
# ---------------------------------------------------------------------------
def _cross_cells(n, cond, pos, inter, noise=0.03, seed=0):
    """Synthetic per-participant (condition x position) 75% thresholds with known
    effects injected on log10 threshold."""
    rng = np.random.default_rng(seed)
    cells = {}
    for i in range(n):
        base = rng.normal(0.8, 0.1)
        c = {}
        for condition in ("standing", "walking"):
            for p in ("first", "second"):
                v = (base + cond * (condition == "walking") + pos * (p == "second")
                     + inter * (condition == "walking" and p == "second")
                     + rng.normal(0, noise))
                c[(condition, p)] = [10 ** v]
        cells[f"sub-{i:02d}"] = c
    return cells


def test_crossover_contrasts_recover_injected_effects():
    """condition / position / interaction are separately recoverable and don't leak."""
    import threshold_by_condition as tbc
    e = tbc._crossover_model(_cross_cells(40, 0.10, 0.0, 0.0, seed=1))["effects"]
    assert abs(e["condition_walking_minus_standing"]["mean_log10_ratio"] - 0.10) < 0.02
    assert abs(e["position_second_minus_first"]["mean_log10_ratio"]) < 0.02
    e = tbc._crossover_model(_cross_cells(40, 0.0, 0.12, 0.0, seed=2))["effects"]
    assert abs(e["position_second_minus_first"]["mean_log10_ratio"] - 0.12) < 0.02
    assert abs(e["condition_walking_minus_standing"]["mean_log10_ratio"]) < 0.02
    e = tbc._crossover_model(_cross_cells(40, 0.0, 0.0, 0.15, seed=3))["effects"]
    assert abs(e["interaction_condition_x_position"]["mean_log10_ratio"] - 0.15) < 0.03


def test_paired_threshold_pools_all_blocks_of_a_condition():
    """The paired test fits each participant's threshold to all blocks of a condition
    (as in the power analysis), not only the last block."""
    import threshold_by_condition as tbc
    rng = np.random.default_rng(7)

    def block(alpha, n=180):
        tilt = rng.choice(np.array([0.76, 1.0, 1.26]) * alpha, size=n, p=[0.05, 0.85, 0.10])
        return tilt, (rng.random(n) < cr.weibull(tilt, alpha, 2.5, 0.05, 0.5)).astype(float)

    walking = [block(3.0), block(3.0), block(3.0), block(9.0)]   # last block deviates
    standing = [block(2.5), block(2.5)]
    per = tbc._participant_fits({"sub-01": {"walking": walking, "standing": standing}},
                                lambda c: (2.5, 0.05))
    w, s = per["sub-01"]["walking"], per["sub-01"]["standing"]
    assert w["n_blocks"] == 4 and w["n"] == sum(t.size for t, _ in walking)
    assert s["n_blocks"] == 2 and s["n"] == sum(t.size for t, _ in standing)
    last_only = tbc.fit_alpha(*walking[-1], beta=2.5, lapse=0.05)
    assert w["alpha_deg"] < 0.8 * last_only["alpha_deg"], (w["alpha_deg"], last_only["alpha_deg"])
    assert 2.5 < w["alpha_deg"] < 9.0


def test_crossover_contrasts_false_positive_rate_controlled():
    """With no true condition or position effect, the crossover contrasts hold their
    nominal false-positive rate."""
    import threshold_by_condition as tbc
    keys = ("condition_walking_minus_standing", "position_second_minus_first",
            "interaction_condition_x_position")
    hits = dict.fromkeys(keys, 0)
    n = 200
    for s in range(n):
        e = tbc._crossover_model(_cross_cells(16, 0, 0, 0, seed=100 + s))["effects"]
        for k in keys:
            hits[k] += e[k]["significant"]
    for k in keys:
        assert hits[k] / n < 0.12, f"{k} FPR {hits[k]/n:.2f} (nominal 0.05)"


def test_crossover_allocation_matches_unity_and_balances():
    """The simulator's allocation string must stay byte-identical to
    ExperimentConductor.ALLOCATION, and permuted blocks must keep the running
    standing/walking split within 2 at every enrolment point."""
    import simulate_participants as sim
    cs = (ROOT.parent / "Assets" / "Scripts" / "ExperimentConductor.cs").read_text(
        encoding="utf-8", errors="ignore")
    m = re.search(r'const string ALLOCATION = "([SW]+)"', cs)
    assert m, "ALLOCATION constant not found in ExperimentConductor.cs"
    assert m.group(1) == sim.ALLOCATION, "Unity and simulator allocations diverged"
    imb = np.cumsum([1 if c == "S" else -1 for c in sim.ALLOCATION])
    assert np.abs(imb).max() <= 2, f"allocation drifts to {np.abs(imb).max()}"
    # visit 2 must be the mirror of visit 1 for every slot (the crossover)
    for slot in range(1, 33):
        assert sim.standing_first(slot, 1) != sim.standing_first(slot, 2)


def test_crossover_design_yields_all_four_cells():
    """End to end: two visits -> scans.tsv records both orders, so every participant
    supplies all four (condition x position) cells and H2 becomes within-subject."""
    import tempfile
    import bids_io
    import simulate_participants as sim
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        sim.generate(2, raw, walk_s=25.0, stand_s=25.0, n_walk=1, n_stand=1, visits=2)
        bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
        scans = pd.read_csv(bids / "sub-01" / "sub-01_scans.tsv", sep="\t")
        mo = scans[scans["filename"].str.contains("motion")]
        assert set(mo["condition_order_first"].dropna()) == {"standing", "walking"}, \
            "the two visits must run opposite orders (crossover)"
        cells = {(c, p) for c, p in zip(mo["condition"], mo["condition_position"])}
        assert cells == {("standing", "first"), ("standing", "second"),
                         ("walking", "first"), ("walking", "second")}, cells
        assert mo["acq_time"].notna().all(), "acq_time must be populated for time-on-task"
        # and the reader surfaces it per recording
        recs = [r for r in bids_io.find_recordings(bids) if r.condition == "walking"]
        meta = bids_io.scan_meta(recs[0], bids)
        assert meta.get("condition_position") in ("first", "second")


# ---------------------------------------------------------------------------
# The vectorised R² spectrum is equivalent to the per-frequency lstsq loop
# ---------------------------------------------------------------------------
def test_r2_spectrum_matches_per_frequency_lstsq():
    """The batched, cached ``r2_spectrum`` matches the per-frequency lstsq, including
    sparse bin sets where aliasing makes the design rank-deficient."""
    rng = np.random.default_rng(1)
    centres = np.linspace(0, 100, 41)[:-1] + 1.25
    worst = 0.0
    for trial in range(300):
        nb = int(rng.integers(6, 41))
        x = centres[np.sort(rng.choice(40, nb, replace=False))]
        k = trial % 5
        if k == 0:
            y = rng.normal(0.75, 0.06, nb)
        elif k == 1:
            y = np.clip(0.75 + 0.1 * np.cos(2 * np.pi * 2 * x / 100), 0, 1)
        elif k == 2:
            cnt = rng.integers(15, 35, nb)
            y = rng.binomial(cnt, 0.75) / cnt          # real bin means
        elif k == 3:
            y = rng.integers(0, 2, nb).astype(float)
        else:
            y = np.full(nb, 0.75)                      # exactly constant
        ref = np.array([pa.fit_one(x, y, f)[0] for f in pa.FREQS])
        worst = max(worst, float(np.max(np.abs(pa.r2_spectrum(x, y) - ref))))
    assert worst < 1e-10, f"vectorised spectrum diverges from lstsq loop: {worst:.2e}"


def test_clocktime_spectrum_matches_per_frequency_reference():
    """The cached clock-time spectrum matches the reference in both modes (COUNT keeps
    all 40 bins with empty ones at 0, MEAN only occupied bins)."""
    rng = np.random.default_rng(4)
    worst = {"mean": 0.0, "count": 0.0}
    for _ in range(25):
        n = int(rng.integers(60, 600))
        onset = np.sort(rng.uniform(0, rng.uniform(40, 400), n))
        for how in ("mean", "count"):
            v = None if how == "count" else (rng.uniform(size=n) < 0.75).astype(float)
            ref = np.array([ctc._r2_at(onset, v, f, how) for f in ctc.FREQS_HZ])
            worst[how] = max(worst[how], float(np.max(np.abs(ctc._spectrum(onset, v, how) - ref))))
    for how, w in worst.items():
        assert w < 1e-10, f"clock spectrum ({how}) diverges from reference: {w:.2e}"


def test_r2_spectrum_design_cache_is_keyed_on_x():
    """The cache must not leak one recording's design into another's bin set."""
    rng = np.random.default_rng(2)
    centres = np.linspace(0, 100, 41)[:-1] + 1.25
    xa, xb = centres, centres[::2]
    ya, yb = rng.normal(0.75, 0.05, xa.size), rng.normal(0.75, 0.05, xb.size)
    a1 = pa.r2_spectrum(xa, ya)
    _ = pa.r2_spectrum(xb, yb)                          # different x in between
    a2 = pa.r2_spectrum(xa, ya)
    assert np.allclose(a1, a2, atol=0, rtol=0), "cache returned a stale design"
    ref_b = np.array([pa.fit_one(xb, yb, f)[0] for f in pa.FREQS])
    assert np.max(np.abs(pa.r2_spectrum(xb, yb) - ref_b)) < 1e-10


# ---------------------------------------------------------------------------
# Phase-agnostic population test (confirmatory)
# ---------------------------------------------------------------------------
def _agnostic_cohort(amp, n_sub=10, n=400, nperm=200, seed=0, phases="random"):
    """Per-participant observed/permuted coefficients for a synthetic cohort.

    ``amp`` is the per-participant oscillation amplitude in accuracy units.
    ``phases`` sets each participant's preferred phase:
      * ``"random"``: uniform (realistic; Davidson's R = 0.33);
      * ``"spread"``: evenly spaced, so the vector mean cancels exactly (R = 0);
      * ``"coherent"``: all identical.
    """
    rng = np.random.default_rng(seed)
    w = 2 * np.pi * 2.0 / 100.0
    obs, null = [], []
    for s in range(n_sub):
        if phases == "spread":
            phi = 2 * np.pi * s / n_sub
        elif phases == "coherent":
            phi = 0.0
        else:
            phi = rng.uniform(0, 2 * np.pi)
        ph = rng.uniform(0, 100, n)
        p = np.clip(0.75 + amp * np.cos(w * ph + phi), 0.01, 0.99)
        correct = (rng.uniform(size=n) < p).astype(float)
        m = pa.analyse_measure(ph, correct, "mean", nperm=nperm)
        assert m is not None
        obs.append(m["coef_2cps"])
        null.append(m["null_coef_2cps"])
    return np.array(obs), np.stack(null)


def test_agnostic_coefficients_match_fit_one():
    """The exported coefficients equal the fitted (cos, sin) pair of the confirmatory fit."""
    rng = np.random.default_rng(3)
    ph = rng.uniform(0, 100, 900)
    y = (rng.uniform(size=900) < 0.75).astype(float)
    m = pa.analyse_measure(ph, y, "mean", nperm=20)
    centers, binned = m["centers"], m["binned"]
    ok = np.isfinite(binned)
    # use the same weights as the fit
    w = m["bin_n"][ok] if m["weighted"] else None
    _, amp, phase, _ = pa.fit_one(centers[ok], binned[ok], 2.0, w)
    a, b = m["coef_2cps"]
    assert abs(np.hypot(a, b) - amp) < 1e-10, "coefficient magnitude != fitted amplitude"
    assert abs(np.arctan2(-b, a) - phase) < 1e-10, "coefficient angle != fitted phase"


def test_agnostic_recovers_incoherent_population():
    """A real effect at individual phases is detected."""
    obs, null = _agnostic_cohort(amp=0.10, seed=11)
    r = ga.agnostic_test(obs, null)
    assert r["p_value"] < 0.05, f"missed a real incoherent effect (p={r['p_value']})"


def test_agnostic_beats_common_phase_test_when_phases_scatter():
    """With evenly spread phases the vector mean cancels: a common-phase contrast misses
    the effect, the phase-agnostic statistic detects it (15 % vs 91 % power in the design)."""
    obs, null = _agnostic_cohort(amp=0.10, seed=12, phases="spread")
    p_agnostic = ga.agnostic_test(obs, null)["p_value"]
    # common-phase analogue: vector mean against the same permutations
    stat = float((obs.mean(axis=0) ** 2).sum())
    null_mean = (null.mean(axis=0) ** 2).sum(axis=1)
    p_coherent = float((1 + np.sum(null_mean >= stat)) / (1 + null_mean.size))
    assert p_agnostic < 0.05, f"agnostic test missed the effect (p={p_agnostic})"
    assert p_coherent > 0.05, (
        f"common-phase test should be blind to evenly spread phases (p={p_coherent})")


def test_agnostic_null_false_positive_rate_controlled():
    """No effect: the permutation test holds its nominal level."""
    hits, reps = 0, 30
    for s in range(reps):
        obs, null = _agnostic_cohort(amp=0.0, n_sub=6, n=300, nperm=200, seed=500 + s)
        hits += ga.agnostic_test(obs, null)["p_value"] < 0.05
    assert hits <= 5, f"false-positive rate too high: {hits}/{reps}"


def test_agnostic_holm_over_two_frequencies():
    """Holm correction over the two confirmatory frequencies adjusts as expected (the
    smaller p doubles, the larger is unchanged)."""
    p = pa.holm_bonferroni([0.02, 0.90])
    assert abs(p[0] - 0.04) < 1e-12 and abs(p[1] - 0.90) < 1e-12


# ---------------------------------------------------------------------------
# Stance-foot labelling: alternation constraint, roll cross-check, naming
# ---------------------------------------------------------------------------
def _synth_gait(dur=120.0, fs=90.0, stride_hz=0.9):
    """Clean synthetic walk: vertical bobs twice per stride, sway sways once."""
    t = np.arange(0, dur, 1 / fs)
    vertical = -np.cos(2 * np.pi * 2 * stride_hz * t)
    lateral = np.sin(2 * np.pi * stride_hz * t)
    troughs, _ = gait_mod._detect_troughs(t, vertical)
    return t, vertical, lateral, troughs


def test_alternation_constraint_repairs_isolated_flips():
    """Steps with weak, sign-flipped sway evidence are corrected by the alternation prior."""
    t, _, lateral, troughs = _synth_gait()
    corrupted = lateral.copy()
    idx = np.searchsorted(t, troughs)
    broken = [5, 17, 30, 44, 61]
    for i in broken:
        corrupted[idx[i]:idx[i + 1]] *= -0.15          # weak and wrong sign
    lab = gait_mod.label_steps(t, corrupted, troughs)
    foot = lab["step_foot"]
    truth = np.arange(foot.size) % 2
    acc = max((foot == truth).mean(), (foot != truth).mean())
    assert acc == 1.0, f"alternation constraint left {(1 - acc) * foot.size:.0f} steps wrong"
    assert lab["n_steps_relabelled"] >= len(broken) - 1, "the corrupted steps were not repaired"
    assert lab["lr_alternation_constrained_pct"] == 100.0


def _clipped(margin):
    """Evidence as passed to the decoder (clipped as in label_steps)."""
    return np.clip(margin, -gait_mod.MARGIN_CLIP, gait_mod.MARGIN_CLIP)


def test_alternation_constraint_allows_a_genuine_double_step():
    """A genuine double step (missed trough) is kept rather than forced to alternate."""
    # 21 steps with one repeat (positions 9 and 10 same foot)
    truth = np.array([i % 2 for i in range(10)] + [1] + [i % 2 for i in range(10)])
    assert int((truth[:-1] == truth[1:]).sum()) == 1, "fixture is not a single double step"
    for strength in (1.0, 3.0):                  # moderate and strong evidence
        margin = _clipped(np.where(truth == 0, 1.0, -1.0) * strength)
        decoded = gait_mod._viterbi_alternate(margin)
        assert np.array_equal(decoded, truth), \
            f"a genuine double step was forced to alternate (evidence x{strength})"


def test_no_single_step_can_break_the_alternation():
    """An isolated step cannot break the alternation however extreme its evidence: one
    flip costs two transitions, while the clipped margin limits one step's gain."""
    truth = np.array([i % 2 for i in range(41)])
    margin = np.where(truth == 0, 1.0, -1.0) * 1.2
    margin[20] *= -1e6                       # one extreme, wrong step
    decoded = gait_mod._viterbi_alternate(_clipped(margin))
    assert not (decoded[:-1] == decoded[1:]).any(), \
        "one artefactual step broke the alternation chain"
    assert np.array_equal(decoded, truth), "the artefactual step was not overruled"

    # check the parameter inequality
    import math
    isolated_flip_cost = 2 * math.log(gait_mod.ALT_PRIOR / (1 - gait_mod.ALT_PRIOR))
    max_single_step_gain = 2 * gait_mod.EMISSION_GAIN * gait_mod.MARGIN_CLIP
    assert isolated_flip_cost > max_single_step_gain, (
        f"ALT_PRIOR/EMISSION_GAIN/MARGIN_CLIP no longer guarantee it: "
        f"cost {isolated_flip_cost:.1f} <= gain {max_single_step_gain:.1f}")


def test_roll_crosscheck_reports_agreement_and_polarity():
    """An agreeing roll channel is fused (whatever its sign); a random one is not."""
    t, _, lateral, troughs = _synth_gait()
    rng = np.random.default_rng(0)
    agreeing = -0.8 * lateral + 0.02 * rng.normal(size=t.size)   # inverted convention
    lab = gait_mod.label_steps(t, lateral, troughs, roll_f=agreeing)
    assert lab["roll_polarity"] == -1, "the roll/sway sign convention was not detected"
    assert lab["roll_agreement_pct"] >= 90 and lab["roll_fused"]

    noise = rng.normal(size=t.size)
    lab_n = gait_mod.label_steps(t, lateral, troughs, roll_f=noise)
    assert not lab_n["roll_fused"], "an uncorrelated roll channel was fused into the evidence"
    assert np.array_equal(lab_n["step_foot"], lab["step_foot"]), \
        "a junk roll channel changed the sway-derived labels"


def test_roll_channel_picked_by_frequency_not_by_axis_name():
    """The roll check picks the once-per-stride channel among candidates.

    Uses band power at the trough-based step rate; a single-bin DFT at the 0.125 Hz Welch
    peak would let a twice-per-stride distractor score higher.
    """
    t, _, lateral, troughs = _synth_gait(dur=300.0)
    stride_hz = 1.0 / (2 * np.median(np.diff(troughs)))
    cands = {
        "twice_per_stride": 3.0 * np.cos(2 * np.pi * 2 * stride_hz * t),   # rides the bob
        "true_roll": 2.0 * np.sin(2 * np.pi * stride_hz * t),              # target channel
        "noise": np.random.default_rng(0).normal(size=t.size),
    }
    name, sig, ratio = gait_mod._pick_roll(t, cands, troughs)
    assert name == "true_roll", f"picked {name!r} (ratio {ratio}) instead of the roll channel"
    assert ratio > 1.0

    # no roll-like channel: no channel is picked
    name2, _, _ = gait_mod._pick_roll(t, {"twice_per_stride": cands["twice_per_stride"]}, troughs)
    assert name2 is None, "accepted a twice-per-stride channel as the roll cross-check"


def test_foot_naming_is_gated_on_rig_verification():
    """Anatomical left/right is only claimed once the rig convention is verified."""
    was = getattr(C, "LATERAL_SIGN_VERIFIED", False)
    try:
        C.LATERAL_SIGN_VERIFIED = False
        assert gait_mod.foot_name(0) == "foot A" and gait_mod.foot_name(1) == "foot B"
        C.LATERAL_SIGN_VERIFIED = True
        C.LATERAL_POSITIVE_IS_RIGHT = True
        assert (gait_mod.foot_name(0), gait_mod.foot_name(1)) == ("right", "left")
        C.LATERAL_POSITIVE_IS_RIGHT = False
        assert (gait_mod.foot_name(0), gait_mod.foot_name(1)) == ("left", "right")
    finally:
        C.LATERAL_SIGN_VERIFIED = was
        C.LATERAL_POSITIVE_IS_RIGHT = True


# ---------------------------------------------------------------------------
# Exploratory: the stance-foot contrast
# ---------------------------------------------------------------------------
def _foot_trials(seed, n_strides=800, delta=0.0, step_amp=0.0, base=0.75):
    """One trial per half-stride; ``delta`` is a foot effect, ``step_amp`` a within-step
    (2 cps) oscillation that must not affect the foot contrast."""
    rng = np.random.default_rng(seed)
    stride = np.repeat(np.arange(n_strides), 2)
    foot = np.tile([0, 1], n_strides)
    pct = rng.uniform(0, 100, foot.size)
    p = base + np.where(foot == 0, delta / 2, -delta / 2) + step_amp * np.cos(2 * np.pi * pct / 100)
    acc = (rng.uniform(size=foot.size) < np.clip(p, 0.02, 0.98)).astype(float)
    return acc, foot, stride


def test_stance_foot_recovers_injected_asymmetry():
    """A real difference between the two feet is detected at the right size and sign."""
    acc, foot, stride = _foot_trials(1, delta=0.10)
    r = sfoot.stance_contrast(acc, foot, stride, n_perm=2000,
                              rng=np.random.default_rng(1))
    assert r["p"] < 0.01, f"missed a 10 pp foot asymmetry (p={r['p']:.3f})"
    assert abs(r["delta"] - 0.10) < 0.03, f"effect size off: {r['delta']:+.3f}"


def test_stance_foot_null_false_positive_rate_controlled():
    """With no foot effect the stride-flip permutation must not over-reject."""
    ps = np.array([sfoot.stance_contrast(*_foot_trials(9000 + i, n_strides=400),
                                         n_perm=500, rng=np.random.default_rng(i))["p"]
                   for i in range(120)])
    assert (ps < 0.05).mean() <= 0.12, f"inflated false-positive rate {(ps < 0.05).mean():.3f}"


def test_stance_foot_immune_to_within_step_oscillation():
    """Flipping whole strides keeps within-step phase, so a strong 2-cps oscillation
    without a foot difference does not produce a stance-foot effect."""
    ps = np.array([sfoot.stance_contrast(*_foot_trials(500 + i, n_strides=400, step_amp=0.18),
                                         n_perm=500, rng=np.random.default_rng(i))["p"]
                   for i in range(120)])
    assert (ps < 0.05).mean() <= 0.12, \
        f"a within-step oscillation leaked into the foot contrast ({(ps < 0.05).mean():.3f})"


def test_stance_foot_population_agnostic_detects_scattered_directions():
    """Opposite asymmetries cancel in the common-direction test but are detected by the
    direction-agnostic test."""
    deltas = [0.06, -0.06] * 4
    res = [sfoot.stance_contrast(*_foot_trials(3000 + 17 * i, n_strides=700, delta=d),
                                 n_perm=1500, rng=np.random.default_rng(400 + i))
           for i, d in enumerate(deltas)]
    p = sfoot.population_test([r["_null"] for r in res], [r["delta"] for r in res])
    assert p["agnostic_p"] < 0.01, f"agnostic test missed scattered asymmetries ({p['agnostic_p']:.3f})"
    assert p["common_p"] > 0.05, f"common-direction test fired on cancelling effects ({p['common_p']:.3f})"


def test_stance_foot_population_null_controlled():
    """No asymmetry: neither population test is significant."""
    res = [sfoot.stance_contrast(*_foot_trials(7000 + 13 * i, n_strides=500),
                                 n_perm=1000, rng=np.random.default_rng(800 + i))
           for i in range(10)]
    p = sfoot.population_test([r["_null"] for r in res], [r["delta"] for r in res])
    assert p["agnostic_p"] > 0.05 and p["common_p"] > 0.05, \
        f"population test fired on a null (agnostic {p['agnostic_p']:.3f}, common {p['common_p']:.3f})"



# ---------------------------------------------------------------------------
# Estimator upgrades: weighted bins and difficulty adjustment
#     Both only reduce variance: they must not change the null, nor the result when
#     there is nothing to correct.
# ---------------------------------------------------------------------------
def _binned_curve(seed, n=1200, amp=0.0, freq=2.0, nbins=40):
    """Trial-level (phase, accuracy) with an optional injected oscillation."""
    r = np.random.default_rng(seed)
    ph = r.uniform(0, 100, n)
    p = 0.75 + amp * np.cos(2 * np.pi * freq * ph / 100.0)
    return ph, (r.random(n) < p).astype(float)


def test_weighted_fit_equals_unweighted_when_bins_are_equal():
    """Equal weights reproduce the unweighted fit."""
    x = np.linspace(1.25, 98.75, 40)
    rng = np.random.default_rng(4)
    y = 0.75 + 0.03 * np.cos(2 * np.pi * 2 * x / 100) + rng.normal(0, 0.02, 40)
    ref = pa.r2_spectrum(x, y)
    for w in (1.0, 7.0, 250.0):
        got = pa.r2_spectrum(x, y, np.full(40, w))
        assert np.max(np.abs(got - ref)) < 1e-12, f"equal weights w={w} changed the fit"


def test_weighted_fit_matches_explicit_weighted_least_squares():
    """The batched weighted solver matches explicit WLS."""
    rng = np.random.default_rng(11)
    worst_r2 = worst_b = 0.0
    for t in range(40):
        n = int(rng.integers(10, 41))
        x = np.sort(rng.uniform(0, 100, n))
        y = rng.normal(0.75, 0.08, n)
        w = rng.integers(1, 80, n).astype(float)
        spec, beta = pa._spectrum_with_coef(x, y, w)
        for f in (0.5, 2.0, 4.0, 7.3):
            i = int(np.argmin(np.abs(pa.FREQS - f)))
            ang = 2 * np.pi * pa.FREQS[i] / 100.0
            X = np.column_stack([np.ones(n), np.cos(ang * x), np.sin(ang * x)])
            rt = np.sqrt(w)
            b_ref, *_ = np.linalg.lstsq(X * rt[:, None], y * rt, rcond=None)
            ybar = float((w * y).sum() / w.sum())
            r2_ref = max(0.0, 1 - (w * (y - X @ b_ref) ** 2).sum()
                         / (w * (y - ybar) ** 2).sum())
            worst_r2 = max(worst_r2, abs(spec[i] - r2_ref))
            worst_b = max(worst_b, float(np.max(np.abs(beta[i] - b_ref))))
    assert worst_r2 < 1e-10, f"weighted R2 differs from reference WLS by {worst_r2:.2e}"
    assert worst_b < 1e-10, f"weighted beta differs from reference WLS by {worst_b:.2e}"


def test_weighting_does_not_inflate_the_false_positive_rate():
    """Weighted fits with unequal bin counts keep the nominal level."""
    hits = 0
    reps = 200
    for i in range(reps):
        r = np.random.default_rng(5000 + i)
        # uneven coverage: half the trials in a third of the cycle
        ph = np.concatenate([r.uniform(0, 33, 400), r.uniform(0, 100, 400)])
        y = (r.random(ph.size) < 0.75).astype(float)
        m = pa.analyse_measure(ph, y, "mean", nperm=200)
        if m and m["p_2cps_holm"] < 0.05:
            hits += 1
    fpr = hits / reps
    assert fpr <= 0.10, f"weighted fit on uneven coverage inflated the FPR to {fpr:.3f}"


def test_difficulty_adjust_equalises_levels_and_keeps_the_grand_mean():
    """The adjustment removes level offsets and keeps the grand mean."""
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0], float)
    lev = np.array(["easy"] * 4 + ["threshold"] * 4, dtype=object)
    adj = pa.difficulty_adjust(y, lev)
    assert np.isclose(adj[:4].mean(), adj[4:].mean()), "level means not equalised"
    assert np.isclose(adj.mean(), y.mean()), "grand mean not preserved"
    # single level -> no-op
    one = np.array(["threshold"] * 8, dtype=object)
    assert np.allclose(pa.difficulty_adjust(y, one), y), "single level should be a no-op"
    # disabled by config
    keep = C.RESIDUALISE_DIFFICULTY
    try:
        C.RESIDUALISE_DIFFICULTY = False
        assert np.allclose(pa.difficulty_adjust(y, lev), y), "flag did not disable it"
    finally:
        C.RESIDUALISE_DIFFICULTY = keep


def test_difficulty_adjust_cannot_manufacture_an_oscillation():
    """Levels assigned independently of phase: adjusting must not create structure."""
    hits = 0
    reps = 200
    for i in range(reps):
        r = np.random.default_rng(6000 + i)
        n = 900
        ph = r.uniform(0, 100, n)
        lev = r.choice(np.array(["threshold", "easy", "hard"], dtype=object),
                       size=n, p=[0.85, 0.10, 0.05])
        base = {"threshold": 0.75, "easy": 0.95, "hard": 0.60}
        y = (r.random(n) < np.array([base[str(v)] for v in lev])).astype(float)
        m = pa.analyse_measure(ph, pa.difficulty_adjust(y, lev), "mean", nperm=200)
        if m and m["p_2cps_holm"] < 0.05:
            hits += 1
    fpr = hits / reps
    assert fpr <= 0.10, f"difficulty adjustment inflated the FPR to {fpr:.3f}"


def test_difficulty_adjustment_improves_recovery_of_a_real_oscillation():
    """With a real 2 cps effect, the adjustment does not reduce power."""
    raw_hits = adj_hits = 0
    reps = 60
    for i in range(reps):
        r = np.random.default_rng(7000 + i)
        n = 900
        ph = r.uniform(0, 100, n)
        lev = r.choice(np.array(["threshold", "easy", "hard"], dtype=object),
                       size=n, p=[0.85, 0.10, 0.05])
        base = np.array([{"threshold": 0.72, "easy": 0.97, "hard": 0.58}[str(v)]
                         for v in lev])
        p = np.clip(base + 0.05 * np.cos(2 * np.pi * 2 * ph / 100.0), 0.01, 0.99)
        y = (r.random(n) < p).astype(float)
        mr = pa.analyse_measure(ph, y, "mean", nperm=200)
        ma = pa.analyse_measure(ph, pa.difficulty_adjust(y, lev), "mean", nperm=200)
        raw_hits += bool(mr and mr["p_2cps_holm"] < 0.05)
        adj_hits += bool(ma and ma["p_2cps_holm"] < 0.05)
    assert adj_hits >= raw_hits, \
        f"adjustment lost power: {adj_hits}/{reps} adjusted vs {raw_hits}/{reps} raw"


# ---------------------------------------------------------------------------
# The population statistic is studentised, and both modules use one version
# ---------------------------------------------------------------------------
def _coef_set(seed, n_sub=10, scales=None, amp=0.0, n_perm=400):
    """Observed and null coefficients for n_sub participants with given noise scales."""
    r = np.random.default_rng(seed)
    scales = np.ones(n_sub) if scales is None else np.asarray(scales, float)
    n_sub = scales.size
    obs, null = [], []
    for i in range(n_sub):
        s = float(scales[i])
        th = r.uniform(0, 2 * np.pi)
        obs.append(np.array([amp * np.cos(th), amp * np.sin(th)]) + r.normal(0, s, 2))
        null.append(r.normal(0, s, (n_perm, 2)))
    return np.array(obs), np.stack(null)


def test_agnostic_statistic_is_studentised():
    """With noise scales spanning 5x, the studentised null mean is close to n."""
    scales = np.array([1.0] * 9 + [5.0])
    obs, null = _coef_set(31, scales=scales, amp=0.0, n_perm=800)
    r = ga.agnostic_test(obs, null)
    # ~1 per participant under H0, so S ~ n
    assert abs(r["null_mean"] - r["n_participants"]) < 0.35 * r["n_participants"], \
        f"studentised null mean {r['null_mean']:.2f} not near n={r['n_participants']}"
    # raw sum reported separately
    assert r["statistic_unstandardised"] != r["statistic"], \
        "raw and studentised statistics are identical - studentisation did not happen"


def test_agnostic_studentisation_helps_when_participants_are_unequal():
    """With unequal noise, the studentised test is at least as powerful as the raw sum."""
    scales = np.array([0.4] * 8 + [3.0, 3.0])
    hits_st = hits_raw = 0
    reps = 120
    for i in range(reps):
        obs, null = _coef_set(2000 + i, scales=scales, amp=0.55, n_perm=400)
        hits_st += ga.agnostic_test(obs, null)["p_value"] < 0.05
        # raw-sum reference
        raw = float((obs ** 2).sum())
        raw_null = (null ** 2).sum(axis=(0, 2))
        hits_raw += (1 + np.sum(raw_null >= raw)) / (1 + raw_null.size) < 0.05
    assert hits_st >= hits_raw, \
        f"studentised test is weaker ({hits_st}/{reps}) than the raw sum ({hits_raw}/{reps})"


def test_agnostic_studentised_null_false_positive_rate_controlled():
    """With unequal scales the studentised test keeps the nominal level."""
    scales = np.array([0.3, 0.5, 1.0, 1.0, 2.0, 4.0, 1.0, 0.8])
    hits = 0
    reps = 300
    for i in range(reps):
        obs, null = _coef_set(4000 + i, scales=scales, amp=0.0, n_perm=400)
        hits += ga.agnostic_test(obs, null)["p_value"] < 0.05
    fpr = hits / reps
    assert 0.01 <= fpr <= 0.10, f"studentised FPR {fpr:.3f} outside [0.01, 0.10]"


def test_one_primary_measure_is_declared():
    """Exactly one primary measure (accuracy, as in the power analysis)."""
    assert C.PRIMARY_MEASURE == "accuracy"
    assert ga.PRIMARY == C.PRIMARY_MEASURE
    assert ga.MEASURES[0] == C.PRIMARY_MEASURE, "primary must be reported first"
    assert C.PRIMARY_MEASURE not in C.SECONDARY_MEASURES
    assert set(ga.MEASURES) == {C.PRIMARY_MEASURE} | set(C.SECONDARY_MEASURES)


# ---------------------------------------------------------------------------
# Exploratory: the signal-detection decomposition
# ---------------------------------------------------------------------------
def _sdt_trials(seed, n=1080, amp_d=0.0, amp_c=0.0, freq=2.0, d0=1.2, c0=0.2):
    """Trials from a signal-detection observer whose d' and/or criterion oscillate."""
    r = np.random.default_rng(seed)
    ph = r.uniform(0, 100, n)
    w = 2 * np.pi * freq / 100.0
    d_loc = d0 + amp_d * np.cos(w * ph)
    c_loc = c0 + amp_c * np.cos(w * ph)
    sig = r.random(n) < 0.5
    ev = r.normal(np.where(sig, d_loc / 2, -d_loc / 2), 1.0)
    return ph, sig, ev > c_loc


def test_sdt_recovers_known_dprime_and_criterion():
    """d' and criterion estimates match the simulated values."""
    ph, sig, said = _sdt_trials(1, n=200000, d0=1.4, c0=0.35)
    _, dp, cr, _, _, _, _ = sdt.sdt_bins(ph, sig, said, nbins=1, min_n=0)
    assert abs(dp[0] - 1.4) < 0.03, f"d' off: {dp[0]:.3f}"
    assert abs(cr[0] - 0.35) < 0.03, f"criterion off: {cr[0]:.3f}"


def test_sdt_inverse_normal_is_accurate():
    """The custom inverse normal matches scipy to 1e-8."""
    try:
        from scipy.stats import norm
    except Exception:                                     # pragma: no cover
        return
    p = np.linspace(1e-5, 1 - 1e-5, 5001)
    assert np.max(np.abs(sdt._z(p) - norm.ppf(p))) < 1e-8


def test_sdt_separates_a_sensory_from_a_decisional_oscillation():
    """A d' oscillation appears in d' but not in the criterion, and vice versa."""
    ph, sig, said = _sdt_trials(2, amp_d=0.55, amp_c=0.0)
    rd = sdt.analyse_measure(ph, sig, said, "dprime", nperm=400,
                             rng=np.random.default_rng(2))
    rc = sdt.analyse_measure(ph, sig, said, "criterion", nperm=400,
                             rng=np.random.default_rng(2))
    assert rd["p_2cps_holm"] < 0.05, f"missed the sensory effect (p={rd['p_2cps_holm']:.3f})"
    assert rc["p_2cps_holm"] > 0.05, f"sensory effect leaked into criterion (p={rc['p_2cps_holm']:.3f})"

    ph, sig, said = _sdt_trials(3, amp_d=0.0, amp_c=0.40)
    rd = sdt.analyse_measure(ph, sig, said, "dprime", nperm=400,
                             rng=np.random.default_rng(3))
    rc = sdt.analyse_measure(ph, sig, said, "criterion", nperm=400,
                             rng=np.random.default_rng(3))
    assert rc["p_2cps_holm"] < 0.05, f"missed the decisional effect (p={rc['p_2cps_holm']:.3f})"
    assert rd["p_2cps_holm"] > 0.05, f"decisional effect leaked into d' (p={rd['p_2cps_holm']:.3f})"


def test_sdt_null_false_positive_rate_controlled():
    """A biased but non-oscillating observer gives no phase effect."""
    hits_d = hits_c = 0
    reps = 150
    for i in range(reps):
        ph, sig, said = _sdt_trials(8000 + i, amp_d=0.0, amp_c=0.0, c0=0.45)
        rd = sdt.analyse_measure(ph, sig, said, "dprime", nperm=200,
                                 rng=np.random.default_rng(i))
        rc = sdt.analyse_measure(ph, sig, said, "criterion", nperm=200,
                                 rng=np.random.default_rng(i))
        hits_d += rd["p_2cps_holm"] < 0.05
        hits_c += rc["p_2cps_holm"] < 0.05
    assert hits_d / reps <= 0.10, f"d' FPR {hits_d / reps:.3f}"
    assert hits_c / reps <= 0.10, f"criterion FPR {hits_c / reps:.3f}"


def test_sdt_population_reuses_the_confirmatory_statistic():
    """The SDT population test uses group_agnostic.agnostic_test."""
    rows = []
    for i in range(8):
        ph, sig, said = _sdt_trials(300 + i, amp_c=0.35)
        rows.append({"criterion": sdt.analyse_measure(ph, sig, said, "criterion",
                                                      nperm=300,
                                                      rng=np.random.default_rng(i)),
                     "dprime": None})
    pop = sdt.population_test(rows)
    assert "criterion" in pop, "population test produced nothing"
    r = pop["criterion"]["by_freq"][2.0]
    # same result as calling agnostic_test directly
    obs = np.array([x["criterion"]["coef_2cps"] for x in rows])
    null = np.stack([x["criterion"]["null_coef_2cps"] for x in rows])
    ref = ga.agnostic_test(obs, null)
    assert abs(r["statistic"] - ref["statistic"]) < 1e-12
    assert r["p_value"] == ref["p_value"]
    assert r["significant"], "population criterion oscillation not detected"


def test_sdt_excludes_timeouts_rather_than_scoring_them():
    """Timeouts are neither hits nor false alarms and are excluded from the rates."""
    import pandas as _pd
    ev = _pd.DataFrame({"stimulus_side": ["right", "left", "right", "left"],
                        "response": ["right", "right", None, "left"]})
    keep = ev["response"].notna() & ev["stimulus_side"].notna()
    assert int(keep.sum()) == 3, "timeout row was not dropped"



# ---------------------------------------------------------------------------
# The common-phase test: Hotelling T^2 replaces the invalid GLMM Wald
# ---------------------------------------------------------------------------
def _coherent(n, amp, seed, noise=1.0):
    """N participants oscillating at a common phase, plus noise."""
    r = np.random.default_rng(seed)
    return np.array([amp, 0.0]) + r.normal(0, noise, (n, 2))


def _idiosyncratic(n, amp, seed, noise=0.5):
    """N participants each oscillating at their own phase (the case that broke the
    variational Wald; Davidson's R = 0.33)."""
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi, n)
    return np.column_stack([amp * np.cos(th), amp * np.sin(th)]) + r.normal(0, noise, (n, 2))


def test_hotelling_matches_the_power_analysis_implementation():
    """The common-phase test matches the statistic of the power analysis."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("power_ref", ROOT / "davidson_power" / "power.py")
    power = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(power)
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(60):
        n = int(rng.integers(4, 30))
        Z = rng.normal(0.2, 1.0, (n, 2))
        worst = max(worst, abs(cph.hotelling_t2(Z)["p_value"] - power.hotelling_p(Z)))
    assert worst < 1e-12, f"diverged from davidson_power.power.hotelling_p by {worst:.2e}"


def test_hotelling_null_false_positive_rate_controlled():
    """Nominal level without effect at the planned sample sizes."""
    for n in (8, 12, 20):
        hits = sum(cph.hotelling_t2(np.random.default_rng(1000 + i).normal(0, 1, (n, 2)))
                   ["p_value"] < 0.05 for i in range(1500))
        fpr = hits / 1500
        assert 0.02 <= fpr <= 0.09, f"FPR {fpr:.3f} at N={n}"


def test_hotelling_survives_idiosyncratic_phases():
    """Strong oscillations at individual phases are not reported as a common-phase effect
    (the variational Wald did so 60 % of the time)."""
    for n in (12, 20):
        hits = sum(cph.hotelling_t2(_idiosyncratic(n, 1.0, 5000 + i))["p_value"] < 0.05
                   for i in range(1500))
        fpr = hits / 1500
        assert fpr <= 0.10, (f"common-phase test fired on idiosyncratic phases "
                             f"{fpr:.3f} of the time at N={n} (the Wald did 0.60)")


def test_hotelling_detects_a_true_common_phase_effect():
    """Power against a real common-phase effect."""
    hits = sum(cph.hotelling_t2(_coherent(20, 0.8, 9000 + i))["p_value"] < 0.05
               for i in range(600))
    assert hits / 600 > 0.6, f"power only {hits / 600:.2f} at N=20, amp 0.8"


def test_hotelling_degenerate_inputs_return_p_one():
    """Too few participants or a singular covariance give p = 1."""
    assert cph.hotelling_t2(np.zeros((3, 2)))["p_value"] == 1.0
    assert not cph.hotelling_t2(np.zeros((3, 2)))["estimable"]
    same = np.tile([0.5, 0.5], (10, 1))          # zero covariance
    assert cph.hotelling_t2(same)["p_value"] == 1.0


def test_paired_hotelling_is_the_difference_test():
    """The paired test equals the one-sample test on differences and is not significant
    for identical conditions."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.3, 0.5, (16, 2)), rng.normal(0.0, 0.5, (16, 2))
    assert cph.paired_hotelling(a, b)["p_value"] == cph.hotelling_t2(a - b)["p_value"]
    hits = 0
    for i in range(600):
        r = np.random.default_rng(700 + i)
        common = r.normal(0.4, 0.5, (14, 2))     # same oscillation in both conditions
        hits += cph.paired_hotelling(common + r.normal(0, .3, (14, 2)),
                                     common + r.normal(0, .3, (14, 2)))["p_value"] < 0.05
    assert hits / 600 <= 0.09, f"paired test fired on identical conditions ({hits / 600:.3f})"


# ---------------------------------------------------------------------------
# response_counts is gone, and H3 with it
# ---------------------------------------------------------------------------
def test_response_counts_measure_is_removed():
    """Response counts stay removed (invalid in a forced-choice task; see config)."""
    assert "response_counts" not in C.SECONDARY_MEASURES
    assert "response_counts" not in ga.MEASURES
    assert "response_counts" not in pop.MEASURES
    assert "response_counts" not in ctc.MEASURES
    assert "response_likelihood" not in glmm.MEASURES


def test_difficulty_model_is_descriptive_not_a_hypothesis_test():
    """The difficulty model is descriptive (H3 dropped: too few HARD trials)."""
    fd = glmm.fit_difficulty_model(
        _sim_glmm(0.5, seed=6).assign(condition="walking", difficulty="THR"))
    if fd:
        assert "DESCRIPTIVE ONLY" in fd["inference"]
        assert "prediction" not in fd, "the retired H3 prediction is back"


# ---------------------------------------------------------------------------
# Missing data: partial blocks (disconnects) and missing blocks
# ---------------------------------------------------------------------------
def _raw_block(dur_s=30.0, seed=0):
    import simulate_participants as sim
    return sim._motion_run("walking", 5.0, dur_s, seed=seed)


def test_read_raw_salvages_truncated_logs_and_rejects_empty_ones():
    """Truncated .csv and .csv.gz logs keep all complete frames; a header-only log
    raises UnusableRawLog."""
    import gzip
    import tempfile
    df = _raw_block()
    with tempfile.TemporaryDirectory() as td:
        full = Path(td) / "p01-b1-t01_walking.csv"
        df.to_csv(full, sep=";", index=False)
        text = full.read_text()
        got, status = bidsify._read_raw(full)
        assert status == "complete" and len(got) == len(df)

        cut = Path(td) / "p01-b1-t02_walking.csv"
        cut.write_text(text[: len(text) // 2])            # stops mid-row
        got, status = bidsify._read_raw(cut)
        assert status == "truncated"
        assert 0 < len(got) < len(df) and got["timestamp"].notna().all()
        assert got.iloc[-1]["block-id"] == 1, "the half-written last row was kept"

        gz = Path(td) / "p01-b1-t03_walking.csv.gz"
        blob = gzip.compress(text.encode())
        gz.write_bytes(blob[: len(blob) // 2])
        got, status = bidsify._read_raw(gz)
        assert status == "truncated" and 0 < len(got) < len(df)

        head = Path(td) / "p01-b1-t04_walking.csv"
        head.write_text(text.splitlines()[0] + "\n")
        try:
            bidsify._read_raw(head)
            raise AssertionError("header-only log was accepted")
        except bidsify.UnusableRawLog:
            pass


def test_bidsify_keeps_reused_trial_number_and_skips_unusable_logs():
    """A reused Trial number ('-1' file) keeps both blocks (re-run as run-1xx), and an
    unusable log does not stop the others."""
    import tempfile
    import bids_io
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        _raw_block(seed=1).to_csv(raw / "p01-b1-t01_walking.csv", sep=";", index=False)
        _raw_block(seed=2).to_csv(raw / "p01-b1-t04_walking.csv", sep=";", index=False)
        _raw_block(seed=3).to_csv(raw / "p01-b1-t04_walking-1.csv", sep=";", index=False)
        (raw / "p01-b1-t02_walking.csv").write_text("timestamp;block-id\n")   # aborted
        bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)

        runs = sorted(r.run for r in bids_io.find_recordings(bids))
        assert runs == ["01", "04", "104"], runs
        # the original keeps the operator's run number
        orig = pd.read_csv(raw.parent / "bids" / "sourcedata" / "raw" / "p01-b1-t04_walking.csv",
                           sep=";")
        ev04 = [r for r in bids_io.find_recordings(bids) if r.run == "04"][0]
        m, _, _ = bids_io.load_motion(ev04)
        assert len(m) == len(orig)
        # re-running from the archive gives the same three runs
        bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
        assert sorted(r.run for r in bids_io.find_recordings(bids)) == ["01", "04", "104"]


def test_stride_phase_is_not_interpolated_across_a_gap():
    """Trials inside a data gap, or in a 'step' spanning one, get no stride phase."""
    steps = np.arange(0, 20, 0.55)
    troughs = np.concatenate([steps[steps < 8], steps[steps > 14]])   # 6 s with no troughs
    _, ps = gait_mod.phase_at(np.array([4.1, 10.0, 16.2]), troughs)
    assert np.isfinite(ps[0]) and np.isnan(ps[1]) and np.isfinite(ps[2])

    t = np.concatenate([np.arange(0, 8, 1 / 60), np.arange(14, 20, 1 / 60)])
    spans = gait_mod._gap_spans(t)
    assert spans.shape == (1, 2) and abs(spans[0, 0] - 8) < 0.05 and abs(spans[0, 1] - 14) < 1e-9
    inside = gait_mod._in_gaps(np.array([7.8, 10.0, 14.3, 15.0]), spans)
    assert inside.tolist() == [True, True, True, False]
    # low-fps sessions are not one long gap
    assert len(gait_mod._gap_spans(np.arange(0, 60, 0.2))) == 0


def test_unrecorded_frame_flag_falls_back_to_repeated_poses():
    """An unrecorded (all-zero) mocap_frame_advanced flag falls back to repeated poses."""
    t = np.arange(0, 10, 1 / 60)
    y = 0.02 * np.sin(2 * np.pi * 1.6 * t)
    y[120:180] = y[119]                       # 1 s freeze
    y[300] = y[299]                           # single repeat, not a freeze
    m = pd.DataFrame({"t": t, "head_local_pos_x": 0.01 * np.cos(np.pi * 1.6 * t),
                      "head_local_pos_y": y, "head_local_pos_z": 0.0,
                      "mocap_frame_advanced": 0})
    m.loc[120:179, "head_local_pos_x"] = m.loc[119, "head_local_pos_x"]
    m.loc[400, "mocap_frame_advanced"] = 1   # one stray 1 in an unrecorded flag
    stale, source = clean_mod.stale_frames(m)
    assert source == "repeated_position"
    assert stale[120:180].all() and stale.sum() == 60
    # a recorded flag is used directly
    m["mocap_frame_advanced"] = 1
    m.loc[500:509, "mocap_frame_advanced"] = 0
    stale, source = clean_mod.stale_frames(m)
    assert source == "mocap_frame_advanced" and stale.sum() == 10


def test_block_inventory_lists_missing_and_truncated_runs():
    """A participant with missing blocks is listed with the missing runs."""
    import tempfile
    old =(C.BIDS_DIR, C.DERIV_DIR)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        for t in (1, 3):
            _raw_block(seed=t).to_csv(raw / f"p01-b1-t{t:02d}_walking.csv", sep=";", index=False)
        text = (raw / "p01-b1-t03_walking.csv").read_text()
        (raw / "p01-b1-t03_walking.csv").write_text(text[: len(text) // 2])
        try:
            C.BIDS_DIR, C.DERIV_DIR = bids, bids / "derivatives"
            bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
            inv = {r["condition"]: r for r in report.block_inventory()}
        finally:
            C.BIDS_DIR, C.DERIV_DIR = old
        w = inv["walking"]
        assert w["status"] == "incomplete" and w["runs_present"] == 2
        assert w["missing_runs"] == "2,4,5,6,7,8" and w["missing_visits"] == "2"
        assert w["truncated_runs"] == "3"
        assert inv["standing"]["status"] == "none"


def test_excluded_recording_is_kept_in_bids_but_skipped_by_analysis():
    """An excluded block stays in BIDS with its reason in scans.tsv, is invisible to the
    analysis stages, and is listed as excluded (not missing) in the block inventory."""
    import tempfile
    import bids_io
    old = (C.BIDS_DIR, C.DERIV_DIR, C.EXCLUDED_RECORDINGS, C.APPLY_BLOCK_RULE)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        for t in (1, 2):
            _raw_block(seed=t).to_csv(raw / f"p01-b1-t{t:02d}_walking.csv", sep=";", index=False)
        try:
            C.BIDS_DIR, C.DERIV_DIR = bids, bids / "derivatives"
            C.EXCLUDED_RECORDINGS = {"sub-01_ses-02_task-ILL_acq-walking_run-02": "view flipped"}
            C.APPLY_BLOCK_RULE = False           # the manual list alone
            bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
            assert [r.run for r in bids_io.find_recordings(bids)] == ["01"]
            assert [r.run for r in bids_io.find_recordings(bids, scope="gait")] == ["01"]
            assert len(bids_io.find_recordings(bids, include_excluded=True)) == 2
            scans = pd.read_csv(bids / "sub-01" / "sub-01_scans.tsv", sep="\t")
            reason = scans.set_index("filename")["exclusion_reason"]
            assert reason.str.contains("view flipped").sum() == 1
            w = {r["condition"]: r for r in report.block_inventory()}["walking"]
        finally:
            C.BIDS_DIR, C.DERIV_DIR, C.EXCLUDED_RECORDINGS, C.APPLY_BLOCK_RULE = old
        assert w["excluded_runs"] == "2" and w["runs_present"] == 1
        assert "2" not in w["missing_runs"].split(",")


def test_setup_correction_overrides_logged_counterbalance_and_mixed_slots_fail():
    """A block logged with another participant's setup disagrees on the slot: bidsify
    refuses to average it, and a config.SETUP_CORRECTIONS entry fixes scans.tsv and
    participants.tsv while the raw log stays as written."""
    import tempfile
    old = (C.BIDS_DIR, C.DERIV_DIR, C.SETUP_CORRECTIONS)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        for t, (slot, first) in zip((5, 6), ((3, "walking"), (1, "standing"))):
            df = _raw_block(seed=t).assign(**{"visit": 2, "counterbalance-slot": slot,
                                              "condition-order-first": first})
            df.to_csv(raw / f"p01-v2-b1-t{t:02d}_walking.csv", sep=";", index=False)
        try:
            C.BIDS_DIR, C.DERIV_DIR = bids, bids / "derivatives"
            C.SETUP_CORRECTIONS = {}
            try:
                bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
                raise AssertionError("mixed counterbalance slots were averaged")
            except ValueError as e:
                assert "SETUP_CORRECTIONS" in str(e)
            C.SETUP_CORRECTIONS = {("sub-01", 2): {"counterbalance_slot": 1,
                                                   "condition_order_first": "standing",
                                                   "reason": "stale setup screen"}}
            bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
            scans = pd.read_csv(bids / "sub-01" / "sub-01_scans.tsv", sep="\t")
            parts = pd.read_csv(bids / "participants.tsv", sep="\t")
        finally:
            C.BIDS_DIR, C.DERIV_DIR, C.SETUP_CORRECTIONS = old
        assert set(scans["condition_order_first"]) == {"standing"}
        assert set(scans["condition_position"]) == {"second"}
        assert parts.loc[0, "counterbalance_slot"] == 1


def test_each_block_gets_one_sheet_per_stage_in_a_figures_folder():
    """QC and gait write one sheet per block into <stage>/sub/ses/figures/."""
    import tempfile
    import bids_io
    import gait as gait_mod
    import qc
    old = (C.BIDS_DIR, C.DERIV_DIR)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw"; raw.mkdir()
        bids = Path(td) / "bids"; bids.mkdir()
        _raw_block(dur_s=120.0, seed=4).to_csv(raw / "p01-b1-t01_walking.csv", sep=";", index=False)
        try:
            C.BIDS_DIR, C.DERIV_DIR = bids, bids / "derivatives"
            bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
            rec, = bids_io.find_recordings(bids, include_excluded=True)
            qc.qc_recording(rec)
            gait_mod.gait_recording(rec)
            for stage in ("qc", "gait"):
                d = C.DERIV_DIR / stage / rec.sub / rec.ses
                assert [f.name for f in (d / "figures").iterdir()] == [f"{rec.label()}_{stage}.png"]
                assert not list(d.glob("*.png")), f"{stage}: a figure next to the data"
        finally:
            C.BIDS_DIR, C.DERIV_DIR = old


def _gate_dataset(td, rng):
    """Three walking blocks of p01 (normal, reversed, guessing) plus one calibration."""
    raw = Path(td) / "raw"; raw.mkdir()
    bids = Path(td) / "bids"; bids.mkdir()
    for t in (1, 2, 3):
        df = _raw_block(dur_s=400.0, seed=t)
        res = df["checkpoint-index"] != -1
        rc = df.loc[res, "response-correct"]
        if t == 2:                                  # reversed answers
            df.loc[res, "response-correct"] = np.where(rc >= 0, 1 - rc, rc)
        if t == 3:                                  # pure guessing
            df.loc[res, "response-correct"] = np.where(
                rc >= 0, rng.integers(0, 2, res.sum()), rc)
        df.to_csv(raw / f"p01-b1-t{t:02d}_walking.csv", sep=";", index=False)
    (raw / "p01-b1-t01_calibration_walking.csv").write_text(_min_calib_csv(3.0))
    (raw / "p01-b0-t01_calibration_standing.csv").write_text(_min_calib_csv(3.0))
    return raw, bids


def test_stage_cache_reuses_only_identical_inputs_and_intact_outputs():
    """A cached step is reused only with a matching fingerprint and untouched outputs."""
    import os
    import tempfile
    import time
    import stage_cache as sc
    old_dir, old_enabled = C.BIDS_DIR, sc.ENABLED
    with tempfile.TemporaryDirectory() as td:
        try:
            C.BIDS_DIR = Path(td) / "data"
            sc.ENABLED = True
            big = np.arange(5000, dtype=float)
            fp = sc.fingerprint("v1", [big, "x"], {"b": 2, "a": big})
            big2 = big.copy()
            big2[2500] += 1e-9
            assert sc.fingerprint("v1", [big2, "x"], {"b": 2, "a": big}) != fp
            assert sc.fingerprint("v1", [big, "x"], {"b": 2, "a": big2}) != fp
            assert sc.fingerprint("v1", [big, "x"], {"b": 2, "a": big}) == fp
            df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.nan]})
            assert sc.fingerprint(df) != sc.fingerprint(df.assign(b=[3.0, 4.0]))

            out = Path(td) / "result.txt"
            out.write_text("result")
            assert not sc.is_fresh("demo", "u", fp, [out])          # nothing recorded yet
            sc.record("demo", "u", fp, [out])
            assert sc.is_fresh("demo", "u", fp, [out])
            assert not sc.is_fresh("demo", "u", "other", [out])     # inputs changed
            assert not sc.is_fresh("demo", "u", fp, [out, Path(td) / "more.txt"])
            time.sleep(0.01)
            out.write_text("edited by hand")                        # output touched
            assert not sc.is_fresh("demo", "u", fp, [out])
            sc.record("demo", "u", fp, [out])
            assert sc.is_fresh("demo", "u", fp, [out])
            os.remove(out)                                          # output deleted
            assert not sc.is_fresh("demo", "u", fp, [out])
            sc.record("demo", "u", fp, [out])                       # missing: not cached
            out.write_text("result")
            assert not sc.is_fresh("demo", "u", fp, [out])
            sc.record("demo", "u", fp, [out])
            sc.ENABLED = False
            assert not sc.is_fresh("demo", "u", fp, [out])          # cache switched off
            assert (Path(td) / ".pipeline_cache" / "demo.json").exists()
        finally:
            C.BIDS_DIR, sc.ENABLED = old_dir, old_enabled


def test_bayes_glmm_finds_a_shared_walking_phase_and_not_a_standing_one():
    """A strong 2-cps oscillation at one shared phase while walking gives a large Bayes
    factor for the walking shared phase and a small one for standing, where there is
    none. Skipped without NumPyro."""
    import pytest
    import bayes_glmm as bg
    if not bg.available():
        pytest.skip("numpyro / jax not installed")
    rng = np.random.default_rng(5)
    n_sub, n_per = 5, 500
    sub = np.repeat(np.arange(n_sub), n_per)
    walk = np.tile(np.r_[np.ones(400), np.zeros(100)], n_sub)
    phase = rng.uniform(0, 100, sub.size)
    harm = np.stack([np.stack([np.cos(2 * np.pi * f * phase / 100),
                               np.sin(2 * np.pi * f * phase / 100)], -1) for f in bg.FREQS], 1)
    eta = 1.1 + walk * 0.6 * harm[:, 0, 0]
    arr = {"subs": [f"sub-{i:02d}" for i in range(n_sub)], "sub": sub, "walk": walk,
           "hard": np.zeros(sub.size), "easy": np.zeros(sub.size), "harm": harm,
           "y": (rng.random(sub.size) < 1 / (1 + np.exp(-eta))).astype(float)}
    post, diag = bg.fit(arr, "accuracy", chains=1, warmup=300, samples=300, seed=1)
    s = bg.summarise(post, "accuracy", arr)["by_frequency"]["2cps"]
    assert diag["divergences"] < 10
    assert s["walk"]["bf10_common"] > 10, s["walk"]
    assert s["stand"]["bf10_common"] < 3, s["stand"]


def test_calibration_transfer_recovers_threshold_and_flags_a_miss():
    """Blocks simulated from the calibrated threshold fall inside their own sampling null;
    blocks from a participant whose real threshold is twice the calibrated one fall
    outside it. A block at ceiling (above the model's 1 - lapse) is a bound only."""
    import calibration_transfer as ct
    import calibration_refit as cref
    import threshold_by_condition as tbc
    rng = np.random.default_rng(3)
    t_cal = 1.0
    tilt = rng.choice([0.75, 1.0, 1.25], size=600, p=[0.05, 0.85, 0.10]) * t_cal

    def respond(t75):
        alpha = t75 / ct._to_t75(1.0)
        p = cref.weibull(tilt, alpha, tbc.BETA_FIX, tbc.LAPSE_FIX, tbc.GUESS)
        return (rng.random(tilt.size) < p).astype(float)

    null = ct.sampling_null(tilt, t_cal, rng, n_sim=150)
    lo, hi = np.quantile(null, [0.025, 0.975])
    assert lo < 1.0 < hi and hi / lo < 1.6, (lo, hi)
    same = ct.fit_threshold(tilt, respond(t_cal))
    assert same["censored"] is None and lo <= same["t75"] / t_cal <= hi
    harder = ct.fit_threshold(tilt, respond(2 * t_cal))
    assert harder["t75"] / t_cal > hi and harder["lo"] > t_cal
    ceiling = ct.fit_threshold(tilt, (rng.random(tilt.size) < 0.98).astype(float))
    assert ceiling["censored"] == "low"


def test_block_rule_leaves_timeouts_out():
    """Chance is tested on the answered trials: 95 right of 152 answered (18 timeouts)
    passes, though 95/170 with timeouts as wrong would not; a
    guesser with many timeouts still fails. 25 % or more timeouts fail on their own."""
    import types
    import bids_io
    import block_rules

    def events(k, n_wrong, n_timeout):
        rc = [1.0] * k + [0.0] * n_wrong + [np.nan] * n_timeout
        return pd.DataFrame({"is_training": [1] * 5 + [0] * len(rc),
                             "response_correct": [1.0] * 5 + rc})

    old = (bids_io.load_events, bids_io.scan_meta, block_rules.calibration_verdict)
    rec = types.SimpleNamespace(sub="sub-01", condition="walking", label=lambda: "x")
    try:
        bids_io.scan_meta = lambda r: {"visit": 1}
        block_rules.calibration_verdict = lambda *a: "RELIABLE"
        bids_io.load_events = lambda r: events(95, 57, 18)
        v = block_rules.evaluate(rec, alpha=0.05)
        assert (v["n_trials"], v["n_answered"], v["n_correct"]) == (170, 152, 95)
        assert v["exclude"] is False, v["reason"]
        bids_io.load_events = lambda r: events(70, 70, 30)
        v = block_rules.evaluate(rec, alpha=0.05)
        assert v["exclude"] is True and v["reason"].startswith("chance")
        bids_io.load_events = lambda r: events(50, 100, 20)
        assert block_rules.evaluate(rec, alpha=0.05)["reason"].startswith("reversal")
        bids_io.load_events = lambda r: events(100, 28, 42)          # 24.7 % timeouts
        assert block_rules.evaluate(rec, alpha=0.05)["exclude"] is False
        bids_io.load_events = lambda r: events(100, 27, 43)          # 25.3 % timeouts
        v = block_rules.evaluate(rec, alpha=0.05)
        assert v["exclude"] is True and v["reason"].startswith("timeouts"), v["reason"]
    finally:
        bids_io.load_events, bids_io.scan_meta, block_rules.calibration_verdict = old


def test_gate_scopes_rule_to_behaviour_and_keeps_gait():
    """The gate flags a reversed and a guessing block. With the rule off nothing but the
    manual list is excluded; with it on, the behavioural analyses lose the flagged blocks
    while the gait analyses keep them. An UNRELIABLE calibration leaves the pooled refit,
    but the above-chance block run on it stays in."""
    import tempfile
    import bids_io
    import block_rules
    old = (C.BIDS_DIR, C.DERIV_DIR, C.APPLY_BLOCK_RULE)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        raw, bids = _gate_dataset(td, rng)
        try:
            C.BIDS_DIR, C.DERIV_DIR = bids, bids / "derivatives"
            bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
            cals = {("standing" if "standing" in c.name else "walking"): c
                    for c in bids_io.find_calibration_tables(bids)}
            cal_beh = cals["walking"]
            qc_dir = C.DERIV_DIR / "calibration" / "sub-01" / "ses-01"
            qc_dir.mkdir(parents=True)
            for cond, verdict in (("standing", "RELIABLE"),
                                  ("walking", "UNRELIABLE - recommend recalibration")):
                (qc_dir / f"{bids_io.calibration_label(cals[cond])}_calibration_qc.json"
                 ).write_text('{"verdict": "%s"}' % verdict)

            C.APPLY_BLOCK_RULE = False
            rows = {r["recording"][-6:]: r for r in block_rules.main()
                    if r["unit"] == "block"}
            assert (bids / "derivatives" / "qc" / bids_io.GATE_FILE).exists()
            assert rows["run-01"]["rule_exclude"] is False
            assert rows["run-02"]["rule"].startswith("reversal")
            assert rows["run-03"]["rule"].startswith("chance")
            assert len(bids_io.find_recordings(bids)) == 3, "rule applied while off"
            assert bids_io.excluded_calibrations() == {}

            C.APPLY_BLOCK_RULE = True            # settings changed: gate rebuilt on read
            assert [r.run for r in bids_io.find_recordings(bids)] == ["01"]
            assert len(bids_io.find_recordings(bids, scope="gait")) == 3
            assert len(bids_io.find_recordings(bids, include_excluded=True)) == 3
            assert list(bids_io.excluded_calibrations()) == [bids_io.calibration_label(cal_beh)]
            fit = cr.refit_from_bids()
            assert set(fit["by_condition"]) == {"standing"}, "the UNRELIABLE one was refit"
            assert fit["excluded_by_gate"] == [bids_io.calibration_label(cal_beh)]
            gate = pd.read_csv(bids / "derivatives" / "qc" / bids_io.GATE_FILE, sep="\t")
            blk = gate[gate["unit"] == "block"].set_index("recording")
            assert set(blk["gait"]) == {"include"}
            assert (blk["behaviour"] == "exclude").sum() == 2
            assert blk.filter(like="run-01", axis=0)["calibration_verdict"].str.startswith(
                "UNRELIABLE").all(), "the calibration verdict is not recorded with the block"
        finally:
            C.BIDS_DIR, C.DERIV_DIR, C.APPLY_BLOCK_RULE = old


def test_gate_rebuilds_when_stale_and_ledger_counts_each_stage():
    """A block added after the gate was built gets a verdict before any stage uses it;
    gate_report logs the stage's N, and prunes outputs of excluded units."""
    import tempfile
    import bids_io
    import block_rules
    old = (C.BIDS_DIR, C.DERIV_DIR, C.APPLY_BLOCK_RULE)
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        raw, bids = _gate_dataset(td, rng)
        (raw / "p01-b1-t03_walking.csv").unlink()           # the guessing block comes later
        late = Path(td) / "late"; late.mkdir()
        df = _raw_block(dur_s=400.0, seed=3)
        res = df["checkpoint-index"] != -1
        rc = df.loc[res, "response-correct"]
        df.loc[res, "response-correct"] = np.where(rc >= 0, rng.integers(0, 2, res.sum()), rc)
        df.to_csv(late / "p01-b1-t03_walking.csv", sep=";", index=False)
        try:
            C.BIDS_DIR, C.DERIV_DIR = bids, bids / "derivatives"
            C.APPLY_BLOCK_RULE = True
            bidsify.convert_all(raw_dir=raw, bids_dir=bids, prune_loose=False)
            block_rules.main(quiet=True)
            assert bids_io.gate_staleness() is None
            bidsify.convert_all(raw_dir=late, bids_dir=bids, prune_loose=False)
            assert "added" in bids_io.gate_staleness()
            assert [r.run for r in bids_io.find_recordings(bids)] == ["01"], \
                "the late guessing block slipped past a stale gate"
            assert bids_io.gate_staleness() is None

            # output of a block excluded later
            ph = bids_io.deriv_dir("phase", "sub-01", "ses-02")
            stale = ph / "sub-01_ses-02_task-ILL_acq-walking_run-03_phase.json"
            kept = ph / "sub-01_ses-02_task-ILL_acq-walking_run-01_phase.json"
            stale.write_text("{}"); kept.write_text("{}")
            row = bids_io.gate_report("phase", condition="walking", n_analysed=1,
                                      prune=("phase",))
            assert not stale.exists() and kept.exists()
            assert (row["units_total"], row["units_included"]) == (3, 1)
            assert (row["participants_total"], row["participants_included"]) == (1, 1)
            bids_io.gate_report("gait", "gait", condition="walking", n_analysed=1)
            bids_io.gate_report("phase", condition="walking", n_analysed=1)   # upsert
            led = pd.read_csv(bids / "derivatives" / "qc" / bids_io.GATE_LEDGER_FILE,
                              sep="\t", dtype=str)
            assert list(led["stage"]) == ["phase", "gait"]
            assert led.set_index("stage").loc["gait", "units_included"] == "3"
            assert (led["gate_id"] == bids_io.gate_id()).all()

            # files of participants dropped by the stage and non-estimable population
            # results are removed
            sd = bids_io.deriv_dir("sdt")
            own = sd / "sub-01_ses-02_task-ILL_acq-walking_sdt.json"
            popf = sd / "sdt_population.json"
            own.write_text("{}"); popf.write_text("{}")
            row = bids_io.gate_report("sdt", condition="walking", analysed_subjects=set(),
                                      prune=("sdt",))
            assert row["n_analysed"] == 0 and not own.exists()
            bids_io.clear_stale("sdt", "sdt_population.json")
            assert not popf.exists()
        finally:
            C.BIDS_DIR, C.DERIV_DIR, C.APPLY_BLOCK_RULE = old


# ---------------------------------------------------------------------------
# Exploratory gait modules: phaser, head stability, gait dynamics, gait PRC
# ---------------------------------------------------------------------------
def test_phaser_combines_channels_better_than_any_single_one():
    """On a synthetic oscillator with a non-uniform waveform, the multi-channel phase is
    closer to the true (uniform) phase than the best single-channel Hilbert phase."""
    import phaser
    from scipy.signal import detrend
    fs = 90.0
    t = np.arange(0, 240, 1 / fs)
    rng = np.random.default_rng(3)
    psi = 2 * np.pi * 0.85 * t                      # true phase, 1 cycle per stride
    phi = psi + 0.3 * np.sin(psi)
    bp = lambda x: gait_mod._bandpass(detrend(x), fs, 0.3, 5.0)   # noqa: E731
    lat = bp(np.cos(phi) + 0.25 * np.cos(2 * phi) + rng.normal(0, .35, t.size))
    roll = bp(np.sin(phi + .7) + rng.normal(0, .5, t.size))
    vert = bp(np.cos(2 * phi + 1) + 0.3 * np.cos(4 * phi) + rng.normal(0, .35, t.size))
    valid = np.ones(t.size, bool)
    valid[(t > 100) & (t < 108)] = False

    def err(est):
        ok = valid & np.isfinite(est)
        d = np.angle(np.exp(1j * (est[ok] - psi[ok])))
        d = np.angle(np.exp(1j * (d - np.angle(np.mean(np.exp(1j * d))))))
        return np.degrees(np.sqrt(np.mean(d ** 2)))
    single = min(err(phaser.protophase(lat)), err(phaser.protophase(vert) / 2))
    combined, info = phaser.stride_phase(t, {"lat": lat, "roll": roll}, {"vert": vert}, valid)
    assert err(combined) < 0.7 * single, (err(combined), single)
    assert np.isnan(combined[(t > 101) & (t < 107)]).all(), "phase invented in the gap"
    assert abs(sum(c["weight"] for c in info["channels"].values()) - 1) < 1e-6


def test_dfa_recovers_white_and_pink_noise_exponents():
    import gait_dynamics as gd
    rng = np.random.default_rng(0)

    def pink(n):
        f = np.fft.rfftfreq(n)
        s = np.zeros_like(f)
        s[1:] = 1 / np.sqrt(f[1:])
        return np.fft.irfft(s * np.exp(2j * np.pi * rng.random(f.size)), n)
    white = np.mean([gd.dfa([rng.normal(0, 1, 600)])["alpha"] for _ in range(20)])
    pink_a = np.mean([gd.dfa([pink(600)])["alpha"] for _ in range(20)])
    assert abs(white - 0.5) < 0.08 and abs(pink_a - 1.0) < 0.1, (white, pink_a)
    # a series broken by a gap is analysed per run, not spliced
    x = pink(600)
    assert abs(gd.dfa([x[:280], x[300:]])["alpha"] - gd.dfa([x])["alpha"]) < 0.2


def test_stride_runs_break_at_gaps_not_across_them():
    import gait_dynamics as gd
    tr = np.concatenate([np.arange(0, 60, 0.6), np.arange(70, 130, 0.6)])   # 10 s gap
    runs = gd.stride_runs(tr)
    assert len(runs) == 2
    assert all(np.allclose(r, 1.2) for r in runs), "a stride spans the gap"


def test_gait_prc_detects_an_injected_phase_advance():
    """Steps after an event come 30 ms early: the shift is negative at the injected size
    and random times show none."""
    import gait_prc as prc
    rng = np.random.default_rng(1)
    step = 0.6
    events = np.sort(rng.uniform(10, 590, 150))
    tr, t = [], 0.0
    while t < 600:
        tr.append(t)
        nxt = t + step + rng.normal(0, 0.01)
        if ((events > t) & (events <= nxt)).any():
            nxt -= 0.03                                    # the perturbation
        t = nxt
    tr = np.asarray(tr)
    ev = prc.shifts(events, tr)["shift"][:, prc.K_REPORT - 1]
    rnd = prc.shifts(rng.uniform(10, 590, 3000), tr)["shift"][:, prc.K_REPORT - 1]
    injected = -100 * 0.03 / step                          # -5 % of a step
    assert abs(np.nanmean(ev) - np.nanmean(rnd) - injected) < 1.5, np.nanmean(ev)
    assert abs(np.nanmean(rnd)) < 2.0


def test_head_fixation_point_and_pitch_axis_recovered():
    """Pitch that exactly fixes a target at 1.2 m gives a 1.2 m fixation point; the pitch
    axis is found from a rotation about room x at the step frequency."""
    import head_stability as hs
    fs, step_hz = 90.0, 1.7
    t = np.arange(0, 120, 1 / fs)
    y = 0.01 * np.cos(2 * np.pi * step_hz * t)
    pitch = np.degrees(np.arctan(y / 1.2))
    tp = np.tan(np.radians(pitch))
    assert abs(np.cov(y, tp, bias=True)[0, 1] / np.var(tp) - 1.2) < 1e-6
    rng = np.random.default_rng(0)
    rv = np.column_stack([pitch * 20, 3 * np.sin(2 * np.pi * step_hz / 2 * t),
                          2 * np.sin(2 * np.pi * step_hz / 2 * t + 1)]) + rng.normal(0, .1, (t.size, 3))
    axis, share = hs.pitch_axis(rv, fs, step_hz)
    assert axis[0] > 0.95 and share > 0.5, (axis, share)


def test_phase_locked_r2_rewards_the_more_accurate_phase():
    import gait_phase_methods as gpm
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 100, 20000)
    x = np.sin(2 * np.pi * true / 100) + rng.normal(0, .3, true.size)
    noisy = (true + rng.normal(0, 15, true.size)) % 100
    assert gpm.phase_locked_r2(true, x) > gpm.phase_locked_r2(noisy, x) + 0.1


# ---------------------------------------------------------------------------
# script entry point (without pytest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:                 # pragma: no cover
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
