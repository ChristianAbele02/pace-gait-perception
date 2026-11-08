"""Validation tests for the analysis pipeline (b9).

Checks that injected effects are recovered, that null data keep nominal false-positive
rates, and that supporting algorithms (sequence shuffle, phase methods, calibration
refit, BIDS conversion, exclusion gate, caching) behave as documented.

Run::

    python -m pytest Analysis/tests -q
    python Analysis/tests/test_pipeline.py      # also works without pytest
"""
from __future__ import annotations

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
import population as pop             # noqa: E402
import calibration as cal       # noqa: E402
import clean as clean_mod            # noqa: E402
import config as C                   # noqa: E402
import bidsify                       # noqa: E402
import report                        # noqa: E402
import response_bias as rbias        # noqa: E402


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
# Calibration slope/lapse: the refit at the rig's calibration design
# ---------------------------------------------------------------------------


def test_fit_joint_fixed_lapse():
    """A fixed lapse is held, not fitted, and still gives a usable slope."""
    obs = _sim_cond_observers(3.5, n_part=8, seed=11)
    fixed = cal.fit_joint(obs, lapse=0.03)
    assert fixed["ok"] and fixed["lapse"] == 0.03
    assert abs(fixed["slope_beta"] - cal.fit_joint(obs)["slope_beta"]) < 1.5


def _sim_cond_observers(slope, n_part=10, n_trials=90, seed=0):
    """A condition's worth of QUEST-like observers (tilts spread around threshold)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_part):
        alpha = float(max(1.0, rng.normal(5.0, 1.0)))
        levels = alpha * np.geomspace(0.4, 2.2, 7)
        x = rng.choice(levels, size=n_trials)
        p = cal.weibull(x, alpha, slope, 0.03)
        out.append((x, (rng.uniform(size=n_trials) < p).astype(float)))
    return out


def test_slope_invariance_lrt():
    """Equal slopes across conditions -> n.s.; clearly different slopes -> detected."""
    same = cal.fit_slope_invariance({"standing": _sim_cond_observers(3.5, seed=1),
                                    "walking": _sim_cond_observers(3.5, seed=2)})
    assert same["ok"] and same["p_value"] > 0.05, f"equal slopes flagged (p={same['p_value']})"
    diff = cal.fit_slope_invariance({"standing": _sim_cond_observers(5.0, seed=3),
                                    "walking": _sim_cond_observers(1.5, seed=4)})
    assert diff["ok"] and diff["slope_differs"], f"clear slope difference missed (p={diff['p_value']})"
    # needs >=2 conditions
    assert not cal.fit_slope_invariance({"standing": _sim_cond_observers(3.0, seed=5)})["ok"]


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


def test_data_quality_labels_and_counts():
    """Each exclusion gets one standard label (manual first, then reversal, timeouts,
    chance), kept trials are the answered trials of kept blocks, and planned runs
    without a recording are 'Not recorded' or, after the last visit, 'Visit pending'."""
    import report
    rec = "sub-01_ses-02_task-ILL_acq-walking_run-{:02d}"
    def row(run, beh, rule="pass", manual="-", answered=160, correct=120):
        return dict(unit="block", recording=rec.format(run), participant="sub-01",
                    condition="walking", visit="1", n_trials="170", n_answered=str(answered),
                    n_correct=str(correct), rule=rule, manual=manual, behaviour=beh,
                    gait="exclude" if manual != "-" else "include")
    gate = pd.DataFrame([
        row(1, "include"),
        row(2, "exclude", "timeouts: 60/170 unanswered; chance: 55/110", answered=110,
            correct=55),
        row(3, "exclude", "reversal: 40/160 below chance", manual="flipped view",
            correct=40),
        dict(unit="calibration", recording="sub-01_ses-01_task-ILL_acq-walking_run-01_beh",
             participant="sub-01", condition="walking", visit="1", n_trials="60",
             n_answered="n/a", n_correct="40", rule="calibration: UNRELIABLE",
             manual="-", behaviour="exclude", gait="n/a"),
    ])
    planned = {"walking": {1: (1, 2, 3, 4), 2: (5, 6)}}
    units = report.unit_rows(gate, {rec.format(r): 5 for r in (1, 2, 3)}, planned,
                         {rec.format(3): "Technical failure"})
    lab = dict(zip(units["recording"] + "/" + units["run"].astype(str), units["label"]))
    assert lab[rec.format(1) + "/1"] == "-"
    assert lab[rec.format(2) + "/2"] == "Too many timeouts"
    assert lab[rec.format(3) + "/3"] == "Technical failure"
    missing = units[units["status"] == "missing"].set_index("run")["label"].to_dict()
    assert missing == {4: "Not recorded", 5: "Visit pending", 6: "Visit pending"}
    tot = report.summary_rows(units).query("level == 'total'").iloc[0]
    assert (tot["blocks_recorded"], tot["blocks_kept"], tot["blocks_missing"]) == (3, 1, 3)
    assert (tot["trials_recorded"], tot["trials_kept"]) == (510, 160)
    assert tot["timeouts_in_kept_blocks"] == 10 and tot["calibrations_kept"] == 0
    assert "## Labels" in report.markdown(units, report.summary_rows(units))


def test_participant_roster_status():
    """The roster marks a participant useful only with every planned block recorded and
    kept, pending while a visit is to come, partial after a lost block or a cancelled
    visit, and not useful with no block kept or a "not useful" rating."""
    import report
    planned = {"walking": {1: (1, 2), 2: (3, 4)}}

    def gate_for(sub, runs, excluded=()):
        return [dict(unit="block", participant=sub, condition="walking",
                     recording=f"{sub}_ses-02_task-ILL_acq-walking_run-{r:02d}",
                     visit="1" if r <= 2 else "2", n_trials="170", n_answered="160",
                     n_correct="120", rule="chance: x" if r in excluded else "pass",
                     manual="-", behaviour="exclude" if r in excluded else "include",
                     gait="include") for r in runs]
    gate = pd.DataFrame(gate_for("sub-01", (1, 2, 3, 4)) + gate_for("sub-02", (1, 2))
                        + gate_for("sub-03", (1, 2)) + gate_for("sub-04", (1, 2, 3, 4), (3,))
                        + gate_for("sub-05", (1, 2), (1, 2)) + gate_for("sub-06", (1, 2, 3, 4)))
    units = report.unit_rows(gate, {}, planned, {}, cancelled={"sub-03": "stopped"})
    assert set(units.loc[units["participant"] == "sub-03", "label"]) == {"-", "Visit cancelled"}
    roster = report.roster_rows(units, {"sub-06": "not useful", "sub-01": "very good"})
    status = dict(zip(roster["participant"], roster["status"]))
    assert status == {"sub-01": "useful", "sub-02": "pending", "sub-03": "partial",
                      "sub-04": "partial", "sub-05": "not useful", "sub-06": "not useful"}
    rating = dict(zip(roster["participant"], roster["rating"]))
    assert rating["sub-01"] == "very good" and rating["sub-02"] == "unrated"
    assert set(roster["status"]) <= set(report.STATUS_LEVELS)
    assert "## Participant roster" in report.markdown(units, report.summary_rows(units), roster=roster)


def test_not_useful_participants_are_excluded(monkeypatch, tmp_path):
    """A participant rated "not useful" is a manual exclusion in every scope while the
    switch is on, and the data-quality overview labels it 'Participant not useful'."""
    import bids_io
    import report
    monkeypatch.setattr(C, "PARTICIPANT_RATINGS", {"sub-01": "good", "sub-02": "not useful"})
    monkeypatch.setattr(C, "EXCLUDE_NOT_USEFUL_PARTICIPANTS", True)
    monkeypatch.setattr(C, "EXCLUDED_RECORDINGS", {})
    assert bids_io.excluded_participants() == {"sub-02": bids_io.NOT_USEFUL_REASON}
    for sub in ("sub-01", "sub-02"):
        d = tmp_path / sub / "ses-02" / "motion"
        d.mkdir(parents=True)
        (d / f"{sub}_ses-02_task-ILL_tracksys-optitrack_acq-walking_run-01_motion.tsv").touch()
    for scope in ("behaviour", "gait"):
        out = bids_io.excluded_recordings(tmp_path, apply_block_rule=False, scope=scope)
        assert list(out) == ["sub-02_ses-02_task-ILL_acq-walking_run-01"]
    gate = pd.DataFrame([dict(unit="block", participant="sub-02", condition="walking",
                              recording="sub-02_ses-02_task-ILL_acq-walking_run-01",
                              visit="1", n_trials="170", n_answered="160", n_correct="120",
                              rule="pass", manual=bids_io.NOT_USEFUL_REASON,
                              behaviour="exclude", gait="exclude")])
    units = report.unit_rows(gate, {}, {"walking": {1: (1,)}}, {}, {})
    assert units.loc[0, "label"] == "Participant not useful"
    monkeypatch.setattr(C, "EXCLUDE_NOT_USEFUL_PARTICIPANTS", False)
    assert bids_io.excluded_participants() == {}


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
    import conditions as cond
    e = cond._crossover_model(_cross_cells(40, 0.10, 0.0, 0.0, seed=1))["effects"]
    assert abs(e["condition_walking_minus_standing"]["mean_log10_ratio"] - 0.10) < 0.02
    assert abs(e["position_second_minus_first"]["mean_log10_ratio"]) < 0.02
    e = cond._crossover_model(_cross_cells(40, 0.0, 0.12, 0.0, seed=2))["effects"]
    assert abs(e["position_second_minus_first"]["mean_log10_ratio"] - 0.12) < 0.02
    assert abs(e["condition_walking_minus_standing"]["mean_log10_ratio"]) < 0.02
    e = cond._crossover_model(_cross_cells(40, 0.0, 0.0, 0.15, seed=3))["effects"]
    assert abs(e["interaction_condition_x_position"]["mean_log10_ratio"] - 0.15) < 0.03


def test_paired_threshold_pools_all_blocks_of_a_condition():
    """The paired test fits each participant's threshold to all blocks of a condition
    (as in the power analysis), not only the last block."""
    import conditions as cond
    rng = np.random.default_rng(7)

    def block(alpha, n=180):
        tilt = rng.choice(np.array([0.76, 1.0, 1.26]) * alpha, size=n, p=[0.05, 0.85, 0.10])
        return tilt, (rng.random(n) < cal.weibull(tilt, alpha, 2.5, 0.05, 0.5)).astype(float)

    walking = [block(3.0), block(3.0), block(3.0), block(9.0)]   # last block deviates
    standing = [block(2.5), block(2.5)]
    per = cond._participant_fits({"sub-01": {"walking": walking, "standing": standing}},
                                lambda c: (2.5, 0.05))
    w, s = per["sub-01"]["walking"], per["sub-01"]["standing"]
    assert w["n_blocks"] == 4 and w["n"] == sum(t.size for t, _ in walking)
    assert s["n_blocks"] == 2 and s["n"] == sum(t.size for t, _ in standing)
    last_only = cond.fit_alpha(*walking[-1], beta=2.5, lapse=0.05)
    assert w["alpha_deg"] < 0.8 * last_only["alpha_deg"], (w["alpha_deg"], last_only["alpha_deg"])
    assert 2.5 < w["alpha_deg"] < 9.0


def test_crossover_contrasts_false_positive_rate_controlled():
    """With no true condition or position effect, the crossover contrasts hold their
    nominal false-positive rate."""
    import conditions as cond
    keys = ("condition_walking_minus_standing", "position_second_minus_first",
            "interaction_condition_x_position")
    hits = dict.fromkeys(keys, 0)
    n = 200
    for s in range(n):
        e = cond._crossover_model(_cross_cells(16, 0, 0, 0, seed=100 + s))["effects"]
        for k in keys:
            hits[k] += e[k]["significant"]
    for k in keys:
        assert hits[k] / n < 0.12, f"{k} FPR {hits[k]/n:.2f} (nominal 0.05)"


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
      * ``"random"``: uniform (individual phases, as in Davidson's data);
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
    r = pop.agnostic_test(obs, null)
    assert r["p_value"] < 0.05, f"missed a real incoherent effect (p={r['p_value']})"


def test_agnostic_beats_common_phase_test_when_phases_scatter():
    """With evenly spread phases the vector mean cancels: a common-phase contrast misses
    the effect, the phase-agnostic statistic detects it."""
    obs, null = _agnostic_cohort(amp=0.10, seed=12, phases="spread")
    p_agnostic = pop.agnostic_test(obs, null)["p_value"]
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
        hits += pop.agnostic_test(obs, null)["p_value"] < 0.05
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
    r = rbias.stance_contrast(acc, foot, stride, n_perm=2000,
                              rng=np.random.default_rng(1))
    assert r["p"] < 0.01, f"missed a 10 pp foot asymmetry (p={r['p']:.3f})"
    assert abs(r["delta"] - 0.10) < 0.03, f"effect size off: {r['delta']:+.3f}"


def test_stance_foot_null_false_positive_rate_controlled():
    """With no foot effect the stride-flip permutation must not over-reject."""
    ps = np.array([rbias.stance_contrast(*_foot_trials(9000 + i, n_strides=400),
                                         n_perm=500, rng=np.random.default_rng(i))["p"]
                   for i in range(120)])
    assert (ps < 0.05).mean() <= 0.12, f"inflated false-positive rate {(ps < 0.05).mean():.3f}"


def test_stance_foot_immune_to_within_step_oscillation():
    """Flipping whole strides keeps within-step phase, so a strong 2-cps oscillation
    without a foot difference does not produce a stance-foot effect."""
    ps = np.array([rbias.stance_contrast(*_foot_trials(500 + i, n_strides=400, step_amp=0.18),
                                         n_perm=500, rng=np.random.default_rng(i))["p"]
                   for i in range(120)])
    assert (ps < 0.05).mean() <= 0.12, \
        f"a within-step oscillation leaked into the foot contrast ({(ps < 0.05).mean():.3f})"


def test_stance_foot_population_agnostic_detects_scattered_directions():
    """Opposite asymmetries cancel in the common-direction test but are detected by the
    direction-agnostic test."""
    deltas = [0.06, -0.06] * 4
    res = [rbias.stance_contrast(*_foot_trials(3000 + 17 * i, n_strides=700, delta=d),
                                 n_perm=1500, rng=np.random.default_rng(400 + i))
           for i, d in enumerate(deltas)]
    p = rbias.population_test([r["_null"] for r in res], [r["delta"] for r in res])
    assert p["agnostic_p"] < 0.01, f"agnostic test missed scattered asymmetries ({p['agnostic_p']:.3f})"
    assert p["common_p"] > 0.05, f"common-direction test fired on cancelling effects ({p['common_p']:.3f})"


def test_stance_foot_population_null_controlled():
    """No asymmetry: neither population test is significant."""
    res = [rbias.stance_contrast(*_foot_trials(7000 + 13 * i, n_strides=500),
                                 n_perm=1000, rng=np.random.default_rng(800 + i))
           for i in range(10)]
    p = rbias.population_test([r["_null"] for r in res], [r["delta"] for r in res])
    assert p["agnostic_p"] > 0.05 and p["common_p"] > 0.05, \
        f"population test fired on a null (agnostic {p['agnostic_p']:.3f}, common {p['common_p']:.3f})"


# ---------------------------------------------------------------------------
# Estimator upgrades: weighted bins and difficulty adjustment
#     Both only reduce variance: they must not change the null, nor the result when
#     there is nothing to correct.
# ---------------------------------------------------------------------------


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
    r = pop.agnostic_test(obs, null)
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
        hits_st += pop.agnostic_test(obs, null)["p_value"] < 0.05
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
        hits += pop.agnostic_test(obs, null)["p_value"] < 0.05
    fpr = hits / reps
    assert 0.01 <= fpr <= 0.10, f"studentised FPR {fpr:.3f} outside [0.01, 0.10]"


def test_one_primary_measure_is_declared():
    """Exactly one primary measure (accuracy, as in the power analysis)."""
    assert C.PRIMARY_MEASURE == "accuracy"
    assert pop.PRIMARY == C.PRIMARY_MEASURE
    assert pop.MEASURES[0] == C.PRIMARY_MEASURE, "primary must be reported first"
    assert C.PRIMARY_MEASURE not in C.SECONDARY_MEASURES
    assert set(pop.MEASURES) == {C.PRIMARY_MEASURE} | set(C.SECONDARY_MEASURES)


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
    _, dp, cr, _, _, _, _ = rbias.sdt_bins(ph, sig, said, nbins=1, min_n=0)
    assert abs(dp[0] - 1.4) < 0.03, f"d' off: {dp[0]:.3f}"
    assert abs(cr[0] - 0.35) < 0.03, f"criterion off: {cr[0]:.3f}"


def test_sdt_inverse_normal_is_accurate():
    """The custom inverse normal matches scipy to 1e-8."""
    try:
        from scipy.stats import norm
    except Exception:                                     # pragma: no cover
        return
    p = np.linspace(1e-5, 1 - 1e-5, 5001)
    assert np.max(np.abs(rbias._z(p) - norm.ppf(p))) < 1e-8


def test_sdt_separates_a_sensory_from_a_decisional_oscillation():
    """A d' oscillation appears in d' but not in the criterion, and vice versa."""
    ph, sig, said = _sdt_trials(2, amp_d=0.55, amp_c=0.0)
    rd = rbias.analyse_measure(ph, sig, said, "dprime", nperm=400,
                             rng=np.random.default_rng(2))
    rc = rbias.analyse_measure(ph, sig, said, "criterion", nperm=400,
                             rng=np.random.default_rng(2))
    assert rd["p_2cps_holm"] < 0.05, f"missed the sensory effect (p={rd['p_2cps_holm']:.3f})"
    assert rc["p_2cps_holm"] > 0.05, f"sensory effect leaked into criterion (p={rc['p_2cps_holm']:.3f})"

    ph, sig, said = _sdt_trials(3, amp_d=0.0, amp_c=0.40)
    rd = rbias.analyse_measure(ph, sig, said, "dprime", nperm=400,
                             rng=np.random.default_rng(3))
    rc = rbias.analyse_measure(ph, sig, said, "criterion", nperm=400,
                             rng=np.random.default_rng(3))
    assert rc["p_2cps_holm"] < 0.05, f"missed the decisional effect (p={rc['p_2cps_holm']:.3f})"
    assert rd["p_2cps_holm"] > 0.05, f"decisional effect leaked into d' (p={rd['p_2cps_holm']:.3f})"


def test_sdt_null_false_positive_rate_controlled():
    """A biased but non-oscillating observer gives no phase effect."""
    hits_d = hits_c = 0
    reps = 150
    for i in range(reps):
        ph, sig, said = _sdt_trials(8000 + i, amp_d=0.0, amp_c=0.0, c0=0.45)
        rd = rbias.analyse_measure(ph, sig, said, "dprime", nperm=200,
                                 rng=np.random.default_rng(i))
        rc = rbias.analyse_measure(ph, sig, said, "criterion", nperm=200,
                                 rng=np.random.default_rng(i))
        hits_d += rd["p_2cps_holm"] < 0.05
        hits_c += rc["p_2cps_holm"] < 0.05
    assert hits_d / reps <= 0.10, f"d' FPR {hits_d / reps:.3f}"
    assert hits_c / reps <= 0.10, f"criterion FPR {hits_c / reps:.3f}"


def test_sdt_population_reuses_the_confirmatory_statistic():
    """The SDT population test uses population.agnostic_test."""
    rows = []
    for i in range(8):
        ph, sig, said = _sdt_trials(300 + i, amp_c=0.35)
        rows.append({"criterion": rbias.analyse_measure(ph, sig, said, "criterion",
                                                      nperm=300,
                                                      rng=np.random.default_rng(i)),
                     "dprime": None})
    res = rbias.sdt_population_test(rows)
    assert "criterion" in res, "population test produced nothing"
    r = res["criterion"]["by_freq"][2.0]
    # same result as calling agnostic_test directly
    obs = np.array([x["criterion"]["coef_2cps"] for x in rows])
    null = np.stack([x["criterion"]["null_coef_2cps"] for x in rows])
    ref = pop.agnostic_test(obs, null)
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
    """N participants each oscillating at their own phase."""
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi, n)
    return np.column_stack([amp * np.cos(th), amp * np.sin(th)]) + r.normal(0, noise, (n, 2))


def test_hotelling_null_false_positive_rate_controlled():
    """Nominal level without effect at the planned sample sizes."""
    for n in (8, 12, 20):
        hits = sum(pop.hotelling_t2(np.random.default_rng(1000 + i).normal(0, 1, (n, 2)))
                   ["p_value"] < 0.05 for i in range(1500))
        fpr = hits / 1500
        assert 0.02 <= fpr <= 0.09, f"FPR {fpr:.3f} at N={n}"


def test_hotelling_survives_idiosyncratic_phases():
    """Strong oscillations at individual phases are not reported as a common-phase effect
    (the variational Wald did so 60 % of the time)."""
    for n in (12, 20):
        hits = sum(pop.hotelling_t2(_idiosyncratic(n, 1.0, 5000 + i))["p_value"] < 0.05
                   for i in range(1500))
        fpr = hits / 1500
        assert fpr <= 0.10, (f"common-phase test fired on idiosyncratic phases "
                             f"{fpr:.3f} of the time at N={n} (the Wald did 0.60)")


def test_hotelling_detects_a_true_common_phase_effect():
    """Power against a real common-phase effect."""
    hits = sum(pop.hotelling_t2(_coherent(20, 0.8, 9000 + i))["p_value"] < 0.05
               for i in range(600))
    assert hits / 600 > 0.6, f"power only {hits / 600:.2f} at N=20, amp 0.8"


def test_hotelling_degenerate_inputs_return_p_one():
    """Too few participants or a singular covariance give p = 1."""
    assert pop.hotelling_t2(np.zeros((3, 2)))["p_value"] == 1.0
    assert not pop.hotelling_t2(np.zeros((3, 2)))["estimable"]
    same = np.tile([0.5, 0.5], (10, 1))          # zero covariance
    assert pop.hotelling_t2(same)["p_value"] == 1.0


def test_paired_hotelling_is_the_difference_test():
    """The paired test equals the one-sample test on differences and is not significant
    for identical conditions."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.3, 0.5, (16, 2)), rng.normal(0.0, 0.5, (16, 2))
    assert pop.paired_hotelling(a, b)["p_value"] == pop.hotelling_t2(a - b)["p_value"]
    hits = 0
    for i in range(600):
        r = np.random.default_rng(700 + i)
        common = r.normal(0.4, 0.5, (14, 2))     # same oscillation in both conditions
        hits += pop.paired_hotelling(common + r.normal(0, .3, (14, 2)),
                                     common + r.normal(0, .3, (14, 2)))["p_value"] < 0.05
    assert hits / 600 <= 0.09, f"paired test fired on identical conditions ({hits / 600:.3f})"


# ---------------------------------------------------------------------------
# response_counts is gone, and H3 with it
# ---------------------------------------------------------------------------
def test_response_counts_measure_is_removed():
    """Response counts stay removed (invalid in a forced-choice task; see config)."""
    assert "response_counts" not in C.SECONDARY_MEASURES
    assert "response_counts" not in pop.MEASURES
    assert "response_counts" not in pop.PREVALENCE_MEASURES
    assert "response_counts" not in ctc.MEASURES


# ---------------------------------------------------------------------------
# Missing data: partial blocks (disconnects) and missing blocks
# ---------------------------------------------------------------------------


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
    import conditions as cond
    import calibration as cal
    rng = np.random.default_rng(3)
    t_cal = 1.0
    tilt = rng.choice([0.75, 1.0, 1.25], size=600, p=[0.05, 0.85, 0.10]) * t_cal

    def respond(t75):
        alpha = t75 / cond._to_t75(1.0)
        p = cal.weibull(tilt, alpha, cond.BETA_FIX, cond.LAPSE_FIX, cond.GUESS)
        return (rng.random(tilt.size) < p).astype(float)

    null = cond.sampling_null(tilt, t_cal, rng, n_sim=150)
    lo, hi = np.quantile(null, [0.025, 0.975])
    assert lo < 1.0 < hi and hi / lo < 1.6, (lo, hi)
    same = cond.fit_threshold(tilt, respond(t_cal))
    assert same["censored"] is None and lo <= same["t75"] / t_cal <= hi
    harder = cond.fit_threshold(tilt, respond(2 * t_cal))
    assert harder["t75"] / t_cal > hi and harder["lo"] > t_cal
    ceiling = cond.fit_threshold(tilt, (rng.random(tilt.size) < 0.98).astype(float))
    assert ceiling["censored"] == "low"


def test_block_rule_judges_each_block_on_frame_rate_reversal_and_timeouts():
    """Per block: answered trials are counted with timeouts left out (95 right of 152
    answered, 18 timeouts); a block significantly below chance is a reversal; 25 % or more
    timeouts fail on their own; a block logged at 6 Hz, or whose stimuli stayed up over
    twice their nominal time, fails the frame-rate check, which excludes it from both
    scopes. Chance is not judged per block."""
    import types
    import bids_io
    import exclusion

    def events(k, n_wrong, n_timeout, dur=0.15):
        rc = [1.0] * k + [0.0] * n_wrong + [np.nan] * n_timeout
        return pd.DataFrame({"is_training": [1] * 5 + [0] * len(rc),
                             "response_correct": [1.0] * 5 + rc,
                             "duration": dur})

    old = (bids_io.load_events, bids_io.scan_meta, exclusion.calibration_verdict)
    rec = types.SimpleNamespace(sub="sub-01", condition="walking", label=lambda: "x")
    try:
        bids_io.scan_meta = lambda r: {"visit": 1, "effective_sampling_hz": "58.8"}
        exclusion.calibration_verdict = lambda *a: "RELIABLE"
        bids_io.load_events = lambda r, **k: events(95, 57, 18)
        v = exclusion.evaluate(rec, alpha=0.05)
        assert (v["n_trials"], v["n_answered"], v["n_correct"]) == (170, 152, 95)
        assert v["exclude"] is False, v["reason"]
        bids_io.load_events = lambda r, **k: events(70, 70, 30)      # a guessing block
        assert exclusion.evaluate(rec, alpha=0.05)["exclude"] is False
        bids_io.load_events = lambda r, **k: events(50, 100, 20)
        assert exclusion.evaluate(rec, alpha=0.05)["reason"].startswith("reversal")
        bids_io.load_events = lambda r, **k: events(100, 28, 42)     # 24.7 % timeouts
        assert exclusion.evaluate(rec, alpha=0.05)["exclude"] is False
        bids_io.load_events = lambda r, **k: events(100, 27, 43)     # 25.3 % timeouts
        v = exclusion.evaluate(rec, alpha=0.05)
        assert v["exclude"] is True and v["reason"].startswith("timeouts"), v["reason"]
        assert v["technical_exclude"] is False

        bids_io.load_events = lambda r, **k: events(120, 40, 10)
        bids_io.scan_meta = lambda r: {"visit": 1, "effective_sampling_hz": "6.3"}
        v = exclusion.evaluate(rec, alpha=0.05)
        assert v["technical_exclude"] is True and v["reason"].startswith("frame rate"), v
        bids_io.scan_meta = lambda r: {"visit": 1, "effective_sampling_hz": "58.8"}
        bids_io.load_events = lambda r, **k: events(120, 40, 10, dur=0.53)
        v = exclusion.evaluate(rec, alpha=0.05)
        assert v["technical_exclude"] is True and "stimulus" in v["reason"], v
    finally:
        bids_io.load_events, bids_io.scan_meta, exclusion.calibration_verdict = old


def test_chance_is_judged_per_participant_and_condition():
    """A weak block of a participant who does the task is kept (55 % next to blocks at
    70 %); a participant who guesses in every block loses them all; a block already out
    for a per-block criterion or on the manual list stays out of the pool."""
    import exclusion

    def row(sub, cond, rec, k, n, out=False):
        return {"recording": rec, "participant": sub, "condition": cond,
                "n_answered": n, "n_correct": k, "rule": "reversal" if out else "pass",
                "rule_exclude": out}

    rows = [row("sub-01", "walking", "a1", 112, 160), row("sub-01", "walking", "a2", 88, 160),
            row("sub-01", "walking", "a3", 30, 150, out=True),
            row("sub-01", "standing", "a4", 84, 160),                   # 52.5 % alone
            row("sub-02", "walking", "b1", 80, 160), row("sub-02", "walking", "b2", 83, 160),
            row("sub-02", "walking", "b3", 150, 160)]                   # manual: left out
    exclusion.pool_chance(rows, manual={"b3": "flipped view"}, alpha=0.05)
    by = {r["recording"]: r for r in rows}
    assert not by["a2"]["rule_exclude"] and by["a2"]["chance_n_answered"] == 320
    assert by["a3"]["rule"] == "reversal" and by["a3"]["chance_n_answered"] == "n/a"
    assert by["a4"]["rule_exclude"] and by["a4"]["rule"].startswith("chance")
    assert by["b1"]["rule_exclude"] and by["b2"]["rule_exclude"]
    assert by["b1"]["chance_n_answered"] == 320 and by["b3"]["chance_n_answered"] == "n/a"
    assert not by["b3"]["rule_exclude"]


def test_presentation_rule_flags_artefacts_in_the_presentation_window():
    """A frozen pose, a jump or a gap anywhere from 0.3 s before onset to 0.1 s after the
    Gabor's offset flags the trial; one repeated frame, or an artefact outside the window,
    does not."""
    import exclusion
    fs = 60.0
    t = np.arange(0, 20, 1 / fs)
    m = pd.DataFrame({"t": t, "head_local_pos_x": 0.01 * np.sin(2 * np.pi * 0.9 * t),
                      "head_local_pos_y": 0.02 * np.sin(2 * np.pi * 1.8 * t),
                      "head_local_pos_z": np.zeros(t.size),
                      "mocap_frame_advanced": np.ones(t.size, int)})
    m.loc[(t >= 5.0) & (t < 5.3), "mocap_frame_advanced"] = 0     # 0.3 s freeze
    m.loc[np.argmin(np.abs(t - 10.0)), "mocap_frame_advanced"] = 0  # one repeated frame
    m.loc[(t >= 12.0) & (t < 12.5), "head_local_pos_x"] += 1.0      # 1 m jump
    m = m[~((t > 15.0) & (t < 15.5))].reset_index(drop=True)         # 0.5 s dropout
    onset = np.array([5.35, 4.6, 4.85, 10.0, 12.2, 15.2, 8.0])
    flagged, why = exclusion.flag_trials(onset, np.full(onset.size, 0.1),
                                            exclusion.artefact_spans(m))
    assert flagged.tolist() == [True, False, True, False, True, True, False], why
    assert why[0] == "freeze" and why[2] == "freeze" and why[5] == "gap"
    assert "jump" in why[4] or "spike" in why[4]
    assert why[1] == why[3] == why[6] == "-"


# ---------------------------------------------------------------------------
# Exploratory gait modules: head motion, gait dynamics, gait PRC
# ---------------------------------------------------------------------------


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
    import gait_dynamics as gd
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
    ev = gd.shifts(events, tr)["shift"][:, gd.K_REPORT - 1]
    rnd = gd.shifts(rng.uniform(10, 590, 3000), tr)["shift"][:, gd.K_REPORT - 1]
    injected = -100 * 0.03 / step                          # -5 % of a step
    assert abs(np.nanmean(ev) - np.nanmean(rnd) - injected) < 1.5, np.nanmean(ev)
    assert abs(np.nanmean(rnd)) < 2.0


def test_head_fixation_point_and_pitch_axis_recovered():
    """Pitch that exactly fixes a target at 1.2 m gives a 1.2 m fixation point; the pitch
    axis is found from a rotation about x at the step frequency, oriented towards -x (the
    mocap frame's z points backward, so that is nose-down positive)."""
    import head_motion as hm
    fs, step_hz = 90.0, 1.7
    t = np.arange(0, 120, 1 / fs)
    y = 0.01 * np.cos(2 * np.pi * step_hz * t)
    pitch = np.degrees(np.arctan(y / 1.2))
    tp = np.tan(np.radians(pitch))
    assert abs(np.cov(y, tp, bias=True)[0, 1] / np.var(tp) - 1.2) < 1e-6
    rng = np.random.default_rng(0)
    rv = np.column_stack([pitch * 20, 3 * np.sin(2 * np.pi * step_hz / 2 * t),
                          2 * np.sin(2 * np.pi * step_hz / 2 * t + 1)]) + rng.normal(0, .1, (t.size, 3))
    axis, share = hm.pitch_axis(rv, fs, step_hz)
    assert axis[0] < -0.95 and share > 0.5, (axis, share)
    # the same pitch 60 deg late: the size still says 1.2 m, the lag is read as such,
    # and the least-squares distance shrinks by cos(60 deg)
    late = np.degrees(np.arctan(0.01 * np.cos(2 * np.pi * step_hz * t - np.pi / 3) / 1.2))
    _, phase, gain = hm._step_coupling(y, late, fs, step_hz)
    assert abs(-phase - 60) < 2 and abs(hm.fixation_point_m(gain) - 1.2) < 0.03, (phase, gain)
    tl = np.tan(np.radians(late))
    assert abs(np.cov(y, tl, bias=True)[0, 1] / np.var(tl) - 0.6) < 0.01


def test_head_kinematics_coupling_reads_yaw_and_roll_against_the_sway():
    """Yaw turning back to the midline 10 % of a stride after the head is furthest out, and
    roll tilting to the left 15 % before it gets there, are read as such."""
    import head_motion as hm
    amp = {"lateral": {"amp_1cps": 20.0, "peak_1cps_pct": 25.0},
           "yaw": {"amp_1cps": float(np.degrees(np.arctan(0.02))), "peak_1cps_pct": 85.0},
           "roll": {"amp_1cps": 1.0, "peak_1cps_pct": 60.0}}
    c = hm.coupling(amp)
    assert abs(c["yaw_lag_pct"] - 10) < 1e-9 and abs(c["roll_lead_pct"] + 15) < 1e-9
    assert abs(c["yaw_fixation_m"] - 1.0) < 1e-9
    amp["yaw"]["peak_1cps_pct"] = 2.0                  # wraps across the stride boundary
    assert abs(hm.coupling(amp)["yaw_lag_pct"] - 27) < 1e-9


def test_rig_check_reads_every_segment_of_the_script():
    """A synthetic performer runs the rig-check script with known signs, including a step
    forward made of a weight shift, the step and the shift back, and one leg straight to
    the other: every movement is found, assigned to its segment and signed."""
    import rig_check as rc
    fs = rc.GRID_FS
    sign = {"shift_right": 1, "shift_left": -1, "step_forward": 1, "stand_right_leg": 1,
            "stand_left_leg": -1, "nod_chin_down": 1, "right_ear_to_shoulder": -1,
            "left_ear_to_shoulder": 1, "turn_right": 1, "turn_left": -1}
    amp = {"x": 0.2, "z": 0.2, "pitch": 35.0, "roll": 50.0, "yaw": 60.0}
    n = int(400 * fs)
    p = {c: np.zeros(n) for c in ("x", "z", "pitch", "roll", "yaw", "mocap_z")}
    p["t"] = np.arange(n) / fs
    t = 5.0                                         # 5 s of getting ready, then still
    p["x"][: int(3 * fs)] = 0.3
    t += 15.0

    def hold(c, v, start, dur):
        a, b = int(start * fs), int((start + dur) * fs)
        ramp = np.minimum(1, np.minimum(np.arange(b - a), np.arange(b - a)[::-1]) / (0.5 * fs))
        p[c][a:b] += v * ramp
        if c == "z":
            p["mocap_z"][a:b] -= v * ramp

    for name, chan, n_rep, _ in rc.SCRIPT:
        for k in range(n_rep):
            if name == "step_forward":
                hold("x", -0.12 if k == 0 else 0.12, t, 1.6)
                hold("z", amp["z"], t + 1.3, 3.0)
                hold("x", -0.12 if k == 0 else 0.12, t + 4.0, 1.6)
                t += 7.5
            else:
                hold(chan, sign[name] * amp[chan], t, 3.0)
                if chan in ("roll", "yaw"):          # an ear tilt also moves the head
                    hold("x", 0.06 * sign[name], t, 3.0)
                t += 3.0 if name == "stand_right_leg" and k == n_rep - 1 else 5.0
        if name != "stand_right_leg":
            t += 10.0
    rows = rc.standing_segments(p)
    assert [r["segment"] for r in rows] == [s[0] for s in rc.SCRIPT]
    for r in rows:
        assert r["channel_moved_every_time"] and r["same_sign_all"], r
        assert r["sign"] == sign[r["segment"]], r
    assert all(v < 0 for v in rows[2]["mocap_z_peak_m"])


# ---------------------------------------------------------------------------
# Accuracy over answered trials; timeouts as a measure of their own
# ---------------------------------------------------------------------------
def test_timeout_measure_recovers_a_phase_dependent_timeout_rate():
    """Timeouts concentrated at one stride phase are detected at 2 cps by the timeout
    measure; with too few timeouts the measure is not run at all."""
    rng = np.random.default_rng(5)
    n = 1500
    phase = rng.uniform(0, 100, n)
    p = 0.06 + 0.05 * np.cos(2 * np.pi * 2.0 * phase / 100.0)
    timeout = (rng.random(n) < p).astype(float)
    m = pa._timeout_measure(phase, timeout)
    assert m is not None and m["p_2cps_holm"] < 0.05, "missed a 2-cps timeout modulation"
    few = np.zeros(n); few[:pa.MIN_TIMEOUTS - 1] = 1.0
    assert pa._timeout_measure(phase, few) is None


# ---------------------------------------------------------------------------
# Trial and step tables: the long tables written for analyses outside the pipeline
# ---------------------------------------------------------------------------
def test_trial_table_step_and_foot_follow_the_gait_arrays():
    """Each onset gets the step it falls in, that step's foot and evidence, and the % step
    gait.phase_at gives it; an onset in a gap, or before the first trough, gets none."""
    import tables as tt
    steps = np.arange(0, 20, 0.5)
    troughs = np.concatenate([steps[steps < 8], steps[steps > 14]])
    n_steps = troughs.size - 1
    foot = np.arange(n_steps) % 2
    margin = np.where(foot == 0, 1.0, -1.0) * np.linspace(0.1, 2.0, n_steps)
    onset = np.array([-1.0, 0.25, 0.8, 4.1, 10.0, 16.2])
    st = tt.stance_at(onset, troughs, foot, margin)
    assert st["step"].tolist()[:4] == [-1, 0, 1, 8] and st["step"][4] == -1
    assert st["foot"].tolist() == [-1, 0, 1, 0, -1, foot[st["step"][5]]]
    assert np.isnan(st["margin"][[0, 4]]).all() and st["margin"][3] == margin[8]
    ps, _ = gait_mod.phase_at(onset, troughs)
    assert np.allclose(st["pct_step"], ps, equal_nan=True)
    assert abs(st["pct_step"][1] - 50.0) < 1e-9


def test_trial_table_rt_rule_matches_phase_analysis():
    """rt_valid keeps exactly the trials phase_analysis's RT hygiene keeps: correct,
    answered, no anticipations, no log-RT outliers, training trials never."""
    import tables as tt
    rng = np.random.default_rng(3)
    n = 400
    rt = np.exp(rng.normal(np.log(0.8), 0.25, n))
    rt[:6] = [0.05, 0.1, 9.0, 12.0, 0.12, 15.0]          # anticipations and outliers
    correct = (rng.random(n) < 0.75).astype(float)
    correct[:6] = 1.0
    timeout = rng.random(n) < 0.05
    correct[timeout], rt[timeout] = np.nan, np.nan
    training = np.zeros(n, bool); training[-5:] = True
    ev = pd.DataFrame({"response": np.where(timeout, None, "left"), "rt": rt,
                       "acc": np.where(correct == 1, 1.0, 0.0), "difficulty": "THR",
                       "phase_target": rng.uniform(0, 100, n)})
    ref = pa._clean_rt(ev[~training])[2]["n_rt_used"]
    got = tt.rt_valid(rt, correct, ~training)
    assert got.sum() == ref and not got[training].any() and not got[:6].any()


def test_block_schedule_orders_blocks_and_measures_breaks():
    """Blocks are numbered in recording order within each visit, and the gap is from the
    end of the previous block (start + duration) to the start of this one."""
    from types import SimpleNamespace
    import bids_io
    import tables as tt
    meta = {"w1": ("1", "2026-10-06T10:20:00", "first", "600"),
            "w2": ("1", "2026-10-06T10:00:00", "first", "600"),
            "s1": ("1", "2026-10-06T10:40:00", "second", "700"),
            "s2": ("2", "2026-10-13T09:00:00", "first", "700")}
    recs = [SimpleNamespace(sub="sub-01", run=k[1], label=lambda k=k: k) for k in meta]
    old = bids_io.scan_meta
    try:
        bids_io.scan_meta = lambda r, d=None: dict(zip(
            ("visit", "acq_time", "condition_position", "duration_s"), meta[r.label()]))
        s = tt.block_schedule(recs)
    finally:
        bids_io.scan_meta = old
    assert [s[k]["block_in_visit"] for k in ("w2", "w1", "s1", "s2")] == [1, 2, 3, 1]
    assert s["w1"]["gap_before_block_s"] == 600.0 and s["s1"]["gap_before_block_s"] == 600.0
    assert np.isnan(s["w2"]["gap_before_block_s"]) and np.isnan(s["s2"]["gap_before_block_s"])
    assert s["s1"]["offset_in_visit_s"] == 2400.0


def test_step_table_recovers_steps_feet_and_sway():
    """On a clean synthetic walk every step is one half stride, the feet alternate, the
    sway sign follows the foot (code 0 = positive sway) and the bob spans the full
    vertical swing; a missed trough leaves one long step flagged invalid, with no sway."""
    import tables as tt
    t, vertical, lateral, troughs = _synth_gait()
    troughs = np.delete(troughs, 40)                       # one missed trough
    lab = gait_mod.label_steps(t, lateral, troughs)
    df = tt.steps_frame(t, lateral, vertical, troughs, lab["step_foot"], lab["step_margin"])
    assert len(df) == troughs.size - 1
    bad = df[df["valid_step"] == 0]
    assert bad["step"].tolist() == [40] and bad["lateral_sway_m"].isna().all()
    ok = df[(df["valid_step"] == 1) & df["step"].between(3, len(df) - 3)]
    assert np.allclose(ok["duration_s"], 1 / (2 * 0.9), atol=0.02)
    pos, neg = gait_mod.foot_name(0), gait_mod.foot_name(1)    # right/left on this rig
    assert set(df["stance_foot"]) == {pos, neg}
    sign = np.sign(ok["lateral_sway_m"]).map({1.0: pos, -1.0: neg})
    assert (sign == ok["stance_foot"]).mean() > 0.95
    assert np.allclose(ok["vertical_range_m"], 2.0, atol=0.05)
    assert df["stride"].diff().dropna().isin([0, 1]).all()
    untrusted = tt.steps_frame(t, lateral, vertical, troughs, lab["step_foot"],
                                lab["step_margin"], trust_feet=False)
    assert untrusted["stance_foot"].isna().all() and untrusted["stride"].isna().all()


def test_tables_ignore_gait_output_left_over_for_a_gait_excluded_block():
    """A block outside the gait scope gets no stride data even when a gait file from an
    earlier run is still on disk."""
    import tempfile
    from types import SimpleNamespace
    import bids_io
    import tables as tt
    with tempfile.TemporaryDirectory() as tmp:
        rec = SimpleNamespace(sub="sub-01", ses="ses-02", label=lambda: "blk")
        np.savez(Path(tmp) / "blk_gait.npz", t=np.arange(10.0), troughs=np.arange(5.0))
        old = bids_io.deriv_dir
        try:
            bids_io.deriv_dir = lambda *a, **k: Path(tmp)
            assert tt._gait_arrays(rec, gait_ok=False) == (None, "n/a")
            g, _ = tt._gait_arrays(rec, gait_ok=True)
            assert g is not None and g["troughs"].size == 5
        finally:
            bids_io.deriv_dir = old


# ---------------------------------------------------------------------------
# The operating point: information per trial, the threshold-scale test, its cost
# ---------------------------------------------------------------------------
def test_information_per_trial_peaks_above_threshold_and_vanishes_at_the_ends():
    """A trial at 75 % is the reference; information about a threshold shift peaks near
    84 % and falls to zero at chance and at ceiling."""
    import threshold_phase as tph
    assert abs(float(tph.efficiency(0.75)) - 1.0) < 1e-12
    pc = np.linspace(0.51, 0.94, 2000)
    peak = pc[np.argmax(tph.efficiency(pc))]
    assert 0.82 < peak < 0.86, peak
    assert tph.efficiency(0.52) < 0.02 and tph.efficiency(0.949) < 0.02


def test_threshold_phase_recovers_a_threshold_modulation_at_any_operating_point():
    """The same threshold oscillation, in a block run at the calibrated threshold and in
    one 35 % harder: the accuracy coefficient shrinks with the operating point, the
    information-weighted coefficient in threshold units does not."""
    import threshold_phase as tph
    rng = np.random.default_rng(5)
    n, theta = 20000, 0.10
    alpha_cal = 1.0 / tph._u(0.75) ** (1 / tph.BETA)         # 75 % at tilt 1
    levels = np.array(["THR", "EASY", "HARD"])
    rel_tilt = {"THR": 1.0, "EASY": 1.25, "HARD": 0.8}
    thr_units, acc_amp = [], []
    for ratio in (1.0, 1.35):
        phase = rng.uniform(0, 100, n)
        level = rng.choice(levels, n, p=[0.85, 0.10, 0.05])
        tilt = np.array([rel_tilt[v] for v in level])
        log_alpha = np.log(ratio * alpha_cal) + theta * np.cos(2 * np.pi * 2 * phase / 100)
        y = (rng.random(n) < tph.expected_accuracy(tilt, np.exp(log_alpha))).astype(float)
        p_hat = tph.expected_accuracy(tilt, tph.block_alpha(tilt, y))
        for values, sink in ((tph.weighted_values(y, level, p_hat), thr_units),
                             (pa.difficulty_adjust(y, level.astype(object)), acc_amp)):
            c, b, _, cnt = pa._bin(phase, values, 40, "mean", with_counts=True)
            ok = np.isfinite(b)
            coef = pa._spectrum_with_coef(c[ok], b[ok], cnt[ok])[1][pa.IDX_2CPS, 1:]
            sink.append(np.hypot(*(tph.to_threshold_units(coef, p_hat)
                                   if sink is thr_units else coef)))
    assert all(abs(t - theta) < 0.035 for t in thr_units), thr_units
    assert acc_amp[1] < 0.85 * acc_amp[0], acc_amp          # model: 0.73x at this ratio


# ---------------------------------------------------------------------------
# Head roll, response direction, press timing, head kinematics
# ---------------------------------------------------------------------------
def test_roll_effect_recovers_the_equivalent_tilt_and_is_flat_without_it():
    """A roll that acts as 0.3 deg of tilt per deg is recovered and detected; with no roll
    effect the estimate stays within its error."""
    import response_bias as rbias
    rng = np.random.default_rng(3)
    n = 1200
    block = np.repeat(np.arange(4), n // 4)
    tilt = rng.choice([-1, 1], n) * rng.uniform(0.6, 1.4, n)
    roll = rng.normal(0, 1.2, n)
    offset = np.array([-0.2, 0.0, 0.1, 0.3])[block]
    for g, detect in ((0.3, True), (0.0, False)):
        y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * (tilt + g * roll) + offset)))).astype(float)
        r = rbias.roll_effect(y, tilt, roll, block, n_perm=200, rng=np.random.default_rng(1))
        assert abs(r["equivalent_tilt"] - g) < 3 * r["equivalent_tilt_se"] + 0.02, r
        assert (r["p"] < 0.05) == detect, r["p"]


def test_criterion_contrast_finds_a_stance_foot_lean_the_accuracy_contrast_misses():
    """A lean towards 'right' on foot 0's steps and 'left' on foot 1's moves the criterion
    contrast and the towards/away accuracy contrast, not the plain accuracy contrast."""
    rng = np.random.default_rng(4)
    n_str = 700
    stride = np.repeat(np.arange(n_str), 2)
    foot = np.tile([0, 1], n_str)
    is_right = rng.random(stride.size) < 0.5
    lean = np.where(foot == 0, 0.5, -0.5)
    p_right = 1 / (1 + np.exp(-(1.2 * np.where(is_right, 1, -1) + lean)))
    said = (rng.random(stride.size) < p_right).astype(float)
    acc = (said == is_right).astype(float)
    crit = rbias.criterion_contrast(is_right, said, foot, stride, n_perm=500,
                                    rng=np.random.default_rng(0))
    assert crit["p"] < 0.01 and crit["delta"] < 0, crit["p"]
    towards = rbias.towards_label(foot, is_right)
    tw = rbias.stance_contrast(acc, towards, stride, n_perm=500, rng=np.random.default_rng(0))
    assert tw["p"] < 0.01, tw["p"]
    plain = rbias.stance_contrast(acc, foot, stride, n_perm=500, rng=np.random.default_rng(0))
    assert abs(plain["z"]) < 3, plain["z"]


def test_press_phase_null_holds_and_detects_presses_locked_to_the_step():
    """With reaction times unrelated to the stride the presses are not flagged; presses
    placed at 40 % of each step are flagged at 2 cps."""
    import press_phase as pp
    rng = np.random.default_rng(5)
    troughs = np.arange(0.0, 600.0, 0.6)
    gait = (troughs, np.tile([0, 1], troughs.size // 2 + 1)[:troughs.size - 1], None, None)
    onset = np.sort(rng.uniform(2, 590, 500))
    rt_free = rng.uniform(0.5, 1.2, onset.size)
    nxt = troughs[np.searchsorted(troughs, onset + 0.5)]
    rt_locked = nxt + 0.4 * 0.6 + rng.normal(0, 0.03, onset.size) - onset
    free = pp.analyse_blocks([{"onset": onset, "rt": rt_free, "gait": gait}], n_perm=300,
                             rng=np.random.default_rng(0))
    locked = pp.analyse_blocks([{"onset": onset, "rt": rt_locked, "gait": gait}], n_perm=300,
                               rng=np.random.default_rng(0))
    assert free["2cps"]["p_holm"] > 0.05, free["2cps"]["p_holm"]
    assert locked["2cps"]["p_holm"] < 0.01, locked["2cps"]["p_holm"]
    assert abs(locked["2cps"]["peak_pct_stride"] - 20.0) < 3.0, locked["2cps"]["peak_pct_stride"]


def test_group_curve_fit_matches_lstsq_and_flags_a_shared_step_rate_curve():
    """The batched R^2 equals a direct least-squares fit, and a clear shared 2 cps group
    curve is flagged in the 1.5-2.5 cps window with its peak where it was put."""
    import group_curve as gc
    rng = np.random.default_rng(9)
    y = rng.normal(size=gc.N_BINS)
    f = 3.7
    w = 2 * np.pi * f / 100.0
    X = np.column_stack([np.ones(gc.N_BINS), np.cos(w * gc._X), np.sin(w * gc._X)])
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert abs(gc.r2_spectrum(y)[int(np.argmin(np.abs(gc.FREQS - f)))] - r2) < 1e-9
    x = np.arange(gc.N_PCT) + 0.5
    pct = 0.7 + 0.05 * np.cos(4 * np.pi * (x - 10.0) / 100.0) + rng.normal(0, 0.02, gc.N_PCT)
    v = gc.verdict(pct, np.random.default_rng(0), n_res=300)
    assert v["window_hit"] and v["sig_2cps"], v
    assert abs(v["peak_pct_2cps"] - 10.0) < 3.0, v["peak_pct_2cps"]


def test_sdt_stride_rate_test_detects_a_criterion_that_flips_once_per_stride():
    """A criterion swinging once per stride is caught by the separate 1 cps test."""
    rng = np.random.default_rng(6)
    n = 2000
    phase = rng.uniform(0, 100, n)
    sig = rng.random(n) < 0.5
    bias = 0.5 * np.cos(2 * np.pi * phase / 100.0)
    said = rng.random(n) < 1 / (1 + np.exp(-(1.2 * np.where(sig, 1, -1) + bias)))
    r = rbias.analyse_measure(phase, sig, said, "criterion", nperm=300,
                            rng=np.random.default_rng(0))
    assert r["p_1cps"] < 0.01, r["p_1cps"]


def test_head_kinematics_harmonic_recovers_amplitude_and_peak():
    import head_motion as hm
    x = (np.arange(hm.N_BINS) + 0.5) * 100.0 / hm.N_BINS
    prof = 0.8 * np.cos(2 * np.pi * (x - 30.0) / 100.0) + 0.2 * np.cos(4 * np.pi * x / 100.0)
    a1, p1 = hm.harmonic(prof, 1.0)
    a2, _ = hm.harmonic(prof, 2.0)
    assert abs(a1 - 0.8) < 0.01 and abs(p1 - 30.0) < 0.5 and abs(a2 - 0.2) < 0.01


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
