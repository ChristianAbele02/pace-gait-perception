"""Exclusion gate: which blocks and calibrations each analysis may use.

Builds ``derivatives/qc/gate.tsv``, which every analysis stage reads (via
``bids_io.find_recordings`` and ``bids_io.excluded_calibrations``). It runs as the third
stage, after the calibration verdicts; ``bids_io.load_gate`` rebuilds it when stale, so a
stage run alone never uses an old gate.

Automatic rule:

1. Chance. A block is excluded if accuracy over the answered main trials is not above
   chance (one-sided binomial test against 0.5 at ``BLOCK_RULE_ALPHA``). Timeouts are left
   out rather than scored wrong: a missed response says nothing about whether the tilt was
   seen, and scoring it wrong would push a guesser below 0.5. With ~150-170 answered trials
   this needs ~57 %, so only blocks with no evidence of task performance are removed. A
   fixed cut-off like 60-70 % would drop blocks that were just hard and bias the
   standing-vs-walking comparison.
2. Reversal. A block significantly below chance is excluded as invalid. This indicates
   reversed answers or a broken stimulus (e.g. a flipped CAVE view), not
   guessing; check the session log.
3. Timeouts. A block with ``BLOCK_RULE_MAX_TIMEOUT_FRACTION`` (25 %) or more main trials
   unanswered is excluded: the task was not done as instructed, and accuracy would rest on
   a self-selected subset.
4. Calibration. An UNRELIABLE calibration is left out of the pooled calibration refit
   (slope/lapse fit and standing-vs-walking slope test). Its blocks are not excluded for
   this; they are judged on criteria 1-3, and the verdict is stored with them. An off-target
   threshold makes levels harder or easier than planned, which the difficulty
   residualisation absorbs. If a condition was recalibrated within a visit, the latest
   calibration applies.

Easy-trial accuracy is not a criterion: with ~17 easy trials per block, a binomial test
fails an engaged participant 10-40 % of the time. It remains a QC warning (``qc.py``).

Scopes. The rule concerns responses, so it gates only the ``behaviour`` scope. The
``gait`` scope (gait extraction, phase methods, head stability, gait dynamics, gait PRC)
keeps those blocks, since someone answering at chance can still walk normally. The manual
list ``config.EXCLUDED_RECORDINGS`` excludes from both scopes. With
``config.APPLY_BLOCK_RULE`` False, verdicts are still written but only the manual list
excludes.

Each stage logs its N (``bids_io.gate_report``) to ``derivatives/qc/gate_ledger.tsv``.

Usage::

    python -m Analysis.block_rules
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

try:
    from . import bids_io
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def calibration_verdict(sub: str, condition: str, visit: int | str | None = None) -> str:
    """Reliability verdict of the calibration a block's threshold came from.

    Each condition is calibrated once per visit (INSTRUCTIONS.md §5), filed as
    ``run-<visit>``. A recalibration in the same visit reuses the Trial number, so bidsify
    files it as ``run-<100k + visit>``; the latest one applies to the following blocks.
    Without a visit, return the worst verdict across the condition's calibrations so an
    unreliable one is not hidden.
    """
    d = C.DERIV_DIR / "calibration" / sub / C.CALIBRATION_SESSION
    js = sorted(d.glob("*_calibration_qc.json")) if d.exists() else []
    if not js:
        return "n/a"
    if condition:
        matched = [j for j in js
                   if f"acq-{condition}" in j.name.lower() or f"_{condition}" in j.name.lower()]
    else:
        matched = list(js)
    if not matched:
        return _load(js[0]).get("verdict", "n/a") if len(js) == 1 else "n/a"
    if visit:
        def run_of(j: Path) -> int | None:
            m = re.search(r"_run-(\d+)", j.name)
            return int(m.group(1)) if m else None
        same = [j for j in matched if run_of(j) is not None and run_of(j) % 100 == int(visit)]
        if same:
            return _load(max(same, key=run_of)).get("verdict", "n/a")
        # No calibration for this visit: report that instead of using another visit's.
        if len(matched) > 1:
            return f"n/a (no calibration for visit {visit})"
    verdicts = [str(_load(j).get("verdict", "n/a")) for j in matched]
    bad = [v for v in verdicts if v.startswith("UNRELIABLE")]
    if bad:
        return bad[0] if len(matched) == 1 else f"{bad[0]} [worst of {len(matched)} visits]"
    return verdicts[0]


def evaluate(rec, alpha: float | None = None) -> dict:
    """Rule verdict for one block, with the underlying numbers."""
    alpha = C.BLOCK_RULE_ALPHA if alpha is None else alpha
    ev = bids_io.load_events(rec)
    main = ev[ev["is_training"] == 0] if not ev.empty else ev
    n_main = int(len(main))
    rc = pd.to_numeric(main["response_correct"], errors="coerce") if n_main else pd.Series()
    n = int(rc.isin([0, 1]).sum())                      # answered trials only
    k = int((rc == 1).sum())
    p_above = binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
    p_below = binomtest(k, n, 0.5, alternative="less").pvalue if n else float("nan")
    visit = (bids_io.scan_meta(rec) or {}).get("visit")
    cal = calibration_verdict(rec.sub, rec.condition, visit)

    reasons = []
    max_to = float(getattr(C, "BLOCK_RULE_MAX_TIMEOUT_FRACTION", 0.25))
    n_timeout = n_main - n
    if n_main and n_timeout / n_main >= max_to:
        reasons.append(f"timeouts: {n_timeout}/{n_main} unanswered "
                       f"({n_timeout / n_main:.0%} >= {max_to:.0%})")
    if n and p_below < alpha:
        reasons.append(f"reversal: accuracy {k}/{n} answered significantly BELOW chance "
                       f"(p={p_below:.2g}) - reversed answers or invalid stimulus")
    elif not n or p_above >= alpha:
        reasons.append(f"chance: accuracy {k}/{n} answered not above chance "
                       f"(p={p_above:.2g})")
    return {
        "recording": rec.label(), "participant": rec.sub, "condition": rec.condition,
        "visit": visit if visit is not None else "n/a",
        "n_trials": n_main, "n_answered": n, "n_correct": k,
        "accuracy": round(k / n, 3) if n else "n/a",
        "p_above_chance": round(float(p_above), 6) if n else "n/a",
        "p_below_chance": round(float(p_below), 6) if n else "n/a",
        "calibration_verdict": cal,
        "exclude": bool(reasons),
        "reason": "; ".join(reasons) or "pass",
        "calibration_flagged": str(cal).startswith("UNRELIABLE"),
    }


def evaluate_calibration(path: Path) -> dict:
    """Rule verdict for one calibration table (its QUEST+ reliability)."""
    df = pd.read_csv(path, sep="\t", na_values="n/a")
    if "tracker" in df:
        df = df[~df["tracker"].astype(str).isin(["catch", "warmup"])]
    c = pd.to_numeric(df["correct"], errors="coerce") if "correct" in df else pd.Series(dtype=float)
    n, k = int(c.notna().sum()), int((c == 1).sum())
    sub = path.parts[-4]
    acq = re.search(r"_acq-([a-z]+)", path.name)
    run = re.search(r"_run-(\d+)", path.name)
    qc = (C.DERIV_DIR / "calibration" / sub / path.parts[-3]
          / f"{bids_io.calibration_label(path)}_calibration_qc.json")
    verdict = str(_load(qc).get("verdict", "n/a"))
    bad = verdict.startswith("UNRELIABLE")
    return {
        "unit": "calibration", "recording": bids_io.calibration_label(path),
        "participant": sub, "condition": acq.group(1) if acq else "calibration",
        "visit": int(run.group(1)) % 100 if run else "n/a",
        "n_trials": n, "n_correct": k, "accuracy": round(k / n, 3) if n else "n/a",
        "p_above_chance": "n/a", "p_below_chance": "n/a",
        "calibration_verdict": verdict,
        "rule": f"calibration: {verdict}" if bad else "pass",
        "rule_exclude": bad,
    }


def main(quiet: bool = False) -> list[dict]:
    """Evaluate all blocks and calibrations and write ``qc/gate.tsv``.

    A sidecar ``.json`` stores the rule, its settings and a fingerprint."""
    recs = bids_io.find_recordings(include_excluded=True, apply_block_rule=False)
    cals = bids_io.find_calibration_tables()
    if not recs and not cals:
        print("No BIDS recordings found. Run Analysis/bidsify.py first.")
        return []
    settings = bids_io.gate_settings()
    active, manual = settings["RuleActive"], settings["Manual"]
    rows = []
    for r in recs:
        v = evaluate(r)
        rows.append({"unit": "block", **v, "rule": v.pop("reason"),
                     "rule_exclude": v.pop("exclude")})
    kept = [v.pop("calibration_flagged") and not v["rule_exclude"] for v in rows]
    rows += [evaluate_calibration(c) for c in cals]
    for v in rows:
        man = manual.get(v["recording"])
        rule_out = active and v["rule_exclude"]
        v["manual"] = man or "-"
        v["behaviour"] = "exclude" if (man or rule_out) else "include"
        v["gait"] = ("n/a" if v["unit"] == "calibration"
                     else "exclude" if man else "include")
        why = ([f"manual: {man.rstrip('.')}"] if man else []) + ([v["rule"]] if rule_out else [])
        v["reason"] = "; ".join(why) or "include"

    out = bids_io.deriv_dir("qc")
    for old in ("block_exclusions.tsv", "block_exclusions.json"):   # legacy files
        (out / old).unlink(missing_ok=True)
    df = pd.DataFrame(rows)
    gid = hashlib.sha1(df[["recording", "behaviour", "gait"]].to_csv(index=False)
                       .encode()).hexdigest()[:10]
    df.to_csv(out / bids_io.GATE_FILE, sep="\t", index=False, na_rep="n/a")
    (out / bids_io.GATE_FILE.replace(".tsv", ".json")).write_text(json.dumps({
        "Description": "Exclusion gate (Analysis/block_rules.py): one row per block and "
                       "per calibration, with the decision for each analysis scope. Every "
                       "analysis stage reads it; derivatives/qc/gate_ledger.tsv logs the "
                       "N each stage was left with.",
        "Settings": settings,
        "GateId": gid,
        "Scopes": {
            "behaviour": "analyses of the responses: excluded = manual list, plus the "
                         "automatic rule while RuleActive",
            "gait": "analyses of the walking alone: excluded = manual list only",
        },
        "Criteria": {
            "chance": "accuracy over the answered main trials (timeouts left out) not "
                      "above 0.5, one-sided binomial",
            "reversal": "accuracy over the answered main trials significantly below 0.5, "
                        "one-sided binomial",
            "timeouts": "at least BLOCK_RULE_MAX_TIMEOUT_FRACTION of the main trials "
                        "unanswered",
            "calibration": "an UNRELIABLE calibration is left out of the pooled "
                           "calibration refit; its blocks are judged by chance and "
                           "reversal alone (calibration_verdict records it)",
        },
        "unit": {"Description": "block (a motion recording) or calibration (a _beh.tsv)."},
        "rule": {"Description": "Which criteria failed, with the numbers; 'pass' if none. "
                                "Recorded whether or not the rule is active."},
        "rule_exclude": {"Description": "True if the automatic rule fails this unit."},
        "manual": {"Description": "Reason from config.EXCLUDED_RECORDINGS, or '-'."},
        "behaviour": {"Description": "include / exclude for the behaviour scope."},
        "gait": {"Description": "include / exclude for the gait scope (n/a: calibration)."},
        "reason": {"Description": "Why the unit is excluded from the behaviour scope."},
    }, indent=2), encoding="utf-8")

    blocks = [v for v in rows if v["unit"] == "block"]
    calib = [v for v in rows if v["unit"] == "calibration"]
    bx = [v for v in blocks if v["behaviour"] == "exclude"]
    gx = [v for v in blocks if v["gait"] == "exclude"]
    cx = [v for v in calib if v["behaviour"] == "exclude"]
    subs = sorted({v["participant"] for v in blocks})
    subs_b = [s for s in subs if all(v["behaviour"] == "exclude"
                                     for v in blocks if v["participant"] == s)]
    state = ("rule ACTIVE" if active else
             "rule NOT ACTIVE: verdicts recorded, only the manual list excludes")
    print(f"Gate {gid} ({state}):")
    print(f"  behaviour   : {len(blocks) - len(bx)}/{len(blocks)} blocks, "
          f"{len(subs) - len(subs_b)}/{len(subs)} participants"
          + (f" (out entirely: {', '.join(subs_b)})" if subs_b else ""))
    print(f"  gait        : {len(blocks) - len(gx)}/{len(blocks)} blocks")
    print(f"  calibrations: {len(calib) - len(cx)}/{len(calib)} enter the pooled refit")
    if not quiet:
        for v, k in zip(rows, kept + [False] * len(calib)):
            if v["rule_exclude"] or v["manual"] != "-":
                tag = ("EXCLUDED" if v["behaviour"] == "exclude"
                       else "flagged, rule not active")
                why = v["reason"] if v["reason"] != "include" else v["rule"]
                print(f"  ! {v['recording']} [{tag}]: {why}")
            elif k:
                print(f"  ~ {v['recording']} [kept, above chance at "
                      f"{v['accuracy']:.0%}]: calibration {v['calibration_verdict']}")
        print(f"Gate -> {out / bids_io.GATE_FILE}")
    return rows


if __name__ == "__main__":
    main()
