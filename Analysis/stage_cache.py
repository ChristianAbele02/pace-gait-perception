"""Content-fingerprint cache that lets pipeline stages skip unchanged work.

A stage calls ``is_fresh(stage, unit, fingerprint, outputs)`` before an expensive step and
``record(...)`` after it. The step is skipped only if:

* the fingerprint matches the recorded one. It covers the step's inputs (data values, file
  contents) and the source code of the step, so changed data or code recomputes;
* every recorded output still has its recorded size and mtime, so deleted or hand-edited
  outputs are rewritten.

The cache never deletes or rewrites outputs; a stale entry only causes recomputation.
Manifests live in ``<project>/.pipeline_cache/`` (outside the BIDS dataset, git-ignored)
and can be deleted at any time. ``ILL_NO_CACHE=1`` or ``run_pipeline --no-cache`` disables it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C

ENABLED = os.environ.get("ILL_NO_CACHE", "") not in ("1", "true", "yes")
_SRC_DIR = Path(__file__).resolve().parent
_code_digests: dict[str, str] = {}


def cache_dir(root: Path | None = None) -> Path:
    """Manifest folder next to the BIDS dataset ``root`` (default config.BIDS_DIR).

    Steps writing to another dataset (e.g. test datasets) pass it as ``root``."""
    return Path(root or C.BIDS_DIR).resolve().parent / ".pipeline_cache"


def code_digest(*modules: str) -> str:
    """Digest of the named Analysis source files (e.g. ``"bidsify.py"``) and the library
    versions the results depend on."""
    key = "|".join(modules)
    if key not in _code_digests:
        h = hashlib.sha1()
        for m in modules:
            h.update(m.encode())
            h.update((_SRC_DIR / m).read_bytes())
        import scipy
        for lib in (np, pd, scipy):
            h.update(f"{lib.__name__}={lib.__version__}".encode())
        _code_digests[key] = h.hexdigest()
    return _code_digests[key]


def _feed(h, part) -> None:
    if isinstance(part, pd.DataFrame):
        h.update(repr(list(part.columns)).encode())
        h.update(repr([str(t) for t in part.dtypes]).encode())
        h.update(pd.util.hash_pandas_object(part, index=True).to_numpy().tobytes())
    elif isinstance(part, np.ndarray):
        h.update(repr((part.dtype.str, part.shape)).encode())
        h.update(np.ascontiguousarray(part).tobytes() if part.dtype != object
                 else repr(part.tolist()).encode())
    elif isinstance(part, Path):
        h.update(str(part.name).encode())
        h.update(part.read_bytes() if part.exists() else b"<missing>")
    elif isinstance(part, dict):
        # hash values one by one: repr of a large array is abbreviated and would collide
        h.update(b"{")
        for k in sorted(part, key=str):
            _feed(h, str(k))
            _feed(h, part[k])
        h.update(b"}")
    elif isinstance(part, (list, tuple)):
        h.update(b"[")
        for x in part:
            _feed(h, x)
        h.update(b"]")
    elif isinstance(part, (np.generic, float, int, str, bool, type(None))):
        h.update(repr(part).encode())
    elif isinstance(part, bytes):
        h.update(part)
    else:
        h.update(repr(part).encode())
    h.update(b"\x00")


def fingerprint(*parts) -> str:
    """Digest of the given inputs: DataFrames and arrays by value, Paths by file content,
    lists and dicts element by element (dict keys sorted), scalars by repr."""
    h = hashlib.sha1()
    for p in parts:
        _feed(h, p)
    return h.hexdigest()


def _manifest_path(stage: str, root: Path | None = None) -> Path:
    return cache_dir(root) / f"{stage}.json"


def _load(stage: str, root: Path | None = None) -> dict:
    f = _manifest_path(stage, root)
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except (OSError, ValueError):
        return {}                                   # unreadable manifest: treat as empty


def _stat(path: Path) -> list | None:
    try:
        s = Path(path).stat()
    except OSError:
        return None
    return [s.st_size, s.st_mtime_ns]


def is_fresh(stage: str, unit: str, fp: str, outputs, root: Path | None = None) -> bool:
    """True if ``unit`` was recorded with this fingerprint and all its outputs are intact."""
    if not ENABLED:
        return False
    entry = _load(stage, root).get(unit)
    if not entry or entry.get("fingerprint") != fp:
        return False
    recorded = entry.get("outputs", {})
    want = {str(Path(o)) for o in outputs}
    if not want or set(recorded) != want:
        return False
    return all(_stat(Path(p)) == st for p, st in recorded.items())


def record(stage: str, unit: str, fp: str, outputs, root: Path | None = None) -> None:
    """Remember that ``unit``'s ``outputs`` were just written from inputs ``fp``."""
    if not ENABLED:
        return
    stats = {str(Path(o)): _stat(Path(o)) for o in outputs}
    if any(v is None for v in stats.values()):
        return                                      # missing output: do not cache
    cache_dir(root).mkdir(parents=True, exist_ok=True)
    manifest = _load(stage, root)
    manifest[unit] = {"fingerprint": fp, "outputs": stats}
    target = _manifest_path(stage, root)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)                          # atomic replace
