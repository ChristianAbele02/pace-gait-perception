#!/usr/bin/env python3
"""Start or stop the experiment on every render node.

    rig/ill.py start [EXE]     # EXE relative to ILL_RENDER_DEST, default MainExperiment.exe
    rig/ill.py stop

Each render node runs a small listener (part of the lab installation, not of this
repository) on ILL_CONTROL_PORT. It accepts a msgpack message, ``["start", <path>]`` or
``["stop"]``, signed with an Ed25519 key, and launches or kills the given executable.

The signing seed is a secret and is never stored in the repository. It is read from the
environment variable ILL_SIGNING_KEY (64 hex characters) or, failing that, from the file
named by ILL_SIGNING_KEY_FILE in rig/rig.env. One-time setup on the head node::

    mkdir -p ~/.pace && chmod 700 ~/.pace
    printf '%s' '<64-hex-seed>' > ~/.pace/signing_key && chmod 600 ~/.pace/signing_key

Hosts, users, port and build folder come from rig/rig.env (see rig.env.example).
Requires PyNaCl and msgpack.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import msgpack
import nacl.encoding
import nacl.signing

RIG_DIR = Path(__file__).resolve().parent


def load_rig_env(path: Path = RIG_DIR / "rig.env") -> dict[str, str]:
    """KEY=value pairs from rig.env; quotes stripped, comment lines skipped."""
    if not path.exists():
        sys.exit(f"{path} not found. Copy rig.env.example to rig.env and fill it in.")
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_signing_key(env: dict[str, str]) -> nacl.signing.SigningKey:
    seed = os.environ.get("ILL_SIGNING_KEY", "").strip()
    if not seed:
        key_file = Path(os.path.expanduser(env.get("ILL_SIGNING_KEY_FILE", "~/.pace/signing_key")))
        if not key_file.exists():
            sys.exit("No signing key: set ILL_SIGNING_KEY or create the file named by "
                     f"ILL_SIGNING_KEY_FILE ({key_file}). See the docstring of rig/ill.py.")
        seed = key_file.read_text(encoding="utf-8").strip()
    if len(seed) != 64:
        sys.exit("The signing key must be 64 hexadecimal characters.")
    return nacl.signing.SigningKey(seed.encode(), encoder=nacl.encoding.HexEncoder)


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("start", "stop"):
        sys.exit(__doc__)
    env = load_rig_env()
    key = load_signing_key(env)
    port = int(env.get("ILL_CONTROL_PORT", "9999"))
    dest = env.get("ILL_RENDER_DEST", "Projects/PACE").replace("/", "\\")
    exe = argv[1] if len(argv) > 1 else "MainExperiment.exe"

    for pair in env.get("ILL_RENDER_NODES", "").split():
        host, user = pair.split(":", 1)
        print(f"Connecting to {host}")
        if argv[0] == "start":
            payload = f"C:\\Users\\{user}\\{dest}\\{exe}"
            message = msgpack.packb(["start", payload], use_bin_type=True)
        else:
            message = msgpack.packb(["stop"], use_bin_type=True)
        with socket.create_connection((host, port), timeout=10) as sock:
            sock.sendall(key.sign(message))


if __name__ == "__main__":
    main(sys.argv[1:])
