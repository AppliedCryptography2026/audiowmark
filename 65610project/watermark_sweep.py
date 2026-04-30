#!/usr/bin/env python3
"""
Watermark every file in a manifest with a single key K and record per-file
sync score and decoding error from ``audiowmark get``.

This produces the *baseline* used by the watermark-transfer attack: the
detector must succeed on every individual watermarked file before we try to
estimate-and-subtract the watermark.

Two message modes:
  * ``--message-mode same``  : every file embeds the same 128-bit message
                               (the canonical attack scenario).
  * ``--message-mode unique``: every file embeds a deterministic-but-distinct
                               message derived as HMAC-SHA256(K, track_id)[:16]
                               (lets us check whether per-file message
                               diversity defeats the attack).

Usage examples:
    python watermark_sweep.py --num 80 --message-mode same
    python watermark_sweep.py --num 80 --message-mode unique --strength 12

The script auto-detects the ``audiowmark:latest`` Docker image and invokes it
with the audiowmark repo root mounted at /data. Pass ``--audiowmark-cmd
PATH`` to use a native binary instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
WM_DIR = DATA_DIR / "wm"
KEY_DIR = DATA_DIR / "keys"
RESULTS_DIR = DATA_DIR / "results"
JSON_DIR = DATA_DIR / "wm_json"

DEFAULT_KEY = KEY_DIR / "wmark.key"
DEFAULT_SAME_MESSAGE = "0123456789abcdef0011223344556677"
DEFAULT_DOCKER_IMAGE = "audiowmark:latest"

log = logging.getLogger("watermark_sweep")


# ---------- audiowmark invocation ----------------------------------------

class Audiowmark:
    """Wrapper that runs audiowmark either natively or via Docker.

    All file paths must live under ``REPO_ROOT`` so they can be exposed to the
    container at /data.
    """

    def __init__(self, native_cmd: Optional[str], docker_image: str):
        self._docker = native_cmd is None
        self._cmd = native_cmd
        self._image = docker_image
        if self._docker:
            log.debug("Using Docker image %s", docker_image)
        else:
            log.debug("Using native binary %s", native_cmd)

    def _to_data(self, p: Path) -> str:
        """Translate a host path under REPO_ROOT to its /data-mounted form."""
        rel = Path(p).resolve().relative_to(REPO_ROOT)
        return f"/data/{rel.as_posix()}"

    def run(self, args: List[str], path_args: List[Path]) -> subprocess.CompletedProcess:
        """Run audiowmark with the given CLI args, with paths mapped if needed."""
        if self._docker:
            cmd = [
                "docker", "run", "--rm",
                "-u", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{REPO_ROOT}:/data",
                "--entrypoint", "/usr/local/bin/audiowmark",
                self._image,
                *args,
            ]
        else:
            cmd = [self._cmd, *args]
        log.debug("exec: %s", " ".join(cmd))
        return subprocess.run(cmd, check=False, capture_output=True, text=True)

    # convenience methods -----------------------------------------------

    def gen_key(self, key_file: Path) -> None:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        if self._docker:
            args = ["gen-key", self._to_data(key_file)]
        else:
            args = ["gen-key", str(key_file)]
        proc = self.run(args, [key_file])
        if proc.returncode != 0:
            raise RuntimeError(f"gen-key failed: {proc.stderr.strip()}")

    def add(self, key_file: Path, in_wav: Path, out_wav: Path,
            message_hex: str, strength: int) -> subprocess.CompletedProcess:
        if self._docker:
            args = [
                "add",
                "--key", self._to_data(key_file),
                "--strength", str(strength),
                self._to_data(in_wav),
                self._to_data(out_wav),
                message_hex,
            ]
        else:
            args = [
                "add",
                "--key", str(key_file),
                "--strength", str(strength),
                str(in_wav),
                str(out_wav),
                message_hex,
            ]
        return self.run(args, [key_file, in_wav, out_wav])

    def get_json(self, key_file: Path, wav: Path, json_out: Path) -> dict:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        if self._docker:
            args = [
                "get",
                "--key", self._to_data(key_file),
                "--json", self._to_data(json_out),
                self._to_data(wav),
            ]
        else:
            args = [
                "get",
                "--key", str(key_file),
                "--json", str(json_out),
                str(wav),
            ]
        proc = self.run(args, [key_file, wav, json_out])
        if proc.returncode != 0:
            raise RuntimeError(f"get failed for {wav.name}: {proc.stderr.strip()}")
        return json.loads(json_out.read_text())


# ---------- message derivation ------------------------------------------

def derive_unique_message(track_id: str, secret: bytes) -> str:
    """Deterministic distinct 128-bit message per track id."""
    digest = hmac.new(secret, track_id.encode(), hashlib.sha256).digest()
    return digest[:16].hex()


# ---------- sweep --------------------------------------------------------

def best_match(matches: list, expected_bits: str) -> Optional[dict]:
    """Pick the highest-quality match whose bits equal the expected message.

    Falls back to the highest-quality match overall if none decode correctly.
    """
    correct = [m for m in matches if m.get("bits", "").lower() == expected_bits.lower()]
    if correct:
        return max(correct, key=lambda m: m.get("quality", 0))
    if matches:
        return max(matches, key=lambda m: m.get("quality", 0))
    return None


def best_pattern(matches: list, expected_bits: str, types: tuple) -> Optional[dict]:
    """Among matches whose ``type`` is in ``types``, return the best-quality
    match decoding to ``expected_bits``."""
    candidates = [m for m in matches
                  if m.get("type") in types
                  and m.get("bits", "").lower() == expected_bits.lower()]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.get("quality", 0))


def load_manifest(path: Path, n: Optional[int]) -> List[Path]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    paths = [PROJECT_DIR / ln for ln in lines]
    if n is not None:
        paths = paths[:n]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"manifest entry missing on disk: {p}")
    return paths


def run_sweep(am: Audiowmark, key_file: Path, inputs: List[Path],
              message_mode: str, fixed_message: str,
              unique_secret: bytes, strength: int,
              out_csv: Path,
              wm_dir: Path, json_dir: Path) -> dict:
    wm_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "track_id", "input_wav", "wm_wav", "expected_message", "decoded_message",
        "decode_correct", "best_quality", "best_error", "best_type",
        "all_quality", "all_error", "add_seconds", "get_seconds",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    sync_scores: List[float] = []
    errors: List[float] = []

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i, in_wav in enumerate(inputs, 1):
            track_id = in_wav.stem
            wm_wav = wm_dir / f"{track_id}.wav"
            json_out = json_dir / f"{track_id}.json"

            if message_mode == "same":
                msg = fixed_message
            elif message_mode == "unique":
                msg = derive_unique_message(track_id, unique_secret)
            else:
                raise SystemExit(f"unknown message-mode: {message_mode}")

            log.info("[%d/%d] %s  msg=%s", i, len(inputs), track_id, msg)

            t0 = time.time()
            proc = am.add(key_file, in_wav, wm_wav, msg, strength)
            t_add = time.time() - t0
            if proc.returncode != 0:
                log.warning("add failed for %s: %s", track_id, proc.stderr.strip())
                continue

            t0 = time.time()
            payload = am.get_json(key_file, wm_wav, json_out)
            t_get = time.time() - t0

            matches = payload.get("matches", [])
            best = best_match(matches, msg)
            all_match = best_pattern(matches, msg, ("all",))

            decoded = best.get("bits") if best else ""
            correct = decoded.lower() == msg.lower() if decoded else False
            quality = best.get("quality") if best else None
            err = best.get("error") if best else None
            mtype = best.get("type") if best else None

            if correct:
                successes += 1
                if quality is not None:
                    sync_scores.append(quality)
                if err is not None:
                    errors.append(err)

            w.writerow({
                "track_id": track_id,
                "input_wav": str(in_wav.relative_to(PROJECT_DIR)),
                "wm_wav": str(wm_wav.relative_to(PROJECT_DIR)),
                "expected_message": msg,
                "decoded_message": decoded,
                "decode_correct": int(correct),
                "best_quality": quality if quality is not None else "",
                "best_error": err if err is not None else "",
                "best_type": mtype or "",
                "all_quality": all_match.get("quality") if all_match else "",
                "all_error": all_match.get("error") if all_match else "",
                "add_seconds": f"{t_add:.3f}",
                "get_seconds": f"{t_get:.3f}",
            })
            f.flush()

    summary = {
        "files": len(inputs),
        "decode_success": successes,
        "decode_success_rate": successes / len(inputs) if inputs else 0,
        "median_sync": sorted(sync_scores)[len(sync_scores)//2] if sync_scores else None,
        "median_error": sorted(errors)[len(errors)//2] if errors else None,
        "csv": str(out_csv),
    }
    return summary


# ---------- main ---------------------------------------------------------

def latest_manifest(num: Optional[int], seed: int) -> Path:
    if num is not None:
        candidate = DATA_DIR / f"manifest_n{num}_seed{seed}.txt"
        if candidate.exists():
            return candidate
    found = sorted(DATA_DIR.glob("manifest_n*_seed*.txt"))
    if not found:
        raise SystemExit("No manifest found. Run retreive_soundfiles.py first.")
    return found[-1]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path,
                   help="Path to a manifest file (default: latest in data/)")
    p.add_argument("--num", "-n", type=int,
                   help="Truncate manifest to first N entries")
    p.add_argument("--seed", "-s", type=int, default=6561,
                   help="Seed used to locate the default manifest (default 6561)")
    p.add_argument("--key", type=Path, default=DEFAULT_KEY,
                   help="Watermarking key file (created with gen-key if absent)")
    p.add_argument("--message-mode", choices=("same", "unique"), default="same",
                   help="Same fixed message for all files, or unique per file")
    p.add_argument("--message", default=DEFAULT_SAME_MESSAGE,
                   help="128-bit hex message for --message-mode same")
    p.add_argument("--unique-secret", default="6.5610-watermark-transfer",
                   help="Secret used to derive per-file unique messages")
    p.add_argument("--strength", type=int, default=10,
                   help="audiowmark strength (default 10)")
    p.add_argument("--out", type=Path,
                   help="Output CSV path (default: data/results/sweep_<mode>_<ts>.csv)")
    p.add_argument("--wm-dir", type=Path,
                   help="Where to write watermarked WAVs "
                        "(default: data/wm for mode=same, data/wm_unique for mode=unique)")
    p.add_argument("--json-dir", type=Path,
                   help="Where to write per-file get JSON "
                        "(default: data/wm_json or data/wm_json_unique)")
    p.add_argument("--audiowmark-cmd",
                   help="Path to native audiowmark binary; default uses Docker")
    p.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE,
                   help=f"Docker image to run (default: {DEFAULT_DOCKER_IMAGE})")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = args.manifest or latest_manifest(args.num, args.seed)
    inputs = load_manifest(manifest, args.num)
    log.info("Manifest: %s (%d files)", manifest, len(inputs))

    if args.audiowmark_cmd:
        if not shutil.which(args.audiowmark_cmd) and not Path(args.audiowmark_cmd).exists():
            raise SystemExit(f"audiowmark binary not found: {args.audiowmark_cmd}")
        am = Audiowmark(native_cmd=args.audiowmark_cmd, docker_image=args.docker_image)
    else:
        if shutil.which("docker") is None:
            raise SystemExit("docker not found; pass --audiowmark-cmd <path> instead")
        am = Audiowmark(native_cmd=None, docker_image=args.docker_image)

    if not args.key.exists():
        log.info("Generating watermarking key %s", args.key)
        am.gen_key(args.key)
    else:
        log.info("Using existing key %s", args.key)

    out_csv = args.out
    if out_csv is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_csv = RESULTS_DIR / f"sweep_{args.message_mode}_n{len(inputs)}_s{args.strength}_{ts}.csv"

    suffix = "" if args.message_mode == "same" else f"_{args.message_mode}"
    wm_dir = args.wm_dir or DATA_DIR / f"wm{suffix}"
    json_dir = args.json_dir or DATA_DIR / f"wm_json{suffix}"
    log.info("Watermarked WAVs -> %s", wm_dir)

    summary = run_sweep(
        am, args.key, inputs,
        message_mode=args.message_mode,
        fixed_message=args.message,
        unique_secret=args.unique_secret.encode(),
        strength=args.strength,
        out_csv=out_csv,
        wm_dir=wm_dir, json_dir=json_dir,
    )

    print()
    print("=" * 60)
    print(f"files                    {summary['files']}")
    print(f"decode success           {summary['decode_success']}/{summary['files']} "
          f"({summary['decode_success_rate']*100:.1f}%)")
    if summary["median_sync"] is not None:
        print(f"median sync (quality)    {summary['median_sync']:.3f}")
    if summary["median_error"] is not None:
        print(f"median decoding error    {summary['median_error']:.3f}")
    print(f"csv                      {summary['csv']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
