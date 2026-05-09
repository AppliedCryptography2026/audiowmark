#!/usr/bin/env python3
"""
Retrieve FMA audio files for the 6.5610 watermark-transfer-attack experiments.

Pulls N random tracks from the FMA "small" subset (8000 30-second clips,
hosted at https://os.unil.cloud.switch.ch/fma/fma_small.zip) without
downloading the full 7.2 GB archive: uses HTTP range requests via
``remotezip`` to fetch only the MP3 members we actually need.

Each MP3 is converted to a stereo 44.1 kHz WAV (the format audiowmark was
designed for) using ``ffmpeg``.

Track selection is deterministic given a seed, and subsets nest: running with
``--num 80`` produces a superset of ``--num 40`` produces a superset of
``--num 5`` (with the same seed). Re-running is incremental: anything already
downloaded or converted is reused.

Usage:
    python retreive_soundfiles.py --num 80
    python retreive_soundfiles.py --num 5 --seed 42
    python retreive_soundfiles.py --list-only --num 80
    python 65610project/retreive_soundfiles.py --num 80 --seed 6561
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from remotezip import RemoteZip
from tqdm import tqdm


FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"

# Tracks documented as zero-length / corrupt in the FMA repo.
KNOWN_BAD_TRACKS = {"099134", "108925", "133297"}

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MP3_DIR = DATA_DIR / "mp3"
WAV_DIR = DATA_DIR / "wav"
CACHE_DIR = PROJECT_DIR / "cache"
NAMELIST_CACHE = CACHE_DIR / "fma_small_namelist.json"

log = logging.getLogger("retrieve_soundfiles")


def get_namelist() -> List[str]:
    """Return all MP3 member paths inside ``fma_small.zip``, cached locally.

    The first call costs a single round-trip to fetch the zip's central
    directory (a few MB out of 7.2 GB); subsequent calls hit the JSON cache.
    """
    if NAMELIST_CACHE.exists():
        names = json.loads(NAMELIST_CACHE.read_text())
        log.info("Loaded %d MP3 paths from cache (%s)", len(names), NAMELIST_CACHE)
        return names

    log.info("Fetching central directory from %s ...", FMA_SMALL_URL)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with RemoteZip(FMA_SMALL_URL) as rz:
        names = sorted(n for n in rz.namelist() if n.endswith(".mp3"))
    NAMELIST_CACHE.write_text(json.dumps(names))
    log.info("Cached %d MP3 paths to %s", len(names), NAMELIST_CACHE)
    return names


def select_tracks(namelist: List[str], n: int, seed: int) -> List[str]:
    """Pick ``n`` track paths deterministically from ``namelist``.

    Uses a single shuffle with a fixed seed so subsets nest across runs.
    """
    usable = [p for p in namelist if Path(p).stem not in KNOWN_BAD_TRACKS]
    rng = random.Random(seed)
    shuffled = usable.copy()
    rng.shuffle(shuffled)
    if n > len(shuffled):
        raise SystemExit(f"Requested {n} tracks but only {len(shuffled)} available")
    return shuffled[:n]


def fetch_mp3s(paths: List[str]) -> List[Path]:
    """Download (or reuse) MP3 members from the remote zip into ``MP3_DIR``."""
    MP3_DIR.mkdir(parents=True, exist_ok=True)

    out: List[Path] = []
    todo: List[tuple[str, Path]] = []
    for member in paths:
        local = MP3_DIR / Path(member).name
        out.append(local)
        if not local.exists() or local.stat().st_size == 0:
            todo.append((member, local))

    if not todo:
        log.info("All %d MP3s already cached locally.", len(out))
        return out

    log.info("Fetching %d MP3 files from FMA via range requests ...", len(todo))
    with RemoteZip(FMA_SMALL_URL) as rz:
        for member, local in tqdm(todo, unit="file", desc="mp3"):
            tmp = local.with_suffix(".mp3.part")
            with rz.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.rename(local)
    return out


def convert_to_wav(mp3_paths: List[Path], sample_rate: int, channels: int) -> List[Path]:
    """Convert each MP3 to a 16-bit PCM WAV at the given rate / channel count."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) and retry."
        )

    WAV_DIR.mkdir(parents=True, exist_ok=True)

    out: List[Path] = []
    todo: List[tuple[Path, Path]] = []
    for mp3 in mp3_paths:
        wav = WAV_DIR / (mp3.stem + ".wav")
        out.append(wav)
        if not wav.exists() or wav.stat().st_size == 0:
            todo.append((mp3, wav))

    if not todo:
        log.info("All %d WAVs already converted.", len(out))
        return out

    log.info("Converting %d files to WAV (%d Hz, %d ch) ...",
             len(todo), sample_rate, channels)
    for mp3, wav in tqdm(todo, unit="file", desc="wav"):
        tmp = wav.with_suffix(".wav.part")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp3),
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-c:a", "pcm_s16le",
            "-f", "wav",
            str(tmp),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            log.warning("ffmpeg failed on %s: %s; skipping", mp3.name, e)
            tmp.unlink(missing_ok=True)
            continue
        tmp.rename(wav)
    return [w for w in out if w.exists() and w.stat().st_size > 0]


def write_manifest(wav_paths: List[Path], n: int, seed: int) -> Path:
    """Record which files belong to this run, for reproducibility."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = DATA_DIR / f"manifest_n{n}_seed{seed}.txt"
    rel = [str(p.relative_to(PROJECT_DIR)) for p in wav_paths]
    manifest.write_text("\n".join(rel) + "\n")
    log.info("Wrote manifest %s (%d entries)", manifest, len(rel))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--num", "-n", type=int, default=80,
                   help="Number of tracks to fetch (default 80)")
    p.add_argument("--seed", "-s", type=int, default=6561,
                   help="Random seed for track selection (default 6561)")
    p.add_argument("--sample-rate", type=int, default=44100,
                   help="Output WAV sample rate in Hz (default 44100)")
    p.add_argument("--channels", type=int, default=2,
                   help="Output WAV channel count (default 2 = stereo)")
    p.add_argument("--list-only", action="store_true",
                   help="Print the selected tracks and exit (no download)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    namelist = get_namelist()
    selected = select_tracks(namelist, args.num, args.seed)

    if args.list_only:
        for member in selected:
            print(member)
        return

    log.info("Selected %d tracks (seed=%d).", len(selected), args.seed)
    mp3s = fetch_mp3s(selected)
    wavs = convert_to_wav(mp3s, args.sample_rate, args.channels)
    write_manifest(wavs, args.num, args.seed)

    print(f"\nDone. {len(wavs)} WAV files in {WAV_DIR}")
    if len(wavs) < args.num:
        print(f"(Note: {args.num - len(wavs)} track(s) were skipped due to conversion errors.)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
