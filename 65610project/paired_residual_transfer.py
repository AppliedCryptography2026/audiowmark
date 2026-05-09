#!/usr/bin/env python3
"""
Estimate an audiowmark watermark from original/watermarked pairs, then try to
transfer that estimated watermark onto other audio files.

This is the paired version of transfer_attack.py: instead of estimating the
shared watermark from watermarked outputs only, we create or consume pairs

    cover.wav  ->  watermarked.wav

and estimate the watermark as

    W(t, f) = log |STFT(watermarked)|^2 - log |STFT(cover)|^2

The estimated W is averaged across pair files, optionally median-smoothed along
frequency to remove broad host-audio mismatch, then applied to held-out target
covers by modifying their STFT magnitudes while preserving phase.

Typical use, from the repo root (same Docker /data mount idea as watermark_sweep.py):

    python 65610project/paired_residual_transfer.py --num-pairs 2 --targets first:3
    python 65610project/paired_residual_transfer.py -v --audiowmark-cmd /path/to/audiowmark \\
        --num-pairs 2 --gain 1.0 --median-width 5

If you already have original/watermarked pairs (for example from watermark_sweep),
list them in a two-column file (paths relative to the repo root or to 65610project):

    # pairs.txt
    soundfiles/cover_a.wav  soundfiles/wm_a.wav
    soundfiles/cover_b.wav  soundfiles/wm_b.wav

Then:

    python 65610project/paired_residual_transfer.py --pairs-file pairs.txt \\
        --targets-manifest targets.txt --save-estimate data/paired_transfer/w_est.npy

The script defaults to Docker image audiowmark:latest. Pass --audiowmark-cmd if
you have a native audiowmark binary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sps
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
WAV_DIR = DATA_DIR / "wav"
PAIR_DIR = DATA_DIR / "paired_transfer"
RESULTS_DIR = DATA_DIR / "results"
KEY_DIR = DATA_DIR / "keys"

DEFAULT_KEY = KEY_DIR / "paired_transfer.key"
DEFAULT_MESSAGE = "0123456789abcdef0011223344556677"
DEFAULT_DOCKER_IMAGE = "audiowmark:latest"

SAMPLE_RATE = 44100
N_FFT = 1024
HOP = 256
WINDOW = "hann"
EPS = 1e-12

SYNC_THRESHOLD = 0.35
ERROR_THRESHOLD = 0.5

log = logging.getLogger("paired_residual_transfer")


class Audiowmark:
    """Wrapper that runs audiowmark either natively or via Docker.

    This intentionally mirrors watermark_sweep.py so both experiment scripts
    invoke the same Docker image with the same /data mount convention.
    All file paths must live under REPO_ROOT.
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
            message_hex: str, strength: float) -> None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
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
        proc = self.run(args, [key_file, in_wav, out_wav])
        if proc.returncode != 0:
            raise RuntimeError(f"add failed for {in_wav.name}: {proc.stderr.strip()}")

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


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.T, sr


def save_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = x.T
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = out / peak * 0.999
    sf.write(str(path), out, sr, subtype="PCM_16")


def stft(x: np.ndarray) -> np.ndarray:
    _, _, z = sps.stft(
        x,
        fs=SAMPLE_RATE,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        window=WINDOW,
        boundary="zeros",
        padded=True,
    )
    return z


def istft(z: np.ndarray, expected_samples: int) -> np.ndarray:
    _, x = sps.istft(
        z,
        fs=SAMPLE_RATE,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        window=WINDOW,
        boundary=True,
    )
    if x.shape[-1] >= expected_samples:
        return x[..., :expected_samples]
    pad = expected_samples - x.shape[-1]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])


def log_power(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    x, sr = load_wav(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path.name} has sr={sr}, expected {SAMPLE_RATE}")
    z = stft(x)
    return np.log(np.abs(z) ** 2 + EPS).astype(np.float64), x, sr


def estimate_from_pairs(covers: List[Path], watermarked: List[Path],
                        median_width: int) -> np.ndarray:
    if len(covers) != len(watermarked):
        raise ValueError("covers and watermarked lists must have the same length")

    residual_sum: Optional[np.ndarray] = None
    t_max: Optional[int] = None
    for cover_path, wm_path in zip(covers, watermarked):
        cover_logp, _, _ = log_power(cover_path)
        wm_logp, _, _ = log_power(wm_path)
        t = min(cover_logp.shape[-1], wm_logp.shape[-1])
        residual = wm_logp[..., :t] - cover_logp[..., :t]
        if residual_sum is None:
            residual_sum = residual
            t_max = t
        else:
            t = min(t_max, residual.shape[-1])
            residual_sum = residual_sum[..., :t] + residual[..., :t]
            t_max = t

    if residual_sum is None:
        raise RuntimeError("no pairs provided")

    w_est = residual_sum / len(covers)
    if median_width > 1:
        smooth = ndi.median_filter(w_est, size=(1, median_width, 1), mode="reflect")
        w_est = w_est - smooth
    return w_est


def apply_estimate(target_path: Path, w_est: np.ndarray, gain: float,
                   out_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    orig, sr = load_wav(target_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{target_path.name} has sr={sr}, expected {SAMPLE_RATE}")
    z = stft(orig)
    mag = np.abs(z)
    phase = np.angle(z)
    logp = np.log(mag ** 2 + EPS)

    n_ch = min(logp.shape[0], w_est.shape[0])
    n_f = min(logp.shape[1], w_est.shape[1])
    n_t = min(logp.shape[2], w_est.shape[2])
    logp[:n_ch, :n_f, :n_t] += gain * w_est[:n_ch, :n_f, :n_t]

    forged_z = np.exp(logp / 2.0) * np.exp(1j * phase)
    forged = istft(forged_z, expected_samples=orig.shape[-1])
    save_wav(out_path, forged, sr)
    return orig, forged


def snr_db(orig: np.ndarray, forged: np.ndarray) -> float:
    n = min(orig.shape[-1], forged.shape[-1])
    diff = orig[..., :n] - forged[..., :n]
    return 20.0 * float(np.log10(np.linalg.norm(orig[..., :n]) / (np.linalg.norm(diff) + 1e-12)))


def best_match(matches: list, expected_bits: str) -> Optional[dict]:
    correct = [m for m in matches if m.get("bits", "").lower() == expected_bits.lower()]
    if correct:
        return max(correct, key=lambda m: m.get("quality", 0))
    if matches:
        return max(matches, key=lambda m: m.get("quality", 0))
    return None


def list_wavs(wav_dir: Path) -> List[Path]:
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no WAV files found in {wav_dir}")
    return wavs


def parse_indices(spec: str, max_len: int) -> List[int]:
    if spec.startswith("first:"):
        return list(range(min(int(spec.split(":", 1)[1]), max_len)))
    return [int(x) for x in spec.split(",") if x.strip()]


def resolve_audio_path(spec: str) -> Path:
    """Resolve a path from the pairs/targets manifest; must exist on disk."""
    raw = spec.strip()
    if not raw or raw.startswith("#"):
        raise ValueError("empty path")
    p = Path(raw)
    if p.is_absolute():
        out = p.resolve()
    else:
        out = None
        for base in (REPO_ROOT, PROJECT_DIR):
            cand = (base / p).resolve()
            if cand.exists():
                out = cand
                break
        if out is None:
            raise SystemExit(f"path not found (tried repo root and 65610project): {raw}")
    if not out.exists():
        raise SystemExit(f"missing file: {out}")
    return out


def load_pairs_file(path: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f"{path}:{i}: expected two whitespace-separated paths")
        pairs.append((resolve_audio_path(parts[0]), resolve_audio_path(parts[1])))
    if not pairs:
        raise SystemExit(f"no pairs in {path}")
    return pairs


def load_paths_manifest(path: Path) -> List[Path]:
    out: List[Path] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        out.append(resolve_audio_path(line))
    if not out:
        raise SystemExit(f"no paths in {path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--wav-dir", type=Path, default=WAV_DIR)
    p.add_argument("--out-dir", type=Path, default=PAIR_DIR)
    p.add_argument("--key", type=Path, default=DEFAULT_KEY)
    p.add_argument("--message", default=DEFAULT_MESSAGE)
    p.add_argument("--strength", type=float, default=10.0)
    p.add_argument("--num-pairs", type=int, default=2,
                   help="How many originals to watermark and compare")
    p.add_argument("--targets", default="first:3",
                   help="Held-out target indices after the pair set, or comma-list")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Multiplier for the estimated watermark before applying")
    p.add_argument("--median-width", type=int, default=1,
                   help="Frequency median-filter width; 1 keeps raw pair residual")
    p.add_argument("--reuse-pairs", action="store_true",
                   help="Reuse existing pair watermarked files instead of recreating them")
    p.add_argument("--pairs-file", type=Path,
                   help="Two columns per line: cover.wav watermarked.wav "
                        "(skip synthesizing pairs; paths relative to repo or 65610project)")
    p.add_argument("--targets-manifest", type=Path,
                   help="One WAV path per line for transfer targets "
                        "(default: held-out files from --wav-dir when not using --pairs-file)")
    p.add_argument("--save-estimate", type=Path,
                   help="Write the averaged log-power residual W_hat as .npy (shape ch x freq x time)")
    p.add_argument("--estimate-only", action="store_true",
                   help="Only build W_hat (and optional --save-estimate); skip transfer and decode")
    p.add_argument("--out-csv", type=Path)
    p.add_argument("--audiowmark-cmd",
                   help="Path to native audiowmark binary; default uses Docker")
    p.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    targets: List[Path]

    if args.pairs_file:
        pair_list = load_pairs_file(args.pairs_file.resolve())
        pair_covers = [c for c, _ in pair_list]
        pair_wms = [w for _, w in pair_list]
        cover_set = {p.resolve() for p in pair_covers}

        if args.targets_manifest:
            targets = load_paths_manifest(args.targets_manifest.resolve())
        else:
            wavs = list_wavs(args.wav_dir)
            held_candidates = [w for w in wavs if w.resolve() not in cover_set]
            if not held_candidates:
                raise SystemExit(
                    "no target WAVs left under --wav-dir after excluding pair covers; "
                    "pass --targets-manifest"
                )
            target_indices = parse_indices(args.targets, len(held_candidates))
            targets = [held_candidates[i] for i in target_indices]
    else:
        wavs = list_wavs(args.wav_dir)
        if args.num_pairs >= len(wavs):
            raise SystemExit(f"--num-pairs={args.num_pairs} leaves no held-out targets")

        pair_covers = wavs[:args.num_pairs]
        held_out = wavs[args.num_pairs:]
        if args.targets_manifest:
            targets = load_paths_manifest(args.targets_manifest.resolve())
        else:
            target_indices = parse_indices(args.targets, len(held_out))
            targets = [held_out[i] for i in target_indices]

    if args.estimate_only and args.pairs_file:
        log.info(
            "estimating W only from pairs file (%d pairs); no audiowmark needed",
            len(pair_covers),
        )
        t0 = time.time()
        w_est = estimate_from_pairs(pair_covers, pair_wms, args.median_width)
        log.info(
            "W estimate ready in %.2fs; shape=%s rms=%.5f",
            time.time() - t0,
            w_est.shape,
            float(np.sqrt(np.mean(w_est ** 2))),
        )
        if args.save_estimate:
            args.save_estimate.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_estimate, w_est)
            log.info("saved W_hat to %s", args.save_estimate)
        print()
        print("=" * 80)
        print(f"pairs used: {len(pair_covers)}")
        print("estimate-only: no forged wavs produced (pairs supplied on disk)")
        if args.save_estimate:
            print(f"W_hat: {args.save_estimate}")
        elif not args.save_estimate:
            print("(pass --save-estimate PATH.npy to persist W_hat)")
        print("=" * 80)
        return

    if args.audiowmark_cmd:
        if not (shutil.which(args.audiowmark_cmd) or Path(args.audiowmark_cmd).exists()):
            raise SystemExit(f"audiowmark binary not found: {args.audiowmark_cmd}")
        am = Audiowmark(args.audiowmark_cmd, args.docker_image)
    else:
        if shutil.which("docker") is None:
            raise SystemExit("docker not found; pass --audiowmark-cmd <path>")
        am = Audiowmark(None, args.docker_image)

    if not args.key.exists():
        log.info("generating key: %s", args.key)
        am.gen_key(args.key)

    pair_wm_dir = args.out_dir / "pairs_wm"
    forged_dir = args.out_dir / f"forged_g{args.gain:0.2f}".replace(".", "p")
    json_dir = forged_dir / "json"

    if not args.pairs_file:
        pair_wms = [pair_wm_dir / p.name for p in pair_covers]

        for cover, wm in zip(pair_covers, pair_wms):
            if args.reuse_pairs and wm.exists() and wm.stat().st_size > 0:
                log.info("reusing pair watermark: %s", wm.name)
                continue
            log.info("watermarking pair cover: %s", cover.name)
            am.add(args.key, cover, wm, args.message, args.strength)

    log.info("estimating W from %d original/watermarked pairs", len(pair_covers))
    t0 = time.time()
    w_est = estimate_from_pairs(pair_covers, pair_wms, args.median_width)
    log.info("W estimate ready in %.2fs; shape=%s rms=%.5f",
             time.time() - t0, w_est.shape, float(np.sqrt(np.mean(w_est ** 2))))

    if args.save_estimate:
        args.save_estimate.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_estimate, w_est)
        log.info("saved W_hat to %s", args.save_estimate)

    if args.estimate_only:
        print()
        print("=" * 80)
        print(f"pairs used: {len(pair_covers)}")
        print("estimate-only: no forged wavs produced")
        if args.save_estimate:
            print(f"W_hat: {args.save_estimate}")
        print("=" * 80)
        return

    if args.out_csv is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_csv = RESULTS_DIR / f"paired_transfer_{ts}.csv"
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "target", "forged_wav", "decoded_message", "decode_correct",
        "best_quality", "best_error", "best_type", "sync_ok", "snr_db",
    ]

    successes = 0
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for target in targets:
            forged = forged_dir / target.name
            json_out = json_dir / f"{target.stem}.json"
            log.info("applying estimate to target: %s", target.name)
            orig, forged_audio = apply_estimate(target, w_est, args.gain, forged)
            snr = snr_db(orig, forged_audio)

            payload = am.get_json(args.key, forged, json_out)
            best = best_match(payload.get("matches", []), args.message)
            decoded = best.get("bits", "") if best else ""
            quality = float(best.get("quality", 0.0)) if best else 0.0
            err = float(best.get("error", 1.0)) if best else 1.0
            mtype = best.get("type", "") if best else ""
            correct = decoded.lower() == args.message.lower()
            sync_ok = quality >= SYNC_THRESHOLD and err <= ERROR_THRESHOLD
            if correct and sync_ok:
                successes += 1

            writer.writerow({
                "target": target.name,
                "forged_wav": str(forged),
                "decoded_message": decoded,
                "decode_correct": int(correct),
                "best_quality": f"{quality:.4f}",
                "best_error": f"{err:.4f}",
                "best_type": mtype,
                "sync_ok": int(sync_ok),
                "snr_db": f"{snr:.2f}",
            })
            f.flush()

    print()
    print("=" * 80)
    print(f"pairs used: {len(pair_covers)}")
    print(f"targets tried: {len(targets)}")
    print(f"successful transferred decodes: {successes}/{len(targets)}")
    print(f"forged wavs: {forged_dir}")
    print(f"csv: {args.out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
