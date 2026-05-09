#!/usr/bin/env python3
"""
Watermark-transfer attack against audiowmark.

Premise
-------
audiowmark's patchwork algorithm perturbs a key-dependent set of "up" and
"down" frequency bins per frame: the *bin selection* is deterministic in
``K`` (and the frame index), and so is identical across every file watermarked
with the same ``K`` and message. In the log-power spectrogram domain that
perturbation is approximately additive:

    log |X_i(t,f)|^2  =  log |C_i(t,f)|^2  +  W(t,f)

where ``X_i`` is watermarked file i, ``C_i`` is its cover, and ``W`` is the
key-induced bias that is constant across i. Averaging across many files
suppresses the cover content (covers are independent, watermark is consistent),
and a frequency-axis median-smoothing isolates the *spiky* bin-by-bin pattern
that is the watermark signature. Subtract that estimate from a target file's
spectrogram, ISTFT, and you have a "cleaned" file that should fail to decode.

Outputs
-------
* one cleaned WAV per (N, target) into data/attack/N{N}/<track>.wav
* per-attack ``audiowmark get`` JSON in data/attack/N{N}/json/<track>.json
* a single CSV summarising every (N, target) pair: detector success, sync
  score, decoding error, and audio-quality metrics (SNR, log-spectral distance)

Usage
-----
    python transfer_attack.py --num-list 5,10,20,40,80
    python transfer_attack.py --num-list 80 --gain 1.0 --median-width 9 \\
        --targets first:5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import hashlib
import hmac

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sps
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
WM_DIR = DATA_DIR / "wm"
ATTACK_DIR = DATA_DIR / "attack"
RESULTS_DIR = DATA_DIR / "results"
DEFAULT_KEY = DATA_DIR / "keys" / "wmark.key"

DEFAULT_DOCKER_IMAGE = "audiowmark:latest"
DEFAULT_MESSAGE = "0123456789abcdef0011223344556677"

# Match audiowmark's analysis frame size (Params::frame_size).
N_FFT = 1024
# 4x overlap gives more frames to average and a smoother spectrogram.
HOP = 256
WINDOW = "hann"
EPS = 1e-12
SAMPLE_RATE = 44100
# audiowmark only embeds in FFT bins [MIN_BAND, MAX_BAND] (Params::min_band/max_band).
MIN_BAND = 20
MAX_BAND = 100
# Sync threshold from Params::sync_threshold2 default; matches were considered
# valid above ~0.6 in audiowmark's normal output.
SYNC_THRESHOLD = 0.6
ERROR_THRESHOLD = 0.5  # any error > 0.5 means the message is unrecoverable

log = logging.getLogger("transfer_attack")


# ---------- audiowmark wrapper (thin) -----------------------------------

class Audiowmark:
    """Same Docker/native shim as in watermark_sweep.py, kept self-contained."""

    def __init__(self, native_cmd: Optional[str], docker_image: str):
        self._docker = native_cmd is None
        self._cmd = native_cmd
        self._image = docker_image

    def _data(self, p: Path) -> str:
        return f"/data/{Path(p).resolve().relative_to(REPO_ROOT).as_posix()}"

    def run(self, args: List[str], path_args: List[Path]) -> subprocess.CompletedProcess:
        """Run audiowmark subcommands with Docker / native invocation."""
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
            args = ["gen-key", self._data(key_file)]
        else:
            args = ["gen-key", str(key_file)]
        proc = self.run(args, [key_file])
        if proc.returncode != 0:
            raise RuntimeError(f"gen-key failed: {proc.stderr.strip()}")

    def add(self, key_file: Path, in_wav: Path, out_wav: Path,
            message_hex: str, strength: int) -> None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if self._docker:
            args = [
                "add",
                "--key", self._data(key_file),
                "--strength", str(strength),
                self._data(in_wav),
                self._data(out_wav),
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
            cmd = [
                "docker", "run", "--rm",
                "-u", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{REPO_ROOT}:/data",
                "--entrypoint", "/usr/local/bin/audiowmark",
                self._image,
                "get",
                "--key", self._data(key_file),
                "--json", self._data(json_out),
                self._data(wav),
            ]
        else:
            cmd = [
                self._cmd, "get",
                "--key", str(key_file),
                "--json", str(json_out),
                str(wav),
            ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"audiowmark get failed on {wav.name}: {proc.stderr.strip()}")
        return json.loads(json_out.read_text())


# ---------- STFT helpers ------------------------------------------------

def _stft(x: np.ndarray) -> np.ndarray:
    """Forward STFT for a (n_channels, n_samples) array, returning (n_ch, n_f, n_t)."""
    _, _, Z = sps.stft(
        x,
        fs=SAMPLE_RATE,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        window=WINDOW,
        boundary="zeros",
        padded=True,
    )
    return Z


def _istft(Z: np.ndarray, expected_samples: int) -> np.ndarray:
    """Inverse STFT, truncated/padded to the requested sample count."""
    _, x = sps.istft(
        Z,
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


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a WAV as float32 and return (n_channels, n_samples), sr."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.T, sr  # (n_ch, n_samples)


def save_wav(path: Path, x: np.ndarray, sr: int) -> None:
    """Write a (n_channels, n_samples) array as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = x.T  # (n_samples, n_channels)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = out / peak * 0.999
    sf.write(str(path), out, sr, subtype="PCM_16")


# ---------- watermark estimation ----------------------------------------

class WatermarkEstimator:
    """Online estimator over log-power spectrograms.

    Tracks per-(channel, freq, frame) running mean and second moment so we can
    compute the bin-wise standard error and apply a z-test mask: bins whose
    averaged log-power deviation is not significant are zeroed, ensuring that
    when the watermark cancels under averaging (e.g. unique per-file messages),
    almost no bins clear the threshold and the attack barely touches the
    target.
    """

    def __init__(self, median_width: int):
        self.median_width = median_width
        self._sum: Optional[np.ndarray] = None
        self._sumsq: Optional[np.ndarray] = None
        self._count = 0
        self._t_max: Optional[int] = None

    def update(self, wm_path: Path) -> None:
        x, sr = load_wav(wm_path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{wm_path.name} sr={sr}, expected {SAMPLE_RATE}")
        Z = _stft(x)
        logp = np.log(np.abs(Z) ** 2 + EPS).astype(np.float64)  # (ch, f, t)
        if self._sum is None:
            self._sum = logp.copy()
            self._sumsq = logp ** 2
            self._t_max = logp.shape[-1]
        else:
            t = min(self._t_max, logp.shape[-1])
            self._sum = self._sum[..., :t] + logp[..., :t]
            self._sumsq = self._sumsq[..., :t] + logp[..., :t] ** 2
            self._t_max = t
        self._count += 1

    @property
    def n(self) -> int:
        return self._count

    def estimate(self, z_threshold: float = 0.0,
                 active_band_only: bool = False) -> np.ndarray:
        """Return W_est with shape (n_channels, n_freq, n_frames).

        Parameters
        ----------
        z_threshold:
            If > 0, zero out bins whose |residual| / SEM is below this
            threshold (per-bin z-test). 0 disables masking.
        active_band_only:
            If True, restrict W_est to FFT bins [MIN_BAND, MAX_BAND] and
            zero everything else.
        """
        if self._count == 0:
            raise RuntimeError("no files seen yet")
        mean_logp = self._sum / self._count
        # Median-smooth along the frequency axis to get the smooth cover
        # envelope; the spiky residual is our watermark signature.
        smoothed = ndi.median_filter(
            mean_logp, size=(1, self.median_width, 1), mode="reflect",
        )
        residual = mean_logp - smoothed

        if z_threshold > 0 and self._count > 1:
            var = (self._sumsq / self._count) - mean_logp ** 2
            var = np.clip(var, 0.0, None)
            sem = np.sqrt(var / self._count) + EPS
            mask = np.abs(residual) >= z_threshold * sem
            residual = np.where(mask, residual, 0.0)

        if active_band_only:
            band_mask = np.zeros(residual.shape[1], dtype=bool)
            band_mask[MIN_BAND:MAX_BAND + 1] = True
            residual = residual * band_mask[None, :, None]

        return residual


# ---------- attack -------------------------------------------------------

def apply_attack(target_path: Path, W_est: np.ndarray, gain: float,
                 out_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract gain * W_est from target log-power spectrogram, ISTFT, save.

    Returns (original_signal, cleaned_signal), both shape (n_ch, n_samples).
    """
    orig, sr = load_wav(target_path)
    Z = _stft(orig)
    mag = np.abs(Z)
    phase = np.angle(Z)
    logp = np.log(mag ** 2 + EPS)

    n_ch = min(W_est.shape[0], logp.shape[0])
    n_f = min(W_est.shape[1], logp.shape[1])
    n_t = min(W_est.shape[2], logp.shape[2])
    logp[:n_ch, :n_f, :n_t] -= gain * W_est[:n_ch, :n_f, :n_t]

    new_mag = np.exp(logp / 2.0)
    Z_clean = new_mag * np.exp(1j * phase)
    cleaned = _istft(Z_clean, expected_samples=orig.shape[-1])
    save_wav(out_path, cleaned, sr)
    return orig, cleaned


# ---------- audio-quality metrics ---------------------------------------

def snr_db(orig: np.ndarray, clean: np.ndarray) -> float:
    n = min(orig.shape[-1], clean.shape[-1])
    o = orig[..., :n]
    c = clean[..., :n]
    diff = o - c
    num = np.linalg.norm(o)
    den = np.linalg.norm(diff) + 1e-12
    return 20.0 * float(np.log10(num / den))


def log_spectral_distance(orig: np.ndarray, clean: np.ndarray) -> float:
    """Mean per-frame log-spectral distance (dB), averaged over channels & frames."""
    n = min(orig.shape[-1], clean.shape[-1])
    Zo = _stft(orig[..., :n])
    Zc = _stft(clean[..., :n])
    log_o = 10.0 * np.log10(np.abs(Zo) ** 2 + EPS)
    log_c = 10.0 * np.log10(np.abs(Zc) ** 2 + EPS)
    return float(np.sqrt(np.mean((log_o - log_c) ** 2)))


# ---------- decoding-result helpers --------------------------------------

def best_match(matches: list, expected_bits: str) -> Optional[dict]:
    correct = [m for m in matches if m.get("bits", "").lower() == expected_bits.lower()]
    if correct:
        return max(correct, key=lambda m: m.get("quality", 0))
    if matches:
        return max(matches, key=lambda m: m.get("quality", 0))
    return None


def derive_unique_message(track_id: str, secret: bytes) -> str:
    """Match watermark_sweep.py.derive_unique_message exactly."""
    return hmac.new(secret, track_id.encode(), hashlib.sha256).digest()[:16].hex()


# ---------- driver -------------------------------------------------------

def parse_target_spec(spec: str, n_total: int) -> List[int]:
    """`all`, `first:K`, `random:K`, or comma-list of indices."""
    if spec == "all":
        return list(range(n_total))
    if spec.startswith("first:"):
        k = int(spec.split(":")[1])
        return list(range(min(k, n_total)))
    if spec.startswith("random:"):
        import random
        k = int(spec.split(":")[1])
        rng = random.Random(0xA1)
        return sorted(rng.sample(range(n_total), min(k, n_total)))
    return [int(x) for x in spec.split(",")]


def load_manifest(manifest_path: Path) -> List[Path]:
    lines = [ln.strip() for ln in manifest_path.read_text().splitlines() if ln.strip()]
    return [PROJECT_DIR / ln for ln in lines]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path,
                   default=DATA_DIR / "manifest_n80_seed6561.txt",
                   help="Manifest of cover WAVs (used to locate watermarked twins)")
    p.add_argument("--wm-dir", type=Path, default=WM_DIR,
                   help="Directory of watermarked WAVs (one per manifest entry)")
    p.add_argument("--num-list", default="5,10,20,40,80",
                   help="Comma-separated N values to evaluate")
    p.add_argument("--targets", default="all",
                   help="`all`, `first:K`, `random:K`, or comma-list of indices")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Single multiplier on the estimated watermark before subtraction")
    p.add_argument("--gain-list", default=None,
                   help="Comma-separated list of gains to sweep per N (overrides --gain)")
    p.add_argument("--median-width", type=int, default=9,
                   help="Width of frequency-axis median filter for cover envelope")
    p.add_argument("--z-threshold", type=float, default=0.0,
                   help="Per-bin z-test threshold; bins below |z| are zeroed in W_est. "
                        "0 disables masking (default).")
    p.add_argument("--active-band-only", action="store_true",
                   help="Restrict W_est to FFT bins [%d, %d] (audiowmark's watermarking band)"
                        % (MIN_BAND, MAX_BAND))
    p.add_argument("--key", type=Path, default=DEFAULT_KEY)
    p.add_argument("--message", default=DEFAULT_MESSAGE,
                   help="Expected 128-bit message (used when --message-mode same)")
    p.add_argument("--message-mode", choices=("same", "unique"), default="same",
                   help="`same`: every file has --message; "
                        "`unique`: per-file message derived as in watermark_sweep.py")
    p.add_argument("--unique-secret", default="6.5610-watermark-transfer",
                   help="Secret for deriving unique messages (must match watermark_sweep.py)")
    p.add_argument("--out", type=Path,
                   help="Output CSV (default: data/results/attack_<ts>.csv)")
    p.add_argument("--audiowmark-cmd",
                   help="Path to native audiowmark binary; default uses Docker")
    p.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    p.add_argument("--keep-cleaned", action="store_true",
                   help="Keep all per-(N, target) cleaned WAVs (default: keep only one per N)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cover_paths = load_manifest(args.manifest)
    wm_paths = [args.wm_dir / p.name for p in cover_paths]
    n_total = len(cover_paths)
    for w in wm_paths:
        if not w.exists():
            raise SystemExit(f"missing watermarked file: {w}\n"
                             "Run watermark_sweep.py first.")
    log.info("Manifest %s with %d entries; watermarked files in %s",
             args.manifest, n_total, args.wm_dir)

    num_list = sorted({int(x) for x in args.num_list.split(",") if x.strip()})
    if max(num_list) > n_total:
        raise SystemExit(f"requested N={max(num_list)} > manifest size {n_total}")
    target_indices = parse_target_spec(args.targets, n_total)
    log.info("Ns to evaluate: %s; targets: %d files", num_list, len(target_indices))

    if args.audiowmark_cmd:
        if not (shutil.which(args.audiowmark_cmd) or Path(args.audiowmark_cmd).exists()):
            raise SystemExit(f"audiowmark binary not found: {args.audiowmark_cmd}")
        am = Audiowmark(native_cmd=args.audiowmark_cmd, docker_image=args.docker_image)
    else:
        if shutil.which("docker") is None:
            raise SystemExit("docker not found; pass --audiowmark-cmd <path>")
        am = Audiowmark(native_cmd=None, docker_image=args.docker_image)

    out_csv = args.out
    if out_csv is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_csv = RESULTS_DIR / f"attack_{ts}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    gain_list = ([float(g) for g in args.gain_list.split(",")]
                 if args.gain_list else [args.gain])
    log.info("Gains to evaluate: %s", gain_list)

    unique_secret = args.unique_secret.encode()

    def expected_for(track_id: str) -> str:
        if args.message_mode == "same":
            return args.message
        return derive_unique_message(track_id, unique_secret)

    fields = [
        "N", "gain", "target_idx", "track_id",
        "expected_message", "decoded_message",
        "decode_correct", "best_quality", "best_error", "best_type",
        "sync_below_threshold", "error_above_threshold",
        "attack_succeeded", "snr_db", "lsd_db",
    ]

    estimator = WatermarkEstimator(args.median_width)
    fed = 0
    summary_rows = []

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for N in num_list:
            while fed < N:
                estimator.update(wm_paths[fed])
                fed += 1
            log.info("[N=%d] estimating watermark from %d files (%.1f s spec)",
                     N, estimator.n, estimator._t_max * HOP / SAMPLE_RATE)
            t0 = time.time()
            W_est = estimator.estimate(
                z_threshold=args.z_threshold,
                active_band_only=args.active_band_only,
            )
            nonzero = float(np.mean(W_est != 0))
            log.info("[N=%d] estimate ready (%.2fs); W_est shape=%s, "
                     "rms=%.4f, nonzero=%.1f%%",
                     N, time.time() - t0, W_est.shape,
                     float(np.sqrt(np.mean(W_est ** 2))), 100 * nonzero)

            for gain in gain_list:
                tag = f"N{N:03d}_g{gain:0.2f}".replace(".", "p")
                n_dir = ATTACK_DIR / tag
                json_dir = n_dir / "json"
                n_dir.mkdir(parents=True, exist_ok=True)

                attack_successes = 0
                snr_list, lsd_list, q_list, e_list = [], [], [], []

                for ti in target_indices:
                    target_wm = wm_paths[ti]
                    track_id = target_wm.stem
                    cleaned_path = n_dir / f"{track_id}.wav"
                    json_path = json_dir / f"{track_id}.json"

                    orig, clean = apply_attack(target_wm, W_est, gain, cleaned_path)
                    snr = snr_db(orig, clean)
                    lsd = log_spectral_distance(orig, clean)

                    expected = expected_for(track_id)
                    payload = am.get_json(args.key, cleaned_path, json_path)
                    matches = payload.get("matches", [])
                    best = best_match(matches, expected)
                    decoded = best.get("bits", "") if best else ""
                    quality = float(best.get("quality", 0.0)) if best else 0.0
                    err = float(best.get("error", 1.0)) if best else 1.0
                    mtype = best.get("type", "") if best else ""
                    correct = decoded.lower() == expected.lower()

                    sync_below = quality < SYNC_THRESHOLD
                    err_above = err > ERROR_THRESHOLD
                    attack_ok = (not correct) or sync_below or err_above
                    if attack_ok:
                        attack_successes += 1
                    snr_list.append(snr)
                    lsd_list.append(lsd)
                    q_list.append(quality)
                    e_list.append(err)

                    writer.writerow({
                        "N": N, "gain": f"{gain:.2f}",
                        "target_idx": ti, "track_id": track_id,
                        "expected_message": expected,
                        "decoded_message": decoded,
                        "decode_correct": int(correct),
                        "best_quality": f"{quality:.4f}",
                        "best_error": f"{err:.4f}",
                        "best_type": mtype,
                        "sync_below_threshold": int(sync_below),
                        "error_above_threshold": int(err_above),
                        "attack_succeeded": int(attack_ok),
                        "snr_db": f"{snr:.2f}",
                        "lsd_db": f"{lsd:.2f}",
                    })
                    f.flush()

                    if not args.keep_cleaned and ti != target_indices[0]:
                        cleaned_path.unlink(missing_ok=True)

                rate = attack_successes / len(target_indices)
                log.info("[N=%d gain=%.2f] attack succ %d/%d = %.1f%%; "
                         "med sync=%.3f, err=%.3f, snr=%.1f dB, lsd=%.2f dB",
                         N, gain, attack_successes, len(target_indices), 100 * rate,
                         float(np.median(q_list)), float(np.median(e_list)),
                         float(np.median(snr_list)), float(np.median(lsd_list)))
                summary_rows.append((N, gain, rate,
                                     float(np.median(q_list)),
                                     float(np.median(e_list)),
                                     float(np.median(snr_list)),
                                     float(np.median(lsd_list))))

    print()
    print("=" * 80)
    print(f"{'N':>4}  {'gain':>5}  {'attack_succ':>11}  {'med_sync':>9}  "
          f"{'med_err':>8}  {'med_snr_dB':>10}  {'med_lsd_dB':>10}")
    for N, gain, rate, q, e, s, l in summary_rows:
        print(f"{N:>4}  {gain:>5.2f}  {rate*100:>10.1f}%  {q:>9.3f}  {e:>8.3f}  "
              f"{s:>10.2f}  {l:>10.2f}")
    print("=" * 80)
    print(f"csv: {out_csv}")


if __name__ == "__main__":
    main()
