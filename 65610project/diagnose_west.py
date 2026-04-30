#!/usr/bin/env python3
"""Compare W_est between same-msg and unique-msg watermarked sets.

Prints, for each scenario:
  * rms of W_est over the whole spectrogram,
  * rms inside audiowmark's active band (FFT bins 20..100),
  * rms outside the active band,
  * the active/outside ratio (>1 means watermark signature is detectable),
  * and the same statistics for the per-bin z-score abs(residual) / SEM.

If the cryptographic premise holds, the same-msg case should have a
substantially higher active/outside ratio and a fatter right tail in the
z-score distribution than the unique-msg case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_attack import (  # noqa: E402
    EPS, HOP, MIN_BAND, MAX_BAND, N_FFT, SAMPLE_RATE, WatermarkEstimator,
    load_wav, _stft, PROJECT_DIR,
)


def stats(name: str, est: WatermarkEstimator) -> None:
    if est._sum is None:
        print(f"{name}: empty estimator")
        return
    mean_logp = est._sum / est._count
    var = (est._sumsq / est._count) - mean_logp ** 2
    var = np.clip(var, 0.0, None)
    sem = np.sqrt(var / est._count) + EPS

    import scipy.ndimage as ndi
    smoothed = ndi.median_filter(mean_logp, size=(1, 9, 1), mode="reflect")
    residual = mean_logp - smoothed
    z = np.abs(residual) / sem

    band_idx = np.arange(residual.shape[1])
    in_band = (band_idx >= MIN_BAND) & (band_idx <= MAX_BAND)
    in_idx = np.where(in_band)[0]
    out_idx = np.where(~in_band)[0]

    rms_all = float(np.sqrt(np.mean(residual ** 2)))
    rms_in = float(np.sqrt(np.mean(residual[:, in_idx, :] ** 2)))
    rms_out = float(np.sqrt(np.mean(residual[:, out_idx, :] ** 2)))
    z_in_p95 = float(np.percentile(z[:, in_idx, :], 95))
    z_out_p95 = float(np.percentile(z[:, out_idx, :], 95))
    z_in_p99 = float(np.percentile(z[:, in_idx, :], 99))
    z_out_p99 = float(np.percentile(z[:, out_idx, :], 99))

    print(f"{name}")
    print(f"  files                {est._count}")
    print(f"  W_est rms (all)      {rms_all:.4f}")
    print(f"  W_est rms (in-band)  {rms_in:.4f}")
    print(f"  W_est rms (out-band) {rms_out:.4f}")
    print(f"  in/out ratio         {rms_in/rms_out:.3f}")
    print(f"  |z| p95  in/out      {z_in_p95:.3f} / {z_out_p95:.3f}  "
          f"(ratio {z_in_p95/z_out_p95:.3f})")
    print(f"  |z| p99  in/out      {z_in_p99:.3f} / {z_out_p99:.3f}  "
          f"(ratio {z_in_p99/z_out_p99:.3f})")
    print()


def feed(estimator: WatermarkEstimator, paths: list) -> None:
    for p in paths:
        estimator.update(p)


def main() -> None:
    manifest = PROJECT_DIR / "data" / "manifest_n80_seed6561.txt"
    cover_paths = [PROJECT_DIR / ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
    names = [p.name for p in cover_paths]

    same_dir = PROJECT_DIR / "data" / "wm"
    uniq_dir = PROJECT_DIR / "data" / "wm_unique"

    e_same = WatermarkEstimator(median_width=9)
    e_uniq = WatermarkEstimator(median_width=9)
    feed(e_same, [same_dir / n for n in names])
    feed(e_uniq, [uniq_dir / n for n in names])

    stats("SAME message (same-K, same-m, N=80)", e_same)
    stats("UNIQUE message (same-K, distinct-m_i, N=80)", e_uniq)


if __name__ == "__main__":
    main()
