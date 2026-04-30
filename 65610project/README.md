# 6.5610 Watermark Transfer Attack on audiowmark

Pipeline:

1. **Fetch covers** from FMA-small (80 × 30 s stereo WAVs).
2. **Watermark** each cover with the *same* key `K` and *same* message `m`,
   recording sync score and decoding error baseline.
3. **Estimate `W`** from N watermarked files via log-power-spectrogram
   averaging + frequency-axis median-filter spike extraction.
4. **Attack**: subtract `gain · Ŵ` from a target's spectrogram, ISTFT, run
   `audiowmark get` on the cleaned file.
5. **Sweep** over N ∈ {5, 10, 20, 40, 80} and gain to draw the SNR-vs-detector
   trade-off curve.
6. **Defense check**: re-watermark with per-file unique messages and re-run
   step 3-5; the attack should fail because `W` is no longer common.

## Quick start

```bash
# from the audiowmark repo root
source ../65610venv/bin/activate
pip install -r 65610project/requirements.txt   # one-time

# requires ffmpeg on PATH (`brew install ffmpeg`)
#       and audiowmark:latest as a Docker image (built from the repo Dockerfile)

# 1. fetch & convert 80 FMA tracks to 44.1 kHz stereo WAV
python 65610project/retreive_soundfiles.py --num 80

# 2. baseline (canonical attack scenario): same K, same message
python 65610project/watermark_sweep.py --num 80 --message-mode same

# 3-5. attack sweep over (N, gain) on same-message twins
python 65610project/transfer_attack.py \
    --num-list 5,10,20,40,80 \
    --gain-list 1.0,1.5,2.0,2.5,3.0 \
    --targets first:25 \
    --message-mode same \
    --out 65610project/data/results/attack_same.csv

# 6. defense check: same K, unique per-file messages
python 65610project/watermark_sweep.py --num 80 --message-mode unique

python 65610project/transfer_attack.py \
    --num-list 5,10,20,40,80 \
    --gain-list 1.0,1.5,2.0,2.5,3.0 \
    --targets first:25 \
    --wm-dir 65610project/data/wm_unique \
    --message-mode unique \
    --out 65610project/data/results/attack_unique.csv

# 7. plots + operating-point summary
python 65610project/plot_attack.py \
    --same   65610project/data/results/attack_same.csv \
    --unique 65610project/data/results/attack_unique.csv \
    --compare-gain 2.0
```

## Cryptographic context

`audiowmark` uses key `K` to derive *all* of:

* the per-frame "up" / "down" frequency-band selection (see
  `UpDownGen` in `src/wmcommon.hh`),
* the per-frame bit-position permutation (see `BitPosGen`),
* the global bit-order reshuffling (see `randomize_bit_order` in
  `src/wmcommon.hh`).

Every value is deterministic in `K` and the (block, frame) index. So when one
key is reused across many tracks with the same message, the *same* multiplicative
spectral perturbation `M(t, f)` is applied to every cover. In log-power that's
an additive bias `W(t, f)` shared across files - exactly the symmetry we exploit
by averaging.

## Layout

```
65610project/
├── retreive_soundfiles.py        # FMA fetch + ffmpeg-convert
├── watermark_sweep.py            # add-then-get sweep, baseline CSV
├── transfer_attack.py            # spectrogram-mean estimator + attack
├── plot_attack.py                # plots + operating-point summary
├── requirements.txt
├── data/
│   ├── manifest_n80_seed6561.txt
│   ├── wav/   wm/   attack/  wm_json/
│   ├── keys/wmark.key
│   └── results/sweep_*.csv  attack_*.csv  figs/*.png
└── cache/fma_small_namelist.json
```

Subsets nest under the default seed: `--num 80` is a superset of `--num 40`,
etc., so the same watermarked twin is reused across N values without
re-watermarking.

## Findings (N=80 cover set, gain sweep 1.0–3.0)

* **Baseline (no attack)**: 80 / 80 watermarked files decode cleanly with
  median sync 1.39 and median bit error 0.26.

* **Attack frontier**: `transfer_attack.py` produces curves of
  (SNR, attack-success-rate) per N. Higher N shifts the frontier
  to the right - the same detector failure rate is reachable at much higher
  audio quality (e.g. 96 % failure at SNR ≈ +2 dB for N=80, vs SNR ≈ –22 dB
  for N=5).

* **Defense check (same K, unique per-file messages)**: under the standard
  `sync < 0.6 OR err > 0.5 OR wrong-bits` criterion, the attack still
  reports ~96 % "success" - but only because the spectrogram subtraction
  is large enough to drop sync. The decoding error is the right
  cryptographic indicator:

  | N  | true-cancellation rate, **same** message | **unique** per-file message |
  |---:|-----------------------------------------:|----------------------------:|
  | 5  | 96 %                                     | 52 %                        |
  | 10 | 88 %                                     | **12 %**                    |
  | 20 | 72 %                                     | **4 %**                     |
  | 40 | 68 %                                     | **4 %**                     |
  | 80 | 76 %                                     | **8 %**                     |

  Once N >= 20 the unique-message case essentially never sees true
  cancellation (the watermark bits remain decodable at ~0.28 error,
  vs ~0.49 for same-message), because averaging cancels the per-file
  watermarks. Diagnostic confirms it: `diagnose_west.py` reports the
  in-band / out-of-band rms ratio of `W_est` at N=80 as
  **1.105** for same-message vs **0.981** for unique-message - the
  watermark signature exists in the estimator only when the message is
  shared.

## Notes

- We skip three FMA tracks that are documented as zero-length (`099134`,
  `108925`, `133297`).
- 30 s clips trigger audiowmark's clip mode (`CLIP-A`/`CLIP-B`) - one
  watermark block per file, which is the cleanest possible setting for the
  averaging attack (`x_i = c_i + W` with the same `W`).
- `audiowmark` is invoked through the project's Docker image; pass
  `--audiowmark-cmd <path>` to either of the scripts to use a native binary.
- Our STFT uses 75 % overlap (hop 256) while audiowmark uses no overlap
  (hop 1024). This frame-grid mismatch causes a residual cancellation
  error that the SNR-vs-N curve plateaus into around N=40 - it's an
  estimator-design artifact, not a sample-size limit.

## Documentation: 
```python 65610project/retreive_soundfiles.py --num 80 --seed 6561 ```
to get the same data files that were used (also converts from mp3 to wav as well). 

Run ```audiowmark gen-key wmark.key``` to get the key. 

Then add watermark to each .wav soundfile using watermark_sweep.py 

