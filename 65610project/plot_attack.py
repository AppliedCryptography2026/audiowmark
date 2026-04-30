#!/usr/bin/env python3
"""
Plots for the watermark-transfer-attack experiment.

Single-CSV mode (legacy):

    python plot_attack.py 65610project/data/results/attack_<ts>.csv

Comparison mode (same-message vs unique-message defense check):

    python plot_attack.py \
        --same   65610project/data/results/attack_same_<ts>.csv \
        --unique 65610project/data/results/attack_unique_<ts>.csv

The comparison mode produces three panels at a fixed gain:

  * ``compare_attack_succ.png`` - standard attack success
    (decoded != expected OR sync < 0.6 OR error > 0.5).
  * ``compare_true_cancel.png`` - "true cancellation" rate, defined as
    median decoding error >= ``--cancel-threshold`` (default 0.45).
    This isolates the cryptographic exploit: the watermark message bits
    are actually scrambled, not just sync-degraded.
  * ``compare_sync_drop.png`` - rate of sync below ``--sync-threshold``.

Plus an ``attack_frontier.png`` per CSV (SNR vs success-rate parametric
in gain), and the legacy single-CSV plots if no comparison CSVs are passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SYNC_THRESHOLD = 0.6
DEFAULT_ERROR_THRESHOLD = 0.5
DEFAULT_CANCEL_THRESHOLD = 0.45  # error this high => watermark bits scrambled


def load(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def aggregate(df: pd.DataFrame,
              cancel_threshold: float = DEFAULT_CANCEL_THRESHOLD,
              sync_threshold: float = DEFAULT_SYNC_THRESHOLD,
              error_threshold: float = DEFAULT_ERROR_THRESHOLD) -> pd.DataFrame:
    """Per-(N, gain) aggregates including standard / cancel / sync metrics."""
    df = df.copy()
    df["sync_drop"] = (df["best_quality"] < sync_threshold).astype(int)
    df["err_high"] = (df["best_error"] > error_threshold).astype(int)
    df["true_cancel"] = (df["best_error"] >= cancel_threshold).astype(int)
    grp = df.groupby(["N", "gain"], as_index=False).agg(
        attack_succ=("attack_succeeded", "mean"),
        sync_drop_rate=("sync_drop", "mean"),
        err_high_rate=("err_high", "mean"),
        true_cancel_rate=("true_cancel", "mean"),
        med_sync=("best_quality", "median"),
        med_err=("best_error", "median"),
        med_snr=("snr_db", "median"),
        med_lsd=("lsd_db", "median"),
        n=("attack_succeeded", "size"),
    )
    return grp


# ---------- single-CSV plots (legacy) ----------------------------------

def plot_success_vs_n(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for gain, sub in summary.groupby("gain"):
        sub = sub.sort_values("N")
        ax.plot(sub["N"], 100 * sub["attack_succ"], marker="o", label=f"gain={gain:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(summary["N"].unique()))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("N (number of watermarked files used to estimate W)")
    ax.set_ylabel("attack success rate (%)")
    ax.set_title("Detector failure rate vs N")
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.3)
    ax.legend(title="W_est gain", loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_snr_vs_n(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for gain, sub in summary.groupby("gain"):
        sub = sub.sort_values("N")
        ax.plot(sub["N"], sub["med_snr"], marker="o", label=f"gain={gain:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(summary["N"].unique()))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("N (number of watermarked files used to estimate W)")
    ax.set_ylabel("median SNR of cleaned vs original (dB)")
    ax.set_title("Audio fidelity of cleaned files vs N")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(title="W_est gain", loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_frontier(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for N, sub in summary.groupby("N"):
        sub = sub.sort_values("gain")
        ax.plot(sub["med_snr"], 100 * sub["attack_succ"], marker="o", label=f"N={int(N)}")
    ax.set_xlabel("median SNR (dB) - higher = better audio")
    ax.set_ylabel("attack success rate (%) - higher = watermark broken")
    ax.set_title("Attack frontier: SNR vs detector failure")
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.3)
    ax.legend(title="averaging set size")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def best_operating_point(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for N, sub in summary.groupby("N"):
        sub = sub.sort_values("gain")
        ok = sub[sub["attack_succ"] >= threshold]
        if ok.empty:
            row = sub.loc[sub["attack_succ"].idxmax()].to_dict()
            row["meets_threshold"] = False
        else:
            row = ok.iloc[0].to_dict()
            row["meets_threshold"] = True
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- comparison plots -------------------------------------------

def _plot_two_curves(summary_a: pd.DataFrame, label_a: str,
                     summary_b: pd.DataFrame, label_b: str,
                     metric: str, ylabel: str, title: str,
                     out: Path, gain: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for summary, label, marker in [(summary_a, label_a, "o"),
                                    (summary_b, label_b, "s")]:
        sub = summary[np.isclose(summary["gain"], gain)].sort_values("N")
        if sub.empty:
            continue
        ax.plot(sub["N"], 100 * sub[metric], marker=marker, label=label, linewidth=2)
    ax.set_xscale("log", base=2)
    ns = sorted(set(summary_a["N"]).union(summary_b["N"]))
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("N (number of watermarked files used to estimate W)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (gain = {gain:.2f})")
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_comparison(summary_same: pd.DataFrame, summary_uniq: pd.DataFrame,
                    out_dir: Path, gain: float) -> None:
    _plot_two_curves(
        summary_same, "same K, same message",
        summary_uniq, "same K, unique per-file message",
        metric="attack_succ",
        ylabel="attack success rate (%)",
        title="Standard attack success (sync<0.6 OR err>0.5 OR wrong-bits)",
        out=out_dir / "compare_attack_succ.png",
        gain=gain,
    )
    _plot_two_curves(
        summary_same, "same K, same message",
        summary_uniq, "same K, unique per-file message",
        metric="true_cancel_rate",
        ylabel="true watermark cancellation rate (%)",
        title="Cryptographic exploit: decoding-error >= 0.45",
        out=out_dir / "compare_true_cancel.png",
        gain=gain,
    )
    _plot_two_curves(
        summary_same, "same K, same message",
        summary_uniq, "same K, unique per-file message",
        metric="sync_drop_rate",
        ylabel="sync-below-threshold rate (%)",
        title="Sync-drop rate (audio-damage proxy)",
        out=out_dir / "compare_sync_drop.png",
        gain=gain,
    )


# ---------- main --------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, nargs="?", help="single attack CSV (legacy)")
    p.add_argument("--same", type=Path, help="same-message attack CSV (comparison mode)")
    p.add_argument("--unique", type=Path, help="unique-message attack CSV (comparison mode)")
    p.add_argument("--success-threshold", type=float, default=0.95)
    p.add_argument("--cancel-threshold", type=float, default=DEFAULT_CANCEL_THRESHOLD)
    p.add_argument("--sync-threshold", type=float, default=DEFAULT_SYNC_THRESHOLD)
    p.add_argument("--error-threshold", type=float, default=DEFAULT_ERROR_THRESHOLD)
    p.add_argument("--compare-gain", type=float, default=2.0,
                   help="Gain to use for comparison-plot panels (default 2.0)")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if args.csv is not None and args.same is None and args.unique is None:
        df = load(args.csv)
        summary = aggregate(df, args.cancel_threshold, args.sync_threshold, args.error_threshold)
        out_dir = args.out_dir or args.csv.parent / "figs"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_success_vs_n(summary, out_dir / "success_vs_n.png")
        plot_snr_vs_n(summary, out_dir / "snr_vs_n.png")
        plot_frontier(summary, out_dir / "frontier.png")
        print(f"figures: {out_dir}\n")
        op = best_operating_point(summary, args.success_threshold)
        print(f"Best operating point for >= {args.success_threshold*100:.0f}% attack success:")
        print(op[["N", "gain", "attack_succ", "med_sync", "med_err",
                  "med_snr", "med_lsd", "meets_threshold"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
        return

    if args.same is None or args.unique is None:
        raise SystemExit("Provide either a single CSV or both --same and --unique CSVs.")

    df_same = load(args.same)
    df_uniq = load(args.unique)
    summary_same = aggregate(df_same, args.cancel_threshold,
                              args.sync_threshold, args.error_threshold)
    summary_uniq = aggregate(df_uniq, args.cancel_threshold,
                              args.sync_threshold, args.error_threshold)

    out_dir = args.out_dir or args.same.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(summary_same, summary_uniq, out_dir, args.compare_gain)
    plot_frontier(summary_same, out_dir / "frontier_same.png")
    plot_frontier(summary_uniq, out_dir / "frontier_unique.png")

    print(f"figures: {out_dir}\n")
    cols = ["N", "gain", "attack_succ", "true_cancel_rate", "sync_drop_rate",
            "med_sync", "med_err", "med_snr", "med_lsd"]
    g = args.compare_gain
    print(f"Comparison at gain = {g:.2f}:\n")
    print("--- same-message (canonical attack scenario) ---")
    print(summary_same[np.isclose(summary_same["gain"], g)][cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print("--- unique-per-file message (defense scenario) ---")
    print(summary_uniq[np.isclose(summary_uniq["gain"], g)][cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
