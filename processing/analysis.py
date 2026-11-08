"""Reusable plotting and spectral-analysis helpers for segment comparisons.

All functions take mne Raw segments (as returned by `load_session(...).segments`) and work
at any sampling rate.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import mannwhitneyu

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
# Relative power is a band's share of 4–40 Hz power. Delta (1–4 Hz) is left out on purpose:
# in these recordings it is dominated by slow drift and movement, and its share drops in later
# segments, which would inflate every other band's share (see DELTA for checking it separately).
TOTAL = (4, 40)
DELTA = (1, 4)
BAND_COLORS = {"theta": "C2", "alpha": "C1", "beta": "C4"}


# --------------------------------------------------------------------- time domain
def band_filter(seg, l_freq, h_freq, notch=None):
    """Copy of `seg` band-pass filtered (zero-phase FIR), optionally notched first."""
    out = seg.copy()
    if notch:
        out.notch_filter(freqs=notch, verbose="error")
    return out.filter(l_freq=l_freq, h_freq=h_freq, verbose="error")


def plot_segments(segs, labels, title, channels=None, sharey_rows=True):
    """Grid of channels (rows) x segments (columns), every sample plotted, in µV."""
    channels = channels or segs[0].ch_names
    n_rows, n_cols = len(channels), len(segs)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(9 * n_cols, 1.6 * n_rows), sharex="col",
        sharey="row" if sharey_rows else False, squeeze=False,
    )
    for col, (seg, label) in enumerate(zip(segs, labels)):
        data = seg.get_data(picks=channels) * 1e6
        for row, ch in enumerate(channels):
            ax = axes[row, col]
            ax.plot(seg.times, data[row], lw=0.3, color="C0", rasterized=True)
            ax.set_ylabel(f"{ch}\n(µV)", rotation=0, ha="right", va="center")
            ax.margins(x=0)
            ax.grid(alpha=0.3)
        axes[0, col].set_title(f"{label}  ({seg.times[-1]:.1f} s)")
        axes[-1, col].set_xlabel("Time from segment start (s)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    plt.show()


# ------------------------------------------------------------------ frequency domain
def window_spectra(seg, bad_channels=(), window_s=2.0, overlap=None, reject_uv=150, notch=60.0):
    """Welch-style per-window PSD (µV²/Hz) with per-channel artifact rejection.

    Steps: drop bad channels, 1 Hz high-pass, notch, cut into `window_s` windows
    (50% overlap by default), Hann-windowed FFT per window. A (window, channel) pair is
    rejected (set to NaN) if that channel's peak-to-peak in the window exceeds `reject_uv`.

    Returns psd[window, channel, freq], freqs, channel names, fraction kept per channel.
    """
    overlap = window_s / 2 if overlap is None else overlap
    clean = seg.copy().drop_channels([c for c in bad_channels if c in seg.ch_names])
    clean.filter(l_freq=1.0, h_freq=None, verbose="error")
    if notch:
        clean.notch_filter(notch, verbose="error")
    ep = mne.make_fixed_length_epochs(clean, duration=window_s, overlap=overlap,
                                      preload=True, verbose="error")
    spec = ep.compute_psd(method="welch", fmin=1, fmax=45, window="hann",
                          n_fft=int(round(window_s * clean.info["sfreq"])), verbose="error")
    psd, freqs = spec.get_data(return_freqs=True)
    psd = psd * 1e12
    bad = np.ptp(ep.get_data(), axis=-1) * 1e6 > reject_uv
    psd[bad] = np.nan
    return psd, freqs, clean.ch_names, 1 - bad.mean(axis=0)


def band_power(psd, freqs, band):
    """Integrate the PSD over a band -> µV²."""
    m = (freqs >= band[0]) & (freqs < band[1])
    return psd[..., m].sum(axis=-1) * (freqs[1] - freqs[0])


def relative_power(spectrum, band):
    """Per-channel band power as a share of TOTAL power, averaged over clean windows."""
    psd, freqs, _, _ = spectrum
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # channels with no clean windows -> NaN
        return np.nanmean(band_power(psd, freqs, band), axis=0) / np.nanmean(band_power(psd, freqs, TOTAL), axis=0)


def plot_spectra(spectra, labels, title, min_keep=0.5):
    """Mean PSD per channel (rows) for each segment (columns), middle 50% of windows shaded."""
    channels = spectra[0][2]
    n = len(channels)
    fig, axes = plt.subplots(n, len(spectra), figsize=(7 * len(spectra), 1.7 * n),
                             sharex=True, sharey="row", squeeze=False)
    for col, ((psd, f, _, keep), label) in enumerate(zip(spectra, labels)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN channels
            mean = np.nanmean(psd, axis=0)
            lo, hi = np.nanpercentile(psd, [25, 75], axis=0)
        for row, ch in enumerate(channels):
            ax = axes[row, col]
            for name, (b0, b1) in BANDS.items():
                ax.axvspan(b0, b1, color=BAND_COLORS[name], alpha=0.12, lw=0,
                           label=f"{name} {b0}–{b1} Hz" if row == 0 else None)
            if keep[row] > 0:
                ax.fill_between(f, lo[row], hi[row], color="C0", alpha=0.25, lw=0)
                ax.plot(f, mean[row], color="C0", lw=1)
                ax.set_yscale("log")
            ax.set_ylabel(f"{ch}\n(µV²/Hz)", rotation=0, ha="right", va="center")
            ax.text(0.99, 0.9, f"{100 * keep[row]:.0f}% windows kept", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="red" if keep[row] < min_keep else "0.4")
            ax.grid(alpha=0.3, which="both")
        axes[0, col].set_title(label)
        axes[-1, col].set_xlabel("Frequency (Hz)")
    axes[0, 0].legend(loc="lower left", fontsize=7, ncol=3)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()


def plot_relative_bars(spectra, labels, title, bands=("theta", "alpha"), min_keep=0.5):
    """Grouped bars of relative band power per channel for two segments.

    Channels where either segment kept < `min_keep` of its windows are greyed out and
    left out of the "higher in segment 2" count.
    """
    channels = spectra[0][2]
    ok = (spectra[0][3] >= min_keep) & (spectra[1][3] >= min_keep)
    x = np.arange(len(channels))
    fig, axes = plt.subplots(len(bands), 1, figsize=(16, 3.5 * len(bands)), sharex=True, squeeze=False)
    for ax, band in zip(axes[:, 0], bands):
        r = [relative_power(s, BANDS[band]) for s in spectra]
        for j, (vals, color, dx) in enumerate(zip(r, ["C0", "C3"], [-0.2, 0.2])):
            ax.bar(x[ok] + dx, vals[ok], 0.4, color=color, label=labels[j] if ok.any() else None)
            ax.bar(x[~ok] + dx, np.nan_to_num(vals[~ok]), 0.4, color="0.85", hatch="//", edgecolor="0.6",
                   label="too few clean windows" if j == 0 and (~ok).any() else None)
        up = int((r[1][ok] > r[0][ok]).sum())
        verdict = (f"higher in {labels[1]} on {up}/{int(ok.sum())} usable channels" if ok.any()
                   else "no channel has enough clean windows in both segments to compare")
        ax.set_title(f"Relative {band} power ({BANDS[band][0]}–{BANDS[band][1]} Hz): {verdict}")
        ax.set_ylabel(f"Share of {TOTAL[0]}–{TOTAL[1]} Hz power")
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xticks(x, channels)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()


def fz_workload_test(segs, labels, bad_channels=(), min_windows=20, **spectra_kwargs):
    """Compare Fz theta between two segments on non-overlapping windows.

    One-sided Mann–Whitney U: is the second segment higher? Skips the test (and says so)
    if either segment has fewer than `min_windows` clean Fz windows.
    """
    measures = {"Absolute theta at Fz (µV²)": [], "Relative theta at Fz": [], "Fz theta / Pz alpha": []}
    for seg in segs:
        psd, f, chs, _ = window_spectra(seg, bad_channels, overlap=0.0, **spectra_kwargs)
        fz, pz = chs.index("Fz"), chs.index("Pz")
        theta_fz = band_power(psd, f, BANDS["theta"])[:, fz]
        with np.errstate(invalid="ignore", divide="ignore"):
            values = [theta_fz,
                      theta_fz / band_power(psd, f, TOTAL)[:, fz],
                      theta_fz / band_power(psd, f, BANDS["alpha"])[:, pz]]
        for key, v in zip(measures, values):
            measures[key].append(v[np.isfinite(v)])

    n = [len(v) for v in measures["Relative theta at Fz"]]
    if min(n) < min_windows:
        print(f"Not enough clean Fz windows to test: {labels[0]} n={n[0]}, {labels[1]} n={n[1]} "
              f"(need ≥{min_windows} each). Skipped.")
        return None

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    rng = np.random.default_rng(0)
    for ax, (label, (s1, s2)) in zip(axes, measures.items()):
        p = mannwhitneyu(s2, s1, alternative="greater").pvalue
        results[label] = (np.median(s1), np.median(s2), p, len(s1), len(s2))
        ax.boxplot([s1, s2], showfliers=False)
        for i, (s, c) in enumerate([(s1, "C0"), (s2, "C3")], start=1):
            ax.scatter(rng.normal(i, 0.05, len(s)), s, s=8, alpha=0.4, color=c)
        ax.set_xticks([1, 2], [f"{labels[0]}\nn={len(s1)}", f"{labels[1]}\nn={len(s2)}"])
        ax.set_title(f"{label}\nmedian {np.median(s1):.3g} → {np.median(s2):.3g}, one-sided p = {p:.3g}")
        ax.grid(axis="y", alpha=0.3)
        print(f"{label:28} {labels[0]}: median {np.median(s1):.3f} (n={len(s1)}); "
              f"{labels[1]}: median {np.median(s2):.3f} (n={len(s2)}); one-sided p = {p:.3g}")
    fig.suptitle(f"Fz theta: {labels[0]} vs {labels[1]} (non-overlapping windows)", fontsize=13)
    fig.tight_layout()
    plt.show()
    return results


def plot_quality(spectra, labels, title):
    """Heatmap of the % of clean windows per channel (rows) and segment (columns)."""
    channels = spectra[0][2]
    keep = np.array([s[3] for s in spectra]).T * 100
    fig, ax = plt.subplots(figsize=(1.6 * len(spectra) + 3, 0.32 * len(channels) + 1.5))
    im = ax.imshow(keep, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for i in range(keep.shape[0]):
        for j in range(keep.shape[1]):
            ax.text(j, i, f"{keep[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(channels)), channels)
    fig.colorbar(im, ax=ax, label="% of 2 s windows kept")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
