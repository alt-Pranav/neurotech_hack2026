"""Reusable plotting and spectral-analysis helpers for segment comparisons.

All functions take mne Raw segments (as returned by `load_session(...).segments`) and work
at any sampling rate.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
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


def plot_segments(segs, labels, title, channels=None, sharey_rows=True, marks=None):
    """Grid of channels (rows) x segments (columns), every sample plotted, in µV.

    marks: optional list (one per column) of [(time_s, label), ...] drawn as vertical lines,
    or [(start_s, end_s, label), ...] drawn as shaded spans.
    """
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
            for m in (marks[col] if marks else []):
                if len(m) == 3:
                    ax.axvspan(m[0], m[1], color="C3", alpha=0.15, lw=0)
                else:
                    ax.axvline(m[0], color="C3", lw=0.8, ls="--")
        for m in (marks[col] if marks else []):
            axes[0, col].text(m[0], 1.02, m[-1], transform=axes[0, col].get_xaxis_transform(),
                              fontsize=7, color="C3", ha="left", va="bottom")
        axes[0, col].set_title(f"{label}  ({seg.times[-1]:.1f} s)", pad=14 if marks else 6)
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

    Reports medians and a one-sided Mann–Whitney U p-value (is the second segment higher?).
    The p-value is DESCRIPTIVE only: windows cut from one continuous recording are not
    independent samples, so it is not a valid significance test (pseudoreplication).
    Real inference needs several blocks per condition or several people. Skips (and says so)
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
        ax.set_title(f"{label}\nmedian {np.median(s1):.3g} → {np.median(s2):.3g} (descriptive p = {p:.2g})")
        ax.grid(axis="y", alpha=0.3)
        print(f"{label:28} {labels[0]}: median {np.median(s1):.3f} (n={len(s1)}); "
              f"{labels[1]}: median {np.median(s2):.3f} (n={len(s2)}); descriptive p = {p:.2g}")
    fig.suptitle(f"Fz theta: {labels[0]} vs {labels[1]} (non-overlapping windows; p-values descriptive only)", fontsize=13)
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


# ------------------------------------------------------------- FOOOF / specparam
# Separates oscillatory peaks from the aperiodic (1/f) background (Donoghue et al., 2020).
# Muscle activity and drift shift the whole spectrum; measuring theta/alpha as power ABOVE
# the fitted 1/f line keeps those broadband shifts from looking like band changes.
FOOOF_RANGE = (2, 40)
FOOOF_SETTINGS = dict(peak_width_limits=(1.0, 8.0), max_n_peaks=6, min_peak_height=0.05,
                      aperiodic_mode="fixed", verbose=False)


def fit_fooof(spectrum, channels=None, freq_range=FOOOF_RANGE):
    """Fit FOOOF to each channel's mean spectrum (over clean windows). Returns {channel: model}."""
    # fooof 1.1 resets the warning filter and emits a deprecation notice on import (it is
    # superseded by `specparam`, not yet on conda-forge); record=True swallows it.
    with warnings.catch_warnings(record=True):
        from fooof import FOOOF
    psd, freqs, chs, keep = spectrum
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(psd, axis=0)
    models = {}
    for ch in channels or chs:
        k = chs.index(ch)
        if not np.all(np.isfinite(mean[k])):
            continue  # no clean windows on this channel
        fm = FOOOF(**FOOOF_SETTINGS)
        fm.fit(freqs, mean[k], freq_range)
        models[ch] = fm
    return models


def periodic_power_db(fm, band):
    """Mean power above the 1/f fit within `band`, in dB (10·log10 ratio to the aperiodic fit)."""
    m = (fm.freqs >= band[0]) & (fm.freqs < band[1])
    return 10 * fm.get_data("peak", "log")[m].mean()


# Taking the log of a single-window periodogram is biased low by Euler's constant
# (E[ln X] = ln E[X] - 0.5772 for exponentially distributed power), i.e. about 2.51 dB.
# It affects every window equally, so differences are unaffected; we add it back so
# per-window values are on the same scale as values from the averaged spectrum.
LOG_PERIODOGRAM_BIAS_DB = 10 * np.euler_gamma / np.log(10)


def window_periodic_db(spectrum, fm, ch, band):
    """Per-window power above the segment's 1/f fit in `band` (dB, log-bias corrected)."""
    psd, freqs, chs, _ = spectrum
    k = chs.index(ch)
    m = (fm.freqs >= band[0]) & (fm.freqs < band[1])
    fm_idx = np.searchsorted(freqs, fm.freqs[m])
    aperiodic = fm.get_model("aperiodic", "log")[m]
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = 10 * (np.log10(psd[:, k, fm_idx]) - aperiodic).mean(axis=1) + LOG_PERIODOGRAM_BIAS_DB
    return vals[np.isfinite(vals)]


def plot_fooof_fits(spectra, labels, title, channels=("Fz", "Pz")):
    """Per channel: log spectra with their 1/f fits (left) and the spectra with 1/f removed (right)."""
    models = [fit_fooof(s, channels) for s in spectra]
    fig, axes = plt.subplots(len(channels), 2, figsize=(15, 3.6 * len(channels)), squeeze=False)
    for row, ch in enumerate(channels):
        for (fms, label, color) in zip(models, labels, ["C0", "C3"]):
            if ch not in fms:
                continue
            fm = fms[ch]
            exp = fm.get_params("aperiodic_params", "exponent")
            axes[row, 0].plot(fm.freqs, fm.power_spectrum, color=color, lw=1.2, label=f"{label}")
            axes[row, 0].plot(fm.freqs, fm.get_model("aperiodic", "log"), color=color, ls="--", lw=1,
                              label=f"1/f fit (exponent {exp:.2f})")
            axes[row, 1].plot(fm.freqs, 10 * fm.get_data("peak", "log"), color=color, lw=1.2, label=label)
        for ax in axes[row]:
            for name in ("theta", "alpha"):
                ax.axvspan(*BANDS[name], color=BAND_COLORS[name], alpha=0.12, lw=0)
            ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel(f"{ch}\nlog10 power (µV²/Hz)")
        axes[row, 1].set_ylabel("Power above 1/f (dB)")
        axes[row, 1].axhline(0, color="k", lw=0.6)
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].legend(fontsize=7)
    axes[0, 0].set_title("Spectrum and fitted 1/f background (shaded: theta, alpha)")
    axes[0, 1].set_title("Spectrum with the 1/f background removed")
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()
    return models


def plot_periodic_bars(spectra, labels, title, min_keep=0.5):
    """Per channel: theta and alpha above 1/f (dB) and the 1/f exponent, for two segments.

    A lower (flatter) exponent means relatively more high-frequency power, which in these
    tasks usually means more muscle activity.
    """
    channels = spectra[0][2]
    ok = (spectra[0][3] >= min_keep) & (spectra[1][3] >= min_keep)
    models = [fit_fooof(s) for s in spectra]
    rows = {
        "Theta (4–8 Hz) above 1/f (dB)": lambda fm: periodic_power_db(fm, BANDS["theta"]),
        "Alpha (8–13 Hz) above 1/f (dB)": lambda fm: periodic_power_db(fm, BANDS["alpha"]),
        "1/f exponent (lower = flatter, more muscle)": lambda fm: fm.get_params("aperiodic_params", "exponent"),
    }
    x = np.arange(len(channels))
    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 3.3 * len(rows)), sharex=True)
    table = {}
    for ax, (name, fn) in zip(axes, rows.items()):
        vals = [np.array([fn(m[ch]) if ch in m else np.nan for ch in channels]) for m in models]
        table[name] = vals
        for j, (v, color, dx) in enumerate(zip(vals, ["C0", "C3"], [-0.2, 0.2])):
            ax.bar(x[ok] + dx, v[ok], 0.4, color=color, label=labels[j] if ok.any() else None)
            ax.bar(x[~ok] + dx, np.nan_to_num(v[~ok]), 0.4, color="0.85", hatch="//", edgecolor="0.6",
                   label="too few clean windows" if j == 0 and (~ok).any() else None)
        up = int((vals[1][ok] > vals[0][ok]).sum())
        ax.set_title(f"{name}: higher in {labels[1]} on {up}/{int(ok.sum())} usable channels"
                     if ok.any() else f"{name}: no channel usable in both segments")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xticks(x, channels)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame({f"{n} [{lab}]": v for n, vals in table.items() for lab, v in zip(labels, vals)},
                        index=channels).round(2)


def periodic_workload_comparison(spectra_nonoverlap, labels, title, min_windows=20,
                                 theta_ch="Fz", alpha_ch="Pz"):
    """Per-window frontal theta and parietal alpha above each segment's 1/f fit (dB), two segments.

    theta_ch / alpha_ch: channel names present in the spectra (use `roi_spectrum` to add an
    averaged region such as "F3+F4"). Medians and their difference are the result; the
    Mann–Whitney p-value is descriptive only (windows from one recording are not independent).
    """
    measures = {f"Theta above 1/f at {theta_ch} (dB)": (theta_ch, "theta"),
                f"Alpha above 1/f at {alpha_ch} (dB)": (alpha_ch, "alpha")}
    data = {}
    for name, (ch, band) in measures.items():
        vals = []
        for sp in spectra_nonoverlap:
            fm = fit_fooof(sp, [ch]).get(ch)
            vals.append(window_periodic_db(sp, fm, ch, BANDS[band]) if fm else np.array([]))
        data[name] = vals
    n = [len(v) for v in data[f"Theta above 1/f at {theta_ch} (dB)"]]
    if min(n) < min_windows:
        print(f"Not enough clean {theta_ch} windows: {labels[0]} n={n[0]}, {labels[1]} n={n[1]} (need ≥{min_windows}). Skipped.")
        return None
    rows = []
    fig, axes = plt.subplots(1, len(measures), figsize=(13, 5))
    rng = np.random.default_rng(0)
    for ax, (name, (s1, s2)) in zip(axes, data.items()):
        p_up = mannwhitneyu(s2, s1, alternative="greater").pvalue
        p_down = mannwhitneyu(s2, s1, alternative="less").pvalue
        ax.boxplot([s1, s2], showfliers=False)
        for i, (s, c) in enumerate([(s1, "C0"), (s2, "C3")], start=1):
            ax.scatter(rng.normal(i, 0.05, len(s)), s, s=8, alpha=0.4, color=c)
        ax.set_xticks([1, 2], [f"{labels[0]}\nn={len(s1)}", f"{labels[1]}\nn={len(s2)}"])
        diff = np.median(s2) - np.median(s1)
        ax.set_title(f"{name}\nmedian {np.median(s1):.2f} → {np.median(s2):.2f} dB (change {diff:+.2f} dB)")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
        rows.append(dict(measure=name, **{labels[0]: round(np.median(s1), 2), labels[1]: round(np.median(s2), 2)},
                         change_dB=round(diff, 2), descriptive_p_higher=float(f"{p_up:.2g}"),
                         descriptive_p_lower=float(f"{p_down:.2g}"), n=f"{len(s1)} / {len(s2)}"))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(rows).set_index("measure")


def roi_spectrum(spectrum, channels, name=None):
    """Add a region-of-interest pseudo-channel: the PSD averaged over `channels` per window.

    A window counts as clean for the region only if it is clean on every channel in it.
    Returns a new spectrum tuple with the extra channel appended (named e.g. "F3+F4").
    """
    psd, freqs, chs, keep = spectrum
    name = name or "+".join(channels)
    idx = [chs.index(c) for c in channels]
    roi = psd[:, idx].mean(axis=1, keepdims=True)          # NaN if any channel was rejected
    roi_keep = np.mean(np.all(np.isfinite(psd[:, idx, 0]), axis=1))
    return np.concatenate([psd, roi], axis=1), freqs, chs + [name], np.append(keep, roi_keep)


def periodic_metrics(spectrum, theta_ch="Fz", alpha_ch="Pz"):
    """Theta above 1/f at `theta_ch`, alpha above 1/f at `alpha_ch` (dB, from the averaged
    spectrum) and the 1/f exponent at both. NaN where a channel has no clean windows."""
    out = {}
    models = fit_fooof(spectrum, [theta_ch, alpha_ch])
    for key, ch, band in [("theta_db", theta_ch, "theta"), ("alpha_db", alpha_ch, "alpha")]:
        fm = models.get(ch)
        out[key] = periodic_power_db(fm, BANDS[band]) if fm else np.nan
        out[f"exponent_{ch}"] = fm.get_params("aperiodic_params", "exponent") if fm else np.nan
    return out


# ------------------------------------------------------------- theta, three ways
THETA_DEFINITIONS_MD = """**Theta power** at a channel is the total power between 4 and 8 Hz: the power spectral density
P(f) (Welch: 2 s Hann windows, 50% overlap, averaged over the clean windows) integrated over
4–8 Hz, in µV², reported as 10·log₁₀(θ / 1 µV²) dB. For a region such as F3+F4 the PSD is averaged
over its channels first. It is compared with the two derived measures used elsewhere:
**relative theta** (θ ÷ 4–40 Hz power) and **theta above 1/f** (the 4–8 Hz peak above the fitted
background). If all three move together, the change is in theta itself; if they disagree, part of
the change is in the rest of the spectrum."""


def theta_definitions(spectrum, ch):
    """Theta at `ch` (a channel or a `roi_spectrum` region) under three definitions:
    absolute power (dB µV²), relative power (share of TOTAL), and power above the 1/f fit (dB)."""
    psd, freqs, chs, _ = spectrum
    k = chs.index(ch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_psd = np.nanmean(psd[:, k], axis=0)
    m = (freqs >= BANDS["theta"][0]) & (freqs < BANDS["theta"][1])
    absolute = mean_psd[m].sum() * (freqs[1] - freqs[0])
    fm = fit_fooof(spectrum, [ch]).get(ch)
    return {"absolute (dB µV²)": 10 * np.log10(absolute),
            "relative (share of 4–40 Hz)": relative_power(spectrum, BANDS["theta"])[k],
            "above 1/f (dB)": periodic_power_db(fm, BANDS["theta"]) if fm else np.nan}


def plot_theta_definitions(spectra, labels, ch, title, colors=None):
    """Bar chart of theta at `ch` under the three definitions, one bar per spectrum.
    Returns the values as a DataFrame (rows = labels)."""
    table = pd.DataFrame([theta_definitions(s, ch) for s in spectra], index=labels)
    colors = colors or ["C0", "C3"] * (len(labels) // 2 + 1)
    fig, axes = plt.subplots(1, 3, figsize=(max(12, 1.6 * len(labels) * 3), 4.3))
    for ax, col in zip(axes, table.columns):
        vals = table[col].values
        ax.bar(range(len(labels)), vals, color=colors[: len(labels)])
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        pad = 0.25 * (hi - lo) if hi > lo else 0.5
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(len(labels)), labels, rotation=20 if len(labels) > 2 else 0, ha="right" if len(labels) > 2 else "center")
        ax.set_title(f"Theta at {ch}: {col}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()
    return table
