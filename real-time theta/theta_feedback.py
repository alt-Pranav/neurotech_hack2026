#!/usr/bin/env python3
"""
theta_feedback.py - real-time frontal theta "mastery" score from an LSL EEG stream.

Pipeline (same settings as the 2026-11-07 offline analysis):
  1. Pull all EEG channels from LSL, convert to microvolts.
  2. Causal 1-40 Hz band-pass + 60 Hz notch (filter state carried across chunks).
  3. Every 0.25 s take the most recent 2-s window:
       - detect flat/dead channels (e.g. Fz stuck at the amplifier rail) and drop them,
       - average reference over all working EEG channels,
       - for F3/F4/Fz: demean, Hann taper, FFT, one-sided PSD (uV^2/Hz),
         theta = sum of PSD over 4 <= f < 8 Hz x bin width  (uV^2),
       - combined theta = mean over the working target channels,
       - reject the window if any target channel exceeds 150 uV peak-to-peak.
  4. Calibration:
       - BASELINE  = median combined theta of the first 15 s of the stream,
       - Press SPACE when the task starts -> NO-MASTERY = median theta of the
         first 15 s of the task, after removing theta spikes (typing mistakes) above
         median + 2 robust SDs on a log scale, floored at 3 x median.
       - During scoring the spike cut-off is re-learned every 2 s from all task data so far.
  5. Scoring: current value = median of clean windows in the last 15 s;
       score = 100 * (no_mastery - value) / (no_mastery - baseline), clipped to 0..100
       (100 = at/below baseline, 0 = at/above no-mastery).

Keys (plot window must have focus):
  SPACE  task has started -> begin 15-s no-mastery calibration
  R      redo the no-mastery (task) calibration starting now
  B      redo the baseline starting now (also clears the task calibration)
  Q/Esc  quit (log is saved)

Example:
  python theta_feedback.py                       # first stream with type 'EEG'
  python theta_feedback.py --name "EE225-000000" # a specific LSL stream
  python theta_feedback.py --no-plot --task-start-after 30 --duration 300   # headless test
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
from scipy import signal

# ----------------------------------------------------------------------------- settings
DEFAULTS = dict(
    channels="F3,F4,Fz",
    band_lo=4.0, band_hi=8.0,        # theta band, Hz  (4 <= f < 8)
    win_s=2.0, step_s=0.25,          # FFT window length / hop
    median_s=15.0,                   # rolling median length for the live value
    baseline_s=15.0,                 # baseline calibration length
    calib_s=15.0,                    # no-mastery (task) calibration length
    hp=1.0, lp=40.0, notch=60.0,     # filters
    ptp_uv=150.0,                    # artifact rejection: peak-to-peak per 2-s window
    spike_k=2.0,                     # theta spike cut-off: median + k * robust SD (log10 scale)
    spike_min_ratio=3.0,             # ...but never below 3 x median task theta (today's cut-off was ~3.5-4 x)
    spike_pad_s=1.0,                 # also drop windows within +-1 s of a spike
    min_clean_frac=0.3,              # need >=30 % clean windows in the 15 s to report a value
    flat_uv=0.5,                     # channel is "flat" if its 2-s raw SD is below this (uV)
)
NON_EEG = re.compile(r"(trig|counter|marker|aux|bip|exg|status|event|stim|ecg|eog|emg|gsr|resp|acc)", re.I)


# ----------------------------------------------------------------------------- processor
class ThetaProcessor:
    """Pure signal-processing + calibration logic (no LSL, no GUI) so it can be tested offline."""

    def __init__(self, fs, ch_names, cfg, ref_mask=None):
        self.fs = float(fs)
        self.cfg = cfg
        self.names = list(ch_names)
        lower = [c.lower() for c in self.names]
        self.targets = [t.strip() for t in cfg.channels.split(",") if t.strip()]
        missing = [t for t in self.targets if t.lower() not in lower]
        if missing:
            raise ValueError(f"Target channel(s) {missing} not found in stream. Channels: {self.names}")
        self.tidx = [lower.index(t.lower()) for t in self.targets]
        n = len(self.names)
        self.ref_mask = np.ones(n, bool) if ref_mask is None else np.asarray(ref_mask, bool)

        # causal filters: band-pass + notch, as one SOS cascade
        sos_bp = signal.butter(4, [cfg.hp, cfg.lp], btype="bandpass", fs=self.fs, output="sos")
        b, a = signal.iirnotch(cfg.notch, 30.0, fs=self.fs)
        self.sos = np.vstack([sos_bp, signal.tf2sos(b, a)])
        self.zi = None

        self.W = int(round(cfg.win_s * self.fs))
        self.S = int(round(cfg.step_s * self.fs))
        self.hann = signal.get_window("hann", self.W)
        self.psd_scale = 2.0 / (self.fs * np.sum(self.hann ** 2))
        f = np.fft.rfftfreq(self.W, 1.0 / self.fs)
        self.band = (f >= cfg.band_lo) & (f < cfg.band_hi)
        self.df = f[1] - f[0]

        self.raw_buf = np.zeros((0, n))
        self.filt_buf = np.zeros((0, n))
        self.ts_buf = np.zeros(0)
        self.n_total = 0
        self.next_end = self.W

        self.windows = []          # all window records (dicts)
        self.events = []           # (sample, lsl_time, text)
        self.state = "BASELINE"
        self.message = "Recording baseline - sit still"
        self.base_start = 0
        self.baseline = np.nan
        self.task_start = None
        self.no_mastery = np.nan
        self.spike_cut = np.nan
        self.suppress_until = -1
        self.value = np.nan        # current 15-s median
        self.score = np.nan
        self.flat_now = []
        self._log_event(0, np.nan, "baseline_start")

    # ------------------------------------------------------------------ public API
    def push(self, chunk_uv, timestamps):
        """chunk_uv: (n_samples, n_channels) microvolts. Returns list of new window records."""
        x = np.asarray(chunk_uv, float)
        if x.ndim != 2 or len(x) == 0:
            return []
        if self.zi is None:  # start filters in steady state for the first sample (huge DC offsets)
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        self.raw_buf = np.vstack([self.raw_buf, x])
        self.filt_buf = np.vstack([self.filt_buf, y])
        self.ts_buf = np.concatenate([self.ts_buf, np.asarray(timestamps, float)])
        self.n_total += len(x)
        new = []
        while self.n_total >= self.next_end:
            off = len(self.raw_buf) - (self.n_total - self.next_end)
            rec = self._process_window(self.raw_buf[off - self.W:off], self.filt_buf[off - self.W:off],
                                       self.ts_buf[off - 1])
            new.append(rec)
            self.next_end += self.S
        keep = self.W
        self.raw_buf, self.filt_buf, self.ts_buf = self.raw_buf[-keep:], self.filt_buf[-keep:], self.ts_buf[-keep:]
        return new

    def start_task(self):
        """SPACE: task has started -> calibrate no-mastery over the next calib_s seconds."""
        if self.state == "BASELINE":
            self.message = "Baseline still recording - wait for it to finish before starting the task"
            return False
        self.task_start = self.n_total
        self.no_mastery = np.nan
        self.spike_cut = np.nan
        self.state = "TASK_CALIB"
        self.message = f"Task calibration (no-mastery) - {self.cfg.calib_s:.0f} s"
        self._log_event(self.n_total, self._last_ts(), "task_start")
        return True

    def redo_baseline(self):
        self.base_start = self.n_total
        self.baseline = self.no_mastery = self.spike_cut = np.nan
        self.task_start = None
        self.state = "BASELINE"
        self.message = "Recording baseline - sit still"
        self._log_event(self.n_total, self._last_ts(), "baseline_restart")

    # ------------------------------------------------------------------ internals
    def _last_ts(self):
        return float(self.ts_buf[-1]) if len(self.ts_buf) else np.nan

    def _log_event(self, sample, ts, text):
        self.events.append(dict(sample=int(sample), elapsed_s=sample / self.fs, lsl_time=ts, event=text))

    def _process_window(self, raw, filt, ts_end):
        cfg = self.cfg
        end = self.next_end
        flat = raw.std(axis=0) < cfg.flat_uv
        ref = self.ref_mask & ~flat
        if ref.sum() >= 2:
            filt = filt - filt[:, ref].mean(axis=1, keepdims=True)   # average reference
        thetas = {}
        ptps = []
        for name, i in zip(self.targets, self.tidx):
            if flat[i]:
                thetas[name] = np.nan
                continue
            s = filt[:, i] - filt[:, i].mean()
            X = np.fft.rfft(s * self.hann)
            psd = self.psd_scale * np.abs(X) ** 2
            thetas[name] = float(psd[self.band].sum() * self.df)
            ptps.append(np.ptp(s))
        good_vals = [thetas[name] for name, i in zip(self.targets, self.tidx) if not flat[i]]
        combined = float(np.mean(good_vals)) if good_vals else np.nan
        flat_targets = [self.targets[k] for k, i in enumerate(self.tidx) if flat[i]]
        self.flat_now = flat_targets
        rec = dict(window=len(self.windows), end_sample=end, elapsed_s=end / self.fs,
                   center_s=(end - self.W / 2) / self.fs, lsl_time=float(ts_end),
                   state=self.state, theta=thetas, combined=combined,
                   flat=";".join(flat_targets), n_ref=int(ref.sum()),
                   ptp_reject=bool(len(ptps) and max(ptps) > cfg.ptp_uv), spike=False)
        self.windows.append(rec)
        self._update_state(rec)
        return rec

    def _clean(self, recs, use_spike=True):
        return np.array([r["combined"] for r in recs
                         if np.isfinite(r["combined"]) and not r["ptp_reject"] and not (use_spike and r["spike"])])

    def _spike_cutoff(self, vals):
        """median + k robust SDs on log10 scale, floored at spike_min_ratio x median."""
        lv = np.log10(vals[vals > 0])
        med = np.median(lv)
        mad = 1.4826 * np.median(np.abs(lv - med))
        return float(max(10 ** (med + self.cfg.spike_k * mad), self.cfg.spike_min_ratio * 10 ** med))

    def _flag_spike(self, rec):
        pad = int(round(self.cfg.spike_pad_s * self.fs))
        rec["spike"] = True
        for w in reversed(self.windows):                      # windows within +-pad of the spike
            if w["end_sample"] < rec["end_sample"] - pad:
                break
            if w["end_sample"] <= rec["end_sample"] + pad:
                w["spike"] = True
        self.suppress_until = max(self.suppress_until, rec["end_sample"] + pad)   # future windows

    def _update_state(self, rec):
        cfg, fs = self.cfg, self.fs
        end = rec["end_sample"]

        # spike rejection once a cut-off exists (scoring phase). The cut-off keeps being
        # re-learned from all task windows so far (as in the offline analysis), every 2 s.
        if self.state == "SCORING" and rec["window"] % 8 == 0:
            vals = self._clean([r for r in self.windows if r["end_sample"] - self.W >= self.task_start], use_spike=False)
            if len(vals) >= 3:
                self.spike_cut = self._spike_cutoff(vals)
        if np.isfinite(self.spike_cut) and self.state == "SCORING":
            if np.isfinite(rec["combined"]) and rec["combined"] > self.spike_cut:
                self._flag_spike(rec)
            elif end <= self.suppress_until:
                rec["spike"] = True

        if self.state == "BASELINE":
            b_end = self.base_start + int(round(cfg.baseline_s * fs))
            if end >= b_end:
                recs = [r for r in self.windows if r["end_sample"] - self.W >= self.base_start and r["end_sample"] <= b_end]
                vals = self._clean(recs, use_spike=False)
                if len(vals) == 0:
                    self.message = "Baseline had no clean data (too many artifacts) - press B to redo"
                    self.state = "BAD_BASELINE"
                else:
                    self.baseline = float(np.median(vals))
                    self.state = "WAIT_TASK"
                    self.message = f"Baseline = {self.baseline:.2f} µV² - press SPACE when the task starts"
                    if len(vals) < cfg.min_clean_frac * len(recs):
                        self.message += f"  (warning: only {len(vals)}/{len(recs)} clean windows)"
                self._log_event(end, rec["lsl_time"], f"baseline_done value={self.baseline:.4f} clean={len(vals)}/{len(recs)}")

        elif self.state == "TASK_CALIB":
            c_end = self.task_start + int(round(cfg.calib_s * fs))
            if end >= c_end:
                recs = [r for r in self.windows if r["end_sample"] - self.W >= self.task_start and r["end_sample"] <= c_end]
                vals = self._clean(recs, use_spike=False)
                if len(vals) < 3:
                    self.state = "BAD_CALIB"
                    self.message = "Task calibration had too few clean windows - press R to redo"
                    self._log_event(end, rec["lsl_time"], "task_calib_failed too_few_clean")
                    return
                self.spike_cut = self._spike_cutoff(vals)
                for r in recs:                         # remove spikes (+-pad) inside the calibration
                    if np.isfinite(r["combined"]) and r["combined"] > self.spike_cut:
                        self._flag_spike(r)
                kept = self._clean(recs, use_spike=True)
                self.no_mastery = float(np.median(kept)) if len(kept) else float(np.median(vals))
                self._log_event(end, rec["lsl_time"],
                                f"task_calib_done no_mastery={self.no_mastery:.4f} spike_cut={self.spike_cut:.4f} "
                                f"clean={len(kept)}/{len(recs)}")
                if not np.isfinite(self.baseline):
                    self.state = "BAD_CALIB"
                    self.message = "No valid baseline - press B to redo baseline"
                elif self.no_mastery <= self.baseline:
                    self.state = "BAD_CALIB"
                    self.message = (f"No-mastery theta ({self.no_mastery:.2f}) <= baseline ({self.baseline:.2f}): "
                                    f"no range to score. Press R to redo the task calibration (or B for baseline).")
                else:
                    self.state = "SCORING"
                    self.message = (f"Scoring  |  baseline {self.baseline:.2f}  no-mastery {self.no_mastery:.2f}  "
                                    f"spike cut-off {self.spike_cut:.1f} µV²")

        # rolling 15-s median + score
        lo = end - int(round(cfg.median_s * fs))
        recent = [r for r in self.windows if r["end_sample"] > lo]
        vals = self._clean(recent, use_spike=True)
        if len(recent) and len(vals) >= cfg.min_clean_frac * len(recent):
            self.value = float(np.median(vals))
        else:
            self.value = np.nan
        if self.state == "SCORING" and np.isfinite(self.value):
            s = 100.0 * (self.no_mastery - self.value) / (self.no_mastery - self.baseline)
            self.score = float(np.clip(s, 0.0, 100.0))
        else:
            self.score = np.nan
        rec["median15"] = self.value
        rec["score"] = self.score
        rec["state"] = self.state


# ----------------------------------------------------------------------------- CSV logger
class Logger:
    """Writes one row per 2-s window. Rows are written ~1 s late so spike padding is final."""

    def __init__(self, path, targets, delay_samples):
        self.path = path
        self.targets = targets
        self.delay = delay_samples
        self.pending = []
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["window", "lsl_time", "elapsed_s", "window_center_s", "state"]
                        + [f"theta_{t}" for t in targets]
                        + ["theta_combined", "flat_channels", "n_ref_channels", "ptp_reject", "spike_reject",
                           "median15", "score"])

    def add(self, recs, current_end):
        self.pending.extend(recs)
        self.flush(current_end - self.delay)

    def flush(self, upto=None):
        while self.pending and (upto is None or self.pending[0]["end_sample"] <= upto):
            r = self.pending.pop(0)
            self.w.writerow([r["window"], f"{r['lsl_time']:.4f}", f"{r['elapsed_s']:.3f}", f"{r['center_s']:.3f}",
                             r["state"]] + [_fmt(r["theta"][t]) for t in self.targets]
                            + [_fmt(r["combined"]), r["flat"], r["n_ref"], int(r["ptp_reject"]), int(r["spike"]),
                               _fmt(r.get("median15")), _fmt(r.get("score"))])
        self.f.flush()

    def close(self):
        self.flush(None)
        self.f.close()


def _fmt(v):
    return "" if v is None or not np.isfinite(v) else f"{v:.4f}"


# ----------------------------------------------------------------------------- LSL input
def open_inlet(args):
    import pylsl
    print("Looking for LSL stream "
          + (f"named '{args.name}'" if args.name else f"of type '{args.type}'") + " ...", flush=True)
    if args.name:
        streams = pylsl.resolve_byprop("name", args.name, timeout=args.timeout)
    else:
        streams = pylsl.resolve_byprop("type", args.type, timeout=args.timeout)
    if not streams:
        sys.exit("No matching LSL stream found. Is the amplifier/LSL app streaming? (use --name or --type)")
    inlet = pylsl.StreamInlet(streams[0], max_buflen=60, recover=True)
    info = inlet.info()
    fs = info.nominal_srate()
    n = info.channel_count()
    labels, types, units = [], [], []
    ch = info.desc().child("channels").child("channel")
    for _ in range(n):
        labels.append(ch.child_value("label") or "")
        types.append(ch.child_value("type") or "")
        units.append(ch.child_value("unit") or "")
        ch = ch.next_sibling()
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != n:
            sys.exit(f"--labels has {len(labels)} names but the stream has {n} channels")
    if not all(labels):
        sys.exit("The stream has no channel labels. Pass them with --labels Fp1,Fp2,... in stream order.")
    # eego labels its EEG channels type "Referential", so exclude known non-EEG channels
    # by label/type instead of requiring type "EEG".
    ref_mask = np.array([not (NON_EEG.search(l) or NON_EEG.search(t)) for l, t in zip(labels, types)])
    if ref_mask.sum() < 2:
        sys.exit(f"Fewer than 2 EEG channels found for the average reference. Labels: {labels}, types: {types}")
    print(f"Connected: '{info.name()}'  {n} ch @ {fs:g} Hz  units={set(u for u in units if u) or 'unknown'}")
    print(f"Average reference uses {int(ref_mask.sum())} channels; excluded: "
          f"{[l for l, m in zip(labels, ref_mask) if not m] or 'none'}")
    return inlet, fs, labels, units, ref_mask


def unit_factor(args, units, first_chunk, ref_mask):
    """Return multiplier that converts the stream values to microvolts.

    In auto mode the signal size decides, not the unit label: eego labels its channels
    "Volt" but streams microvolts. EEG varies by ~1e-5 V, i.e. ~10 uV.
    """
    if args.units != "auto":
        return {"uV": 1.0, "mV": 1e3, "V": 1e6}[args.units]
    x = np.asarray(first_chunk, float)[:, ref_mask]
    spread = np.median(x.std(axis=0)) if len(x) > 1 else np.median(np.abs(x))
    f = 1e6 if spread < 0.01 else 1.0
    labelled = ", ".join(sorted({u for u, m in zip(units, ref_mask) if m and u})) or "none"
    print(f"Units: median channel SD = {spread:.3g} (stream label: {labelled}) -> treating values as "
          f"{'volts' if f == 1e6 else 'microvolts'} (override with --units)")
    return f


# ----------------------------------------------------------------------------- live plot
def run_gui(args, inlet, proc, logger, factor_holder):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    for k in ("keymap.home", "keymap.back", "keymap.save", "keymap.quit"):   # free R/B/Q for our controls
        plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in ("r", "b", "q", "c", "s")]

    C_SCORE, C_THETA, C_MED, C_BASE, C_NM, INK, MUTED = "#2a78d6", "#9a9994", "#0b0b0b", "#1baf7a", "#eb6834", "#0b0b0b", "#52514e"
    fig = plt.figure(figsize=(12, 7.5), facecolor="#fcfcfb")
    fig.canvas.manager.set_window_title("Theta mastery score")
    gs = fig.add_gridspec(3, 4, height_ratios=[0.55, 1.2, 1.0], hspace=0.45, wspace=0.3)
    ax_big = fig.add_subplot(gs[0, :])
    ax_score = fig.add_subplot(gs[1, :])
    ax_theta = fig.add_subplot(gs[2, :], sharex=ax_score)
    for a in (ax_score, ax_theta):
        a.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="y", color="#e6e5e0")
    ax_big.axis("off")
    big = ax_big.text(0.0, 0.55, "--", fontsize=44, fontweight="bold", color=INK, va="center")
    status = ax_big.text(0.22, 0.75, "", fontsize=11, color=INK, va="center")
    sub = ax_big.text(0.22, 0.35, "", fontsize=9.5, color=MUTED, va="center")
    keys = ax_big.text(0.22, 0.02, "SPACE task started   R redo task calibration   B redo baseline   Q quit",
                       fontsize=9, color=MUTED, va="center")
    ln_score, = ax_score.plot([], [], color=C_SCORE, lw=2)
    ax_score.set_ylim(-3, 103)
    ax_score.set_ylabel("Score (0-100)")
    ax_score.set_title("Mastery score (100 = at/below baseline theta, 0 = at/above no-mastery theta)",
                       loc="left", fontsize=10, color=INK)
    ln_theta, = ax_theta.plot([], [], color=C_THETA, lw=0.8, label="2-s window theta")
    ln_rej, = ax_theta.plot([], [], ls="none", marker="x", ms=4, color="#b8462e", label="rejected window")
    ln_med, = ax_theta.plot([], [], color=C_MED, lw=2, label="15-s median")
    ln_base = ax_theta.axhline(np.nan, color=C_BASE, lw=1.5, ls="--", label="baseline")
    ln_nm = ax_theta.axhline(np.nan, color=C_NM, lw=1.5, ls="--", label="no-mastery")
    ax_theta.set_ylabel("Theta 4-8 Hz (uV²)")
    ax_theta.set_xlabel("Time since start (s)")
    ax_theta.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncol=5)
    spans = []

    def on_key(ev):
        k = (ev.key or "").lower()
        if k == " ":
            proc.start_task()
        elif k == "r":
            if proc.state not in ("BASELINE",):
                proc.start_task()
        elif k == "b":
            proc.redo_baseline()
        elif k in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        pull_and_process(inlet, proc, logger, factor_holder)
        W = proc.windows
        if not W:
            status.set_text("Waiting for data ...")
            return
        t = np.array([r["center_s"] for r in W])
        th = np.array([r["combined"] for r in W])
        md = np.array([r.get("median15", np.nan) for r in W])
        sc = np.array([r.get("score", np.nan) for r in W])
        rej = np.array([r["ptp_reject"] or r["spike"] for r in W])
        lo = max(0.0, t[-1] - args.plot_window)
        ln_score.set_data(t, sc)
        ln_theta.set_data(t, np.where(rej, np.nan, th))
        ln_rej.set_data(t[rej], th[rej])
        ln_med.set_data(t, md)
        ln_base.set_ydata([proc.baseline] * 2)
        ln_nm.set_ydata([proc.no_mastery] * 2)
        ax_score.set_xlim(lo, max(lo + 10, t[-1] + 1))
        vis = (t >= lo) & np.isfinite(th) & ~rej
        top = np.nanpercentile(th[vis], 98) if vis.any() else 20
        cands = [top, proc.baseline, proc.no_mastery]
        ax_theta.set_ylim(0, 1.15 * np.nanmax([c for c in cands if np.isfinite(c)] + [1.0]))
        for s in spans:
            s.remove()
        spans.clear()
        for e0, e1, col in calib_spans(proc):
            spans.append(ax_theta.axvspan(e0, e1, color=col, alpha=0.12, lw=0))
        s = proc.score
        big.set_text(f"{s:.0f}" if np.isfinite(s) else "--")
        big.set_color(INK if np.isfinite(s) else MUTED)
        status.set_text(proc.message)
        info = f"t = {t[-1]:.0f} s   15-s median theta = " + (f"{proc.value:.2f} uV²" if np.isfinite(proc.value) else "n/a (too few clean windows)")
        info += f"   channels: {', '.join(c for c in proc.targets if c not in proc.flat_now)}"
        if proc.flat_now:
            info += f"   FLAT/DROPPED: {', '.join(proc.flat_now)}"
        sub.set_text(info)
        sub.set_color("#b8462e" if proc.flat_now else MUTED)

    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)  # noqa: F841
    plt.show()


def calib_spans(proc):
    fs = proc.fs
    out = []
    out.append((proc.base_start / fs, (proc.base_start / fs) + proc.cfg.baseline_s, "#1baf7a"))
    if proc.task_start is not None:
        out.append((proc.task_start / fs, proc.task_start / fs + proc.cfg.calib_s, "#eb6834"))
    return out


# ----------------------------------------------------------------------------- main loop helpers
def pull_and_process(inlet, proc, logger, factor_holder):
    data, ts = inlet.pull_chunk(timeout=0.0, max_samples=100000)
    if not data:
        return []
    x = np.asarray(data, float)
    if factor_holder[0] is None:
        # hold the first 0.5 s so the unit guess sees enough samples, then process it all
        held = factor_holder[3]
        held.append((x, ts))
        if sum(len(h[0]) for h in held) < proc.fs / 2:
            return []
        x = np.vstack([h[0] for h in held])
        ts = [t for h in held for t in h[1]]
        held.clear()
        factor_holder[0] = unit_factor(factor_holder[1], factor_holder[2], x, proc.ref_mask)
    recs = proc.push(x * factor_holder[0], ts)
    if recs:
        logger.add(recs, proc.n_total)
    return recs


def run_headless(args, inlet, proc, logger, factor_holder):
    t0 = time.time()
    last_print = 0
    task_sent = False
    try:
        while True:
            pull_and_process(inlet, proc, logger, factor_holder)
            el = proc.n_total / proc.fs
            if args.task_start_after is not None and not task_sent and el >= args.task_start_after:
                task_sent = proc.start_task()
            if el - last_print >= 5 and proc.windows:
                last_print = el
                print(f"[{el:7.1f} s] {proc.state:11s} value={proc.value:7.2f}  score="
                      + (f"{proc.score:5.1f}" if np.isfinite(proc.score) else "  -- ")
                      + (f"  flat={proc.flat_now}" if proc.flat_now else ""), flush=True)
            if args.duration and el >= args.duration:
                break
            if time.time() - t0 > 5 and proc.n_total == 0:
                print("No data arriving ...", flush=True)
                t0 = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass


def main():
    p = argparse.ArgumentParser(description="Real-time frontal theta mastery score from LSL EEG.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--name", help="LSL stream name (default: first stream of --type)")
    p.add_argument("--type", default="EEG", help="LSL stream type to look for")
    p.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for the stream")
    p.add_argument("--labels", help="comma-separated channel labels if the stream has none")
    p.add_argument("--units", choices=["auto", "uV", "mV", "V"], default="auto", help="stream units")
    p.add_argument("--log-dir", default="theta_logs", help="folder for CSV/JSON logs")
    p.add_argument("--no-plot", action="store_true", help="run without the live plot (console only)")
    p.add_argument("--plot-window", type=float, default=180.0, help="seconds of history shown in the plot")
    p.add_argument("--task-start-after", type=float, help="(testing) auto-press SPACE after this many seconds of data")
    p.add_argument("--duration", type=float, help="(headless) stop after this many seconds of data")
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = p.parse_args()

    inlet, fs, labels, units, ref_mask = open_inlet(args)
    proc = ThetaProcessor(fs, labels, args, ref_mask)
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(args.log_dir, f"theta_{stamp}.csv")
    logger = Logger(csv_path, proc.targets, int(round((args.spike_pad_s + args.step_s) * fs)))
    factor_holder = [None, args, units, []]   # factor, args, unit labels, chunks held before the unit guess
    print(f"Logging to {csv_path}")
    print("Baseline: first %.0f s - sit still." % args.baseline_s)
    try:
        if args.no_plot:
            run_headless(args, inlet, proc, logger, factor_holder)
        else:
            run_gui(args, inlet, proc, logger, factor_holder)
    finally:
        logger.close()
        summary = dict(started=stamp, stream=inlet.info().name(), fs=fs, channels=labels, targets=proc.targets,
                       settings={k: getattr(args, k) for k in DEFAULTS}, baseline_uV2=proc.baseline,
                       no_mastery_uV2=proc.no_mastery, spike_cutoff_uV2=proc.spike_cut, final_state=proc.state,
                       events=proc.events, unit_factor_to_uV=factor_holder[0], csv=os.path.basename(csv_path))
        js = csv_path.replace(".csv", "_session.json")
        with open(js, "w") as f:
            json.dump(summary, f, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o))
        print(f"Saved {csv_path} and {js}")


if __name__ == "__main__":
    main()
