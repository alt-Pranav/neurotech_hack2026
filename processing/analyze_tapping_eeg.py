"""Finger tapping sequence learning: line up EEG with the tapping log and measure band power.

Inputs
  --eeg     EEG recording (ANT Neuro .cnt, or anything mne.io.read_raw can open)
  --blocks  "Blocks CSV" exported from the finger tapping page (one row per round)

Outputs (in --out)
  rounds.csv     one row per round: times, typing speed, tap vs rest power change per signal
  timeline.csv   band power every 1 s, in dB relative to the still baseline
  pooled.csv     all tap windows vs all rest windows, with permutation p-values
  alignment.txt  clock check: do movement artifacts follow the round starts?
  summary.png    typing speed and the C4 timeline

Example
  python analyze_tapping_eeg.py --eeg rec.cnt --blocks P01_session3_blocks.csv \
      --out ../results/P01_session3 --exclude 377.7-381.0
"""
import argparse
import csv
import datetime as dt
import os

import numpy as np
import mne
from scipy.signal import welch

mne.set_log_level("ERROR")

BANDS = {  # name: (channel, low Hz, high Hz)
    "mu_C3": ("C3", 8, 12),
    "mu_C4": ("C4", 8, 12),
    "beta_C3": ("C3", 13, 30),
    "beta_C4": ("C4", 13, 30),
    "theta_Fz": ("Fz", 4, 7),
}
NON_SCALP = ("Fp1", "Fp2", "F9", "F10", "M1", "M2")  # left out of the average reference
EMG_CHANNELS = ["T7", "T8", "F7", "F8"]
WIN = 2.0           # analysis window, s
PTP_LIMIT = 150.0   # reject windows over this peak-to-peak, uV
TAP_SKIP = 1.0      # ignore the first second of each tap round (reaction)
REST_LEN = 9.0      # rest window taken right before each round, s


def load_eeg(path):
    if path.lower().endswith(".cnt"):
        raw = mne.io.read_raw_ant(path, preload=True)
    else:
        raw = mne.io.read_raw(path, preload=True)
    return raw


def load_blocks(path):
    rows = list(csv.DictReader(open(path, newline="")))
    out = []
    for r in rows:
        out.append({
            "round": int(r["block"]),
            "phase": "new" if r["phase"] in ("control", "new") else "practice",
            "sequence": r["sequence"],
            "start_unix": float(r["start_epoch_ms"]) / 1000,
            "end_unix": float(r["end_epoch_ms"]) / 1000,
            "pattern_ms": float(r["mean_sequence_ms"]) if r["mean_sequence_ms"] else None,
            "full_patterns": int(r["correct_sequences"]),
            "presses": int(r["presses"]),
            "wrong_keys": int(r["error_presses"]),
        })
    return sorted(out, key=lambda r: r["round"])


def parse_ranges(items):
    out = []
    for it in items or []:
        a, b = it.split("-")
        out.append((float(a), float(b)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eeg", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", help="EEG-time ranges to drop, in seconds, e.g. 377.7-381.0")
    ap.add_argument("--utc-offset", type=float, default=-5, help="hours, for printing clock times (default EST)")
    ap.add_argument("--line", type=float, default=60, help="mains frequency, Hz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tz = dt.timezone(dt.timedelta(hours=args.utc_offset))
    rng = np.random.default_rng(0)

    raw = load_eeg(args.eeg)
    sf = raw.info["sfreq"]
    t0 = raw.info["meas_date"].timestamp()
    dur = raw.n_times / sf
    blocks = load_blocks(args.blocks)
    for b in blocks:
        b["start_s"] = b["start_unix"] - t0
        b["end_s"] = min(b["end_unix"] - t0, dur)
    excl = parse_ranges(args.exclude)
    first = blocks[0]["start_s"]
    BASE = (max(0.0, first - 25.0), first)

    # ---------- clock check: big artifacts vs round starts ----------
    art = raw.copy().pick([c for c in ("Fp1", "Fp2", "F7", "F8", "T7", "T8") if c in raw.ch_names])
    art.notch_filter([args.line, 2 * args.line, 3 * args.line]).filter(1, 100)
    amp = np.abs(art.get_data()).max(0) * 1e6
    half = int(sf / 2)
    seg = amp[: len(amp) // half * half].reshape(-1, half).max(1)
    times = np.arange(len(seg)) * 0.5
    inside = (times > first - 25) & (times < dur - 5)
    top = np.argsort(np.where(inside, seg, 0))[::-1][:15]
    lines = [f"EEG recording started {dt.datetime.fromtimestamp(t0, tz):%H:%M:%S.%f} and ran {dur:.1f} s.",
             f"Round 1 starts {first:.2f} s into the recording.", "",
             "Largest artifacts, and how long after the nearest round start they fall:"]
    after = []
    starts = np.array([b["start_s"] for b in blocks])
    for i in sorted(top, key=lambda i: times[i]):
        t = times[i]
        prev = starts[starts <= t]
        lag = t - prev.max() if len(prev) else float("nan")
        after.append(lag)
        lines.append(f"  {t:7.1f} s  ({dt.datetime.fromtimestamp(t0 + t, tz):%H:%M:%S})  {seg[i]:5.0f} uV  {lag:5.1f} s after a round start")
    share = np.mean([0 <= a <= 4.5 for a in after if not np.isnan(a)])
    lines += ["", f"{share:.0%} of these fall 0 to 4.5 s after a round start (about 18% would by chance).",
              "A high share means the laptop and EEG clocks agree."]
    open(os.path.join(args.out, "alignment.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------- preprocessing ----------
    raw.notch_filter([args.line, 2 * args.line, 3 * args.line]).filter(1, 45)
    raw.pick([c for c in raw.ch_names if c not in NON_SCALP])
    raw.set_eeg_reference("average")
    X = {c: raw.get_data(picks=[c])[0] * 1e6 for c in {v[0] for v in BANDS.values()}}

    emg_raw = load_eeg(args.eeg).pick(EMG_CHANNELS)
    emg_raw.notch_filter([args.line, 2 * args.line, 3 * args.line]).filter(30, 100)
    emg = np.sqrt((emg_raw.get_data() ** 2).mean(0)) * 1e6
    n = int(WIN * sf)
    emg_ref = np.percentile([emg[int(t * sf): int(t * sf) + n].mean() for t in np.arange(BASE[0], dur - WIN, 1.0)], 90)

    def excluded(t):
        return any(a - WIN < t < b for a, b in excl)

    def clean(t):
        if excluded(t):
            return False
        i = int(t * sf)
        if any(np.ptp(X[c][i: i + n]) > PTP_LIMIT for c in X):
            return False
        return emg[i: i + n].mean() <= emg_ref * 1.5

    def windows(a, b, step=1.0):
        ts, t = [], a
        while t + WIN <= b:
            ts.append(t)
            t += step
        return ts

    def power(ch, t, lo, hi):
        i = int(t * sf)
        f, p = welch(X[ch][i: i + n], fs=sf, nperseg=n)
        return p[(f >= lo) & (f <= hi)].mean()

    # ---------- per round: tap vs the rest right before it ----------
    for k, b in enumerate(blocks):
        tap = windows(b["start_s"] + TAP_SKIP, b["end_s"])
        rest = windows(*(BASE if k == 0 else (b["start_s"] - REST_LEN - 0.5, b["start_s"] - 0.5)))
        tg, rg = [t for t in tap if clean(t)], [t for t in rest if clean(t)]
        b["eeg_windows_kept"], b["eeg_windows_total"] = len(tg), len(tap)
        for name, (ch, lo, hi) in BANDS.items():
            if len(tg) >= 3 and len(rg) >= 3:
                tp = np.mean([power(ch, t, lo, hi) for t in tg])
                rp = np.mean([power(ch, t, lo, hi) for t in rg])
                b[name + "_pct"] = round((tp - rp) / rp * 100, 1)
            else:
                b[name + "_pct"] = ""

    cols = ["round", "phase", "sequence", "clock_start", "start_s", "end_s", "pattern_ms", "full_patterns",
            "presses", "wrong_keys"] + [k + "_pct" for k in BANDS] + ["eeg_windows_kept", "eeg_windows_total"]
    with open(os.path.join(args.out, "rounds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for b in blocks:
            b["clock_start"] = dt.datetime.fromtimestamp(b["start_unix"], tz).strftime("%H:%M:%S.%f")[:-3]
            w.writerow([round(b[c], 2) if isinstance(b[c], float) else b[c] for c in cols])

    # ---------- timeline ----------
    base_t = [t for t in windows(*BASE) if clean(t)]
    base_p = {k: np.median([power(ch, t, lo, hi) for t in base_t]) for k, (ch, lo, hi) in BANDS.items()}
    with open(os.path.join(args.out, "timeline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + [k + "_db" for k in BANDS] + ["muscle_uv", "clean"])
        for t in np.arange(BASE[0], dur - WIN, 1.0):
            vals = [round(10 * np.log10(power(ch, t, lo, hi) / base_p[k]), 2) for k, (ch, lo, hi) in BANDS.items()]
            w.writerow([round(t + WIN / 2, 1)] + vals + [round(emg[int(t * sf): int(t * sf) + n].mean(), 2), int(clean(t))])

    # ---------- pooled, non-overlapping windows, permutation test ----------
    def pooled_windows(phase, kind):
        out = []
        for k, b in enumerate(blocks):
            if b["phase"] != phase:
                continue
            a, e = (b["start_s"] + TAP_SKIP, b["end_s"]) if kind == "tap" else (b["end_s"] + 0.5, min(b["end_s"] + REST_LEN + 0.5, dur))
            out += [t for t in windows(a, e, step=WIN) if not excluded(t)]
        return out

    rows = []
    for name, (ch, lo, hi) in BANDS.items():
        for phase in ("practice", "new"):
            def vals(kind):
                v = []
                for t in pooled_windows(phase, kind):
                    i = int(t * sf)
                    if np.ptp(X[ch][i: i + n]) < PTP_LIMIT:
                        v.append(power(ch, t, lo, hi))
                return np.array(v)
            tp, rp = vals("tap"), vals("rest")
            if len(tp) < 3 or len(rp) < 3:
                continue
            d = (tp.mean() - rp.mean()) / rp.mean() * 100
            allv = np.r_[tp, rp]
            null = []
            for _ in range(5000):
                rng.shuffle(allv)
                null.append((allv[: len(tp)].mean() - allv[len(tp):].mean()) / allv[len(tp):].mean() * 100)
            p = float(np.mean(np.abs(null) >= abs(d)))
            rows.append([name.replace("_", " "), phase, round(d, 1), round(p, 3), len(tp), len(rp)])
    with open(os.path.join(args.out, "pooled.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "rounds", "change_pct", "p_value", "tap_windows", "rest_windows"])
        w.writerows(rows)

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 1.3]})
    col = ["#3b6fb6" if b["phase"] == "practice" else "#d9822b" for b in blocks]
    ax[0].bar([b["round"] for b in blocks], [b["pattern_ms"] or 0 for b in blocks], color=col)
    ax[0].set_ylabel("ms per pattern")
    ax[0].set_xticks([b["round"] for b in blocks])
    ax[0].set_xlabel("round (blue = practice, orange = new pattern)")
    ax[0].set_title("Typing: time per 7-key pattern")
    tl = list(csv.DictReader(open(os.path.join(args.out, "timeline.csv"))))
    t = np.array([float(r["t_s"]) for r in tl])
    ok = np.array([r["clean"] == "1" for r in tl])
    for b in blocks:
        ax[1].axvspan(b["start_s"], b["end_s"], color="#3b6fb6" if b["phase"] == "practice" else "#d9822b", alpha=0.12)
    for a, e in excl:
        ax[1].axvspan(a, e, color="red", alpha=0.3)
    for key, c in (("mu_C4_db", "#3b6fb6"), ("beta_C4_db", "#2a9d8f")):
        y = np.array([float(r[key]) for r in tl])
        y[~ok] = np.nan
        ax[1].plot(t, np.convolve(np.nan_to_num(y), np.ones(3) / 3, "same") * np.where(ok, 1, np.nan), color=c, lw=1.5, label=key.replace("_db", "").replace("_", " at "))
    ax[1].axhline(0, color="grey", ls="--", lw=1)
    ax[1].set_xlabel("seconds from EEG start (shaded = tap rounds)")
    ax[1].set_ylabel("dB vs still baseline")
    ax[1].set_title("EEG at C4 (right motor cortex, left hand)")
    ax[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "summary.png"), dpi=120)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
