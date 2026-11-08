#!/usr/bin/env python3
"""
lsl_replay_cnt.py - stream an ANT Neuro / eego .cnt recording over LSL as if it were live.
Use it to test theta_feedback.py without the amplifier.

Example (today's tapping session, starting at the still baseline, real-time speed):
  python lsl_replay_cnt.py team6_gary_2026-11-07_19-24-29.cnt --start 22.6
Then, in another terminal:
  python theta_feedback.py --name CNT-replay
  (press SPACE ~30 s after starting to mark the first tapping round)
"""
import argparse
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Replay a .cnt EEG file over LSL in real time.")
    p.add_argument("cnt", help="path to the .cnt file")
    p.add_argument("--start", type=float, default=0.0, help="start this many seconds into the recording")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    p.add_argument("--name", default="CNT-replay", help="LSL stream name")
    p.add_argument("--units", choices=["uV", "V"], default="uV", help="units to send")
    p.add_argument("--chunk-ms", type=float, default=20.0, help="chunk size in ms")
    args = p.parse_args()

    import mne
    from pylsl import StreamInfo, StreamOutlet, local_clock

    mne.set_log_level("error")
    raw = mne.io.read_raw_ant(args.cnt, preload=True)
    fs = raw.info["sfreq"]
    scale = 1e6 if args.units == "uV" else 1.0
    data = (raw.get_data() * scale).T.astype(np.float32)        # (n_samples, n_channels)
    data = data[int(round(args.start * fs)):]
    labels = raw.ch_names

    info = StreamInfo(args.name, "EEG", len(labels), fs, "float32", f"cnt-replay-{abs(hash(args.cnt)) % 10**8}")
    chans = info.desc().append_child("channels")
    for lab in labels:
        c = chans.append_child("channel")
        c.append_child_value("label", lab)
        c.append_child_value("unit", "microvolts" if args.units == "uV" else "volts")
        c.append_child_value("type", "EEG")
    outlet = StreamOutlet(info, chunk_size=int(fs * args.chunk_ms / 1000))

    n = len(data)
    step = max(1, int(round(fs * args.chunk_ms / 1000)))
    print(f"Streaming '{args.name}': {len(labels)} ch @ {fs:g} Hz, {n / fs:.1f} s from t={args.start:g} s, "
          f"speed x{args.speed:g}. Ctrl+C to stop.", flush=True)
    t_start = time.perf_counter()
    lsl0 = local_clock()
    i = 0
    try:
        while i < n:
            due = (i / fs) / args.speed
            wait = due - (time.perf_counter() - t_start)
            if wait > 0:
                time.sleep(wait)
            j = min(n, i + step)
            stamps = lsl0 + (np.arange(i, j) / fs) / args.speed
            outlet.push_chunk(data[i:j].tolist(), stamps.tolist())
            i = j
            if i % int(fs * 10) < step:
                print(f"  sent {i / fs:6.1f} s  (recording time {args.start + i / fs:6.1f} s)", flush=True)
    except KeyboardInterrupt:
        pass
    print("Done.")
    time.sleep(2.0)   # let the receiver drain


if __name__ == "__main__":
    main()
