"""Publish a fake eego-like EEG stream over LSL, for development without the headset.

    python acquisition/simulate_eego.py [--channels 32] [--srate 500]

Each channel is noise plus a 10 Hz alpha rhythm, in volts. A TRIGGER channel
(pulse every 2 s) is appended so code that excludes non-EEG channels can be tested.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

# Standard 10-20 / 10-10 labels; extra channels get generic names.
LABELS_10_20 = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "M1", "T7", "C3", "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--srate", type=float, default=500.0)
    p.add_argument("--name", default="EE225-SIMULATED")
    args = p.parse_args()

    labels = [LABELS_10_20[i] if i < len(LABELS_10_20) else f"E{i + 1}" for i in range(args.channels)]
    info = StreamInfo(args.name, "EEG", args.channels + 1, args.srate, "float32", "eego-sim")
    chans = info.desc().append_child("channels")
    for label in labels:
        c = chans.append_child("channel")
        c.append_child_value("label", label)
        c.append_child_value("type", "EEG")
        c.append_child_value("unit", "V")
    trig = chans.append_child("channel")
    trig.append_child_value("label", "TRIGGER")
    trig.append_child_value("type", "TRIGGER")
    trig.append_child_value("unit", "")

    outlet = StreamOutlet(info, chunk_size=int(args.srate // 50))
    print(f"Streaming {args.name!r}: {args.channels} EEG + TRIGGER @ {args.srate:g} Hz. Ctrl+C to stop.")

    rng = np.random.default_rng()
    phase = rng.uniform(0, 2 * np.pi, args.channels)
    n_sent = 0
    start = local_clock()
    try:
        while True:
            n_due = int((local_clock() - start) * args.srate) - n_sent
            if n_due > 0:
                t = (n_sent + np.arange(n_due)) / args.srate
                eeg = 10e-6 * np.sin(2 * np.pi * 10 * t[:, None] + phase) + 5e-6 * rng.standard_normal((n_due, args.channels))
                trigger = ((n_sent + np.arange(n_due)) % int(2 * args.srate) == 0).astype(float)
                outlet.push_chunk(np.column_stack([eeg, trigger]).astype(np.float32))
                n_sent += n_due
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
