"""Receive EEG from ANT Neuro eego software (EE-2xx amplifier) over Lab Streaming Layer.

The eego software publishes the amplifier data as an LSL stream (type "EEG"). This
module finds that stream, reads its channel metadata, and keeps the most recent
samples in a ring buffer that processing / ML / dashboard code can read from.

Command line:
    python acquisition/eego_lsl_receiver.py --list
    python acquisition/eego_lsl_receiver.py [--name NAME] [--save rec.csv] [--duration 60]

Library:
    from acquisition.eego_lsl_receiver import EegoReceiver
    rx = EegoReceiver.connect()
    rx.update()
    data, times = rx.latest(seconds=2)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from pylsl import (
    StreamInfo,
    StreamInlet,
    proc_clocksync,
    proc_dejitter,
    resolve_byprop,
    resolve_streams,
)

# Channels eego may append after the electrodes. They are not EEG and should be
# excluded from filtering / ML (see EegoReceiver.eeg_picks).
NON_EEG_LABELS = {"TRIGGER", "TRG", "COUNTER", "SAMPLE_COUNTER", "SAMPLECOUNTER"}


@dataclass
class ChannelInfo:
    label: str
    type: str = ""
    unit: str = ""


def _read_channels(info: StreamInfo) -> list[ChannelInfo]:
    """Read channel labels/types/units from the stream's XML description.

    Falls back to generic names (ch1, ch2, ...) if the sender gave no metadata.
    """
    channels: list[ChannelInfo] = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        channels.append(
            ChannelInfo(
                label=ch.child_value("label"),
                type=ch.child_value("type"),
                unit=ch.child_value("unit"),
            )
        )
        ch = ch.next_sibling("channel")
    if len(channels) != info.channel_count():
        channels = [ChannelInfo(f"ch{i + 1}") for i in range(info.channel_count())]
    return channels


def list_streams(wait: float = 2.0) -> list[StreamInfo]:
    return resolve_streams(wait_time=wait)


@dataclass
class EegoReceiver:
    inlet: StreamInlet
    info: StreamInfo
    channels: list[ChannelInfo]
    buffer_seconds: float = 30.0
    _buf: np.ndarray = field(init=False, repr=False)
    _ts: np.ndarray = field(init=False, repr=False)
    _write_pos: int = field(init=False, default=0)
    n_received: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        n = max(1, int(round(self.buffer_seconds * self.sfreq)))
        self._buf = np.zeros((self.n_channels, n), dtype=np.float64)
        self._ts = np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------------ setup
    @classmethod
    def connect(
        cls,
        name: str | None = None,
        stream_type: str = "EEG",
        timeout: float = 10.0,
        buffer_seconds: float = 30.0,
    ) -> "EegoReceiver":
        """Find the eego stream and open an inlet.

        name: exact LSL stream name to pick (see --list). If omitted, the first stream
              of `stream_type` is used.
        """
        if name:
            found = resolve_byprop("name", name, timeout=timeout)
        else:
            found = resolve_byprop("type", stream_type, timeout=timeout)
        if not found:
            what = f"name={name!r}" if name else f"type={stream_type!r}"
            raise RuntimeError(
                f"No LSL stream with {what} found within {timeout}s. Is eego streaming "
                "to LSL, and are you on the same local network? Try --list."
            )
        if len(found) > 1 and not name:
            print(
                f"[warn] {len(found)} {stream_type} streams found, using "
                f"{found[0].name()!r}. Pick one with --name.",
                file=sys.stderr,
            )

        # clocksync: map sender timestamps to our local clock.
        # dejitter: smooth timestamps for a regular-rate stream.
        inlet = StreamInlet(
            found[0],
            max_buflen=int(max(buffer_seconds, 10)),
            processing_flags=proc_clocksync | proc_dejitter,
        )
        info = inlet.info(timeout=timeout)  # full info, including channel metadata
        inlet.open_stream(timeout=timeout)  # start buffering now, not at the first pull
        return cls(inlet, info, _read_channels(info), buffer_seconds)

    # ------------------------------------------------------------- properties
    @property
    def sfreq(self) -> float:
        return self.info.nominal_srate()

    @property
    def n_channels(self) -> int:
        return self.info.channel_count()

    @property
    def ch_names(self) -> list[str]:
        return [c.label for c in self.channels]

    @property
    def eeg_picks(self) -> list[int]:
        """Indices of real EEG channels (drops trigger / sample-counter channels)."""
        return [
            i
            for i, c in enumerate(self.channels)
            if c.label.upper() not in NON_EEG_LABELS
            and c.type.upper() not in {"TRIGGER", "MISC", "STIM"}
        ]

    # ------------------------------------------------------------------- data
    def update(self, timeout: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Pull all samples that have arrived since the last call.

        Returns the new chunk as (data[n_channels, n_new], timestamps[n_new]) and
        also appends it to the ring buffer.
        """
        samples, stamps = self.inlet.pull_chunk(timeout=timeout)
        if not stamps:
            return np.empty((self.n_channels, 0)), np.empty(0)
        chunk = np.asarray(samples, dtype=np.float64).T
        ts = np.asarray(stamps, dtype=np.float64)
        self._append(chunk, ts)
        return chunk, ts

    def _append(self, chunk: np.ndarray, ts: np.ndarray) -> None:
        size = self._buf.shape[1]
        n = chunk.shape[1]
        self.n_received += n
        if n >= size:  # chunk larger than buffer: keep only its tail
            chunk, ts, n = chunk[:, -size:], ts[-size:], size
        idx = (self._write_pos + np.arange(n)) % size
        self._buf[:, idx] = chunk
        self._ts[idx] = ts
        self._write_pos = (self._write_pos + n) % size

    def latest(self, seconds: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Most recent samples in time order: (data[n_channels, n], timestamps[n])."""
        size = self._buf.shape[1]
        available = min(self.n_received, size)
        n = available if seconds is None else min(available, int(round(seconds * self.sfreq)))
        idx = (self._write_pos - n + np.arange(n)) % size
        return self._buf[:, idx], self._ts[idx]

    def close(self) -> None:
        self.inlet.close_stream()


# ---------------------------------------------------------------------- CLI
def _print_streams() -> None:
    streams = list_streams()
    if not streams:
        print("No LSL streams found on the network.")
        return
    for s in streams:
        print(
            f"name={s.name()!r} type={s.type()!r} channels={s.channel_count()} "
            f"srate={s.nominal_srate()} format={s.channel_format()} "
            f"host={s.hostname()!r} source_id={s.source_id()!r}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list LSL streams on the network and exit")
    p.add_argument("--name", help="exact LSL stream name to connect to (default: first EEG stream)")
    p.add_argument("--type", default="EEG", help="LSL stream type to look for (default: EEG)")
    p.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the stream")
    p.add_argument("--duration", type=float, help="stop after this many seconds (default: until Ctrl+C)")
    p.add_argument("--save", metavar="CSV", help="write every sample to this CSV file")
    args = p.parse_args()

    if args.list:
        _print_streams()
        return 0

    try:
        rx = EegoReceiver.connect(name=args.name, stream_type=args.type, timeout=args.timeout)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    units = sorted({c.unit for c in rx.channels if c.unit}) or ["unknown"]
    print(
        f"Connected to {rx.info.name()!r} on {rx.info.hostname()!r}: "
        f"{rx.n_channels} channels @ {rx.sfreq:g} Hz, units: {', '.join(units)}"
    )
    print("Channels:", ", ".join(rx.ch_names))
    non_eeg = sorted(set(range(rx.n_channels)) - set(rx.eeg_picks))
    if non_eeg:
        print("Non-EEG channels:", ", ".join(rx.ch_names[i] for i in non_eeg))

    writer = None
    f = None
    if args.save:
        f = open(args.save, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["lsl_timestamp", *rx.ch_names])

    picks = rx.eeg_picks
    start = last_report = time.monotonic()
    count_at_report = 0
    try:
        while args.duration is None or time.monotonic() - start < args.duration:
            chunk, ts = rx.update(timeout=0.2)
            if writer is not None and ts.size:
                writer.writerows(np.column_stack([ts, chunk.T]).tolist())

            now = time.monotonic()
            if now - last_report >= 1.0:
                rate = (rx.n_received - count_at_report) / (now - last_report)
                data, _ = rx.latest(seconds=1.0)
                rms = np.sqrt(np.mean(data[picks] ** 2)) if data.size and picks else float("nan")
                print(
                    f"t={now - start:6.1f}s  samples={rx.n_received:8d}  "
                    f"rate={rate:7.1f} Hz  mean RMS (EEG, last 1s)={rms:.4g}"
                )
                if rate == 0:
                    print("  [warn] no data in the last second - is eego still streaming?")
                last_report, count_at_report = now, rx.n_received
    except KeyboardInterrupt:
        pass
    finally:
        if f is not None:
            f.close()
            print(f"Saved {rx.n_received} samples to {args.save}")
        rx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
