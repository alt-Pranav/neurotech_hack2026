"""Load an eego recording session (.cnt + .seg) and split it into its segments.

eego writes a session as three files sharing one base name:
    <name>.cnt   EEG data + event annotations (read with MNE / antio)
    <name>.seg   recording segments: the recording was paused (e.g. for an impedance
                 check) and resumed, so the samples in the .cnt are discontinuous
    <name>.evt   binary event log; its events are already in the .cnt, so it is ignored

Each segment is returned as its own mne Raw object, because filtering across a
segment boundary would smear the discontinuity into the signal. Filter segments
separately.

Usage:
    from processing import load_session
    session = load_session("data/Session 1_ Handwriting")
    seg1, seg2 = session.segments
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import mne

# .seg start times are OLE automation dates: days since 1899-12-30 (UTC).
_OLE_EPOCH = datetime(1899, 12, 30)


@dataclass
class SegmentInfo:
    start_sample: int
    n_samples: int
    start_time: datetime | None = None  # wall-clock UTC start, if known

    @property
    def stop_sample(self) -> int:
        return self.start_sample + self.n_samples


@dataclass
class Session:
    name: str
    raw: mne.io.BaseRaw  # the whole recording, segments concatenated as stored
    segment_info: list[SegmentInfo]
    segments: list[mne.io.BaseRaw]  # one Raw per segment (copies, unfiltered)

    @property
    def sfreq(self) -> float:
        return self.raw.info["sfreq"]

    @property
    def ch_names(self) -> list[str]:
        return self.raw.ch_names


def _find_cnt(path: Path) -> Path:
    if path.is_file():
        return path
    cnts = sorted(path.glob("*.cnt"))
    if len(cnts) != 1:
        raise FileNotFoundError(f"Expected exactly one .cnt in {path}, found {len(cnts)}: {cnts}")
    return cnts[0]


def read_seg_file(seg_path: Path, n_total: int) -> list[SegmentInfo]:
    """Parse an eego .seg file into segment sample ranges.

    Format (text, CRLF):
        NumberSegments=      N
        <start, OLE days>  <unknown>  <n_samples>    # one line per segment after a break
    The first segment is implicit: it holds the samples before the listed ones.
    """
    lines = [ln.strip() for ln in seg_path.read_text().splitlines() if ln.strip()]
    n_segments = int(lines[0].split("=")[1])
    listed = []
    for ln in lines[1:]:
        start_days, _unknown, n_samples = ln.split()
        start = _OLE_EPOCH + timedelta(days=float(start_days))
        listed.append((start, int(n_samples)))
    if len(listed) != n_segments - 1:
        raise ValueError(
            f"{seg_path.name}: NumberSegments={n_segments} but {len(listed)} segment lines; "
            "format not understood"
        )

    first_len = int(n_total) - sum(n for _, n in listed)
    if first_len <= 0:
        raise ValueError(f"{seg_path.name}: segment lengths exceed the {n_total} samples in the .cnt")
    segments = [SegmentInfo(0, first_len)]
    for start, n in listed:
        segments.append(SegmentInfo(segments[-1].stop_sample, n, start))
    return segments


def load_session(path: str | Path, verbose: str = "error") -> Session:
    """Load an eego session directory (or its .cnt file) and split it into segments.

    Data are in volts (MNE convention). Nothing is filtered.
    """
    cnt = _find_cnt(Path(path))
    raw = mne.io.read_raw_ant(cnt, preload=True, verbose=verbose)
    _normalize_channel_names(raw)
    n_total = raw.n_times

    seg_path = cnt.with_suffix(".seg")
    if seg_path.exists():
        info = read_seg_file(seg_path, n_total)
    else:
        info = [SegmentInfo(0, n_total)]
    info[0].start_time = raw.info["meas_date"].replace(tzinfo=None)

    sfreq = raw.info["sfreq"]
    segments = [
        raw.copy().crop(
            tmin=s.start_sample / sfreq,
            # the last segment runs to the end of the recording (tmax=None avoids
            # asking for one sample past the end)
            tmax=None if s.stop_sample >= n_total else s.stop_sample / sfreq,
            include_tmax=False,
            verbose=verbose,
        )
        for s in info
    ]
    return Session(cnt.stem, raw, info, segments)


def _normalize_channel_names(raw: mne.io.BaseRaw) -> None:
    """Fix the capitalization of standard 10-05 labels in place (e.g. 'pz' -> 'Pz', 'o1' -> 'O1').

    Some eego cap configurations export lowercase labels; analysis code looks channels up
    by their standard names.
    """
    standard = {name.lower(): name for name in mne.channels.make_standard_montage("colin27_1005").ch_names}
    mapping = {ch: standard[ch.lower()] for ch in raw.ch_names
               if ch not in standard.values() and ch.lower() in standard}
    if mapping:
        raw.rename_channels(mapping)
