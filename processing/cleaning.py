"""Artifact cleaning: ICA with ICLabel (eye/muscle components) and ASR (movement bursts).

Typical use for a pair of segments to compare:

    from processing.cleaning import prepare, ica_clean, asr_clean
    segs = [prepare(s, bad_channels) for s in (seg_a, seg_b)]
    segs = asr_clean(segs, calibration=prepare(clean_seg, bad_channels))   # only for movement tasks
    cleaned, report, ica = ica_clean(segs)

ICLabel was trained on 1–100 Hz band-passed, average-referenced data, so `prepare` puts
every segment in that form first. Cleaned segments stay average-referenced.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components

# ICLabel classes treated as artifacts when the classifier is confident enough.
ARTIFACT_LABELS = ("eye blink", "muscle artifact", "heart beat", "line noise", "channel noise")


def prepare(seg, bad_channels=(), line_freq=60.0):
    """Copy of `seg` ready for ICA/ICLabel: bad channels dropped, standard 10-05 positions,
    1–100 Hz band-pass, line-noise notch (and harmonics below 100 Hz), average reference."""
    out = seg.copy().drop_channels([c for c in bad_channels if c in seg.ch_names])
    out.set_montage("standard_1005", on_missing="raise", verbose="error")
    out.filter(l_freq=1.0, h_freq=100.0, verbose="error")
    out.notch_filter(np.arange(line_freq, 100.0, line_freq), verbose="error")
    out.set_eeg_reference("average", projection=False, verbose="error")
    return out


def asr_clean(segs, calibration, cutoff=20):
    """Artifact Subspace Reconstruction (meegkit) for short, large movement bursts.

    ASR is calibrated on `calibration` (ideally clean data from the same session and cap;
    meegkit picks its clean windows automatically) and then applied to each segment.
    cutoff: standard deviations above calibration before a burst is reconstructed
    (20 is a commonly used, fairly conservative value).
    """
    from meegkit.asr import ASR

    sfreq = calibration.info["sfreq"]
    asr = ASR(sfreq=sfreq, cutoff=cutoff)
    asr.fit(calibration.get_data())
    out = []
    for seg in segs:
        if seg.ch_names != calibration.ch_names:
            raise ValueError("segments and calibration data must have the same channels")
        cleaned = seg.copy()
        cleaned._data = asr.transform(seg.get_data())
        out.append(cleaned)
    return out


def ica_clean(segs, threshold=0.7, random_state=97):
    """Fit one extended-Infomax ICA on the segments together, label components with
    ICLabel, and remove those confidently (probability >= threshold) labelled as artifacts.

    Returns (cleaned segments, report DataFrame of all components, fitted ICA).
    """
    combined = mne.concatenate_raws([s.copy() for s in segs], verbose="error")
    rank = len(combined.ch_names) - 1  # average reference removes one dimension
    ica = ICA(n_components=rank, method="infomax", fit_params=dict(extended=True),
              random_state=random_state, max_iter="auto", verbose="error")
    ica.fit(combined, verbose="error")
    labels = label_components(combined, ica, method="iclabel")
    report = pd.DataFrame({
        "component": [f"IC{i:02d}" for i in range(ica.n_components_)],
        "label": labels["labels"],
        "probability": np.round(labels["y_pred_proba"], 2),
    })
    report["removed"] = report["label"].isin(ARTIFACT_LABELS) & (report["probability"] >= threshold)
    ica.exclude = [int(c[2:]) for c in report.loc[report["removed"], "component"]]
    cleaned = [ica.apply(s.copy(), verbose="error") for s in segs]
    return cleaned, report, ica
