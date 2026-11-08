# neurotech_hack2026

EEG from an ANT Neuro eego amplifier (EE-2xx), recorded in the eego software and exported
as .cnt files, with data processing, ML, and a visualization dashboard.

## Setup

Requires [Miniconda](https://docs.anaconda.com/miniconda/) (or Anaconda).

```bash
git clone git@github.com:alt-Pranav/neurotech_hack2026.git
cd neurotech_hack2026
conda env create -f environment.yml
conda activate neuro_env
```

Check that LSL works:

```bash
python -c "import pylsl; print('liblsl', pylsl.library_version())"
```

### Adding / updating packages

1. Add the package to `environment.yml` (conda-forge packages under `dependencies`,
   pip-only packages under the `pip:` list) and commit it.
2. Everyone else pulls and runs:

```bash
conda env update -f environment.yml --prune
```

Note: `pylsl` is not on conda-forge, so it is installed via pip. Its wheel bundles the
native `liblsl` library, so nothing else needs installing.

## Loading recorded sessions (eego export)

Sessions are exported from eego as `<name>.cnt` (EEG + events), `<name>.seg` (recording
segments) and `<name>.evt` (event log, redundant with the `.cnt`). Put each session in its
own folder under `data/` (git-ignored; share recordings separately).

```python
from processing import load_session

session = load_session("data/Session 1_ Handwriting")   # folder or .cnt path
seg1, seg2 = session.segments    # one mne Raw per segment, in volts, unfiltered
```

eego pauses the recording during an impedance check, so one `.cnt` can hold several
segments joined end to end. `load_session` splits them using the `.seg` file; filter each
segment separately so the jump between them doesn't distort the signal.

## Analysis code

- `processing/analysis.py`: band plots, Welch spectra with per-channel window rejection,
  relative power (share of 4–40 Hz), and FOOOF (theta/alpha measured above the 1/f background).
- `processing/cleaning.py`: `prepare` (1–100 Hz, notch, average reference), `ica_clean`
  (ICA + ICLabel removes eye/muscle components), `asr_clean` (Artifact Subspace Reconstruction
  for movement bursts).

Notebooks:
- `notebooks/session1_segments.ipynb`: writing with the dominant vs non-dominant hand.
- `notebooks/session2_segments.ipynb`: writing, juggling and throwing pairs.
- `notebooks/session3_pt1_segments.ipynb`: 8 handwriting trials alternating left/right hands.
- `notebooks/session3_pt2_segments.ipynb`: finger-tapping rounds (13 practice + 4 new-pattern rounds,
  with a still baseline); round schedule aligned to the recording from the data.

Session 3 used a different cap layout (adds Cz, M1/M2, Oz; no T9/T10/P9/P10) and its Fz
electrode was saturated throughout, so frontal theta there is measured at F3+F4.

Both compare segments for signs of mental workload (frontal theta up, parietal alpha down).
p-values in them are descriptive only: windows from one recording aren't independent samples.
