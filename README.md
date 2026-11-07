# neurotech_hack2026

Real-time EEG from an ANT Neuro eego amplifier (EE-2xx) over Lab Streaming Layer (LSL),
with data processing, ML, and a visualization dashboard.

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

## Receiving EEG from eego

1. Connect the EE-2xx amplifier and open the eego software.
2. Enable LSL streaming in eego and start the stream (impedance check / recording view).
3. On a machine on the **same local network** (or the same machine):

```bash
conda activate neuro_env
python acquisition/eego_lsl_receiver.py --list           # show available LSL streams
python acquisition/eego_lsl_receiver.py                  # connect + record to data/<date_time>.csv
python acquisition/eego_lsl_receiver.py --no-save        # just watch, do not record
```

Run `python acquisition/eego_lsl_receiver.py --help` for all options. LSL discovers
streams by multicast/broadcast on the local network; some networks (e.g. campus Wi-Fi)
block this, so a wired connection or a phone hotspot is the safest setup.

### No headset? Use the simulator

`acquisition/simulate_eego.py` publishes a fake eego-like stream (32 EEG channels in
volts with a 10 Hz alpha rhythm, plus a TRIGGER channel) so you can develop without the amp:

```bash
python acquisition/simulate_eego.py        # terminal 1
python acquisition/eego_lsl_receiver.py    # terminal 2
```

### Using the receiver in code

To use the receiver from your own code (processing, ML, dashboard):

```python
from acquisition.eego_lsl_receiver import EegoReceiver

rx = EegoReceiver.connect()          # resolves the eego EEG stream
rx.update()                          # pull everything that has arrived
data, times = rx.latest(seconds=2)   # (n_channels, n_samples), LSL timestamps
print(rx.ch_names, rx.sfreq)
```

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

Example analysis: `notebooks/session1_frontal_segments.ipynb` plots the frontal channels
(raw, alpha 8–13 Hz, gamma 30–100 Hz) for both segments side by side.
