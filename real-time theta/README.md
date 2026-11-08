# Real-time theta feedback

A live frontal-theta "mastery" score computed from an eego LSL stream. It is part of
[neurotech_hack2026](../README.md); see the main README for environment setup
(`neuro_env`). Unlike the rest of the repo, which works from exported `.cnt` files, this
program reads the amplifier live over Lab Streaming Layer (LSL).

| File | What it does |
|---|---|
| `theta_feedback.py` | Reads an LSL EEG stream, computes F3/F4/Fz theta (4–8 Hz) and shows a live 0–100 score |
| `lsl_replay_cnt.py` | Replays a recorded eego `.cnt` file over LSL in real time, for testing without the amplifier |

Run both from this folder (the folder name has a space, so quote it):

```bash
conda activate neuro_env
cd "real-time theta"
```

## Live session

1. In the eego software, enable LSL streaming and start the stream. The laptop running
   this program must be on the same local network as the eego laptop. Campus Wi-Fi often
   blocks LSL discovery, so a phone hotspot or wired connection is safest.
2. Run `python theta_feedback.py`. It uses the first stream of type `EEG`; pass
   `--name "<stream name>"` to pick a specific one (eego names it like
   `EE511-..._on_<computer>`).
3. Check the startup lines: the average reference should use all EEG channels (only
   `Trigger` excluded), and units should say "microvolts". eego labels its data "Volt"
   but sends microvolts, so the program judges units from the signal size.
4. **Baseline:** the subject sits still for the first 15 s.
5. **Task calibration:** press **SPACE** as the task starts. The first 15 s of the task
   set the "no mastery" level.
6. The score then updates every 0.25 s. 100 means theta at or below baseline, and
   0 means theta at or above the no-mastery level.

Keys (the plot window must have focus):

| Key | Action |
|---|---|
| SPACE | Task has started; begin the 15 s task calibration |
| R | Redo the task calibration, starting now |
| B | Redo the baseline, starting now (also clears the task calibration) |
| Q / Esc | Quit and save the logs |

## How the score is computed

The settings match the offline analysis in `analysis/`.

1. Causal 1–40 Hz band-pass plus a 60 Hz notch, applied chunk by chunk.
2. Every 0.25 s the most recent 2 s window is processed:
   - flat or dead channels are dropped,
   - the data are average-referenced over the working EEG channels,
   - theta power is computed for each target channel (Hann-tapered FFT) and averaged across them.
3. A window is rejected if any target channel exceeds 150 µV peak to peak. Theta spikes
   (for example from typing mistakes) above median + 2 robust SDs are also rejected,
   along with windows within 1 s of a spike.
4. The live value is the median of the clean windows from the last 15 s:
   `score = 100 × (no_mastery − value) / (no_mastery − baseline)`, clipped to 0–100.

Every threshold is a command-line option (for example `--ptp-uv`, `--channels`,
`--calib-s`). `python theta_feedback.py --help` lists them all with their defaults.

## Testing without the headset

Replay a recording in one terminal and run the feedback in another:

```bash
python lsl_replay_cnt.py path/to/recording.cnt --start 22.6   # terminal 1
python theta_feedback.py --name CNT-replay                    # terminal 2
```

`--start` skips ahead in the recording and `--speed` changes the playback rate.
Reading `.cnt` files needs `mne` and `antio`.

To run headless (no plot) with an automatic task start:

```bash
python theta_feedback.py --no-plot --task-start-after 30 --duration 300
```

## Output

Each session writes two files to `theta_logs/` (change this with `--log-dir`):

- `theta_<timestamp>.csv`: one row per 2 s window, with per-channel theta, the
  rejection flags, the 15 s median and the score.
- `theta_<timestamp>_session.json`: the settings, calibration values, events and the
  stream info.
