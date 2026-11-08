# Protocol

Two motor tasks, recorded with EEG (C3, C4 and Fz are the electrodes of interest). Start the EEG recording before each task, and run the EEG software on the same laptop as the tapping page if you can, so both use one clock.

## Task 1: Alternating-hand handwriting

The participant writes the same three lines 8 times, switching hands each time and starting with the left.

Each trial, write:

1. Full name
2. School name
3. Major

| Trial | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hand | Left | Right | Left | Right | Left | Right | Left | Right |

- Write down the clock time when each trial starts and ends.
- Keep the head and jaw still; only the hand moves.
- Still to decide: rest time between trials, and what (if anything) gets scored.

## Task 2: Finger tapping sequence learning

Run `task/finger_tapping.html`. The defaults are the protocol below. Total time 7 min 35 s.

Finger placement, left hand: pinky on A, ring on S, middle on D, index on F.

| Part | Rounds | Pattern | Timing |
| --- | --- | --- | --- |
| Still baseline | before round 1 | none, sit still | 30 s |
| Practice | 1 to 13 | F A D S F D S | 15 s tap, 10 s rest |
| New pattern (control) | 14 to 17 | S A S D F S A | 15 s tap, 10 s rest |

Steps:

1. Start the EEG recording. Clench the jaw once hard; it leaves a clear spike to check timing against.
2. Name the session with a participant ID (P01, P02), never a real name.
3. Click Start and don't touch the mouse or switch windows until it ends. If the page loses focus, keypresses stop being recorded.
4. Tap during TAP, keep still during REST. Esc twice stops early; finished rounds are kept.
5. Export from the page (Details and export): Blocks CSV for the analysis, Events and Keypresses CSV for anything finer.

What to look for: time per pattern drops over the first few rounds and levels off, then jumps at round 14. On the EEG, compare mu and beta at C3 and C4 across rounds 1 to 2 (early), 10 to 13 (learned) and 14 to 17 (new).
