# Chords to MIDI (CLI)

A single-file Python CLI that reads a plain-text chord sequence (one chord event per line) and renders it as a **standard MIDI** file using **mido**.

- ✅ Supports common chord qualities (maj/min/7/maj7/min7/dim/aug/sus2/sus4)
- ✅ Supports **slash chords** (explicit bass)
- ✅ Supports **rests** (`NC` / `N.C.`)
- ✅ Configurable tempo, time signature, ticks, instrument, channel, velocity
- ✅ Voicing modes: `close`, `open`, `spread` + optional `--voice-leading`

---

## Install

Requires Python 3.9+ recommended.

```bash
pip install mido
````

> Note: `mido` writes MIDI files directly (no external services). Playback depends on your DAW/player.

---

## Quick start

Create an input file `chords.txt`:

```text
# 4 bars of a simple progression
C   | 1
Am7 | 1
F   | 1
G7  | 1
```

Generate MIDI:

```bash
python chords_to_midi.py --input chords.txt --output chords.mid
```

---

## Input file format (Chord Sequence File)

### General rules

* **Plain text**, UTF-8.
* **One chord event per line**
* The script **ignores**:

  * Blank lines
  * Lines starting with `#` (comments)

### Grammar

Each non-comment line:

```text
<chord_symbol> [| <duration_measures>]
```

* `duration_measures` is optional
* If omitted, the script uses the global default from `--chord-length`
* Duration is in **measures (bars)** and can be a float (e.g., `0.5`)

### Examples

```text
C
Am
Fmaj7
D7 | 2
Gm7 | 0.5
```

### Duration semantics

The script converts measures → beats → ticks using:

* `beats_per_measure = numerator * (4 / denominator)`
* `ticks = round(measures * beats_per_measure * ticks_per_beat)`

So in:

* `4/4`: `beats_per_measure = 4`
* `6/8`: `beats_per_measure = 6 * (4/8) = 3` (i.e., 3 quarter-note beats per bar)

---

## Supported chord symbols

### Root notes

* Root is one of: `A B C D E F G`
* Optional accidental:

  * `#` (sharp)
  * `b` (flat)

Examples:

* `F#`, `Bb`, `C`, `Eb`

### Chord qualities (minimum set)

|    Quality | Examples       | Pitch-class intervals (from root) |
| ---------: | -------------- | --------------------------------- |
|      Major | `C`, `Cmaj`    | 0, 4, 7                           |
|      Minor | `Am`, `Amin`   | 0, 3, 7                           |
| Dominant 7 | `G7`           | 0, 4, 7, 10                       |
|    Major 7 | `Cmaj7`        | 0, 4, 7, 11                       |
|    Minor 7 | `Am7`, `Amin7` | 0, 3, 7, 10                       |
| Diminished | `Bdim`         | 0, 3, 6                           |
|  Augmented | `Caug`         | 0, 4, 8                           |
|       Sus2 | `Dsus2`        | 0, 2, 7                           |
|       Sus4 | `Dsus4`        | 0, 5, 7                           |

### Slash chords (optional bass)

Use `/` to provide an explicit bass note:

```text
C/E
Dm/F
G7/B
```

* The bass note must be a valid root note name with optional accidental (`#`/`b`).
* The chord tones remain the same; the bass note is placed in the bass register (see `--bass-octave`).

### Rests (no chord)

Use either:

* `NC`
* `N.C.`

Examples:

```text
NC | 1
N.C. | 2
```

Rests advance the timeline without sounding notes.

---

## CLI usage

```bash
python chords_to_midi.py --input INPUT.txt --output OUTPUT.mid [options]
```

### Required arguments

* `--input`
  Path to chord sequence file

* `--output`
  Path to output `.mid` file (must end in `.mid`)

### Timing options

* `--time-signature 4/4`
  Time signature (default: `4/4`)
  Denominator must be a power of 2 (e.g., 4, 8, 16).

* `--bpm 120`
  Tempo in BPM (default: `120`)

* `--chord-length 1`
  Default chord duration **in measures** for lines without `| duration` (default: `1`)

* `--ticks-per-beat 480`
  MIDI PPQ / ticks per quarter note (default: `480`)

### MIDI rendering options

* `--velocity 80`
  Note velocity `1..127` (default: `80`)

* `--program 0`
  MIDI program/instrument `0..127` (default: `0`)
  Examples:

  * `0` = Acoustic Grand Piano (General MIDI)
  * `24` = Nylon Guitar
  * `32` = Acoustic Bass

* `--channel 0`
  MIDI channel `0..15` (default: `0`)

### Register / voicing options

* `--base-octave 4`
  Base octave used to place chords near a comfortable register (default: `4`)
  Reference: `C4 = MIDI 60`

* `--bass-octave N`
  Octave for the bass note (slash bass or spread bass).
  Default is `base_octave - 1`.

#### `--voicing {close,open,spread}`

Controls how chord tones are distributed across octaves.

* `close` (default)
  Root-position, stacked upward as tightly as possible (within one octave region).

  * If slash chord: bass placed at `--bass-octave`, upper tones kept above it.

* `open`
  Starts from close voicing and applies:

  * Triad: drop the middle note by an octave (if still above bass)
  * 7th chord: drop-2 (2nd highest note down one octave)
  * Notes are re-sorted and clamped into range.

* `spread`
  Wider voicing across 2–3 octaves.

  * Bass is root (or slash bass) at `--bass-octave`
  * Upper voices aim for larger gaps (prefers ≥ 5th between adjacent voices when possible)
  * Total spread is limited by `--max-spread`

#### `--max-spread 24`

Max semitone distance between lowest and highest note in `spread` voicing (default: `24`).

#### `--voice-leading`

If enabled, the script tries to minimize movement from the previous chord by selecting an octave shift that reduces total semitone change, while respecting:

* ascending order
* bass placement
* safe range clamping

### Self test

* `--self-test`
  Runs internal sanity checks (parsing + voicing basics) and exits.

Example:

```bash
python chords_to_midi.py --self-test
```

---

## Examples

### 1) Default settings

```bash
python chords_to_midi.py --input chords.txt --output chords.mid
```

### 2) 3/4 at 90 BPM, half-measure chords

```bash
python chords_to_midi.py \
  --input chords.txt --output waltz.mid \
  --time-signature 3/4 --bpm 90 --chord-length 0.5
```

### 3) Open voicing with voice leading and a guitar program

```bash
python chords_to_midi.py \
  --input chords.txt --output guitar.mid \
  --program 24 --voicing open --voice-leading
```

### 4) Spread voicing with a larger max spread and explicit bass octave

```bash
python chords_to_midi.py \
  --input chords.txt --output spread.mid \
  --voicing spread --max-spread 30 --bass-octave 2
```

---

## Project structure

This project is intentionally minimal:

```text
.
├── chords_to_midi.py     # Single-file CLI tool
├── README.md             # This file
└── chords.txt            # Example input (optional)
```

---

## Notes / constraints

* Unsupported chord qualities cause a **hard error** with line number and the offending content.
* This tool writes a single-track MIDI file with:

  * time signature meta
  * tempo meta
  * program change
  * chord note-on/off events
* Rests (`NC`/`N.C.`) advance time without notes.

---

## License

Add your preferred license (MIT/Apache-2.0/etc.) if you plan to publish.
