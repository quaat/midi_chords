#!/usr/bin/env python3
"""
chords_to_midi.py

Convert a plain-text chord sequence into a standard MIDI file.

Install:
    pip install mido

Input format (one chord event per line):
    - Blank lines are ignored
    - Lines starting with '#' are ignored
    - Each event:
        <chord_symbol> [| <duration_measures>]

Examples:
    C
    Am
    Fmaj7
    D7 | 2
    Gm7 | 0.5
    C/E | 1
    NC | 1      (rest)
    N.C. | 2    (rest)

Supported chord symbols:
    Root: A B C D E F G with optional '#' or 'b' (e.g., F#, Bb)
    Qualities:
        - Major:      C, Cmaj
        - Minor:      Am, Amin
        - Dominant 7: G7
        - Major 7:    Cmaj7
        - Minor 7:    Am7, Amin7
        - Diminished: Bdim
        - Augmented:  Caug
        - Sus:        Dsus2, Dsus4
    Optional slash bass:
        C/E, Dm/F, G7/B, etc.

Usage:
    python chords_to_midi.py --input chords.txt --output chords.mid

    python chords_to_midi.py --input chords.txt --output chords.mid \
        --time-signature 4/4 --bpm 120 --chord-length 1 \
        --voicing open --voice-leading --program 0 --channel 0

    python chords_to_midi.py --self-test
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mido


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class TimeSignature:
    numerator: int
    denominator: int

    @property
    def beats_per_measure(self) -> float:
        # Quarter-note beats per measure (as requested):
        # beats_per_measure = numerator * (4 / denominator)
        return self.numerator * (4.0 / self.denominator)


@dataclass(frozen=True)
class ParsedChord:
    """A chord as parsed from a symbol."""
    symbol: str
    is_rest: bool
    root_pc: Optional[int]                 # 0..11, None for rest
    quality: Optional[str]                 # e.g. "maj", "min7", ...
    pitch_classes: List[int]               # absolute PCs (0..11) in degree-order (root, 3rd, 5th, 7th)
    slash_bass_pc: Optional[int]           # explicit bass pitch class if provided (e.g. C/E -> E)


@dataclass(frozen=True)
class ChordEvent:
    chord: ParsedChord
    duration_measures: float
    line_no: int
    raw_line: str


class ParseError(ValueError):
    def __init__(self, message: str, *, line_no: Optional[int] = None, raw_line: Optional[str] = None) -> None:
        super().__init__(message)
        self.line_no = line_no
        self.raw_line = raw_line


# ----------------------------
# Theory / parsing helpers
# ----------------------------

NOTE_BASE_PC: Dict[str, int] = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

QUALITY_INTERVALS: Dict[str, List[int]] = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "7":    [0, 4, 7, 10],   # dominant 7
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}


def parse_time_signature(value: str) -> TimeSignature:
    """Parse '4/4', '3/4', '6/8' into a TimeSignature."""
    s = value.strip()
    if "/" not in s:
        raise ParseError(f"Invalid time signature '{value}'. Expected format like '4/4'.")
    parts = s.split("/")
    if len(parts) != 2:
        raise ParseError(f"Invalid time signature '{value}'. Expected exactly one '/'.")
    try:
        num = int(parts[0].strip())
        den = int(parts[1].strip())
    except ValueError as e:
        raise ParseError(f"Invalid time signature '{value}'. Numerator/denominator must be integers.") from e
    if num <= 0 or den <= 0:
        raise ParseError(f"Invalid time signature '{value}'. Numerator/denominator must be positive.")
    # MIDI time signature meta expects denominator as power-of-two in most conventional uses.
    if den & (den - 1) != 0:
        raise ParseError(f"Invalid time signature '{value}'. Denominator should be a power of 2 (e.g. 4, 8, 16).")
    return TimeSignature(numerator=num, denominator=den)


def _parse_note_name_to_pc(note: str) -> int:
    """Parse a pitch-class note name like 'C', 'F#', 'Bb'."""
    n = note.strip()
    if not n:
        raise ParseError("Empty note name.")
    letter = n[0].upper()
    if letter not in NOTE_BASE_PC:
        raise ParseError(f"Invalid note letter '{n[0]}'. Expected A–G.")
    accidental = n[1:] if len(n) > 1 else ""
    if accidental not in ("", "#", "b"):
        raise ParseError(f"Invalid accidental in note '{note}'. Only '#' and 'b' are supported.")
    pc = NOTE_BASE_PC[letter]
    if accidental == "#":
        pc = (pc + 1) % 12
    elif accidental == "b":
        pc = (pc - 1) % 12
    return pc


def parse_chord_line(line: str, *, default_duration_measures: float, line_no: int) -> Optional[Tuple[str, float]]:
    """
    Parse one input line into (chord_symbol, duration_measures).
    Returns None for blank/comment lines.
    """
    raw = line.rstrip("\n")
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return None

    # Grammar: <chord_symbol> [| <duration>]
    parts = [p.strip() for p in stripped.split("|")]
    if len(parts) == 1:
        symbol = parts[0]
        duration = default_duration_measures
    elif len(parts) == 2:
        symbol = parts[0]
        dur_s = parts[1]
        if not dur_s:
            raise ParseError("Missing duration after '|'.", line_no=line_no, raw_line=raw)
        try:
            duration = float(dur_s)
        except ValueError as e:
            raise ParseError(f"Invalid duration '{dur_s}'. Expected a number (measures).", line_no=line_no, raw_line=raw) from e
    else:
        raise ParseError("Too many '|' separators. Expected at most one.", line_no=line_no, raw_line=raw)

    if not symbol:
        raise ParseError("Missing chord symbol.", line_no=line_no, raw_line=raw)

    if duration <= 0:
        raise ParseError(f"Duration must be > 0 measures (got {duration}).", line_no=line_no, raw_line=raw)

    return symbol, duration


def parse_chord_symbol(symbol: str) -> ParsedChord:
    """
    Parse a chord symbol into root, quality, optional slash bass, or rest (NC/N.C.).
    """
    s = symbol.strip()

    rest_norm = s.replace(".", "").replace(" ", "").upper()
    if rest_norm == "NC":
        return ParsedChord(
            symbol=s,
            is_rest=True,
            root_pc=None,
            quality=None,
            pitch_classes=[],
            slash_bass_pc=None,
        )

    # Optional slash bass
    if "/" in s:
        left, right = s.split("/", 1)
        chord_part = left.strip()
        bass_part = right.strip()
        if not bass_part:
            raise ParseError(f"Invalid slash chord '{symbol}': missing bass note after '/'.")
        slash_bass_pc = _parse_note_name_to_pc(bass_part)
    else:
        chord_part = s
        slash_bass_pc = None

    chord_part = chord_part.strip()
    if not chord_part:
        raise ParseError(f"Invalid chord symbol '{symbol}'. Missing chord before slash.")

    # Root: letter + optional accidental
    # Everything after that is quality.
    root_letter = chord_part[0].upper()
    if root_letter not in NOTE_BASE_PC:
        raise ParseError(f"Unsupported chord '{symbol}': invalid root '{chord_part[0]}'.")
    accidental = ""
    if len(chord_part) >= 2 and chord_part[1] in ("#", "b"):
        accidental = chord_part[1]
        quality_part = chord_part[2:]
    else:
        quality_part = chord_part[1:]

    root_name = root_letter + accidental
    root_pc = _parse_note_name_to_pc(root_name)
    q = quality_part.strip().lower()

    # Normalize aliases / supported set
    # Important: match longer strings first.
    if q in ("", "maj"):
        quality = "maj"
    elif q in ("m", "min"):
        quality = "min"
    elif q == "7":
        quality = "7"
    elif q == "maj7":
        quality = "maj7"
    elif q in ("m7", "min7"):
        quality = "min7"
    elif q == "dim":
        quality = "dim"
    elif q == "aug":
        quality = "aug"
    elif q == "sus2":
        quality = "sus2"
    elif q == "sus4":
        quality = "sus4"
    else:
        raise ParseError(f"Unsupported chord quality '{quality_part}' in '{symbol}'. Supported: "
                         "maj, min, 7, maj7, min7, dim, aug, sus2, sus4.")

    pcs = chord_symbol_to_pitch_classes(root_pc=root_pc, quality=quality)

    return ParsedChord(
        symbol=s,
        is_rest=False,
        root_pc=root_pc,
        quality=quality,
        pitch_classes=pcs,
        slash_bass_pc=slash_bass_pc,
    )


def chord_symbol_to_pitch_classes(*, root_pc: int, quality: str) -> List[int]:
    """
    Convert (root_pc, quality) into absolute pitch classes in chord-degree order:
    [root, third/sus, fifth, optional seventh]
    """
    if quality not in QUALITY_INTERVALS:
        raise ParseError(f"Internal error: unknown quality '{quality}'.")
    intervals = QUALITY_INTERVALS[quality]
    return [int((root_pc + iv) % 12) for iv in intervals]


# ----------------------------
# Timing helpers
# ----------------------------

def measures_to_ticks(
    measures: float,
    *,
    time_signature: TimeSignature,
    ticks_per_beat: int,
) -> int:
    """Convert measures (bars) into MIDI ticks using quarter-note beats-per-measure."""
    beats = measures * time_signature.beats_per_measure
    ticks = int(round(beats * ticks_per_beat))
    return max(1, ticks)


# ----------------------------
# Voicing / register helpers
# ----------------------------

def _pc_to_midi_in_octave(pc: int, octave: int) -> int:
    """MIDI note for pitch class in a specific octave (C4=60 where octave=4)."""
    return 12 * (octave + 1) + (pc % 12)


def _stack_close(pitch_classes: Sequence[int], start_midi: int) -> List[int]:
    """Stack chord tones in close position ascending, starting at start_midi for the first tone."""
    if not pitch_classes:
        return []
    notes = [start_midi]
    for pc in pitch_classes[1:]:
        prev = notes[-1]
        prev_pc = prev % 12
        delta = (pc - prev_pc) % 12
        if delta == 0:
            delta = 12
        notes.append(prev + delta)
    return notes


def _shift_list(notes: Sequence[int], semitones: int) -> List[int]:
    return [n + semitones for n in notes]


def _within_range(notes: Sequence[int], *, min_note: int, max_note: int) -> bool:
    return all(min_note <= n <= max_note for n in notes)


def _ensure_strictly_ascending_unique(notes: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for n in sorted(notes):
        nn = n
        # Avoid duplicates by lifting by octaves if needed.
        while nn in seen:
            nn += 12
        out.append(nn)
        seen.add(nn)
    # Enforce strict ascending if duplicates forced octaves might have broken order.
    out_sorted = sorted(out)
    for i in range(1, len(out_sorted)):
        if out_sorted[i] <= out_sorted[i - 1]:
            out_sorted[i] = out_sorted[i - 1] + 1
    return out_sorted


def _voice_leading_cost(curr: Sequence[int], prev: Sequence[int]) -> float:
    """
    Small-n brute assignment for minimal movement cost.
    Chords are at most 4 notes here, so brute force is fine.
    """
    curr_s = list(sorted(curr))
    prev_s = list(sorted(prev))
    if not curr_s or not prev_s:
        return 0.0

    n, m = len(curr_s), len(prev_s)

    # We match min(n,m) notes one-to-one (subset/permutation on larger side),
    # and penalize any unmatched notes by distance to nearest note in the other chord.
    k = min(n, m)
    if n <= m:
        smaller = curr_s
        larger = prev_s
        smaller_is_curr = True
    else:
        smaller = prev_s
        larger = curr_s
        smaller_is_curr = False

    best = float("inf")
    for idxs in itertools.combinations(range(len(larger)), k):
        for perm in itertools.permutations(idxs):
            cost = 0.0
            for i in range(k):
                a = smaller[i]
                b = larger[perm[i]]
                cost += abs(a - b)
            if cost < best:
                best = cost

    # Unmatched notes penalty
    unmatched_cost = 0.0
    if n != m:
        larger_notes = curr_s if n > m else prev_s
        smaller_notes = prev_s if n > m else curr_s
        matched_count = k
        # Penalize extra notes gently: distance to nearest in other chord.
        # (We don't know which were matched in 'best', but for small n this approximation is ok.)
        for extra in larger_notes[matched_count:]:
            nearest = min(abs(extra - x) for x in smaller_notes)
            unmatched_cost += nearest * 0.5

    return best + unmatched_cost


def _choose_best_global_shift(
    base_notes: Sequence[int],
    *,
    prev_notes: Optional[Sequence[int]],
    min_note: int,
    max_note: int,
    fixed_bass: Optional[int] = None,
    max_shift_octaves: int = 2,
) -> List[int]:
    """
    Choose a global octave shift (±N octaves) that minimizes movement from prev_notes.
    If fixed_bass is provided, bass stays fixed and only upper voices are shifted.
    """
    shifts = [12 * k for k in range(-max_shift_octaves, max_shift_octaves + 1)]
    best_notes: Optional[List[int]] = None
    best_cost = float("inf")

    for s in shifts:
        if fixed_bass is None:
            cand = _shift_list(base_notes, s)
        else:
            uppers = [n for n in base_notes if n != fixed_bass]
            cand = [fixed_bass] + _shift_list(uppers, s)
            cand = sorted(cand)
        cand = _ensure_strictly_ascending_unique(cand)
        if not _within_range(cand, min_note=min_note, max_note=max_note):
            continue
        if prev_notes:
            cost = _voice_leading_cost(cand, prev_notes)
        else:
            # Prefer staying near original if no previous chord
            cost = abs(s) * 0.1
        if cost < best_cost:
            best_cost = cost
            best_notes = cand

    return best_notes if best_notes is not None else list(sorted(base_notes))


def _enforce_range_with_bass_fixed(
    notes: List[int],
    *,
    bass: Optional[int],
    min_note: int,
    max_note: int,
) -> List[int]:
    """
    Try to keep bass fixed when possible; otherwise shift entire chord.
    """
    if not notes:
        return notes
    notes = sorted(notes)
    if bass is None or bass not in notes:
        # Shift whole chord to fit.
        while min(notes) < min_note:
            notes = [n + 12 for n in notes]
        while max(notes) > max_note:
            notes = [n - 12 for n in notes]
        return _ensure_strictly_ascending_unique(notes)

    # Bass fixed: first try shifting uppers only
    bass_i = notes.index(bass)
    uppers = notes[bass_i + 1 :]
    lowers = notes[: bass_i]
    if lowers:
        # Shouldn't happen (bass should be lowest). Fix by sorting and treating first as bass.
        bass = notes[0]
        bass_i = 0
        uppers = notes[1:]

    def can_shift_uppers(delta: int) -> bool:
        shifted = [u + delta for u in uppers]
        if shifted and shifted[0] <= bass:
            return False
        return _within_range([bass] + shifted, min_note=min_note, max_note=max_note)

    # If out of range, attempt to fix by shifting uppers
    cand = [bass] + uppers
    cand = _ensure_strictly_ascending_unique(cand)
    # Too low (unlikely with fixed bass), or too high:
    while max(cand) > max_note and uppers:
        if can_shift_uppers(-12):
            uppers = [u - 12 for u in uppers]
            cand = [bass] + uppers
            cand = _ensure_strictly_ascending_unique(cand)
        else:
            # Shift entire chord down
            cand = [n - 12 for n in cand]
            bass -= 12
            uppers = [u - 12 for u in uppers]
            cand = _ensure_strictly_ascending_unique(cand)
            break

    while min(cand) < min_note:
        # Shift entire chord up (including bass) if needed
        cand = [n + 12 for n in cand]
        bass += 12
        uppers = [u + 12 for u in uppers]
        cand = _ensure_strictly_ascending_unique(cand)

    return cand


def apply_voicing(
    pitch_classes: Sequence[int],
    *,
    voicing: str,
    base_note: int,
    bass_note: Optional[int],
    bass_octave: int,
    prev_notes: Optional[Sequence[int]] = None,
    voice_leading: bool = False,
    max_spread: int = 24,
    min_note: int = 36,   # C2
    max_note: int = 84,   # C6
) -> List[int]:
    """
    Convert pitch classes into concrete MIDI notes according to voicing.

    Parameters:
        pitch_classes: chord tones as absolute PCs in degree order (root, 3rd, 5th, 7th).
        voicing: 'close', 'open', or 'spread'
        base_note: MIDI note for C in base octave (e.g. C4=60)
        bass_note: optional explicit bass pitch class (slash bass)
        bass_octave: octave number for bass register
        prev_notes: previously voiced chord notes (for voice leading)
    """
    if not pitch_classes:
        return []

    if voicing not in ("close", "open", "spread"):
        raise ValueError(f"Unknown voicing '{voicing}' (expected close/open/spread).")

    # Determine (optional) explicit bass MIDI note.
    bass_midi: Optional[int] = None
    if bass_note is not None:
        bass_midi = _pc_to_midi_in_octave(bass_note, bass_octave)

    # Build close-position upper chord (excluding explicit bass unless spread).
    root_pc = pitch_classes[0]
    root_start = base_note + root_pc  # root in the base octave region
    close_notes = _stack_close(pitch_classes, start_midi=root_start)

    # If explicit bass is used, ensure upper chord is above it.
    if bass_midi is not None:
        while min(close_notes) <= bass_midi:
            close_notes = _shift_list(close_notes, 12)

    if voicing == "close":
        notes = close_notes
        if bass_midi is not None:
            notes = [bass_midi] + notes
        notes = _ensure_strictly_ascending_unique(notes)

        if voice_leading and prev_notes:
            notes = _choose_best_global_shift(
                notes,
                prev_notes=prev_notes,
                min_note=min_note,
                max_note=max_note,
                fixed_bass=bass_midi,
            )

        notes = _enforce_range_with_bass_fixed(notes, bass=bass_midi, min_note=min_note, max_note=max_note)
        return notes

    if voicing == "open":
        # Start from close voicing excluding bass, then apply drop rule.
        upper = list(close_notes)
        if len(upper) == 3:
            # Move middle note down an octave if it stays above bass.
            mid = upper[1] - 12
            if bass_midi is None or mid > bass_midi:
                upper[1] = mid
        elif len(upper) == 4:
            # Drop-2: take 2nd highest note and drop by an octave.
            idx = -2
            dropped = upper[idx] - 12
            if bass_midi is None or dropped > bass_midi:
                upper[idx] = dropped
        upper = sorted(upper)
        upper = _ensure_strictly_ascending_unique(upper)

        notes = upper
        if bass_midi is not None:
            notes = [bass_midi] + upper
        notes = _ensure_strictly_ascending_unique(notes)

        if voice_leading and prev_notes:
            notes = _choose_best_global_shift(
                notes,
                prev_notes=prev_notes,
                min_note=min_note,
                max_note=max_note,
                fixed_bass=bass_midi,
            )

        notes = _enforce_range_with_bass_fixed(notes, bass=bass_midi, min_note=min_note, max_note=max_note)
        return notes

    # voicing == "spread"
    # Always put (slash bass if present else root) in bass register.
    spread_bass_pc = bass_note if bass_note is not None else root_pc
    spread_bass_midi = _pc_to_midi_in_octave(spread_bass_pc, bass_octave)

    # Upper tones: chord tones excluding one instance of the bass pitch class (avoid duplicates).
    upper_pcs: List[int] = list(pitch_classes)
    # Remove a single occurrence matching bass pc
    if spread_bass_pc in upper_pcs:
        upper_pcs.remove(spread_bass_pc)

    # Place uppers with intended gaps (prefer >= 7 semitones) when possible.
    notes = [spread_bass_midi]
    prev = spread_bass_midi

    desired_gap = 7
    for pc in upper_pcs:
        # Try to place at least desired_gap above prev.
        target = prev + desired_gap
        # Find smallest midi >= target with pitch class pc.
        cand = target
        delta = (pc - (cand % 12)) % 12
        cand += delta
        # Ensure strictly above prev.
        while cand <= prev:
            cand += 12
        notes.append(cand)
        prev = cand

    notes = _ensure_strictly_ascending_unique(notes)

    # If spread too wide, compress by octave-dropping top notes where safe.
    def spread_amount(ns: Sequence[int]) -> int:
        return int(max(ns) - min(ns)) if ns else 0

    while spread_amount(notes) > max_spread and len(notes) >= 2:
        notes = sorted(notes)
        # Try dropping the highest note by 12 if it remains above previous.
        hi = notes[-1]
        prev_hi = notes[-2]
        dropped = hi - 12
        if dropped > prev_hi and dropped > notes[0]:
            notes[-1] = dropped
            notes = _ensure_strictly_ascending_unique(notes)
        else:
            # If we can't, stop to avoid breaking order.
            break

    # Optional voice-leading: choose a global shift for uppers (keep bass fixed).
    if voice_leading and prev_notes:
        notes = _choose_best_global_shift(
            notes,
            prev_notes=prev_notes,
            min_note=min_note,
            max_note=max_note,
            fixed_bass=spread_bass_midi,
        )

    notes = _enforce_range_with_bass_fixed(notes, bass=spread_bass_midi, min_note=min_note, max_note=max_note)

    # Re-check max_spread after range shifts; compress again if needed.
    while spread_amount(notes) > max_spread and len(notes) >= 2:
        notes = sorted(notes)
        hi = notes[-1]
        prev_hi = notes[-2]
        dropped = hi - 12
        if dropped > prev_hi and dropped > notes[0]:
            notes[-1] = dropped
            notes = _ensure_strictly_ascending_unique(notes)
        else:
            break

    return notes


# ----------------------------
# MIDI writing
# ----------------------------

def write_midi(
    events: Sequence[ChordEvent],
    *,
    output_path: Path,
    time_signature: TimeSignature,
    bpm: float,
    ticks_per_beat: int,
    channel: int,
    program: int,
    velocity: int,
    base_octave: int,
    bass_octave: int,
    voicing: str,
    voice_leading: bool,
    max_spread: int,
) -> int:
    """
    Write events to a standard MIDI file. Returns number of chord events written (non-comment lines).
    """
    if ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be positive.")
    if not (0 <= channel <= 15):
        raise ValueError("channel must be 0..15.")
    if not (0 <= program <= 127):
        raise ValueError("program must be 0..127.")
    if not (1 <= velocity <= 127):
        raise ValueError("velocity must be 1..127.")
    if bpm <= 0:
        raise ValueError("bpm must be positive.")

    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # Meta at time=0
    track.append(mido.MetaMessage("time_signature", numerator=time_signature.numerator, denominator=time_signature.denominator, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.Message("program_change", program=program, channel=channel, time=0))

    base_note = _pc_to_midi_in_octave(0, base_octave)  # C in base octave (C4=60)
    prev_notes: Optional[List[int]] = None

    written = 0
    for ev in events:
        dur_ticks = measures_to_ticks(
            ev.duration_measures,
            time_signature=time_signature,
            ticks_per_beat=ticks_per_beat,
        )

        if ev.chord.is_rest:
            # Advance time with a harmless meta event.
            track.append(mido.MetaMessage("text", text="rest", time=dur_ticks))
            written += 1
            continue

        # Determine explicit bass pitch class (slash chord), if any.
        bass_pc = ev.chord.slash_bass_pc

        notes = apply_voicing(
            ev.chord.pitch_classes,
            voicing=voicing,
            base_note=base_note,
            bass_note=bass_pc if bass_pc is not None else None,
            bass_octave=bass_octave,
            prev_notes=prev_notes,
            voice_leading=voice_leading,
            max_spread=max_spread,
        )

        # Note-ons at the same time (no strum option)
        for i, n in enumerate(notes):
            track.append(mido.Message("note_on", note=n, velocity=velocity, channel=channel, time=0 if i > 0 else 0))

        # Note-offs after duration
        for i, n in enumerate(notes):
            track.append(mido.Message("note_off", note=n, velocity=0, channel=channel, time=dur_ticks if i == 0 else 0))

        prev_notes = list(notes)
        written += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(output_path))
    return written


# ----------------------------
# CLI / self-test
# ----------------------------

def _run_self_test() -> int:
    # Time signature
    ts = parse_time_signature("6/8")
    assert ts.numerator == 6 and ts.denominator == 8
    assert math.isclose(ts.beats_per_measure, 3.0)

    # Note parsing
    assert _parse_note_name_to_pc("C") == 0
    assert _parse_note_name_to_pc("B") == 11
    assert _parse_note_name_to_pc("F#") == 6
    assert _parse_note_name_to_pc("Bb") == 10

    # Chord parsing
    c = parse_chord_symbol("C")
    assert not c.is_rest
    assert c.pitch_classes == [0, 4, 7]

    am7 = parse_chord_symbol("Am7")
    assert am7.pitch_classes == [9, (9 + 3) % 12, (9 + 7) % 12, (9 + 10) % 12]

    sus2 = parse_chord_symbol("Dsus2")
    assert sus2.pitch_classes == [2, 4, 9]  # D(2), E(4), A(9)

    sl = parse_chord_symbol("C/E")
    assert sl.slash_bass_pc == 4

    rest = parse_chord_symbol("N.C.")
    assert rest.is_rest and rest.pitch_classes == []

    # Voicing sanity
    base_c4 = _pc_to_midi_in_octave(0, 4)
    notes_close = apply_voicing([0, 4, 7], voicing="close", base_note=base_c4, bass_note=None, bass_octave=3)
    assert notes_close == sorted(notes_close)
    assert len(notes_close) == 3

    notes_open = apply_voicing([0, 4, 7, 11], voicing="open", base_note=base_c4, bass_note=None, bass_octave=3)
    assert notes_open == sorted(notes_open)
    assert len(notes_open) == 4

    notes_spread = apply_voicing([0, 4, 7], voicing="spread", base_note=base_c4, bass_note=None, bass_octave=3, max_spread=24)
    assert notes_spread[0] < notes_spread[-1]
    assert (notes_spread[-1] - notes_spread[0]) <= 24 or len(notes_spread) == 0

    print("Self-test passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read a chord sequence text file and write a standard MIDI file (single track)."
    )
    parser.add_argument("--input", required=False, help="Path to input chord sequence text file.")
    parser.add_argument("--output", required=False, help="Path to output .mid file.")

    parser.add_argument("--time-signature", default="4/4", help="Time signature like 4/4, 3/4, 6/8 (default: 4/4).")
    parser.add_argument("--bpm", type=float, default=120.0, help="Tempo in BPM (quarter note) (default: 120).")
    parser.add_argument("--chord-length", type=float, default=1.0, help="Default chord length in measures (default: 1).")
    parser.add_argument("--ticks-per-beat", type=int, default=480, help="MIDI ticks per quarter note (default: 480).")

    parser.add_argument("--velocity", type=int, default=80, help="Note velocity 1..127 (default: 80).")
    parser.add_argument("--program", type=int, default=0, help="MIDI program (instrument) 0..127 (default: 0).")
    parser.add_argument("--channel", type=int, default=0, help="MIDI channel 0..15 (default: 0).")
    parser.add_argument("--base-octave", type=int, default=4, help="Base octave for chord register (C4=60) (default: 4).")

    parser.add_argument("--voicing", choices=["close", "open", "spread"], default="close",
                        help="Voicing style: close, open, spread (default: close).")
    parser.add_argument("--voice-leading", action="store_true",
                        help="Enable simple voice leading (minimize movement from previous chord).")
    parser.add_argument("--max-spread", type=int, default=24,
                        help="Max semitone spread (highest-lowest) for spread voicing (default: 24).")
    parser.add_argument("--bass-octave", type=int, default=None,
                        help="Bass octave for slash bass / spread bass (default: base_octave - 1).")

    parser.add_argument("--self-test", action="store_true", help="Run sanity checks and exit.")

    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --self-test is used.")

    try:
        ts = parse_time_signature(args.time_signature)
    except ParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.chord_length <= 0:
        print("Error: --chord-length must be > 0.", file=sys.stderr)
        return 2
    if args.ticks_per_beat <= 0:
        print("Error: --ticks-per-beat must be > 0.", file=sys.stderr)
        return 2
    if args.max_spread <= 0:
        print("Error: --max-spread must be > 0.", file=sys.stderr)
        return 2

    base_octave = args.base_octave
    bass_octave = args.bass_octave if args.bass_octave is not None else (base_octave - 1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 2
    if input_path.is_dir():
        print(f"Error: input path is a directory: {input_path}", file=sys.stderr)
        return 2
    if output_path.suffix.lower() != ".mid":
        print(f"Error: output file must have .mid extension: {output_path}", file=sys.stderr)
        return 2

    events: List[ChordEvent] = []
    try:
        text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback common encoding
        text = input_path.read_text(encoding="utf-8-sig")

    for i, line in enumerate(text.splitlines(), start=1):
        try:
            parsed = parse_chord_line(line, default_duration_measures=args.chord_length, line_no=i)
            if parsed is None:
                continue
            sym, dur = parsed
            chord = parse_chord_symbol(sym)
            events.append(ChordEvent(chord=chord, duration_measures=dur, line_no=i, raw_line=line.rstrip("\n")))
        except ParseError as e:
            ln = e.line_no if e.line_no is not None else i
            raw = e.raw_line if e.raw_line is not None else line.rstrip("\n")
            print(f"Parse error on line {ln}: {e}\n  >> {raw}", file=sys.stderr)
            return 2

    if not events:
        print("Error: no chord events found (file only contained blanks/comments).", file=sys.stderr)
        return 2

    try:
        written = write_midi(
            events,
            output_path=output_path,
            time_signature=ts,
            bpm=args.bpm,
            ticks_per_beat=args.ticks_per_beat,
            channel=args.channel,
            program=args.program,
            velocity=args.velocity,
            base_octave=base_octave,
            bass_octave=bass_octave,
            voicing=args.voicing,
            voice_leading=args.voice_leading,
            max_spread=args.max_spread,
        )
    except Exception as e:
        print(f"Error writing MIDI: {e}", file=sys.stderr)
        return 1

    # Summary
    print("Wrote MIDI successfully.")
    print(f"  input:          {input_path}")
    print(f"  output:         {output_path}")
    print(f"  time signature: {ts.numerator}/{ts.denominator}  (beats/measure={ts.beats_per_measure:g})")
    print(f"  bpm:            {args.bpm:g}")
    print(f"  chord length:   {args.chord_length:g} measures (default)")
    print(f"  ticks/beat:     {args.ticks_per_beat}")
    print(f"  channel:        {args.channel}")
    print(f"  program:        {args.program}")
    print(f"  velocity:       {args.velocity}")
    print(f"  base octave:    {base_octave}")
    print(f"  bass octave:    {bass_octave}")
    print(f"  voicing:        {args.voicing}")
    print(f"  voice leading:  {'on' if args.voice_leading else 'off'}")
    print(f"  max spread:     {args.max_spread} semitones")
    print(f"  events:         {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
