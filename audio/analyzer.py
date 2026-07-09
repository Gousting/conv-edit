"""
Audio analysis engine for conv-edit.
Input: audio file → Output: structured MusicTimeline JSON.

Components:
  - BPM + beat / downbeat grid
  - Energy curve (RMS + spectral centroid)
  - Sub-bass onset detection
  - Music structure segmentation (intro/verse/chorus/drop)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
from scipy.signal import butter, sosfilt


# ─── Data model ────────────────────────────────────────


@dataclass
class MusicSegment:
    """A labelled section of the music piece."""
    label: str            # "intro" | "verse" | "chorus" | "bridge" | "drop" | "outro"
    start_sec: float
    end_sec: float
    energy: float         # 0.0–1.0  mean energy in this segment


@dataclass
class MusicTimeline:
    """Complete structured music analysis result."""
    bpm: float
    duration_sec: float
    time_signature: str = "4/4"    # detected or assumed
    sample_rate: int = 44100
    beats_sec: list[float] = field(default_factory=list)
    downbeats_sec: list[float] = field(default_factory=list)
    bass_onsets_sec: list[float] = field(default_factory=list)
    energy_curve: list[dict] = field(default_factory=list)   # [{time_sec, value}]
    segments: list[MusicSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm,
            "duration_sec": self.duration_sec,
            "time_signature": self.time_signature,
            "sample_rate": self.sample_rate,
            "beat_count": len(self.beats_sec),
            "downbeat_count": len(self.downbeats_sec),
            "bass_onset_count": len(self.bass_onsets_sec),
            "beats_sec": self.beats_sec,
            "downbeats_sec": self.downbeats_sec,
            "bass_onsets_sec": self.bass_onsets_sec,
            "energy_curve": self.energy_curve,
            "segments": [asdict(s) for s in self.segments],
            "suggested_intro_offset": self._suggest_intro_offset(),
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path:
            path.write_text(text)
        return text

    def _suggest_intro_offset(self) -> float:
        """If first segment is low-energy intro, suggest skipping it."""
        if not self.segments:
            return 0.0
        first = self.segments[0]
        if first.label in ("intro",) and first.energy < 0.35 and len(self.segments) >= 2:
            # Return the end time of the intro as a suggested offset
            return first.end_sec
        # Find first energy peak
        for s in self.segments:
            if s.energy > 0.5:
                return s.start_sec
        return 0.0


# ─── Core analyzer ──────────────────────────────────────


class AudioAnalyzer:
    """Analyze an audio file and produce a MusicTimeline."""

    def __init__(
        self,
        sr: int = 22050,
        bpm_min: float = 60,
        bpm_max: float = 200,
        segment_min_duration: float = 5.0,
    ):
        self.sr = sr
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.segment_min_duration = segment_min_duration

    def analyze(self, audio_path: str | Path) -> MusicTimeline:
        """Run full analysis pipeline on an audio file."""
        audio_path = Path(audio_path)
        y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        duration = len(y) / sr

        timeline = MusicTimeline(bpm=120, duration_sec=duration, sample_rate=sr)

        # 1. BPM + beat grid
        timeline.bpm, timeline.beats_sec = self._detect_beats(y, sr)

        # 2. Downbeats
        timeline.downbeats_sec = self._detect_downbeats(timeline.beats_sec, timeline.bpm)

        # 3. Energy curve
        timeline.energy_curve = self._compute_energy_curve(y, sr)

        # 4. Sub-bass onsets
        timeline.bass_onsets_sec = self._detect_bass_onsets(y, sr)

        # 5. Structure segmentation
        timeline.segments = self._segment_structure(y, sr, timeline)

        return timeline

    # ── BPM + beat grid ─────────────────────────────

    def _detect_beats(self, y: np.ndarray, sr: int) -> tuple[float, list[float]]:
        """Detect BPM and beat timestamps."""
        if y.size == 0 or float(np.max(np.abs(y))) < 1e-6:
            return 120.0, self._metronome(120.0, len(y) / sr)

        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, start_bpm=120.0, tightness=80
        )
        bpm = float(np.atleast_1d(tempo)[0])
        bpm = self._clamp_bpm(bpm)
        beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        return bpm, [round(b, 3) for b in beats]

    def _detect_downbeats(self, beats: list[float], bpm: float, y: np.ndarray = None) -> list[float]:
        """Estimate downbeats. Only use 4/4 if beat count supports it."""
        if len(beats) < 8:
            return beats[:1] if beats else []

        # Try librosa's time signature detection
        ts = 4  # default
        try:
            # beat_track already ran; check if the beat spacing is consistent
            intervals = [beats[i+1] - beats[i] for i in range(len(beats)-1)]
            median_interval = float(np.median(intervals))
            # If beat intervals vary by >30%, treat as irregular time signature
            mad = float(np.median([abs(i - median_interval) for i in intervals]))
            if mad / (median_interval + 1e-6) > 0.30:
                ts = None  # irregular — don't force downbeats
        except Exception:
            pass

        if ts is None:
            return []   # no reliable downbeat detection

        return beats[::ts]

    def _clamp_bpm(self, bpm: float) -> float:
        while bpm > self.bpm_max:
            bpm /= 2.0
        while bpm < self.bpm_min:
            bpm *= 2.0
        return bpm

    def _metronome(self, bpm: float, duration: float) -> list[float]:
        interval = 60.0 / bpm
        n = int(duration / interval) + 1
        return [round(i * interval, 3) for i in range(n)]

    # ── Energy curve ────────────────────────────────

    def _compute_energy_curve(self, y: np.ndarray, sr: int) -> list[dict]:
        """Compute RMS energy at regular intervals (10 Hz)."""
        hop = sr // 10  # 10 samples per second
        if hop < 1:
            hop = sr
        frames = range(0, len(y), hop)
        rms = []
        for i in frames:
            chunk = y[i : i + hop]
            val = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size > 0 else 0.0
            rms.append(val)
        if not rms:
            return []
        max_rms = max(rms) or 1.0
        return [
            {"time_sec": round(i * hop / sr, 2), "value": round(rms[i] / max_rms, 3)}
            for i in range(len(rms))
        ]

    # ── Sub-bass onsets ─────────────────────────────

    def _detect_bass_onsets(self, y: np.ndarray, sr: int) -> list[float]:
        """Bandpass 20–150 Hz and detect bass attack points."""
        if y.size == 0:
            return []
        sos = butter(4, [20.0, 150.0], btype="band", fs=sr, output="sos")
        y_bass = sosfilt(sos, y).astype(np.float32)
        onset_env = librosa.onset.onset_strength(y=y_bass, sr=sr)
        if onset_env.size == 0 or onset_env.max() == 0:
            return []
        frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, backtrack=False,
            wait=int(sr * 0.03 / 512), delta=0.07,
        )
        if frames.size == 0:
            return []
        # Gate: ignore very quiet signals
        rms = float(np.sqrt(np.mean(y_bass ** 2)))
        if rms < 0.01:
            return []
        normalized = onset_env / onset_env.max()
        kept = [f for f in frames if normalized[f] >= 0.7]
        return [round(float(t), 3) for t in librosa.frames_to_time(np.array(kept), sr=sr)]

    # ── Structure segmentation ──────────────────────

    def _segment_structure(
        self, y: np.ndarray, sr: int, timeline: MusicTimeline
    ) -> list[MusicSegment]:
        """Detect music structure using ruptures change-point detection."""
        try:
            import ruptures as rpt
        except ImportError:
            return self._fallback_segmentation(timeline)

        # Build feature: RMS energy per beat
        if len(timeline.beats_sec) < 4:
            return self._fallback_segmentation(timeline)

        beat_energy = self._energy_per_beat(y, sr, timeline.beats_sec)
        if len(beat_energy) < 4:
            return self._fallback_segmentation(timeline)

        signal = np.array(beat_energy).reshape(-1, 1)

        # Detect change points (max 6 segments)
        algo = rpt.Binseg(model="l2", min_size=3).fit(signal)
        n_segs = min(6, max(2, len(beat_energy) // 8))
        try:
            change_points = algo.predict(n_bkps=n_segs - 1)
        except Exception:
            return self._fallback_segmentation(timeline)

        # Build segments from change points
        cp = [0] + change_points
        segments = []
        for i in range(len(cp) - 1):
            start_idx = cp[i]
            end_idx = min(cp[i + 1], len(timeline.beats_sec) - 1)
            if end_idx <= start_idx:
                continue
            start = timeline.beats_sec[start_idx]
            end = min(timeline.beats_sec[end_idx], timeline.duration_sec)
            seg_energy = float(np.mean(beat_energy[start_idx:end_idx]))
            label = self._label_segment(i, len(cp) - 1, seg_energy)
            segments.append(MusicSegment(
                label=label, start_sec=round(start, 2),
                end_sec=round(end, 2), energy=round(seg_energy, 3),
            ))

        # Merge short segments
        return self._merge_short(segments, timeline)

    def _energy_per_beat(
        self, y: np.ndarray, sr: int, beats: list[float]
    ) -> list[float]:
        """Compute mean RMS energy within each beat interval."""
        energies = []
        for i in range(len(beats) - 1):
            s = int(beats[i] * sr)
            e = int(beats[i + 1] * sr)
            chunk = y[s:e]
            rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size > 0 else 0.0
            energies.append(rms)
        if not energies:
            return [0.0]
        mx = max(energies) or 1.0
        return [e / mx for e in energies]

    def _label_segment(self, idx: int, total: int, energy: float) -> str:
        """Heuristic label assignment based on position and energy."""
        if total == 1:
            return "body"
        if idx == 0:
            return "intro" if energy < 0.5 else "verse"
        if idx == total - 1:
            return "outro" if energy < 0.4 else "chorus"
        # Middle segments
        if energy > 0.75:
            return "drop" if total >= 4 else "chorus"
        if energy > 0.5:
            return "chorus"
        if energy > 0.3:
            return "verse"
        return "bridge"

    def _merge_short(
        self, segments: list[MusicSegment], timeline: MusicTimeline
    ) -> list[MusicSegment]:
        """Merge segments shorter than minimum duration."""
        if len(segments) <= 1:
            return segments
        merged = [segments[0]]
        for seg in segments[1:]:
            dur = seg.end_sec - seg.start_sec
            if dur < self.segment_min_duration and merged:
                # Merge into previous
                merged[-1].end_sec = seg.end_sec
                merged[-1].energy = max(merged[-1].energy, seg.energy)
            else:
                merged.append(seg)
        return merged

    def _fallback_segmentation(self, timeline: MusicTimeline) -> list[MusicSegment]:
        """Fallback: split into intro / body / outro by time."""
        d = timeline.duration_sec
        if d < 15:
            return [MusicSegment(label="body", start_sec=0, end_sec=d, energy=0.5)]
        intro_end = d * 0.15
        outro_start = d * 0.80
        segs = [
            MusicSegment(label="intro", start_sec=0, end_sec=intro_end, energy=0.3),
            MusicSegment(label="body", start_sec=intro_end, end_sec=outro_start, energy=0.6),
            MusicSegment(label="outro", start_sec=outro_start, end_sec=d, energy=0.3),
        ]
        # Fill in energy from curve
        for seg in segs:
            vals = [p["value"] for p in timeline.energy_curve
                    if seg.start_sec <= p["time_sec"] <= seg.end_sec]
            if vals:
                seg.energy = round(float(np.mean(vals)), 3)
        return segs


# ─── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <audio_file> [output.json]")
        sys.exit(1)

    analyzer = AudioAnalyzer()
    timeline = analyzer.analyze(sys.argv[1])

    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    json_str = timeline.to_json(Path(out_path) if out_path else None)

    if not out_path:
        print(json_str)
    else:
        print(f"BPM: {timeline.bpm:.1f}  |  {len(timeline.beats_sec)} beats  "
              f"|  {len(timeline.segments)} segments")
        for seg in timeline.segments:
            print(f"  {seg.label:8s} {seg.start_sec:6.1f}s – {seg.end_sec:6.1f}s  "
                  f"energy={seg.energy:.2f}")
        print(f"Saved → {out_path}")
