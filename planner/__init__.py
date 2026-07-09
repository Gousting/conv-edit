"""
Cut planning engine — v5.

Changes from v4:
  - _shift_timestamps filters negative beats/downbeats (offset edge case)
  - protect_end now checks energy tolerance — if clip tail lands in
    a segment with energy gap > 0.3, fall back to original length
  - _loop_clips generates unique IDs per loop iteration (originalId_L1, _L2...)
  - Default scale_mode → "letterbox" (safer for vertical clips)
  - audio_channels: 2 added to RenderTimeline
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════

@dataclass
class ClipInfo:
    clip_id: str
    source_path: str
    start_sec: float          # I-frame aligned — UI & render
    end_sec: float
    duration_sec: float
    tags: list[str] = field(default_factory=list)
    intensity: float = 0.5
    audio_mode: str = "融入BGM"
    description: str = ""
    protect_end: bool = False


@dataclass
class RenderSegment:
    clip: ClipInfo
    timeline_start_sec: float
    timeline_end_sec: float
    bgm_volume_gain_db: float = 0.0
    gain_fade_ms: int = 20
    source_audio_volume: float = 1.0
    source_audio_fade_in_ms: int = 0
    jcut_audio_lead_ms: int = 100
    playback_speed: float = 1.0  # 1.0=normal, >1=slower, <1=faster


@dataclass
class RenderTimeline:
    bgm_path: str
    bgm_start_offset_sec: float = 0.0
    total_duration_sec: float = 0.0
    output_width: int = 1920
    output_height: int = 1080
    output_fps: int = 30
    scale_mode: str = "letterbox"    # letterbox | crop | stretch
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    loop_capped: bool = False       # True when max_loops truncated the result
    segments: list[RenderSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bgm_path": self.bgm_path,
            "bgm_start_offset_sec": self.bgm_start_offset_sec,
            "total_duration_sec": self.total_duration_sec,
            "output_resolution": f"{self.output_width}x{self.output_height}",
            "output_fps": self.output_fps,
            "scale_mode": self.scale_mode,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "loop_capped": self.loop_capped,
            "segment_count": len(self.segments),
            "segments": [
                {
                    "clip_id": s.clip.clip_id,
                    "source_path": s.clip.source_path,
                    "source_start": s.clip.start_sec,
                    "source_end": s.clip.end_sec,
                    "source_duration": s.clip.duration_sec,
                    "timeline_start": s.timeline_start_sec,
                    "timeline_end": s.timeline_end_sec,
                    "playback_speed": s.playback_speed,
                    "bgm_volume_gain_db": s.bgm_volume_gain_db,
                    "gain_fade_ms": s.gain_fade_ms,
                    "source_audio_volume": s.source_audio_volume,
                    "source_audio_fade_in_ms": s.source_audio_fade_in_ms,
                    "audio_mode": s.clip.audio_mode,
                    "intensity": s.clip.intensity,
                    "tags": s.clip.tags,
                    "jcut_audio_lead_ms": s.jcut_audio_lead_ms,
                    "protect_end": s.clip.protect_end,
                }
                for s in self.segments
            ],
        }


# ═══════════════════════════════════════════════════════
#  Audio mode → rendering parameters
# ═══════════════════════════════════════════════════════

AUDIO_MODE_PARAMS = {
    "突出人声": {"bgm_gain_db": -8.0,  "source_vol": 1.0, "gain_fade_ms": 15},
    "融入BGM":  {"bgm_gain_db": -12.0, "source_vol": 0.3, "gain_fade_ms": 30},
    "纯BGM":    {"bgm_gain_db": 0.0,   "source_vol": 0.0, "gain_fade_ms": 50},
}


# ═══════════════════════════════════════════════════════
#  Planner
# ═══════════════════════════════════════════════════════

class CutPlanner:
    """Plan clip placement along a music timeline."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def plan(
        self,
        clips: list[ClipInfo],
        music_timeline: dict,
        bgm_path: str,
        target_duration: Optional[float] = None,
        bgm_start_offset: float = 0.0,
        auto_offset: bool = False,
        intensity_match_strength: float = 0.7,
        duration_strategy: str = "fit",
        output_width: int = 1920,
        output_height: int = 1080,
        output_fps: int = 30,
        scale_mode: str = "letterbox",
        max_loops: int = 3,
    ) -> RenderTimeline:
        if auto_offset and bgm_start_offset == 0.0:
            suggested = music_timeline.get("suggested_intro_offset", 0.0)
            if suggested > 0.5:
                bgm_start_offset = suggested

        effective_duration = music_timeline["duration_sec"] - bgm_start_offset

        beats, downbeats, segments_raw = self._shift_timestamps(
            music_timeline, bgm_start_offset, effective_duration
        )

        segments_raw = self._ensure_segments(segments_raw, effective_duration)

        total_clip_dur = sum(c.duration_sec for c in clips)
        total_duration = target_duration or effective_duration
        loop_capped = False

        if duration_strategy == "loop" and total_clip_dur < total_duration:
            clips = self._loop_clips(clips, total_duration, max_loops)
            # Detect truncation: if loops didn't fill, flag for validator
            if sum(c.duration_sec for c in clips) < total_duration:
                total_duration = sum(c.duration_sec for c in clips)
                loop_capped = True
        elif duration_strategy == "truncate" and total_clip_dur > total_duration:
            clips = self._truncate_clips(clips, total_duration)
        elif duration_strategy == "fit":
            effective = min(total_clip_dur, effective_duration)
            if total_clip_dur > effective:
                clips = self._truncate_clips(clips, effective)
            total_duration = effective

        segments_raw = [s for s in segments_raw if s["start_sec"] < total_duration]
        if segments_raw:
            segments_raw[-1]["end_sec"] = min(segments_raw[-1]["end_sec"], total_duration)

        assignments = self._assign_clips_to_segments(clips, segments_raw, intensity_match_strength)
        render_segments = self._build_gapless_timeline(
            assignments, segments_raw, beats, downbeats
        )

        return RenderTimeline(
            bgm_path=bgm_path,
            bgm_start_offset_sec=bgm_start_offset,
            total_duration_sec=total_duration,
            output_width=output_width,
            output_height=output_height,
            output_fps=output_fps,
            scale_mode=scale_mode,
            audio_sample_rate=48000,
            audio_channels=2,
            loop_capped=loop_capped,
            segments=render_segments,
        )

    # ── Timestamp shift ─────────────────────────────

    def _shift_timestamps(
        self,
        timeline: dict,
        offset: float,
        effective_duration: float,
    ) -> tuple[list[float], list[float], list[dict]]:
        """Shift beats/downbeats/segments left by `offset`.
        Filters negative values and ensures t=0 anchor."""
        beats = timeline.get("beats_sec", [])
        downbeats = timeline.get("downbeats_sec", [])
        segments = timeline.get("segments", [])

        if offset <= 0:
            return beats, downbeats, segments

        # Shift & filter negative beats
        shifted = [round(b - offset, 3) for b in beats if b >= offset - 0.01]
        # Ensure no negative values leak through
        shifted = [b for b in shifted if b >= 0.0]
        if not shifted or shifted[0] > 0.01:
            shifted.insert(0, 0.0)

        # Shift & filter negative downbeats
        shifted_d = [round(d - offset, 3) for d in downbeats if d >= offset - 0.01]
        shifted_d = [d for d in shifted_d if d >= 0.0]

        # Shift & filter segments (drop those entirely shifted out)
        shifted_segs: list[dict] = []
        for s in segments:
            new_start = max(0.0, s["start_sec"] - offset)
            new_end = max(0.5, s["end_sec"] - offset)
            # Drop if the segment fully shifted into negative
            if new_end <= 0:
                continue
            if new_end > effective_duration + 0.5:
                new_end = effective_duration
            shifted_segs.append({
                **s,
                "start_sec": round(new_start, 3),
                "end_sec": round(new_end, 3),
            })
        if shifted_segs:
            shifted_segs[0]["start_sec"] = 0.0

        return shifted, shifted_d, shifted_segs

    # ── Duration strategies ─────────────────────────

    def _loop_clips(
        self, clips: list[ClipInfo], target: float, max_loops: int = 3
    ) -> list[ClipInfo]:
        total = sum(c.duration_sec for c in clips)
        if total >= target:
            return clips

        original = list(clips)
        result = list(clips)
        current = total

        for loop_i in range(1, max_loops + 1):
            for c in original:
                if current >= target:
                    return result
                result.append(ClipInfo(
                    clip_id=f"{c.clip_id}_L{loop_i}",
                    source_path=c.source_path,
                    start_sec=c.start_sec,
                    end_sec=c.end_sec,
                    duration_sec=c.duration_sec,
                    tags=c.tags + [f"loop{loop_i}"],
                    intensity=c.intensity,
                    audio_mode=c.audio_mode,
                    description=c.description,
                    protect_end=c.protect_end,
                ))
                current += c.duration_sec
            if current >= target:
                break

        return result

    def _truncate_clips(self, clips: list[ClipInfo], target: float) -> list[ClipInfo]:
        sorted_clips = sorted(clips, key=lambda c: c.intensity, reverse=True)
        result = []
        total = 0.0
        for c in sorted_clips:
            if total + c.duration_sec <= target:
                result.append(c)
                total += c.duration_sec
            elif total < target:
                remaining = target - total
                if remaining > 0.3:
                    result.append(ClipInfo(
                        clip_id=c.clip_id,
                        source_path=c.source_path,
                        start_sec=c.start_sec,
                        end_sec=c.start_sec + remaining,
                        duration_sec=remaining,
                        tags=c.tags,
                        intensity=c.intensity,
                        audio_mode=c.audio_mode,
                        description=c.description,
                        protect_end=False,
                    ))
                break
        return result

    # ── Segmentation ────────────────────────────────

    def _ensure_segments(self, segments: list[dict], duration: float) -> list[dict]:
        if len(segments) >= 3:
            return segments
        i_end = duration * 0.30
        o_start = duration * 0.70
        return [
            {"label": "intro", "start_sec": 0.0,    "end_sec": i_end,    "energy": 0.30},
            {"label": "body",  "start_sec": i_end,   "end_sec": o_start,  "energy": 0.75},
            {"label": "outro", "start_sec": o_start, "end_sec": duration, "energy": 0.30},
        ]

    # ── Intensity matching ──────────────────────────

    def _assign_clips_to_segments(
        self, clips: list[ClipInfo], music_segments: list[dict], strength: float,
    ) -> list[list[ClipInfo]]:
        if not clips:
            return [[] for _ in music_segments]

        n_segs = len(music_segments)
        sorted_clips = sorted(clips, key=lambda c: c.intensity)
        assignments: list[list[ClipInfo]] = [[] for _ in range(n_segs)]

        for clip in sorted_clips:
            scores = []
            for si in range(n_segs):
                energy = music_segments[si]["energy"]
                iscore = 1.0 - abs(clip.intensity - energy)
                bscore = 1.0 / (len(assignments[si]) + 1)
                score = strength * iscore + (1 - strength) * bscore + self.rng.uniform(0, 0.1)
                scores.append((score, si))
            scores.sort(reverse=True)
            assignments[scores[0][1]].append(clip)

        return assignments

    # ── Gapless timeline build ──────────────────────

    def _build_gapless_timeline(
        self,
        assignments: list[list[ClipInfo]],
        music_segments: list[dict],
        beats: list[float],
        downbeats: list[float],
    ) -> list[RenderSegment]:
        result: list[RenderSegment] = []
        start_t = music_segments[0]["start_sec"] if music_segments else 0.0
        current_time = start_t

        for seg_idx, clips in enumerate(assignments):
            seg = music_segments[seg_idx]
            seg_start = seg["start_sec"]
            seg_end = seg["end_sec"]
            seg_duration = seg_end - seg_start

            if not clips:
                continue

            if current_time < seg_start:
                current_time = seg_start

            for clip in clips:
                clip_dur = clip.duration_sec
                clip_end = current_time + clip_dur

                if beats:
                    clip_end = self._snap_to_beat(
                        clip_end, beats, seg, seg_duration,
                        current_time, clip_dur, clip.protect_end, clip.intensity,
                        music_segments,
                    )

                dur = clip_end - current_time
                if dur < 0.2:
                    continue

                params = AUDIO_MODE_PARAMS.get(clip.audio_mode, AUDIO_MODE_PARAMS["融入BGM"])
                jcut = max(100, params["gain_fade_ms"])
                src_fade = jcut

                result.append(RenderSegment(
                    clip=clip,
                    timeline_start_sec=round(current_time, 3),
                    timeline_end_sec=round(clip_end, 3),
                    bgm_volume_gain_db=params["bgm_gain_db"],
                    gain_fade_ms=params["gain_fade_ms"],
                    source_audio_volume=params["source_vol"],
                    source_audio_fade_in_ms=src_fade,
                    jcut_audio_lead_ms=jcut,
                ))
                current_time = clip_end

        return result

    def _snap_to_beat(
        self,
        time_sec: float,
        beats: list[float],
        current_seg: dict,
        seg_duration: float,
        clip_start: float,
        clip_dur: float,
        protect_end: bool,
        clip_intensity: float,
        all_segments: list[dict],
    ) -> float:
        """Beat alignment with energy-tolerance guard for protect_end."""
        if not beats:
            return time_sec

        seg_end = current_seg["end_sec"]
        distances = [(abs(b - time_sec), b) for b in beats]
        distances.sort()
        candidates = [d[1] for d in distances[:3]]
        forward = [b for b in candidates if b >= time_sec]
        snapped = forward[0] if forward else candidates[0]

        max_allowed = seg_end + seg_duration * 0.2
        if snapped <= max_allowed:
            return snapped

        # --- protect_end with energy tolerance ---
        if protect_end:
            # Check if snapped tail falls into a segment whose energy
            # is too far below this clip's intensity
            tail_seg = self._find_segment_at(snapped, all_segments)
            if tail_seg and abs(clip_intensity - tail_seg["energy"]) > 0.3:
                # Energy mismatch — don't stretch into a low-energy section.
                # Fall back to original unsnapped time.
                return time_sec
            # Energy match is acceptable; accept the beat snap
            return snapped

        # Secondary trim
        beats_in_seg = [b for b in beats if clip_start < b <= seg_end]
        if beats_in_seg:
            trimmed = max(beats_in_seg)
            if trimmed - clip_start >= 0.3:
                return trimmed

        return seg_end

    def _find_segment_at(self, t: float, segments: list[dict]) -> Optional[dict]:
        """Return the segment that contains time `t`, or None."""
        for s in segments:
            if s["start_sec"] <= t <= s["end_sec"]:
                return s
        return None


# ═══════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════

def make_clip_id(source: str, start: float, end: float) -> str:
    return hashlib.sha256(f"{source}:{start:.2f}:{end:.2f}".encode()).hexdigest()[:12]


def load_clips_from_json(path: str | Path) -> list[ClipInfo]:
    with open(path) as f:
        data = json.load(f)
    clips = []
    for c in data.get("clips", []):
        start = c["start_sec"]
        end = c["end_sec"]
        clips.append(ClipInfo(
            clip_id=c.get("clip_id", make_clip_id(c["source_path"], start, end)),
            source_path=c["source_path"],
            start_sec=start,
            end_sec=end,
            duration_sec=c.get("duration_sec", end - start),
            tags=c.get("tags", []),
            intensity=c.get("intensity", 0.5),
            audio_mode=c.get("audio_mode", "融入BGM"),
            description=c.get("description", ""),
            protect_end=c.get("protect_end", False),
        ))
    return clips


def load_music_timeline(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python planner.py <clips.json> <timeline.json> [bgm] [strategy] [seed] [offset|--auto-offset] [--scale-mode mode]")
        print("  strategy:     fit | loop | truncate")
        print("  seed:         random seed (default 42)")
        print("  offset:       BGM start offset in seconds")
        print("  --auto-offset: use suggested_intro_offset")
        print("  --scale-mode:  letterbox (default) | crop | stretch")
        sys.exit(1)

    clips = load_clips_from_json(sys.argv[1])
    music = load_music_timeline(sys.argv[2])
    bgm = sys.argv[3] if len(sys.argv) > 3 else "bgm.mp3"
    strategy = sys.argv[4] if len(sys.argv) > 4 else "fit"

    seed, offset = 42, 0.0
    auto_offset = False
    scale_mode = "letterbox"

    for a in sys.argv[5:]:
        if a == "--auto-offset":
            auto_offset = True

    for i, a in enumerate(sys.argv[5:-1]):
        if a == "--scale-mode":
            scale_mode = sys.argv[6 + i]
            break

    for a in sys.argv[5:]:
        try:
            seed = int(a)
            break
        except ValueError:
            continue

    for a in sys.argv[5:]:
        try:
            offset = float(a)
            break
        except ValueError:
            continue

    planner = CutPlanner(seed=seed)
    timeline = planner.plan(
        clips, music, bgm,
        duration_strategy=strategy,
        bgm_start_offset=offset,
        auto_offset=auto_offset,
        scale_mode=scale_mode,
    )

    print(json.dumps(timeline.to_dict(), indent=2, ensure_ascii=False))
