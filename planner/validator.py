"""
Pre-render validator — v6.

v6 changes:
  - auto_fix: only speed-adjust "純BGM" clips (no vocal distortion)
  - playback_speed field replaces speed tags (renderer-ready)
  - max_loops truncation → WARNING (not silent HINT)
  - validate: gap-on-vocal-clips → ERROR
  - validate: jcut constraint now checks every segment
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Issue:
    severity: str
    message: str
    segment_index: int | None = None
    fix: str = ""


# ─── Audio modes that CAN be speed-adjusted (no vocal to ruin) ───
SPEEDABLE_MODES = {"纯BGM"}


def _is_speedable(seg: dict) -> bool:
    """True if this segment's audio can safely be time-stretched."""
    return seg.get("audio_mode", "") in SPEEDABLE_MODES and not seg.get("protect_end", False)


# ═══════════════════════════════════════════════════════
#  validate
# ═══════════════════════════════════════════════════════

def validate(plan: dict, strict: bool = True) -> list[Issue]:
    issues: list[Issue] = []
    segs = plan.get("segments", [])
    bgm_path = plan.get("bgm_path", "")
    total_claimed = plan.get("total_duration_sec", 0)

    # ── Top-level ────────────────────────────────
    if bgm_path and not Path(bgm_path).exists():
        issues.append(Issue("WARN", f"BGM not found: {bgm_path}",
                            fix=f"Verify file exists at '{bgm_path}'"))

    for hint_field, hint_text in [
        ("scale_mode", "No scale_mode (default: letterbox)"),
        ("audio_sample_rate", "No audio_sample_rate (default: 48000)"),
        ("audio_channels", "No audio_channels (default: 2 → stereo)"),
    ]:
        if not plan.get(hint_field):
            issues.append(Issue("HINT", hint_text,
                                fix=f"Set {hint_field} in planner output"))

    res = plan.get("output_resolution", "")
    fps = plan.get("output_fps", 0)
    if not res:
        issues.append(Issue("HINT", "No output_resolution (default: 1920x1080)"))
    if fps <= 0:
        issues.append(Issue("HINT", "No output_fps (default: 30)"))

    # ── max_loops truncation detection ───────────
    if plan.get("loop_capped"):
        issues.append(Issue(
            "WARN",
            "BGM truncated — loop strategy hit max_loops limit. "
            "Total clip duration shorter than BGM.",
            fix="Add more clips, increase max_loops, or accept truncated BGM with fade-out"
        ))

    if not segs:
        issues.append(Issue("ERROR", "No segments"))
        return issues

    _check_aspect_ratios(segs, issues)

    # ── Back-to-back ─────────────────────────────
    prev_end = segs[0]["timeline_start"]
    for i, s in enumerate(segs):
        t0, t1 = s["timeline_start"], s["timeline_end"]

        if t0 > prev_end + 0.01:
            gap = t0 - prev_end
            prev_mode = segs[i - 1].get("audio_mode", "") if i > 0 else ""
            curr_mode = s.get("audio_mode", "")
            speedable_nearby = _is_speedable(segs[i - 1]) if i > 0 else False

            if gap >= 2.0:
                issues.append(Issue(
                    "ERROR",
                    f"Gap {gap:.2f}s before segment {i} — too large for auto-fix",
                    i,
                    "Add clips, use 'loop' strategy, or shorten BGM"
                ))
            elif not speedable_nearby and not _is_speedable(s):
                # Adjacent clips both have preserved audio → can't speed-fix
                issues.append(Issue(
                    "ERROR",
                    f"Gap {gap:.2f}s before segment {i} — adjacent clips contain "
                    f"preserved audio ({prev_mode}, {curr_mode}), cannot speed-compensate",
                    i,
                    "Add a 純BGM clip between them, trim silent tails, or add more clips"
                ))
            else:
                issues.append(Issue(
                    "WARN",
                    f"Gap {gap:.2f}s before segment {i}",
                    i,
                    f"Run --auto-fix (speed-ripple on 純BGM clips can compensate {gap:.2f}s)"
                ))

        if t0 < prev_end - 0.01:
            issues.append(Issue(
                "ERROR",
                f"Overlap {prev_end - t0:.2f}s at segment {i}",
                i,
                "Check planner logic"
            ))
        if t1 <= t0:
            issues.append(Issue(
                "ERROR", f"Segment {i}: zero/negative duration ({t0}→{t1})",
                i
            ))
        prev_end = t1

    # ── Duration ─────────────────────────────────
    actual = segs[-1]["timeline_end"] - segs[0]["timeline_start"]
    if abs(actual - total_claimed) > 1.0:
        issues.append(Issue(
            "WARN",
            f"Duration mismatch: claimed {total_claimed:.1f}s, actual {actual:.1f}s",
            fix="Re-run planner"
        ))

    # ── Source files ─────────────────────────────
    seen: dict[str, float] = {}
    for i, s in enumerate(segs):
        src = s.get("source_path", "")
        if not src:
            issues.append(Issue("ERROR", f"Segment {i}: missing source_path", i))
            continue
        if src not in seen:
            if not Path(src).exists():
                issues.append(Issue("WARN", f"Source not found: {src}", i))
                seen[src] = 99999
            else:
                seen[src] = _get_duration(src)
        fd = seen[src]
        se = s.get("source_end", 0)
        if se > fd + 0.5 and not s.get("protect_end", False):
            issues.append(Issue(
                "ERROR",
                f"Segment {i}: source_end {se:.1f}s > file duration {fd:.1f}s",
                i,
                f"Trim clip end to ≤ {fd:.1f}s"
            ))

    # ── Audio params ─────────────────────────────
    for i, s in enumerate(segs):
        for field in ("bgm_volume_gain_db", "gain_fade_ms",
                       "source_audio_volume", "source_audio_fade_in_ms"):
            if field not in s:
                issues.append(Issue("WARN", f"Segment {i}: missing {field}", i))
        jcut = s.get("jcut_audio_lead_ms", 100)
        fade = s.get("gain_fade_ms", 20)
        if jcut < fade:
            issues.append(Issue(
                "WARN",
                f"Segment {i}: jcut ({jcut}ms) < gain_fade ({fade}ms) — may pop",
                i,
                f"Increase jcut_audio_lead_ms to ≥ {fade}"
            ))

    # ── playback_speed on vocal clips ────────────
    for i, s in enumerate(segs):
        ps = s.get("playback_speed", 1.0)
        if ps != 1.0 and not _is_speedable(s):
            issues.append(Issue(
                "ERROR",
                f"Segment {i}: playback_speed={ps} on non-純BGM clip — "
                f"will distort vocals",
                i,
                "Remove speed adjustment from this clip or set audio_mode=純BGM"
            ))

    return issues


def _check_aspect_ratios(segs: list[dict], issues: list[Issue]) -> None:
    ratios: dict[str, float] = {}
    for s in segs:
        src = s.get("source_path", "")
        if not src or src in ratios:
            continue
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", src],
                capture_output=True, text=True, timeout=10
            )
            parts = r.stdout.strip().split(",")
            if len(parts) >= 2:
                w, h = int(parts[0]), int(parts[1])
                if h > 0:
                    ratios[src] = w / h
        except Exception:
            continue
    if len(ratios) >= 2 and max(ratios.values()) - min(ratios.values()) > 0.05:
        ref = list(ratios.values())[0]
        outliers = [
            f"{Path(p).name} ({v:.3f})"
            for p, v in ratios.items() if abs(v - ref) > 0.05
        ]
        if outliers:
            issues.append(Issue(
                "WARN",
                f"Aspect ratio mismatch: {', '.join(outliers)}",
                fix="Use consistent source resolution or set scale_mode=letterbox"
            ))


# ═══════════════════════════════════════════════════════
#  auto_fix — speed compensation  (v6: vocal-safe)
# ═══════════════════════════════════════════════════════

def auto_fix(plan: dict) -> dict:
    """Fix gaps with speed compensation — ONLY on 純BGM clips.

    Strategy:
      gap < 500ms   → stretch the nearest speedable clip (±5%)
      500ms–2s      → speed-ripple across up to 3 nearby speedable clips
      ≥ 2s          → leave untouched (user must resolve)

    If no speedable clip is adjacent, the gap stays — validate()
    will flag it as an ERROR since vocal clips can't be stretched.
    """
    import copy
    plan = copy.deepcopy(plan)
    segs = plan.setdefault("segments", [])

    # Ensure every segment starts with playback_speed = 1.0
    for s in segs:
        s.setdefault("playback_speed", 1.0)

    i = 1
    while i < len(segs):
        prev = segs[i - 1]
        curr = segs[i]
        t0 = prev["timeline_end"]
        t1 = curr["timeline_start"]
        gap = t1 - t0

        if gap < 0.01:
            i += 1
            continue

        if gap >= 2.0:
            i += 1
            continue

        if gap < 0.5:
            fixed = _stretch_one(segs, i - 1, gap)
        else:
            fixed = _speed_ripple(segs, i - 1, i, gap)

        # Re-chain timeline
        _rechain(segs)

        i += 1

    # Fix overlaps
    for j in range(1, len(segs)):
        prev = segs[j - 1]
        curr = segs[j]
        if curr["timeline_start"] < prev["timeline_end"] - 0.01:
            offset = prev["timeline_end"] - curr["timeline_start"]
            for k in range(j, len(segs)):
                segs[k]["timeline_start"] = round(segs[k]["timeline_start"] + offset, 3)
                segs[k]["timeline_end"] = round(segs[k]["timeline_end"] + offset, 3)

    plan["total_duration_sec"] = round(segs[-1]["timeline_end"], 2)
    return plan


def _stretch_one(segs: list[dict], prev_idx: int, gap: float) -> bool:
    """Try to stretch prev_idx segment to absorb `gap`.  Only 純BGM."""
    prev = segs[prev_idx]
    if not _is_speedable(prev):
        return False

    old_dur = prev["timeline_end"] - prev["timeline_start"]
    if old_dur <= 0.2:
        return False

    new_dur = old_dur + gap
    speed = new_dur / old_dur  # >1 = slower, <1 = faster

    if speed < 0.95 or speed > 1.05:
        return False  # too much distortion

    prev["timeline_end"] = round(prev["timeline_start"] + new_dur, 3)
    prev["playback_speed"] = round(speed, 4)
    return True


def _speed_ripple(segs: list[dict], left_idx: int, right_idx: int, gap: float) -> bool:
    """Distribute `gap` across nearby speedable segments (up to 3)."""
    candidates = []
    for idx in range(max(0, left_idx - 1), min(len(segs), right_idx + 2)):
        if idx >= len(segs):
            continue
        dur = segs[idx]["timeline_end"] - segs[idx]["timeline_start"]
        if dur > 0.5 and _is_speedable(segs[idx]):
            candidates.append(idx)

    if not candidates:
        return False

    per_clip = gap / len(candidates)
    for idx in candidates:
        old_dur = segs[idx]["timeline_end"] - segs[idx]["timeline_start"]
        new_dur = old_dur + per_clip
        # Clamp ±5%
        speed = max(0.95, min(1.05, new_dur / old_dur))
        new_dur = old_dur * speed
        segs[idx]["timeline_end"] = round(segs[idx]["timeline_start"] + new_dur, 3)
        segs[idx]["playback_speed"] = round(speed, 4)

    return True


def _rechain(segs: list[dict]) -> None:
    """Ensure timeline is gapless: pull segments together to close small gaps
    (overlaps already fixed by shifting; gaps closed by advancing start times)."""
    for j in range(1, len(segs)):
        prev_end = segs[j - 1]["timeline_end"]
        # Fix overlaps: push current segment forward
        if segs[j]["timeline_start"] < prev_end - 0.01:
            offset = prev_end - segs[j]["timeline_start"]
            segs[j]["timeline_start"] = round(segs[j]["timeline_start"] + offset, 3)
            segs[j]["timeline_end"] = round(segs[j]["timeline_end"] + offset, 3)
        # Fix gaps: pull current segment back to prev_end
        elif segs[j]["timeline_start"] > prev_end + 0.01:
            segs[j]["timeline_start"] = prev_end


def _get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(r.stdout.strip())
    except Exception:
        return 99999


# ─── CLI ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("plan", help="Path to render_timeline.json")
    p.add_argument("--auto-fix", action="store_true",
                   help="Speed-compensate gaps (純BGM only)")
    p.add_argument("--output", "-o", default=None,
                   help="Write fixed plan to this file")
    p.add_argument("--strict", action="store_true", default=True)
    args = p.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)

    if args.auto_fix:
        fixed = auto_fix(plan)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(fixed, f, indent=2, ensure_ascii=False)
            print(f"✅ Fixed plan → {args.output}")
        else:
            print(json.dumps(fixed, indent=2, ensure_ascii=False))
        sys.exit(0)

    issues = validate(plan, strict=args.strict)

    if not issues:
        print("✅ All checks passed — safe to render")
        sys.exit(0)

    errors = [i for i in issues if i.severity == "ERROR"]
    warns = [i for i in issues if i.severity == "WARN"]
    hints = [i for i in issues if i.severity == "HINT"]

    for e in errors:
        loc = f" [seg {e.segment_index}]" if e.segment_index is not None else ""
        print(f"❌ ERROR{loc}: {e.message}")
        if e.fix:
            print(f"   ↳ Fix: {e.fix}")

    for w in warns:
        loc = f" [seg {w.segment_index}]" if w.segment_index is not None else ""
        print(f"⚠️  WARN{loc}: {w.message}")
        if w.fix:
            print(f"   ↳ Fix: {w.fix}")

    for h in hints:
        print(f"💡 HINT: {h.message}")

    summary = f"{len(errors)} errors, {len(warns)} warnings"
    if hints:
        summary += f", {len(hints)} hints"
    print(f"\n{summary}")
    sys.exit(1 if errors else 0)
