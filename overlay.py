"""
conv-edit overlay system — text/image overlay rendering engine.

通用层：所有 preset 共用的叠加渲染逻辑。
游戏专属的叠加样式/预设放在 presets/ 目录下。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path


# ─── Data Model ────────────────────────────────────────

@dataclass
class OverlayDef:
    """Single overlay element — text or image with position, timing, style."""

    id: str
    type: str  # "text_bar" | "text_popup" | "subtitle" | "image"
    content: str  # text string or image file path

    # Timing (seconds, in GLOBAL timeline — not per-clip)
    start: float = 0.0
    end: float = 9999.0

    # Position (ffmpeg expressions: "10", "(w-text_w)/2", "w-tw-10")
    x: str = "(w-text_w)/2"
    y: str = "10"

    # Text style
    font_size: int = 28
    font_color: str = "white"
    font_border: int = 2
    font_border_color: str = "black"
    font_file: str = ""  # path to .ttf / .otf

    # Background box (for text_bar type)
    box: bool = False
    box_color: str = "black@0.5"
    box_border: int = 5

    # Image specific
    scale_w: int = 0  # 0 = original width
    scale_h: int = 0

    # Animation preset
    animation: str = ""  # "fade_in" | "pop_in" | "shake" | "typewriter"

    def to_drawtext(self) -> str:
        """Convert to ffmpeg drawtext filter fragment."""
        escaped = self._escape_text(self.content)
        parts = [
            f"text='{escaped}'",
            f"fontsize={self.font_size}",
            f"fontcolor={self.font_color}",
            f"x={self.x}",
            f"y={self.y}",
        ]
        if self.font_file:
            parts.append(f"fontfile='{self.font_file}'")
        if self.font_border > 0:
            parts.append(f"bordercolor={self.font_border_color}")
            parts.append(f"borderw={self.font_border}")
        if self.box:
            parts.append(f"box=1:boxcolor={self.box_color}:boxborderw={self.box_border}")
        return "drawtext=" + ":".join(parts)

    def to_overlay_filter(self) -> str:
        """Convert image overlay to ffmpeg overlay filter chain.
        Returns a filter fragment that should be composed with other filters."""
        scale = ""
        if self.scale_w > 0 and self.scale_h > 0:
            scale = f",scale={self.scale_w}:{self.scale_h}"
        return (
            f"movie='{self.content}'{scale}[ov_{self.id}];"
            f"[base][ov_{self.id}]overlay={self.x}:{self.y}"
        )

    @staticmethod
    def _escape_text(text: str) -> str:
        """Escape special characters for ffmpeg drawtext."""
        return (
            text.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
            .replace("%", "\\%")
        )

    def with_enable(self, local_start: float, local_end: float) -> OverlayDef:
        """Return a copy with enable expression scoped to a clip's local timeline."""
        import copy
        ov = copy.copy(self)
        ov.start = local_start
        ov.end = local_end
        return ov


# ─── Clip Filter Builder ───────────────────────────────

def build_clip_vf(
    base_vf: str,
    overlays: list[OverlayDef],
    clip_global_start: float,
    clip_dur: float,
    clip_sources: list[str] | None = None,
) -> str:
    """Build the video filter chain for a single clip with overlays.

    Args:
        base_vf:  Base ffmpeg video filter (e.g. "scale=1920:1080,fps=30")
        overlays: All overlays defined in the plan
        clip_global_start: This clip's start time in the output timeline
        clip_dur: This clip's duration in seconds
        clip_sources: Extra source files needed (for image overlays)

    Returns:
        Complete vf string for ffmpeg -vf parameter
    """
    filters = [base_vf] if base_vf else []

    for ov in overlays:
        # Convert global timeline → clip-local timeline
        ov_start = ov.start - clip_global_start
        ov_end = ov.end - clip_global_start

        # Skip overlays that don't intersect this clip
        if ov_end <= 0 or ov_start >= clip_dur:
            continue

        # Clamp to clip boundaries
        t0 = max(0.0, ov_start)
        t1 = min(clip_dur, ov_end)

        if ov.type == "image":
            filters.append(
                f"movie='{ov.content}'"
                f"{',scale=' + str(ov.scale_w) + ':' + str(ov.scale_h) if ov.scale_w else ''}"
                f"[ov_{ov.id}];"
                f"[in][ov_{ov.id}]overlay={ov.x}:{ov.y}"
                f":enable='between(t,{t0},{t1})'"
            )
        else:
            enable = f":enable='between(t,{t0},{t1})'" if (t0 > 0 or t1 < clip_dur) else ""
            filters.append(ov.to_drawtext() + enable)

    return ",".join(filters)


# ─── Overlay Collection ────────────────────────────────

def overlays_from_dict(data: list[dict]) -> list[OverlayDef]:
    """Convert JSON/dict list to OverlayDef list."""
    result = []
    for d in data:
        # Filter to only OverlayDef fields
        valid_keys = {f.name for f in OverlayDef.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        result.append(OverlayDef(**filtered))
    return result


def overlays_to_dict(overlays: list[OverlayDef]) -> list[dict]:
    """Convert OverlayDef list to JSON-serializable dicts."""
    return [asdict(ov) for ov in overlays]


# ─── Animation Presets ────────────────────────────────

ANIMATION_PRESETS: dict[str, dict] = {
    "fade_in": {
        "duration": 0.3,
        "drawtext_extra": ":alpha='if(lt(t-{start},0.3), (t-{start})/0.3, 1)'",
    },
    "pop_in": {
        "duration": 0.4,
        "scale_start": 0.3,
    },
    "shake": {
        "duration": 0.5,
        "amplitude": 10,
        "frequency": 20,
    },
}


def apply_animation(ov: OverlayDef, local_start: float) -> str:
    """Apply animation preset to an overlay, returning extra filter modifiers."""
    if not ov.animation or ov.animation not in ANIMATION_PRESETS:
        return ""

    preset = ANIMATION_PRESETS[ov.animation]
    if ov.animation == "fade_in":
        dur = preset["duration"]
        return (
            f":alpha='if(lt(t-{local_start},{dur}),"
            f"(t-{local_start})/{dur},1)'"
        )
    # shake, pop_in would need more complex filter chains
    return ""


# ─── Preset Loader ─────────────────────────────────────

def load_overlays_from_preset(preset_path: str | Path) -> list[OverlayDef]:
    """Load overlay definitions from a YAML preset file."""
    import yaml

    preset_path = Path(preset_path)
    if not preset_path.exists():
        return []

    with open(preset_path) as f:
        data = yaml.safe_load(f)

    overlay_data = data.get("overlays", [])
    return overlays_from_dict(overlay_data)


# ─── Builtin Presets ──────────────────────────────────

def douyin_comedy_overlays() -> list[OverlayDef]:
    """抖音欢乐剧场风格默认叠加层 — 游戏剪辑专用。"""
    return [
        # 顶部标题条（全片恒定）
        OverlayDef(
            id="title_bar",
            type="text_bar",
            content="",
            x="(w-text_w)/2",
            y="10",
            font_size=22,
            font_color="yellow",
            font_border=2,
            font_border_color="black",
            box=True,
            box_color="black@0.55",
            box_border=6,
            start=0.0,
            end=9999.0,
        ),
        # 底部字幕
        OverlayDef(
            id="subtitle",
            type="subtitle",
            content="",
            x="(w-text_w)/2",
            y="h-th-50",
            font_size=26,
            font_color="yellow",
            font_border=2,
            font_border_color="black",
            box=True,
            box_color="black@0.5",
            box_border=5,
            start=0.0,
            end=9999.0,
        ),
    ]
