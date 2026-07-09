"""
conv-edit v2: AI-assisted video clip annotation workbench
Auto scene detection → smart auto-annotation → one-click pipeline
"""

import json
import subprocess
import tempfile
import hashlib
import base64
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import our own modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from planner import CutPlanner, load_clips_from_json, load_music_timeline
from planner.validator import validate, auto_fix
from audio.analyzer import AudioAnalyzer

app = FastAPI(title="conv-edit")
SESSIONS: dict[str, dict] = {}
BGM_SESSIONS: dict[str, dict] = {}
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
RENDER_DIR = Path(__file__).parent / "renders"
RENDER_DIR.mkdir(exist_ok=True)


# ─── Data model ────────────────────────────────────────
@dataclass
class Scene:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    thumbnail_b64: str = ""
    selected: bool = True
    trim_start: float = 0.0
    trim_end: float = 0.0
    
    # Auto-detected (user can override)
    intensity: float = 0.5
    intensity_auto: bool = True                    # True = auto, False = user-set
    audio_mode: str = "融入BGM"
    audio_mode_auto: bool = True
    tags: list[str] = None
    description: str = ""
    protect_end: bool = False

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.trim_end == 0.0:
            self.trim_end = self.duration_sec

    @property
    def effective_start(self) -> float:
        return self.start_sec + self.trim_start

    @property
    def effective_end(self) -> float:
        return self.start_sec + self.trim_end

    @property
    def effective_duration(self) -> float:
        return max(0.1, self.effective_end - self.effective_start)


# ═══════════════════════════════════════════════════════
#  Auto-detection engines
# ═══════════════════════════════════════════════════════

def auto_detect_attributes(video_path: str, scenes: list[Scene]) -> None:
    """Run all auto-detection on scenes. Mutates scenes in-place."""
    for s in scenes:
        mid = s.start_sec + s.duration_sec / 2
        
        # 1. Intensity from audio RMS
        rms_db = _measure_rms(video_path, s.start_sec, s.duration_sec)
        s.intensity = _rms_to_intensity(rms_db)
        s.intensity_auto = True
        
        # 2. Audio mode from silence analysis
        s.audio_mode = _detect_audio_mode(video_path, s.start_sec, s.duration_sec)
        s.audio_mode_auto = True
        
        # 3. Auto-tags
        s.tags = _suggest_tags(s)
        
        # 4. Auto protect_end for short high-intensity clips
        if s.duration_sec < 3.0 and s.intensity > 0.75:
            s.protect_end = True


def _measure_rms(video_path: str, start: float, duration: float) -> float:
    """Measure RMS audio level in dB. Returns e.g. -15.0."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-ss", str(start), "-t", str(min(duration, 10)),
             "-i", video_path,
             "-af", "volumedetect", "-vn", "-sn",
             "-f", "null", "/dev/null"],
            capture_output=True, text=True, timeout=20
        )
        m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", r.stderr)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return -70.0  # effectively silent


def _rms_to_intensity(rms_db: float) -> float:
    """Map RMS dB to 0-1 intensity. -10dB=0.9, -25dB=0.5, -40dB=0.1."""
    return round(max(0.0, min(1.0, (rms_db + 50) / 40)), 2)


def _detect_audio_mode(video_path: str, start: float, duration: float) -> str:
    """Classify audio mode based on silence detection."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-ss", str(start), "-t", str(min(duration, 10)),
             "-i", video_path,
             "-af", "silencedetect=n=-30dB:d=0.5",
             "-vn", "-sn",
             "-f", "null", "/dev/null"],
            capture_output=True, text=True, timeout=20
        )
        # Count silence segments
        silences = re.findall(r"silence_start:\s*([\d.]+)", r.stderr)
        silence_ends = re.findall(r"silence_end:\s*([\d.]+)", r.stderr)
        
        total_silence = 0.0
        for i in range(min(len(silences), len(silence_ends))):
            total_silence += float(silence_ends[i]) - float(silences[i])
        
        if duration <= 0:
            return "融入BGM"
        
        silence_ratio = total_silence / duration
        
        # Near total silence → 纯BGM (source audio is just ambiance)
        if silence_ratio > 0.8:
            return "纯BGM"
        # Lots of short non-silence bursts → likely speech
        non_silence_count = len(silences) - 1
        if non_silence_count >= 3 and silence_ratio < 0.5:
            return "突出人声"
        
        return "融入BGM"
    except Exception:
        return "融入BGM"


def _suggest_tags(s: Scene) -> list[str]:
    """Suggest tags based on clip characteristics."""
    tags = []
    if s.duration_sec < 2.0 and s.intensity > 0.7:
        tags.append("快节奏")
    if s.duration_sec > 6.0 and s.intensity < 0.4:
        tags.append("过渡")
    if s.intensity > 0.8:
        tags.append("高能")
    elif s.intensity < 0.3:
        tags.append("平静")
    return tags


# ═══════════════════════════════════════════════════════
#  FFmpeg helpers
# ═══════════════════════════════════════════════════════

def detect_scenes_ffmpeg(video_path: str, threshold: float = 0.3) -> list[Scene]:
    """Use ffmpeg's scene detection filter to find cut points."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene\\\\,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    timestamps = [0.0]
    for line in result.stderr.split('\n'):
        m = re.search(r'pts_time:([\d.]+)', line)
        if m:
            t = float(m.group(1))
            if t > timestamps[-1] + 0.5:
                timestamps.append(t)
    
    duration = get_video_duration(video_path)
    if timestamps[-1] < duration - 0.5:
        timestamps.append(duration)
    
    MIN_SCENE = 1.5
    scenes = []
    for i in range(len(timestamps) - 1):
        start = timestamps[i]
        end = timestamps[i + 1]
        dur = end - start
        if dur >= MIN_SCENE or len(scenes) == 0:
            scenes.append(Scene(
                index=len(scenes),
                start_sec=start,
                end_sec=end,
                duration_sec=dur
            ))
        else:
            scenes[-1].end_sec = end
            scenes[-1].duration_sec = scenes[-1].end_sec - scenes[-1].start_sec
    
    # Fallback: split by equal time chunks
    if len(scenes) < 3:
        CHUNK_SEC = 5.0
        scenes = []
        t = 0.0
        idx = 0
        while t < duration:
            end = min(t + CHUNK_SEC, duration)
            scenes.append(Scene(index=idx, start_sec=t, end_sec=end, duration_sec=end - t))
            t = end
            idx += 1
    
    return scenes


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return float(result.stdout.strip())


def generate_thumbnail(video_path: str, time_sec: float, width: int = 320) -> str:
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        thumb_path = f.name
    
    cmd = [
        "ffmpeg", "-y", "-ss", str(time_sec), "-i", video_path,
        "-vframes", "1", "-vf", f"scale={width}:-1",
        "-q:v", "5", thumb_path
    ]
    subprocess.run(cmd, capture_output=True, timeout=15)
    
    with open(thumb_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    
    Path(thumb_path).unlink(missing_ok=True)
    return f"data:image/jpeg;base64,{data}"


# ═══════════════════════════════════════════════════════
#  API Routes
# ═══════════════════════════════════════════════════════

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...), threshold: float = Form(0.3)):
    """Upload video, detect scenes, auto-annotate, return results."""
    ext = Path(file.filename).suffix or ".mp4"
    session_id = hashlib.md5(file.filename.encode()).hexdigest()[:12]
    video_path = OUTPUT_DIR / f"{session_id}{ext}"
    
    content = await file.read()
    video_path.write_bytes(content)
    
    try:
        scenes = detect_scenes_ffmpeg(str(video_path), threshold)
        if len(scenes) > 200:
            scenes = scenes[:200]
    except Exception as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Scene detection failed: {e}")
    
    # ── Auto-detection ──────────────────────────
    auto_detect_attributes(str(video_path), scenes)
    
    # ── Thumbnails ──────────────────────────────
    for s in scenes:
        mid = s.start_sec + s.duration_sec / 2
        try:
            s.thumbnail_b64 = generate_thumbnail(str(video_path), mid)
        except Exception:
            s.thumbnail_b64 = ""
    
    SESSIONS[session_id] = {
        "video_path": str(video_path),
        "scenes": [asdict(s) for s in scenes],
        "filename": file.filename
    }
    
    return {
        "session_id": session_id,
        "filename": file.filename,
        "duration": get_video_duration(str(video_path)),
        "scene_count": len(scenes),
        "scenes": [asdict(s) for s in scenes]
    }


@app.post("/api/analyze-bgm")
async def analyze_bgm(file: UploadFile = File(...)):
    """Upload BGM, auto-analyze, return waveform + beats for visualization."""
    bgm_id = hashlib.md5(file.filename.encode()).hexdigest()[:12]
    bgm_path = OUTPUT_DIR / f"bgm_{bgm_id}{Path(file.filename).suffix or '.mp3'}"
    
    content = await file.read()
    bgm_path.write_bytes(content)
    
    try:
        analyzer = AudioAnalyzer()
        timeline = analyzer.analyze(str(bgm_path))
        result = timeline.to_dict()
        result["bgm_id"] = bgm_id
        result["bgm_path"] = str(bgm_path)
        result["filename"] = file.filename
        
        # Downsample energy curve for frontend waveform (max 200 points)
        curve = result.get("energy_curve", [])
        if len(curve) > 200:
            step = len(curve) // 200
            result["energy_curve"] = [curve[i] for i in range(0, len(curve), step)]
        
        BGM_SESSIONS[bgm_id] = result
        return result
    except Exception as e:
        bgm_path.unlink(missing_ok=True)
        raise HTTPException(500, f"BGM analysis failed: {e}")


@app.post("/api/plan")
async def plan_timeline(data: dict):
    """Generate render timeline from clips + BGM selection."""
    session_id = data.get("session_id")
    bgm_id = data.get("bgm_id")
    strategy = data.get("strategy", "fit")
    seed = data.get("seed", 42)
    offset = data.get("offset", 0.0)
    auto_offset = data.get("auto_offset", True)
    scale_mode = data.get("scale_mode", "letterbox")
    
    if session_id not in SESSIONS:
        raise HTTPException(404, "Video session not found")
    if bgm_id not in BGM_SESSIONS:
        raise HTTPException(404, "BGM session not found")
    
    session = SESSIONS[session_id]
    bgm = BGM_SESSIONS[bgm_id]
    
    # Build clips from session scenes
    video_path = session["video_path"]
    clips = []
    for s in session["scenes"]:
        if not s.get("selected", True):
            continue
        from planner import ClipInfo
        start = s["start_sec"] + s.get("trim_start", 0)
        end = s["start_sec"] + s.get("trim_end", s["duration_sec"])
        clips.append(ClipInfo(
            clip_id=f"{session_id}_{s['index']:03d}",
            source_path=video_path,
            start_sec=start,
            end_sec=end,
            duration_sec=max(0.1, end - start),
            tags=s.get("tags", []),
            intensity=s.get("intensity", 0.5),
            audio_mode=s.get("audio_mode", "融入BGM"),
            description=s.get("description", ""),
            protect_end=s.get("protect_end", False),
        ))
    
    if not clips:
        raise HTTPException(400, "No clips selected")
    
    # Build music timeline from BGM analysis
    music_timeline = {
        "bpm": bgm.get("bpm", 120),
        "duration_sec": bgm.get("duration_sec", 60),
        "beats_sec": bgm.get("beats_sec", []),
        "downbeats_sec": bgm.get("downbeats_sec", []),
        "segments": bgm.get("segments", []),
        "suggested_intro_offset": bgm.get("suggested_intro_offset", 0.0),
    }
    
    # Run planner
    planner = CutPlanner(seed=seed)
    timeline = planner.plan(
        clips, music_timeline, bgm["bgm_path"],
        duration_strategy=strategy,
        bgm_start_offset=offset,
        auto_offset=auto_offset,
        scale_mode=scale_mode,
    )
    
    plan_dict = timeline.to_dict()
    
    # Run validator inline
    issues = validate(plan_dict)
    errors = [i for i in issues if i.severity == "ERROR"]
    warnings = [i for i in issues if i.severity == "WARN"]
    
    # Auto-fix if there are warnings but no errors
    if warnings and not errors:
        try:
            plan_dict = auto_fix(plan_dict)
            issues = validate(plan_dict)
            errors = [i for i in issues if i.severity == "ERROR"]
            warnings = [i for i in issues if i.severity == "WARN"]
        except Exception:
            pass
    
    # Build intensity heatmap for frontend
    heatmap = []
    for seg in plan_dict.get("segments", []):
        heatmap.append({
            "start": seg["timeline_start"],
            "end": seg["timeline_end"],
            "intensity": seg.get("intensity", 0.5),
            "audio_mode": seg.get("audio_mode", "融入BGM"),
            "clip_id": seg["clip_id"],
        })
    
    return {
        "plan": plan_dict,
        "heatmap": heatmap,
        "errors": [{"msg": i.message, "seg": i.segment_index, "fix": i.fix} for i in errors],
        "warnings": [{"msg": i.message, "seg": i.segment_index, "fix": i.fix} for i in warnings],
        "ready": len(errors) == 0,
    }


@app.post("/api/render")
async def render_video(data: dict):
    """Render final video from a plan."""
    plan = data.get("plan")
    if not plan:
        raise HTTPException(400, "No plan provided")
    
    render_id = hashlib.md5(json.dumps(plan).encode()).hexdigest()[:12]
    output_path = RENDER_DIR / f"render_{render_id}.mp4"
    
    # Save plan for reference
    plan_path = RENDER_DIR / f"render_{render_id}.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    
    segs = plan.get("segments", [])
    if not segs:
        raise HTTPException(400, "No segments in plan")
    
    bgm = plan.get("bgm_path", "")
    offset = plan.get("bgm_start_offset_sec", 0.0)
    fps = plan.get("output_fps", 30)
    width, height = 1920, 1080
    if plan.get("output_resolution"):
        w, h = plan["output_resolution"].split("x")
        width, height = int(w), int(h)
    scale_mode = plan.get("scale_mode", "letterbox")
    
    # Build filter complex for concatenation
    filter_parts = []
    audio_parts = []
    
    for i, s in enumerate(segs):
        src = s["source_path"]
        src_start = s["source_start"]
        src_dur = s["source_duration"]
        tl_dur = s["timeline_end"] - s["timeline_start"]
        speed = s.get("playback_speed", 1.0)
        audio_mode = s.get("audio_mode", "融入BGM")
        
        # Video: trim + scale + setpts
        scale_filter = ""
        if scale_mode == "letterbox":
            scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        elif scale_mode == "crop":
            scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
        else:
            scale_filter = f"scale={width}:{height}"
        
        if abs(speed - 1.0) > 0.001:
            filter_parts.append(
                f"[{i}:v]trim={src_start}:{src_start+src_dur},setpts={speed}*PTS,"
                f"{scale_filter},fps={fps},format=yuv420p[v{i}]"
            )
        else:
            filter_parts.append(
                f"[{i}:v]trim={src_start}:{src_start+src_dur},setpts=PTS,"
                f"{scale_filter},fps={fps},format=yuv420p[v{i}]"
            )
        
        # Audio: trim + atempo if speed-adjusted
        src_vol = s.get("source_audio_volume", 1.0)
        if audio_mode == "纯BGM" or src_vol <= 0:
            # Mute source audio
            audio_parts.append(f"[{i}:a]atrim={src_start}:{src_start+src_dur},volume=0[a{i}]")
        else:
            vol_str = f"volume={src_vol}"
            if abs(speed - 1.0) > 0.001:
                audio_parts.append(
                    f"[{i}:a]atrim={src_start}:{src_start+src_dur},atempo={1/speed},{vol_str}[a{i}]"
                )
            else:
                audio_parts.append(
                    f"[{i}:a]atrim={src_start}:{src_start+src_dur},{vol_str}[a{i}]"
                )
    
    # Concat all video streams
    v_inputs = "".join(f"[v{i}]" for i in range(len(segs)))
    filter_parts.append(f"{v_inputs}concat=n={len(segs)}:v=1:a=0[vout]")
    
    # Concat all audio streams
    a_inputs = "".join(f"[a{i}]" for i in range(len(segs)))
    filter_parts.append(f"{a_inputs}concat=n={len(segs)}:v=0:a=1[aout]")
    
    filter_complex = ";".join(filter_parts)
    
    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    for s in segs:
        cmd += ["-ss", str(s["source_start"]), "-i", s["source_path"]]
    
    cmd += [
        "-i", bgm,
        "-ss", str(offset),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-map", "1:a",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise HTTPException(500, f"Render failed: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Render timed out (10 min)")
    except Exception as e:
        raise HTTPException(500, f"Render error: {e}")
    
    return {
        "render_id": render_id,
        "output": str(output_path),
        "download_url": f"/api/download/{render_id}",
    }


@app.get("/api/download/{render_id}")
async def download_render(render_id: str):
    """Download rendered video."""
    path = RENDER_DIR / f"render_{render_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Render not found")
    return FileResponse(path, media_type="video/mp4", filename=f"output_{render_id}.mp4")


@app.post("/api/export/{session_id}")
async def export_clips(session_id: str, data: dict):
    """Export selected/trimmed scenes as unified clip format JSON."""
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    
    session = SESSIONS[session_id]
    video_path = session["video_path"]
    
    clips = []
    for s in data.get("scenes", session["scenes"]):
        if not s.get("selected", True):
            continue
        
        start = s.get("start_sec", 0) + s.get("trim_start", 0)
        end = s.get("start_sec", 0) + s.get("trim_end", s.get("duration_sec", 0))
        
        clip = {
            "clip_id": f"{session_id}_{s.get('index', 0):03d}",
            "source_path": video_path,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": max(0.1, end - start),
            "tags": s.get("tags", []),
            "intensity": s.get("intensity", 0.5),
            "audio_mode": s.get("audio_mode", "融入BGM"),
            "description": s.get("description", ""),
            "protect_end": s.get("protect_end", False),
        }
        clips.append(clip)
    
    # Save clips to file
    clips_path = OUTPUT_DIR / f"clips_{session_id}.json"
    output = {"source_video": video_path, "clip_count": len(clips), "clips": clips}
    clips_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    
    return output


@app.get("/api/preview/{session_id}/{scene_index}")
async def preview_scene(session_id: str, scene_index: int):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    
    scenes = SESSIONS[session_id]["scenes"]
    if scene_index >= len(scenes):
        raise HTTPException(404, "Scene not found")
    
    s = scenes[scene_index]
    video_path = SESSIONS[session_id]["video_path"]
    preview_path = OUTPUT_DIR / f"preview_{session_id}_{scene_index}.mp4"
    
    start = s["start_sec"]
    duration = min(s["duration_sec"], 10.0)
    
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
        "-t", str(duration), "-vf", "scale=640:-1",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an", str(preview_path)
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)
    
    return FileResponse(preview_path, media_type="video/mp4")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Static files not found</h1>"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(SESSIONS),
        "bgm_sessions": len(BGM_SESSIONS),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
