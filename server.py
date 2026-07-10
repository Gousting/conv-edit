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

# ─── Logging ──────────────────────────────────────
import logging
LOG_DIR = Path.home() / ".conv-edit"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "server.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("conv-edit")
log.info("=== conv-edit server starting ===")

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

def _merge_high_energy_fragments(video_path: str, scene_times: list[float], duration: float) -> list[float]:
    """方案一：如果两个切点间隔 < 2.5s 且该区间音频响度极高（激战），
    强制删除中间切点，把碎片合并为完整的战斗段落。"""
    if len(scene_times) <= 2:
        return scene_times
    
    # 提取音频 RMS 强度（采样间隔 ~1s）
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-af", "aresample=1000,asetnsamples=1000,astats=metadata=1:reset=1",
            "-vn", "-sn", "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        rms_db = []
        for line in result.stderr.split('\n'):
            m = re.search(r'RMS level dB: ([-\\d.]+)', line)
            if m:
                rms_db.append(float(m.group(1)))
    except Exception:
        return scene_times  # 失败则跳过保护
    
    if len(rms_db) < 10:
        return scene_times
    
    # dB → 0-1 强度
    intensities = [max(0.0, min(1.0, (v + 50) / 40)) for v in rms_db]
    sample_dur = duration / len(intensities)
    
    def _intensity_at(t: float) -> float:
        idx = min(int(t / sample_dur), len(intensities) - 1)
        return intensities[idx]
    
    MIN_GAP = 2.5       # 小于此间隔的碎片候选合并
    HIGH_ENERGY = 0.6   # 强度阈值，超过视为激战
    
    merged = [scene_times[0]]
    skipped = 0
    for i in range(1, len(scene_times) - 1):
        gap = scene_times[i] - scene_times[i - 1]
        mid = (scene_times[i - 1] + scene_times[i]) / 2
        energy = _intensity_at(mid)
        
        if gap < MIN_GAP and energy > HIGH_ENERGY:
            # 激战中的碎切点 → 跳过，合并到前后
            skipped += 1
            continue
        merged.append(scene_times[i])
    
    merged.append(scene_times[-1])
    if skipped:
        print(f"[高能保护] 合并了 {skipped} 个碎切点，保留高光片段完整")
    return merged


def detect_scenes_ffmpeg(video_path: str, threshold: float = 0.3) -> list[Scene]:
    """Use ffmpeg's scene detection filter to find cut points.
    Retries with lower thresholds before falling back to intensity-based splitting."""
    
    def _run_scdet(thresh: float) -> list[float]:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select='gt(scene\\\\,{thresh})',showinfo",
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
        return timestamps

    duration = get_video_duration(video_path)
    
    # ─── 方案二：高阈值策略（游戏内容不需要细切） ───
    # 阈值越高越不敏感，只识别真正的场景切换（如地图切换），忽略枪火/跑动
    for thresh in [threshold, 0.4, 0.35]:
        timestamps = _run_scdet(thresh)
        if timestamps[-1] < duration - 0.5:
            timestamps.append(duration)
        if len(timestamps) >= 3:
            break
    
    # ─── 方案一：高能片段保护合并（防止击杀镜头被切碎） ───
    timestamps = _merge_high_energy_fragments(video_path, timestamps, duration)
    
    # If scdet found nothing useful (0 or 1 cuts), skip to intensity-based
    if len(timestamps) <= 2:
        return _intensity_split(video_path, duration)
    
    MIN_SCENE = 1.5
    MAX_SCENE = 8.0
    
    scenes = []
    for i in range(len(timestamps) - 1):
        start = timestamps[i]
        end = timestamps[i + 1]
        dur = end - start
        
        # Merge very short scenes into neighbors
        if dur < MIN_SCENE:
            if scenes:
                scenes[-1].end_sec = end
                scenes[-1].duration_sec = scenes[-1].end_sec - scenes[-1].start_sec
            continue
        
        # Split overly long scenes
        if dur > MAX_SCENE:
            sub_n = int(dur / MAX_SCENE) + 1
            sub_dur = dur / sub_n
            for j in range(sub_n):
                sub_start = start + j * sub_dur
                sub_end = min(start + (j + 1) * sub_dur, end)
                if sub_end - sub_start >= MIN_SCENE:
                    scenes.append(Scene(
                        index=len(scenes),
                        start_sec=sub_start,
                        end_sec=sub_end,
                        duration_sec=sub_end - sub_start
                    ))
        else:
            scenes.append(Scene(
                index=len(scenes),
                start_sec=start,
                end_sec=end,
                duration_sec=dur
            ))
    
    # Fallback: intensity-based splitting using audio energy
    if len(scenes) < 3:
        scenes = _intensity_split(video_path, duration)
    
    return scenes


def _intensity_split(video_path: str, duration: float) -> list[Scene]:
    """Split video by audio intensity peaks instead of fixed time chunks."""
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-af", "aresample=1000,asetnsamples=1000,astats=metadata=1:reset=1",
            "-vn", "-sn",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        rms_values = []
        for line in result.stderr.split('\n'):
            m = re.search(r'RMS level dB: ([-\d.]+)', line)
            if m:
                rms_values.append(float(m.group(1)))
    except Exception:
        rms_values = []
    
    if len(rms_values) < 10:
        # Ultimate fallback: moderate chunking (not 5s, vary by 3-7s)
        return _adaptive_chunk(duration)
    
    # Convert dB to 0-1 intensity, find peaks
    intensities = [max(0.0, min(1.0, (v + 50) / 40)) for v in rms_values]
    sample_dur = duration / len(intensities)
    
    # Find local minima as split points
    splits = [0.0]
    window = max(3, len(intensities) // 20)  # ~5% of video as window
    
    for i in range(window, len(intensities) - window):
        # Local minimum in a window, lower threshold for more splits
        neighborhood = intensities[i-window:i+window+1]
        if intensities[i] == min(neighborhood) and intensities[i] < 0.5:
            t = (i + 0.5) * sample_dur
            if t - splits[-1] >= 1.5:  # Minimum 1.5s between splits
                splits.append(t)
    
    splits.append(duration)
    
    scenes = []
    MIN_SCENE = 1.5
    for i in range(len(splits) - 1):
        start = splits[i]
        end = splits[i + 1]
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
    
    return scenes


def _adaptive_chunk(duration: float) -> list[Scene]:
    """Adaptive chunking: shorter chunks for short videos, longer for long ones."""
    if duration <= 30:
        chunk = 3.0
    elif duration <= 120:
        chunk = 5.0
    else:
        chunk = 8.0
    
    scenes = []
    t = 0.0
    idx = 0
    while t < duration:
        end = min(t + chunk, duration)
        # Vary chunk size slightly to avoid uniform splits
        if idx > 0 and end < duration:
            import random
            end = min(end + random.uniform(-1.5, 1.5), duration)
        if end - t >= 1.5:
            scenes.append(Scene(index=idx, start_sec=t, end_sec=end, duration_sec=end - t))
            t = end
            idx += 1
        else:
            t = end
    return scenes


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    try:
        return float(result.stdout.strip())
    except (ValueError, TypeError):
        return 0.0


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
    
    # If gaps are too large and strategy was 'fit', retry with 'loop'
    gap_errors = [i for i in errors if "Gap" in i.message and "too large" in i.message]
    if gap_errors and strategy == "fit":
        try:
            timeline2 = planner.plan(
                clips, music_timeline, bgm["bgm_path"],
                duration_strategy="loop",
                bgm_start_offset=offset,
                auto_offset=auto_offset,
                scale_mode=scale_mode,
            )
            plan_dict = timeline2.to_dict()
            issues = validate(plan_dict)
            errors = [i for i in issues if i.severity == "ERROR"]
            warnings = [i for i in issues if i.severity == "WARN"]
        except Exception:
            pass
    
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
    plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    
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
    
    # Deduplicate source files → map each to a unique input index
    unique_sources = []
    source_to_idx = {}
    for s in segs:
        sp = s["source_path"]
        if sp not in source_to_idx:
            source_to_idx[sp] = len(unique_sources)
            unique_sources.append(sp)
    
    # Two-phase approach for reliable rendering:
    # Phase 1: cut individual clips (fast, -c copy where possible)
    # Phase 2: concat via demuxer + mix BGM
    
    # Build scale filter once
    if scale_mode == "letterbox":
        scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    elif scale_mode == "crop":
        scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    else:
        scale_filter = f"scale={width}:{height}"
    
    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp(dir=RENDER_DIR, prefix="tmp_"))
    clip_files = []
    clip_list = tmp_dir / "concat_list.txt"
    lines = []
    
    try:
        for i, s in enumerate(segs):
            src = s["source_path"]
            src_start = s["source_start"]
            src_dur = s["source_duration"]
            speed = s.get("playback_speed", 1.0)
            src_vol = s.get("source_audio_volume", 1.0)
            clip_out = tmp_dir / f"clip_{i:03d}.mp4"
            
            # Cut clip with video scaling + audio (preserves sync)
            # Vocal reduction: notch out 200-500Hz and 800-1500Hz (voice range),
            # keep bass (footsteps) and highs (gunshots/explosions) intact
            vocal_reduce = plan.get("vocal_reduction", True)
            vf = f"{scale_filter},fps={fps},format=yuv420p"
            if vocal_reduce:
                af = f"bandreject=f=300:w=300:width_type=h,bandreject=f=1200:w=600:width_type=h,volume={src_vol}"
            else:
                af = f"volume={src_vol}"
            if abs(speed - 1.0) > 0.001:
                vf = f"setpts={speed}*PTS,{scale_filter},fps={fps},format=yuv420p"
                if vocal_reduce:
                    af = f"bandreject=f=300:w=300:width_type=h,bandreject=f=1200:w=600:width_type=h,atempo={1/speed},volume={src_vol}"
                else:
                    af = f"atempo={1/speed},volume={src_vol}"
            
            subprocess.run(["ffmpeg", "-y", "-vsync", "cfr",
                "-i", src, "-ss", str(src_start), "-t", str(src_dur),
                "-vf", vf, "-af", af,
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                str(clip_out)], capture_output=True, timeout=120)
            
            clip_files.append(str(clip_out))
            lines.append(f"file '{clip_out}'\n")
        
        # Phase 2: concat all clips, then mix with BGM
        clip_list.write_text("".join(lines))
        
        # Concat demuxer gives us video (0:v) + audio (0:a)
        # Mix concat audio with BGM via filter_complex
        audio_filter = f"[0:a][1:a]amix=inputs=2:duration=first:weights=1 {10 ** (plan.get('bgm_volume_db', -12.0) / 20)}[aout]"
        
        cmd = ["ffmpeg", "-y",
            "-fflags", "+genpts",
            "-f", "concat", "-safe", "0", "-i", str(clip_list),
            "-i", bgm,
            "-filter_complex", audio_filter,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", str(plan.get("total_duration_sec", 60)),
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise HTTPException(500, f"Render failed: {result.stderr[-500:]}")
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
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
    clips_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    
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
        "-t", str(duration), "-vf", "scale=854:-2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-b:a", "64k",
        str(preview_path)
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)
    
    return FileResponse(preview_path, media_type="video/mp4")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return "<h1>Static files not found</h1>"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(SESSIONS),
        "bgm_sessions": len(BGM_SESSIONS),
    }


# ═══════════════════════════════════════════════════════
#  LLM / VLM integration
# ═══════════════════════════════════════════════════════

import httpx

LLM_CONFIG_PATH = Path.home() / ".conv-edit" / "llm_config.json"
LLM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_LLM_CONFIG = {
    "api_base": "http://localhost:11434/v1",
    "api_key": "",
    "llm_model": "qwen3.6:27b",
    "vision_model": "minicpm-v:8b",
    "enabled": False,
    "max_tokens": 2048,
}


def _load_llm_config() -> dict:
    if LLM_CONFIG_PATH.exists():
        try:
            cfg = json.loads(LLM_CONFIG_PATH.read_text(encoding='utf-8'))
            log.debug(f"config loaded -> enabled={cfg.get('enabled')} llm={cfg.get('llm_model')} vision={cfg.get('vision_model')} base={cfg.get('api_base')}")
            return cfg
        except Exception as e:
            log.warning(f"config load failed: {e}, using defaults")
    log.debug("config -> using defaults (not found or corrupt)")
    return dict(DEFAULT_LLM_CONFIG)


def _save_llm_config(config: dict) -> None:
    LLM_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')


async def _llm_chat(config: dict, messages: list[dict], model: str = None) -> str:
    """Send a chat request to OpenAI-compatible API."""
    m = model or config.get("llm_model", DEFAULT_LLM_CONFIG["llm_model"])
    api_url = f"{config['api_base'].rstrip('/')}/chat/completions"
    msg_count = len(messages)
    img_count = 0
    if messages and isinstance(messages[-1].get("content"), list):
        img_count = sum(1 for c in messages[-1]["content"] if c.get("type") == "image_url")
    
    log.info(f"LLM request -> model={m} url={api_url} msgs={msg_count} images={img_count} timeout=30s")
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {"Content-Type": "application/json"}
            if config.get("api_key"):
                headers["Authorization"] = f"Bearer {config['api_key']}"
            resp = await client.post(
                api_url,
                headers=headers,
                json={
                    "model": m,
                    "messages": messages,
                    "max_tokens": config.get("max_tokens", 2048),
                    "temperature": 0.3,
                },
            )
            if resp.status_code != 200:
                log.error(f"LLM API error -> status={resp.status_code} body={resp.text[:300]}")
                raise HTTPException(502, f"LLM API error: {resp.text[:300]}")
            data = resp.json()
            content_text = data["choices"][0]["message"]["content"]
            log.info(f"LLM response <- model={m} len={len(content_text)} chars")
            return content_text
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"LLM request failed -> model={m} error={type(e).__name__}: {e}")
        raise


async def _vlm_chat(config: dict, messages: list[dict]) -> str:
    """Send a vision chat request."""
    return await _llm_chat(config, messages, model=config.get("vision_model"))


@app.get("/api/llm-config")
async def get_llm_config():
    return _load_llm_config()


@app.post("/api/llm-config")
async def set_llm_config(data: dict):
    config = _load_llm_config()
    for k in ("api_base", "api_key", "llm_model", "vision_model", "enabled", "max_tokens"):
        if k in data:
            config[k] = data[k]
    _save_llm_config(config)
    return {"status": "saved", "config": config}


@app.get("/api/models")
async def list_models():
    """Fetch available models from the configured Ollama/OpenAI API."""
    config = _load_llm_config()
    api_base = config.get("api_base", "").rstrip("/")
    if not api_base:
        return {"models": [], "error": "API 地址未配置"}
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Try Ollama /api/tags first
            if "11434" in api_base or "ollama" in api_base.lower():
                resp = await client.get(f"{api_base.replace('/v1','')}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [{"name": m["name"], "id": m["name"]} for m in data.get("models", [])]
                    return {"models": models, "source": "ollama"}
            
            # Try OpenAI /v1/models
            headers = {"Content-Type": "application/json"}
            if config.get("api_key"):
                headers["Authorization"] = f"Bearer {config['api_key']}"
            resp = await client.get(f"{api_base}/models", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                models = [{"name": m.get("id", m.get("name", "")), "id": m.get("id", "")}
                         for m in data.get("data", [])]
                return {"models": models, "source": "openai"}
            
            return {"models": [], "error": f"API 返回 {resp.status_code}"}
    except Exception as e:
        return {"models": [], "error": str(e)[:100]}


@app.post("/api/test-connection")
async def test_connection(data: dict):
    """Server-side proxy to test LLM API connection (avoids browser CORS)."""
    api_base = data.get("api_base", "").rstrip("/")
    api_key = data.get("api_key", "")
    model = data.get("model", "")
    
    if not api_base:
        raise HTTPException(400, "API 地址未配置")
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json={
                    "model": model or "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "回复OK"}],
                    "max_tokens": 5
                }
            )
            if resp.status_code == 200:
                return {"status": "ok", "model": model}
            else:
                detail = resp.text[:200]
                raise HTTPException(502, f"API 错误 ({resp.status_code}): {detail}")
    except httpx.ConnectError:
        raise HTTPException(502, "无法连接到 API 服务器，请检查地址和网络")
    except httpx.TimeoutException:
        raise HTTPException(502, "连接超时")
    except Exception as e:
        raise HTTPException(502, str(e)[:200])


@app.post("/api/llm-review")
async def llm_review(data: dict):
    """LLM reviews a timeline plan and suggests improvements."""
    config = _load_llm_config()
    if not config.get("enabled"):
        raise HTTPException(400, "LLM is not enabled. Configure it in settings first.")

    plan = data.get("plan", {})
    issues = data.get("issues", [])
    clips_meta = data.get("clips_meta", [])

    # Build a structured prompt
    prompt = _build_review_prompt(plan, issues, clips_meta)

    try:
        result = await _llm_chat(config, [
            {"role": "system", "content": "你是视频剪辑质量审查专家。分析时间线计划，找出问题并给出具体可操作的调整建议。用中文回复，每条建议一行。"},
            {"role": "user", "content": prompt},
        ])
        return {"review": result, "model": config["llm_model"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"LLM review failed: {e}")


@app.post("/api/vision-review")
async def vision_review(data: dict):
    """VLM reviews rendered video frames for visual issues."""
    config = _load_llm_config()
    if not config.get("enabled"):
        raise HTTPException(400, "LLM is not enabled.")

    render_path = data.get("render_path", "")
    plan = data.get("plan", {})

    if not Path(render_path).exists():
        raise HTTPException(404, "Render file not found")

    # Extract key frames from transitions and midpoints
    frames = _extract_review_frames(render_path, plan)
    if not frames:
        raise HTTPException(400, "Could not extract frames")

    # Build vision prompt with frames
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": _build_vision_prompt(plan)},
        ]
    }]
    for fp in frames:
        b64 = base64.b64encode(Path(fp).read_bytes()).decode()
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    try:
        result = await _vlm_chat(config, messages)
        return {"review": result, "model": config["vision_model"], "frames": len(frames)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Vision review failed: {e}")


def _build_review_prompt(plan: dict, issues: list[dict], clips_meta: list[dict]) -> str:
    """Build LLM prompt for timeline review."""
    segs = plan.get("segments", [])
    total = plan.get("total_duration_sec", 0)
    bgm_offset = plan.get("bgm_start_offset_sec", 0)

    lines = [f"## 时间线概览",
             f"- 总时长: {total:.1f}s",
             f"- BGM偏移: {bgm_offset:.1f}s",
             f"- 片段数: {len(segs)}",
             f"- 问题: {len(issues)} 个",
             f""]

    if issues:
        lines.append("## 校验发现的问题")
        for i in issues:
            lines.append(f"- [{i.get('severity','?')}] {i.get('msg','')}")
        lines.append("")

    lines.append("## 片段时间线")
    lines.append("| # | 入点 | 出点 | 时长 | 强度 | 模式 | 标签 | 变速 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(segs):
        spd = f"{s.get('playback_speed',1.0):.2f}x" if s.get('playback_speed', 1.0) != 1.0 else "-"
        lines.append(f"| {i+1} | {s['timeline_start']:.1f}s | {s['timeline_end']:.1f}s | "
                     f"{s['timeline_end']-s['timeline_start']:.1f}s | "
                     f"{s.get('intensity','?')} | {s.get('audio_mode','?')} | "
                     f"{','.join(s.get('tags',[]))} | {spd} |")

    if clips_meta:
        lines.append("\n## 原始素材")
        for c in clips_meta[:10]:
            lines.append(f"- {c.get('clip_id','?')}: {c.get('duration_sec',0):.1f}s "
                         f"强度={c.get('intensity',0)} 模式={c.get('audio_mode','?')} "
                         f"标签={c.get('tags',[])}")

    lines.append("\n## 审查要求")
    lines.append("请检查以下方面：")
    lines.append("1. 是否有过长/过短的片段？")
    lines.append("2. 强度匹配是否合理（高能片段在高能音乐段、平淡在平淡段）？")
    lines.append("3. 间隙或重叠是否影响观感？")
    lines.append("4. 变速片段是否会导致画面异常？")
    lines.append("5. 给出 3 条以内的具体调整建议（改哪个参数、为什么）。")

    return "\n".join(lines)


def _build_vision_prompt(plan: dict) -> str:
    """Build VLM prompt for visual review."""
    segs = plan.get("segments", [])
    return ("你是视频质量审查专家。以下是渲染视频的关键帧截图。请检查："
            "1) 是否有黑场或花屏？2) 画幅是否一致？3) 相邻镜头过渡是否流畅？"
            "4) 画面内容与标注的片段类型（高能/过渡）是否匹配？"
            f"共 {len(segs)} 个片段。用中文回复，每条发现一行。")


def _extract_review_frames(render_path: str, plan: dict, max_frames: int = 6) -> list[str]:
    """Extract key frames from a rendered video at transition points."""
    segs = plan.get("segments", [])
    times = []
    # Capture at segment starts (transitions) and midpoints
    for i, s in enumerate(segs[:max_frames]):
        t = s.get("timeline_start", 0)
        times.append(t + 0.1)  # just after transition
        mid = (s["timeline_start"] + s["timeline_end"]) / 2
        if abs(mid - t) > 1.0:
            times.append(mid)

    frames = []
    for ti in set(round(t, 2) for t in times[:max_frames]):
        frame_path = RENDER_DIR / f"review_frame_{hashlib.md5(str(ti).encode()).hexdigest()[:8]}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ti), "-i", render_path,
             "-vframes", "1", "-q:v", "5", str(frame_path)],
            capture_output=True, timeout=30
        )
        if frame_path.exists():
            frames.append(str(frame_path))

    return frames


# ═══════════════════════════════════════════════════════
#  Editing presets
# ═══════════════════════════════════════════════════════

EDITING_PRESETS = {
    "gaming": {
        "name": "🎮 游戏集锦",
        "desc": "快节奏击杀/精彩操作合集",
        "strategy": "fit",
        "intensity_match_strength": 0.9,
        "scale_mode": "crop",
        "prefer_short": True,
        "audio_bias": "纯BGM",
        "transition": "hard",
    },
    "film": {
        "name": "🎬 电影对白",
        "desc": "保护人声，长镜头为主",
        "strategy": "fit",
        "intensity_match_strength": 0.3,
        "scale_mode": "letterbox",
        "prefer_short": False,
        "audio_bias": "突出人声",
        "transition": "fade",
    },
    "vlog": {
        "name": "🏔️ 旅行Vlog",
        "desc": "中等节奏，穿插安静段",
        "strategy": "fit",
        "intensity_match_strength": 0.5,
        "scale_mode": "letterbox",
        "prefer_short": False,
        "audio_bias": "融入BGM",
        "transition": "fade",
    },
    "mv": {
        "name": "🎵 MV踩点混剪",
        "desc": "极短切，强制节拍对齐",
        "strategy": "loop",
        "intensity_match_strength": 0.8,
        "scale_mode": "crop",
        "prefer_short": True,
        "audio_bias": "纯BGM",
        "transition": "hard",
    },
    "narrative": {
        "name": "📖 叙事短片",
        "desc": "按顺序排列，保护结尾",
        "strategy": "fit",
        "intensity_match_strength": 0.2,
        "scale_mode": "letterbox",
        "prefer_short": False,
        "audio_bias": "融入BGM",
        "transition": "fade",
    },
    "smart": {
        "name": "🤖 智能推荐",
        "desc": "LLM分析素材后自动选择风格",
        "strategy": "fit",
        "intensity_match_strength": 0.6,
        "scale_mode": "letterbox",
        "prefer_short": None,
        "audio_bias": None,
        "transition": "auto",
    },
}


@app.get("/api/presets")
async def get_presets():
    return {"presets": EDITING_PRESETS}


@app.post("/api/auto-select")
async def auto_select(data: dict):
    """LLM/VLM auto-selects clips + recommends editing style.
    If bgm_id is provided, BGM characteristics influence selection."""
    config = _load_llm_config()
    if not config.get("enabled"):
        raise HTTPException(400, "LLM not enabled")

    session_id = data.get("session_id")
    bgm_id = data.get("bgm_id")            # optional — if present, BGM-aware selection
    style_hint = data.get("style_hint", "")
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")

    session = SESSIONS[session_id]
    scenes = session["scenes"]
    if not scenes:
        raise HTTPException(400, "No scenes to review")

    # Build BGM context if available
    bgm_info = None
    if bgm_id and bgm_id in BGM_SESSIONS:
        bgm = BGM_SESSIONS[bgm_id]
        bgm_info = {
            "bpm": bgm.get("bpm", 120),
            "duration_sec": bgm.get("duration_sec", 60),
            "segments": [{"label": s["label"], "start": s["start_sec"],
                          "end": s["end_sec"], "energy": s.get("energy", 0.5)}
                         for s in bgm.get("segments", [])],
        }

    review_scenes = scenes[:20]
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": _build_autoselect_prompt(review_scenes, style_hint, bgm_info)},
        ]
    }]

    # Attach up to 8 thumbnails (skip large ones to avoid API errors)
    thumb_count = 0
    for s in review_scenes[:8]:
        thumb = s.get("thumbnail_b64", "")
        # Skip thumbnails > 150KB (too large for some APIs)
        if thumb and thumb.startswith("data:image") and len(thumb) < 150000:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": thumb}
            })
            thumb_count += 1

    # Try vision model if thumbs available, otherwise skip straight to text
    result = None
    last_error = None
    
    vis_model = config.get("vision_model", config["llm_model"])
    txt_model = config.get("llm_model", DEFAULT_LLM_CONFIG["llm_model"])
    log.info(f"auto-select -> scenes={len(review_scenes)} thumbnails={thumb_count} vision={vis_model} text={txt_model}")
    
    if thumb_count > 0:
        try:
            log.info(f"auto-select -> trying vision model: {vis_model}")
            result = await _llm_chat(config, messages, model=vis_model)
            log.info(f"auto-select <- vision model OK: {vis_model}")
        except Exception as e:
            last_error = str(e)[:100]
            log.warning(f"auto-select -> vision model failed ({vis_model}): {last_error}")
    
    # Text model fallback (or primary if no thumbs)
    if result is None:
        try:
            log.info(f"auto-select -> trying text model: {txt_model}")
            if len(messages[0]["content"]) > 1:
                messages[0]["content"] = [messages[0]["content"][0]]
            result = await _llm_chat(config, messages)
            log.info(f"auto-select <- text model OK: {txt_model}")
        except Exception as e2:
            last_error = last_error or str(e2)[:100]
            log.warning(f"auto-select -> text model also failed ({txt_model}): {last_error}")
            log.info("auto-select -> falling back to rule-based selection")
            parsed = _rule_based_select(review_scenes, style_hint, bgm_info)
            parsed["reasoning"] = f"规则选片（模型暂时不可用：{last_error}），按风格选中 {len(parsed['selected_indices'])} 个片段"
            return _apply_autoselect(session, scenes, parsed)

    log.info(f"auto-select -> parsing LLM response ({len(result)} chars)")
    parsed = _parse_autoselect_result(result, review_scenes)
    log.info(f"auto-select -> done: preset={parsed.get('preset')} selected={len(parsed['selected_indices'])}")
    return _apply_autoselect(session, scenes, parsed)


@app.post("/api/auto-select-stream")
async def auto_select_stream(data: dict):
    """SSE-streaming version of auto-select with real-time progress updates."""
    from fastapi.responses import StreamingResponse
    import asyncio
    
    config = _load_llm_config()
    
    async def event_stream():
        session_id = data.get("session_id")
        bgm_id = data.get("bgm_id")
        style_hint = data.get("style_hint", "")
        
        log.info(f"auto-select-stream -> start session={session_id} bgm={bgm_id}")
        
        if session_id not in SESSIONS:
            log.error(f"auto-select-stream -> session not found: {session_id}")
            yield f"data: {json.dumps({'step': 'error', 'msg': 'Session not found'})}\n\n"
            return
        
        session = SESSIONS[session_id]
        scenes = session["scenes"]
        
        # Build BGM context
        bgm_info = None
        if bgm_id and bgm_id in BGM_SESSIONS:
            bgm = BGM_SESSIONS[bgm_id]
            bgm_info = {"bpm": bgm.get("bpm", 120), "duration_sec": bgm.get("duration_sec", 60),
                        "segments": [{"label": s["label"], "start": s["start_sec"],
                                      "end": s["end_sec"], "energy": s.get("energy", 0.5)}
                                     for s in bgm.get("segments", [])]}
        
        review_scenes = scenes[:20]
        
        # Build messages
        messages = [{"role": "user", "content": [
            {"type": "text", "text": _build_autoselect_prompt(review_scenes, style_hint, bgm_info)},
        ]}]
        
        thumb_count = 0
        for s in review_scenes[:8]:
            thumb = s.get("thumbnail_b64", "")
            if thumb and thumb.startswith("data:image") and len(thumb) < 150000:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": thumb}})
                thumb_count += 1
        
        model_name = config.get("vision_model", config.get("llm_model", "unknown"))
        yield f"data: {json.dumps({'step': 'progress', 'msg': f'🔗 正在连接 {model_name}（{len(review_scenes)} 个镜头，{thumb_count} 张缩略图）...', 'pct': 10})}\n\n"
        await asyncio.sleep(0.1)
        
        # Try vision model
        result = None
        try:
            yield f"data: {json.dumps({'step': 'progress', 'msg': '⏳ 已发送请求，等待视觉模型回复...', 'pct': 25})}\n\n"
            result = await _llm_chat(config, messages, model=model_name)
            log.info(f"auto-select-stream <- vision model OK: {model_name}")
        except Exception as e1:
            log.warning(f"auto-select-stream -> vision model failed ({model_name}): {e1}")
            yield f"data: {json.dumps({'step': 'progress', 'msg': f'⚠️ 视觉模型无响应，降级到纯文本模型...', 'pct': 30})}\n\n"
            await asyncio.sleep(0.1)
            try:
                if len(messages[0]["content"]) > 1:
                    messages[0]["content"] = [messages[0]["content"][0]]
                txt_model = config.get("llm_model", "unknown")
                yield f"data: {json.dumps({'step': 'progress', 'msg': f'⏳ 正在请求文本模型 ({txt_model})...', 'pct': 35})}\n\n"
                result = await _llm_chat(config, messages)
                log.info(f"auto-select-stream <- text model OK: {txt_model}")
            except Exception as e2:
                log.warning(f"auto-select-stream -> text model also failed ({txt_model}): {e2}")
                log.info("auto-select-stream -> falling back to rule-based")
                yield f"data: {json.dumps({'step': 'progress', 'msg': '⚠️ 文本模型也无响应，使用规则选片...', 'pct': 40})}\n\n"
                await asyncio.sleep(0.1)
                parsed = _rule_based_select(review_scenes, style_hint, bgm_info)
                final = _apply_autoselect(session, scenes, parsed)
                yield f"data: {json.dumps({'step': 'progress', 'msg': '📋 规则选片完成', 'pct': 80})}\n\n"
                await asyncio.sleep(0.1)
                yield f"data: {json.dumps({'step': 'done', 'result': final})}\n\n"
                return
        
        yield f"data: {json.dumps({'step': 'progress', 'msg': '📋 正在解析 LLM 回复...', 'pct': 70})}\n\n"
        await asyncio.sleep(0.1)
        
        parsed = _parse_autoselect_result(result, review_scenes)
        final = _apply_autoselect(session, scenes, parsed)
        
        selected_n = len(parsed["selected_indices"])
        yield f"data: {json.dumps({'step': 'progress', 'msg': f'✅ 选中 {selected_n} 个片段', 'pct': 95})}\n\n"
        await asyncio.sleep(0.1)
        yield f"data: {json.dumps({'step': 'done', 'result': final})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _apply_autoselect(session: dict, all_scenes: list[dict], parsed: dict) -> dict:
    """Apply auto-select results to session scenes."""
    for s in all_scenes:
        s["selected"] = s["index"] in parsed["selected_indices"]
        if s["index"] in parsed.get("intensity_overrides", {}):
            s["intensity"] = parsed["intensity_overrides"][s["index"]]
            s["intensity_auto"] = False
        if s["index"] in parsed.get("mode_overrides", {}):
            s["audio_mode"] = parsed["mode_overrides"][s["index"]]
            s["audio_mode_auto"] = False
    
    return {
        "recommended_preset": parsed.get("preset", "smart"),
        "preset_name": EDITING_PRESETS.get(parsed.get("preset", "smart"), {}).get("name", "智能"),
        "selected_count": len(parsed["selected_indices"]),
        "total_scenes": len(all_scenes),
        "intensity_overrides": len(parsed.get("intensity_overrides", {})),
        "reasoning": parsed.get("reasoning", ""),
        "scenes": all_scenes,
    }


def _rule_based_select(scenes: list[dict], style_hint: str, bgm_info: dict = None) -> dict:
    """Rule-based clip selection when LLM is unavailable."""
    indices = []
    intensity_ov = {}
    mode_ov = {}
    
    # Determine strategy from style hint
    hint_lower = (style_hint or "").lower()
    if any(k in hint_lower for k in ["游戏", "集锦", "gaming", "击杀"]):
        preset = "gaming"
        for s in scenes:
            if s.get("intensity", 0) > 0.6 and s["duration_sec"] <= 5:
                indices.append(s["index"])
    elif any(k in hint_lower for k in ["电影", "对白", "film"]):
        preset = "film"
        for s in scenes:
            if s["duration_sec"] >= 2 and s["duration_sec"] <= 12:
                indices.append(s["index"])
    elif any(k in hint_lower for k in ["mv", "踩点", "音乐"]):
        preset = "mv"
        indices = [s["index"] for s in scenes]  # All clips
        for s in scenes:
            intensity_ov[s["index"]] = max(0.7, s.get("intensity", 0.5))
    else:
        preset = "smart"
        # Smart: select by intensity, prefer variety
        high = [s for s in scenes if s.get("intensity", 0) > 0.6]
        low = [s for s in scenes if s.get("intensity", 0) <= 0.6]
        # Alternate high and low for variety
        for i in range(max(len(high), len(low))):
            if i < len(high):
                indices.append(high[i]["index"])
            if i < len(low):
                indices.append(low[i]["index"])
    
    if not indices:
        indices = [s["index"] for s in scenes[:10]]  # First 10 as fallback
    
    return {
        "preset": preset,
        "reasoning": f"规则选片（自动降级）：按风格 '{style_hint or '智能'}' 选中 {len(indices)} 个片段",
        "selected_indices": indices,
        "intensity_overrides": intensity_ov,
        "mode_overrides": mode_ov,
    }


def _build_autoselect_prompt(scenes: list[dict], style_hint: str, bgm_info: dict = None) -> str:
    """Build prompt for LLM auto-selection, optionally BGM-aware."""
    style_text = f"用户想要的风格: {style_hint}" if style_hint else "请根据素材内容自动判断最合适的剪辑风格"

    lines = [f"{style_text}", f""]

    # BGM section
    if bgm_info:
        lines += [
            "## 🎵 背景音乐分析",
            f"- BPM: {bgm_info['bpm']:.0f}",
            f"- 时长: {bgm_info['duration_sec']:.0f}s",
            f"- 结构段落:",
        ]
        for seg in bgm_info["segments"]:
            lines.append(f"  {seg['label']}: {seg['start']:.0f}s–{seg['end']:.0f}s (能量={seg['energy']:.2f})")
        lines += [
            "",
            "**选片时请考虑 BGM 特征：**",
            f"- BPM {bgm_info['bpm']:.0f} → {'快节奏，优先短切(1-3s)' if bgm_info['bpm'] > 120 else '舒缓，可选中长镜头(4-10s)' if bgm_info['bpm'] < 90 else '中等节奏'}",
            "- 高能量段落(drop)需要足够多的高强度片段",
            "- intro/outro 段配低强度过渡片段",
            "- 确保总素材时长能覆盖 BGM 的关键段落",
            "",
        ]

    lines += [
        f"以下是 {len(scenes)} 个镜头片段的信息：",
        "",
        "| # | 时长 | 自动强度 | 音频模式 | 标签 |",
        "|---|---|---|---|---|",
    ]
    for s in scenes:
        lines.append(
            f"| {s['index']} | {s['duration_sec']:.1f}s | "
            f"{s.get('intensity', 0.5):.1f} | {s.get('audio_mode', '?')} | "
            f"{','.join(s.get('tags', []))} |"
        )

    lines += [
        "",
        "请完成以下任务，用JSON格式回复（不要markdown代码块，纯JSON）：",
        "{",
        '  "preset": "gaming|film|vlog|mv|narrative",',
        '  "reasoning": "为什么选这个风格（一句话）",',
        '  "selected_indices": [0, 2, 5, ...],',
        '  "intensity_overrides": {"3": 0.9, "7": 0.2},',
        '  "mode_overrides": {"1": "突出人声", "4": "纯BGM"}',
        "}",
        "",
        "选片原则：",
        "- 游戏类：优先选高强度(>0.7)、短片段(<4s)，跳过纯过渡",
        "- 电影类：优先选有对话的(突出人声)、中等长度(3-10s)",
        "- Vlog类：均匀选择，保留高低强度交替",
        "- MV类：全选，强度统一拉高",
        "- 叙事类：保留原顺序，不跳片段",
        "- selected_indices 列出所有应该保留的片段索引",
        "- intensity_overrides 只写需要修正的（自动检测不准的）",
        "- mode_overrides 只写需要修正的",
        "",
        "直接返回JSON：",
    ]
    return "\n".join(lines)


def _parse_autoselect_result(result: str, scenes: list[dict]) -> dict:
    """Parse LLM response into structured data."""
    import re
    # Extract JSON from response
    json_match = re.search(r'\{[\s\S]*\}', result)
    if not json_match:
        # Fallback: select all
        return {
            "preset": "smart",
            "reasoning": "无法解析LLM响应，默认全选",
            "selected_indices": [s["index"] for s in scenes],
            "intensity_overrides": {},
            "mode_overrides": {},
        }
    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {
            "preset": "smart",
            "reasoning": "JSON解析失败，默认全选",
            "selected_indices": [s["index"] for s in scenes],
            "intensity_overrides": {},
            "mode_overrides": {},
        }

    # Convert string keys to int
    intensity_overrides = {}
    for k, v in data.get("intensity_overrides", {}).items():
        try:
            intensity_overrides[int(k)] = float(v)
        except (ValueError, TypeError):
            pass

    mode_overrides = {}
    for k, v in data.get("mode_overrides", {}).items():
        try:
            mode_overrides[int(k)] = str(v)
        except (ValueError, TypeError):
            pass

    return {
        "preset": data.get("preset", "smart"),
        "reasoning": data.get("reasoning", ""),
        "selected_indices": [int(i) for i in data.get("selected_indices", []) if str(i).isdigit()],
        "intensity_overrides": intensity_overrides,
        "mode_overrides": mode_overrides,
    }


# ═══════════════════════════════════════════════════════
#  Chat-guided semi-auto workflow
# ═══════════════════════════════════════════════════════

CHAT_SESSIONS: dict[str, list[dict]] = {}  # session_id → message history


def _build_chat_system_prompt(scenes: list[dict], bgm_info: dict = None) -> str:
    """Build system prompt that guides the LLM to act as a video editing assistant."""
    scene_count = len(scenes)
    selected = sum(1 for s in scenes if s.get("selected", True))
    total_dur = sum(s["duration_sec"] for s in scenes if s.get("selected", True))

    # Summarize scenes
    intensities = [s.get("intensity", 0.5) for s in scenes]
    modes = {}
    for s in scenes:
        m = s.get("audio_mode", "融入BGM")
        modes[m] = modes.get(m, 0) + 1
    tags_all = set()
    for s in scenes:
        for t in s.get("tags", []):
            tags_all.add(t)

    prompt = f"""你是视频剪辑助手。用户上传了一段视频，你的任务是引导用户完成剪辑。

## 当前素材
- 总镜头: {scene_count} 个
- 已选: {selected} 个（总时长 {total_dur:.0f}s）
- 强度范围: {min(intensities):.1f}～{max(intensities):.1f}
- 音频模式分布: {modes}
- 标签: {', '.join(tags_all) if tags_all else '无'}"""

    if bgm_info:
        prompt += f"""

## BGM
- BPM: {bgm_info['bpm']:.0f}
- 时长: {bgm_info['duration_sec']:.0f}s
- 段落: {', '.join(f"{s['label']}({s['start']:.0f}～{s['end']:.0f}s)" for s in bgm_info['segments'])}"""

    prompt += f"""

## 你的职责
1. 先问用户这是什么类型的视频、想要什么风格
2. 推荐剪辑风格预设 + 执行智能选片 [ACTION:auto-select]
3. 等用户确认或提出修改
4. 用户满意后执行生成 [ACTION:generate-plan]（需先有BGM）
5. 用户要渲染时执行 [ACTION:render]

## 可用命令（放回复末尾，一行一个）
[ACTION:set-preset gaming|film|vlog|mv|narrative]
[ACTION:auto-select]
[ACTION:generate-plan]
[ACTION:render]

## 规则
- 回复简洁，2-4句话，用中文
- 用户说"行/可以/OK"就是确认，继续下一步
- 用户说"全自动/一键出片/帮我自动规划/全部交给你"时：
  直接在一条回复里依次输出 set-preset → auto-select → generate-plan → render，不追问"""

    return prompt


@app.post("/api/chat")
async def chat(data: dict):
    """Chat endpoint for guided semi-auto editing workflow."""
    config = _load_llm_config()
    if not config.get("enabled"):
        raise HTTPException(400, "LLM not enabled")

    session_id = data.get("session_id")
    user_msg = data.get("message", "").strip()
    bgm_id = data.get("bgm_id")

    if session_id not in SESSIONS:
        raise HTTPException(404, "Video session not found")

    session = SESSIONS[session_id]
    scenes = session["scenes"]

    # Build BGM context
    bgm_info = None
    if bgm_id and bgm_id in BGM_SESSIONS:
        bgm = BGM_SESSIONS[bgm_id]
        bgm_info = {
            "bpm": bgm.get("bpm", 120),
            "duration_sec": bgm.get("duration_sec", 60),
            "segments": [{"label": s["label"], "start": s["start_sec"],
                          "end": s["end_sec"], "energy": s.get("energy", 0.5)}
                         for s in bgm.get("segments", [])],
        }

    # Initialize or load chat history
    chat_id = f"chat_{session_id}"
    if chat_id not in CHAT_SESSIONS or data.get("reset"):
        CHAT_SESSIONS[chat_id] = [
            {"role": "system", "content": _build_chat_system_prompt(scenes, bgm_info)},
            {"role": "assistant", "content": _build_first_message(scenes, bgm_info)},
        ]

    history = CHAT_SESSIONS[chat_id]

    # Update context in system prompt (scenes may have changed)
    history[0]["content"] = _build_chat_system_prompt(scenes, bgm_info)

    if user_msg:
        history.append({"role": "user", "content": user_msg})

    try:
        log.info(f"chat -> session={session_id} msg_len={len(user_msg)} history={len(history)}")
        result = await _llm_chat(config, history[-20:])  # last 20 messages for context
        actions = _parse_chat_actions(result)
        log.info(f"chat <- reply={len(result)} chars actions={[a['action'] for a in actions]}")
        history.append({"role": "assistant", "content": result})
        CHAT_SESSIONS[chat_id] = history

        return {
            "reply": result,
            "actions": actions,
            "history_length": len(history),
        }
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")


def _build_first_message(scenes: list[dict], bgm_info: dict = None) -> str:
    """Generate the assistant's first message."""
    scene_count = len(scenes)
    selected = sum(1 for s in scenes if s.get("selected", True))
    total_dur = sum(s["duration_sec"] for s in scenes if s.get("selected", True))
    bgm_line = f"，BGM {bgm_info['bpm']:.0f}BPM" if bgm_info else ""

    return (f"你好！我看到了你的视频——{scene_count} 个镜头{bgm_line}，"
            f"当前已选 {selected} 个（{total_dur:.0f}s）。"
            f"\n\n先告诉我：这是什么内容的视频？你想做成什么风格？比如游戏击杀集锦、旅行Vlog、电影剪辑……")


def _parse_chat_actions(reply: str) -> list[dict]:
    """Extract action commands from LLM reply."""
    import re
    actions = []
    for m in re.finditer(r'\[ACTION:([^\]]+)\]', reply):
        cmd = m.group(1).strip()
        if cmd == "auto-select":
            actions.append({"action": "auto-select"})
        elif cmd == "generate-plan":
            actions.append({"action": "generate-plan"})
        elif cmd == "render":
            actions.append({"action": "render"})
        elif cmd.startswith("set-preset "):
            preset = cmd.split(" ", 1)[1].strip()
            actions.append({"action": "set-preset", "preset": preset})
    return actions


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
