"""
conv-edit pytest fixtures.


Generates test media + mocks LLM so tests run without remote API.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from server import app, SESSIONS, BGM_SESSIONS, OUTPUT_DIR, RENDER_DIR

TEST_DIR = Path(__file__).parent / ".test_media"
TEST_DIR.mkdir(exist_ok=True)


# ─── Mock LLM responses ───────────────────────────────

MOCK_AUTOSELECT_JSON = json.dumps({
    "preset": "gaming",
    "reasoning": "素材强度高、片段短，适合游戏集锦风格",
    "selected_indices": [0, 2, 4],
    "intensity_overrides": {},
    "mode_overrides": {},
})

MOCK_CHAT_REPLY = (
    "我看了你的素材，推荐游戏集锦风格。[ACTION:set-preset gaming]\n"
    "[ACTION:auto-select]"
)

MOCK_LLM_REVIEW = "1. 片段3过长建议切分\n2. 第二个过渡段强度偏低"


def mock_llm_response(text: str = MOCK_AUTOSELECT_JSON) -> AsyncMock:
    """Return an AsyncMock that behaves like _llm_chat's return."""
    async def _mock(*args, **kwargs):
        return text
    return _mock


# ─── Media generators ─────────────────────────────────

def generate_test_video(path: Path, duration: float = 5.0) -> Path:
    """Generate a small test video with color bars + sine wave audio."""
    if path.exists():
        return path
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-c:a", "aac", "-b:a", "64k",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return path


def generate_test_bgm(path: Path, duration: float = 10.0) -> Path:
    """Generate a test BGM WAV file (PCM for reliable librosa loading)."""
    if path.exists():
        return path
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:a", "pcm_s16le", "-ar", "22050",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return path


# ─── Fixtures ──────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video():
    """Session-scoped: generate once, reuse across tests."""
    p = TEST_DIR / "test_video.mp4"
    generate_test_video(p)
    assert p.exists(), f"Failed to generate {p}"
    return p


@pytest.fixture(scope="session")
def test_bgm():
    """Session-scoped: generate once, reuse across tests."""
    p = TEST_DIR / "test_bgm.wav"
    generate_test_bgm(p)
    assert p.exists(), f"Failed to generate {p}"
    return p


@pytest_asyncio.fixture
async def client():
    """Async HTTP client connected to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def session_with_video(client, test_video):
    """Upload a video and return (client, session_data)."""
    with open(test_video, "rb") as f:
        resp = await client.post(
            "/api/analyze",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"threshold": "0.3"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["scene_count"] > 0
    return client, data


@pytest_asyncio.fixture
async def session_with_bgm(client, test_bgm):
    """Upload a BGM and return (client, bgm_data)."""
    with open(test_bgm, "rb") as f:
        resp = await client.post(
            "/api/analyze-bgm",
            files={"file": ("test.wav", f, "audio/wav")},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    return client, data


@pytest_asyncio.fixture
async def full_session(client, test_video, test_bgm):
    """Upload both video and BGM, return (client, session_id, bgm_id)."""
    # Upload video
    with open(test_video, "rb") as f:
        r = await client.post(
            "/api/analyze",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"threshold": "0.3"},
        )
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # Upload BGM
    with open(test_bgm, "rb") as f:
        r = await client.post(
            "/api/analyze-bgm",
            files={"file": ("test.wav", f, "audio/wav")},
        )
    assert r.status_code == 200
    bid = r.json()["bgm_id"]

    # Enable LLM (mock will override)
    await client.post("/api/llm-config", json={
        "enabled": True,
        "api_base": "http://mock:11434/v1",
        "llm_model": "test-model",
        "vision_model": "test-vision",
    })

    return client, sid, bid


# ─── Helpers ───────────────────────────────────────────

def assert_valid_scene(scene: dict):
    """Check a scene dict has all required fields with valid values."""
    assert isinstance(scene["index"], int)
    assert scene["start_sec"] >= 0
    assert scene["end_sec"] > scene["start_sec"]
    assert scene["duration_sec"] > 0
    assert isinstance(scene["intensity"], (int, float))
    assert 0 <= scene["intensity"] <= 1
    assert scene["audio_mode"] in ("融入BGM", "突出人声", "纯BGM")


def assert_valid_plan(plan: dict):
    """Check a plan dict has valid structure."""
    assert "segments" in plan
    assert len(plan["segments"]) > 0
    assert plan["total_duration_sec"] > 0
    for seg in plan["segments"]:
        assert seg["clip_id"]
        assert seg["source_duration"] > 0
        assert seg["timeline_end"] > seg["timeline_start"]





# ─── Cleanup ──────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """Clean up test media after all tests."""
    import shutil
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)
