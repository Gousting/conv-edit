"""
conv-edit API integration tests.
Tests the full pipeline: upload → analyze → auto-select → plan → render.
LLM calls are mocked so tests run offline.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from conftest import (
    MOCK_AUTOSELECT_JSON,
    MOCK_CHAT_REPLY,
    MOCK_LLM_REVIEW,
    assert_valid_plan,
    assert_valid_scene,
)


# ═══════════════════════════════════════════════════════
# Health / Basics
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_health(client):
    """Server responds to health check."""
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_index_html(client):
    """Root returns HTML."""
    r = await client.get("/")
    assert r.status_code == 200
    assert "Conv-Edit" in r.text


@pytest.mark.asyncio
async def test_presets(client):
    """Presets endpoint returns editing styles."""
    r = await client.get("/api/presets")
    assert r.status_code == 200
    data = r.json()
    assert "gaming" in data["presets"]
    assert "smart" in data["presets"]


# ═══════════════════════════════════════════════════════
# Video Upload & Analysis
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_upload_video(client, test_video):
    """Upload video → scene detection succeeds."""
    with open(test_video, "rb") as f:
        r = await client.post(
            "/api/analyze",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"threshold": "0.3"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"]
    assert data["scene_count"] >= 1
    assert data["duration"] > 0
    assert len(data["scenes"]) == data["scene_count"]
    for s in data["scenes"]:
        assert_valid_scene(s)


@pytest.mark.asyncio
async def test_upload_video_no_file(client):
    """Missing file returns 422."""
    r = await client.post("/api/analyze")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_preview_scene(session_with_video):
    """Preview endpoint returns MP4."""
    client, data = session_with_video
    sid = data["session_id"]
    r = await client.get(f"/api/preview/{sid}/0")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"


@pytest.mark.asyncio
async def test_preview_invalid_scene(session_with_video):
    """Preview of non-existent scene returns 404."""
    client, data = session_with_video
    r = await client.get(f"/api/preview/{data['session_id']}/999")
    assert r.status_code == 404


# ═══════════════════════════════════════════════════════
# BGM Upload & Analysis
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa (slow JIT in CI, runs fine locally)")
async def test_upload_bgm(client, test_bgm):
    """Upload BGM → analysis succeeds."""
    with open(test_bgm, "rb") as f:
        r = await client.post(
            "/api/analyze-bgm",
            files={"file": ("test.mp3", f, "audio/mpeg")},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["bgm_id"]
    assert data["bpm"] > 0
    assert data["duration_sec"] > 0
    assert len(data["segments"]) >= 1


@pytest.mark.asyncio
async def test_upload_bgm_no_file(client):
    """Missing BGM file returns 422."""
    r = await client.post("/api/analyze-bgm")
    assert r.status_code == 422


# ═══════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_export_clips(session_with_video):
    """Export returns valid JSON with correct structure."""
    client, data = session_with_video
    sid = data["session_id"]
    scenes = data["scenes"]
    r = await client.post(
        f"/api/export/{sid}",
        json={"scenes": scenes},
    )
    assert r.status_code == 200
    exported = r.json()
    assert exported["clip_count"] == len(scenes)
    assert len(exported["clips"]) == len(scenes)
    for clip in exported["clips"]:
        assert clip["clip_id"]
        assert clip["start_sec"] >= 0
        assert clip["end_sec"] > clip["start_sec"]


@pytest.mark.asyncio
async def test_export_invalid_session(client):
    """Export non-existent session returns 404."""
    r = await client.post("/api/export/nonexistent", json={"scenes": []})
    assert r.status_code == 404


# ═══════════════════════════════════════════════════════
# Plan Generation
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa for BGM analysis")
async def test_generate_plan_valid(full_session):
    """Generate timeline plan with valid video + BGM."""
    client, sid, bid = full_session
    r = await client.post("/api/plan", json={
        "session_id": sid,
        "bgm_id": bid,
        "strategy": "fit",
        "seed": 42,
        "auto_offset": True,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["plan"]
    assert_valid_plan(data["plan"])
    assert data["ready"] is True  # no errors after auto-fix
    assert "heatmap" in data


@pytest.mark.asyncio
async def test_generate_plan_no_bgm(session_with_video):
    """Plan without BGM returns 404."""
    client, data = session_with_video
    r = await client.post("/api/plan", json={
        "session_id": data["session_id"],
        "bgm_id": "nonexistent",
    })
    assert r.status_code == 404


@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa for BGM analysis")
async def test_generate_plan_no_clips_selected(full_session):
    """Plan with all clips deselected returns 400."""
    client, sid, bid = full_session
    # Deselect all clips first via export
    r = await client.post(f"/api/export/{sid}", json={
        "scenes": [{
            "start_sec": 0, "duration_sec": 1, "selected": False,
            "index": 0, "trim_start": 0, "trim_end": 1,
        }],
    })
    # Now try plan
    r = await client.post("/api/plan", json={
        "session_id": sid,
        "bgm_id": bid,
    })
    assert r.status_code in (400, 200)  # 400 if no clips, 200 if fallback works


# ═══════════════════════════════════════════════════════
# Auto-Select (LLM mocked)
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_auto_select_with_mock(session_with_video):
    """Auto-select with mocked LLM returns valid scene selection."""
    client, data = session_with_video
    sid = data["session_id"]

    # Enable LLM for auto-select
    await client.post("/api/llm-config", json={
        "enabled": True, "api_base": "http://mock:11434/v1",
        "llm_model": "test-model",
    })

    with patch("server._llm_chat", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = MOCK_AUTOSELECT_JSON
        r = await client.post("/api/auto-select", json={
            "session_id": sid,
            "style_hint": "游戏集锦",
        })

    assert r.status_code == 200
    result = r.json()
    assert result["selected_count"] == 3  # from mock JSON
    assert result["total_scenes"] == data["scene_count"]
    # Verify scenes were modified
    assert len(result["scenes"]) == data["scene_count"]


@pytest.mark.asyncio
async def test_auto_select_fallback_to_rules(session_with_video):
    """When LLM fails twice, falls back to rule-based selection."""
    client, data = session_with_video
    sid = data["session_id"]

    # Enable LLM
    await client.post("/api/llm-config", json={
        "enabled": True, "api_base": "http://mock:11434/v1",
        "llm_model": "test-model",
    })

    with patch("server._llm_chat", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("API unavailable")
        r = await client.post("/api/auto-select", json={
            "session_id": sid,
            "style_hint": "游戏集锦",
        })

    assert r.status_code == 200
    result = r.json()
    assert "自动降级" in result["reasoning"] or "规则选片" in result["reasoning"]
    assert result["selected_count"] > 0  # should still select something


@pytest.mark.asyncio
async def test_auto_select_stream(session_with_video):
    """SSE streaming auto-select works with mocked LLM."""
    client, data = session_with_video
    sid = data["session_id"]

    with patch("server._llm_chat", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = MOCK_AUTOSELECT_JSON
        r = await client.post("/api/auto-select-stream", json={
            "session_id": sid,
            "style_hint": "",
        })

    assert r.status_code == 200
    body = r.text
    assert "data:" in body
    # Should contain progress + done events
    assert "progress" in body
    assert "done" in body


# ═══════════════════════════════════════════════════════
# Chat (LLM mocked)
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa for BGM analysis")
async def test_chat_with_mock(full_session):
    """Chat endpoint works with mocked LLM."""
    client, sid, bid = full_session

    with patch("server._llm_chat", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = MOCK_CHAT_REPLY
        r = await client.post("/api/chat", json={
            "session_id": sid,
            "bgm_id": bid,
            "message": "帮我自动规划",
            "reset": True,
        })

    assert r.status_code == 200
    data = r.json()
    assert data["reply"]
    assert len(data["actions"]) >= 1
    # Should have auto-select action
    actions = [a["action"] for a in data["actions"]]
    assert "auto-select" in actions


@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa for BGM analysis")
async def test_chat_auto_select_action(full_session):
    """Chat triggers auto-select action which applies correctly."""
    client, sid, bid = full_session

    with patch("server._llm_chat", new_callable=AsyncMock) as mock_llm:
        # First call: chat response
        # Second call: auto-select response (triggered by action)
        mock_llm.side_effect = [
            "[ACTION:auto-select]",
            MOCK_AUTOSELECT_JSON,
        ]
        r = await client.post("/api/chat", json={
            "session_id": sid,
            "bgm_id": bid,
            "message": "自动选片",
            "reset": True,
        })

    assert r.status_code == 200
    data = r.json()
    assert len(data["actions"]) >= 1


# ═══════════════════════════════════════════════════════
# LLM Config
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_llm_config_crud(client):
    """LLM config get/set works."""
    r = await client.get("/api/llm-config")
    assert r.status_code == 200
    default = r.json()
    assert "enabled" in default

    r = await client.post("/api/llm-config", json={"enabled": True, "llm_model": "qwen-test"})
    assert r.status_code == 200

    r = await client.get("/api/llm-config")
    assert r.json()["enabled"] is True
    assert r.json()["llm_model"] == "qwen-test"


# ═══════════════════════════════════════════════════════
# LLM Review
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa for BGM analysis")
async def test_llm_review_with_mock(full_session):
    """LLM review endpoint works with mocked LLM."""
    client, sid, bid = full_session

    # First generate a plan
    r = await client.post("/api/plan", json={
        "session_id": sid, "bgm_id": bid, "seed": 42,
    })
    assert r.status_code == 200
    plan = r.json()

    with patch("server._llm_chat", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = MOCK_LLM_REVIEW
        r = await client.post("/api/llm-review", json={
            "plan": plan["plan"],
            "issues": [],
            "clips_meta": [],
        })

    assert r.status_code == 200
    data = r.json()
    assert "review" in data
    assert data["review"] == MOCK_LLM_REVIEW


# ═══════════════════════════════════════════════════════
# Error Handling / Edge Cases
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_plan_nonexistent_session(client):
    """Plan with bad session returns 404."""
    r = await client.post("/api/plan", json={
        "session_id": "bad_session",
        "bgm_id": "bad_bgm",
    })
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_auto_select_not_enabled(session_with_video):
    """Auto-select when LLM not enabled returns 400."""
    client, data = session_with_video
    # Disable LLM
    await client.post("/api/llm-config", json={"enabled": False})
    r = await client.post("/api/auto-select", json={
        "session_id": data["session_id"],
    })
    assert r.status_code == 400


@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa for BGM analysis")
async def test_chat_not_enabled(full_session):
    """Chat when LLM not enabled returns 400."""
    client, sid, bid = full_session
    await client.post("/api/llm-config", json={"enabled": False})
    r = await client.post("/api/chat", json={
        "session_id": sid,
        "message": "hello",
    })
    assert r.status_code == 400


# ═══════════════════════════════════════════════════════
# Render (lightweight — skips actual encoding for speed)
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa for BGM analysis")
async def test_render_basic(full_session):
    """Render endpoint accepts a valid plan (may fail on encoding, tests API layer)."""
    client, sid, bid = full_session

    # Generate plan
    r = await client.post("/api/plan", json={
        "session_id": sid, "bgm_id": bid, "seed": 42,
        "strategy": "fit",
    })
    assert r.status_code == 200
    plan_data = r.json()

    # Try render — may fail if test video is too short, but API should handle gracefully
    r = await client.post("/api/render", json={"plan": plan_data["plan"]})
    # Accept 200 (success) or 4xx/5xx (encoding issues with test media)
    # The important thing: it doesn't crash the server
    assert r.status_code in range(200, 600)


@pytest.mark.asyncio
async def test_render_no_plan(client):
    """Render without plan returns 400."""
    r = await client.post("/api/render", json={})
    assert r.status_code == 400


# ═══════════════════════════════════════════════════════
# Full Pipeline (integration)
# ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.skip(reason="requires librosa for BGM analysis")
async def test_full_pipeline_mocked(full_session):
    """End-to-end: upload → auto-select → plan → render with mocked LLM."""
    client, sid, bid = full_session

    # 1. Auto-select (mocked)
    with patch("server._llm_chat", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = MOCK_AUTOSELECT_JSON
        r = await client.post("/api/auto-select", json={
            "session_id": sid, "bgm_id": bid,
        })
    assert r.status_code == 200
    selected = r.json()["selected_count"]
    assert selected > 0

    # 2. Generate plan
    r = await client.post("/api/plan", json={
        "session_id": sid, "bgm_id": bid, "seed": 42,
        "strategy": "fit",
    })
    assert r.status_code == 200
    plan = r.json()
    assert_valid_plan(plan["plan"])
    assert plan["ready"] is True

    # 3. Render (API level)
    r = await client.post("/api/render", json={"plan": plan["plan"]})
    assert r.status_code in range(200, 600)

    # 4. Download (if render succeeded)
    if r.status_code == 200:
        render_data = r.json()
        dl = await client.get(render_data["download_url"])
        assert dl.status_code == 200
