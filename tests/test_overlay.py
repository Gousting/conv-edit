"""Test overlay module — text/image overlay rendering engine."""

import pytest
from overlay import (
    OverlayDef,
    build_clip_vf,
    overlays_from_dict,
    overlays_to_dict,
    douyin_comedy_overlays,
    load_overlays_from_preset,
)


class TestOverlayDef:
    def test_to_drawtext_basic(self):
        ov = OverlayDef(id="test", type="text_bar", content="Hello", x="10", y="20")
        dt = ov.to_drawtext()
        assert "drawtext=" in dt
        assert "text='Hello'" in dt
        assert "x=10" in dt
        assert "y=20" in dt

    def test_to_drawtext_with_box(self):
        ov = OverlayDef(id="bar", type="text_bar", content="Title", box=True, box_color="black@0.5")
        dt = ov.to_drawtext()
        assert "box=1" in dt
        assert "black@0.5" in dt

    def test_to_drawtext_escapes_special_chars(self):
        ov = OverlayDef(id="test", type="subtitle", content="I'm ready: let's go!")
        dt = ov.to_drawtext()
        assert "I\\'m ready\\: let\\'s go" in dt

    def test_to_drawtext_border(self):
        ov = OverlayDef(id="test", type="text_popup", content="Hi", font_border=3, font_border_color="red")
        dt = ov.to_drawtext()
        assert "bordercolor=red" in dt
        assert "borderw=3" in dt

    def test_with_enable(self):
        ov = OverlayDef(id="test", type="subtitle", content="Hi", start=5.0, end=10.0)
        ov2 = ov.with_enable(2.0, 4.0)
        assert ov2.start == 2.0
        assert ov2.end == 4.0
        assert ov.content == "Hi"  # original unchanged


class TestBuildClipVf:
    def test_no_overlays(self):
        vf = build_clip_vf("scale=1920:1080", [], 0.0, 5.0)
        assert vf == "scale=1920:1080"

    def test_overlay_inside_clip(self):
        ov = OverlayDef(id="t1", type="text_bar", content="Title", x="10", y="10", start=2.0, end=4.0)
        vf = build_clip_vf("scale=1920:1080", [ov], 0.0, 5.0)
        assert "drawtext=" in vf
        assert "between(t,2.0,4.0)" in vf

    def test_overlay_outside_clip(self):
        ov = OverlayDef(id="t1", type="text_bar", content="Title", start=10.0, end=15.0)
        vf = build_clip_vf("scale=1920:1080", [ov], 0.0, 5.0)
        # Overlay completely outside clip — should be skipped
        assert "drawtext=" not in vf

    def test_overlay_partial_clip(self):
        ov = OverlayDef(id="t1", type="text_bar", content="Mid", start=3.0, end=10.0)
        vf = build_clip_vf("scale=1920:1080", [ov], 2.0, 3.0)  # clip runs 2s-5s
        assert "between(t,1.0,3.0)" in vf  # local: 3-2=1, 5-2=3

    def test_overlay_spans_entire_clip(self):
        ov = OverlayDef(id="t1", type="text_bar", content="Full", start=0.0, end=9999.0)
        vf = build_clip_vf("scale=1920:1080", [ov], 5.0, 3.0)
        assert "drawtext=" in vf
        # Should NOT have enable expression since it spans entire clip
        assert "between" not in vf

    def test_multiple_overlays(self):
        ov1 = OverlayDef(id="bar", type="text_bar", content="Top", start=0.0, end=9999.0)
        ov2 = OverlayDef(id="sub", type="subtitle", content="Hello", start=1.0, end=2.0)
        vf = build_clip_vf("base", [ov1, ov2], 0.0, 5.0)
        assert "text='Top'" in vf
        assert "text='Hello'" in vf
        assert "between(t,1.0,2.0)" in vf  # only on subtitle


class TestSerialization:
    def test_roundtrip(self):
        ovs = [
            OverlayDef(id="a", type="text_bar", content="Hello", font_size=24),
            OverlayDef(id="b", type="subtitle", content="World", box=True),
        ]
        d = overlays_to_dict(ovs)
        parsed = overlays_from_dict(d)
        assert len(parsed) == 2
        assert parsed[0].id == "a"
        assert parsed[0].content == "Hello"
        assert parsed[1].box is True

    def test_from_dict_with_extra_keys(self):
        data = [{"id": "x", "type": "text_bar", "content": "Hi", "extra_field": "ignored"}]
        parsed = overlays_from_dict(data)
        assert len(parsed) == 1
        assert parsed[0].id == "x"


class TestDouyinComedyPreset:
    def test_default_overlays_count(self):
        ovs = douyin_comedy_overlays()
        assert len(ovs) == 2
        assert ovs[0].id == "title_bar"
        assert ovs[1].id == "subtitle"

    def test_title_bar_style(self):
        ovs = douyin_comedy_overlays()
        bar = ovs[0]
        assert bar.font_color == "yellow"
        assert bar.box is True
        assert bar.y == "10"


class TestPresetLoading:
    def test_load_yaml_preset(self, tmp_path):
        import yaml
        preset_data = {
            "name": "test",
            "overlays": [
                {"id": "t1", "type": "text_bar", "content": "Test Title"}
            ]
        }
        preset_file = tmp_path / "test.yaml"
        with open(preset_file, 'w') as f:
            yaml.dump(preset_data, f)
        
        ovs = load_overlays_from_preset(str(preset_file))
        assert len(ovs) == 1
        assert ovs[0].id == "t1"
        assert ovs[0].content == "Test Title"

    def test_load_nonexistent_preset(self):
        ovs = load_overlays_from_preset("/nonexistent/preset.yaml")
        assert ovs == []
