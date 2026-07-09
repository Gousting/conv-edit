# Conv-Edit 实现文档 (v6)

> 2026-07-10 · v5→v6: 人声保护（变速仅纯BGM）、playback_speed字段、loop_capped警告、jcut约束显式化

## 一、架构

```
用户输入
├── 模式A: 视频 → 标注工作台(server.py) → clips.json
└── 模式B: (预留)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                      三引擎管线                           │
│                                                          │
│  audio/analyzer.py           planner/__init__.py (v6)     │
│  ┌───────────────┐          ┌──────────────────────┐     │
│  │ 音乐分析引擎   │          │ 切点规划引擎          │     │
│  │ BPM/节拍/能量  │ timeline │ · loop_capped标记     │     │
│  │ intro偏移建议  │ ──────→ │ · playback_speed=1.0 │     │
│  └───────────────┘          │ · protect_end能量容差 │     │
│                             └──────────┬───────────┘     │
│                                        │                 │
│                                 render_timeline.json      │
│                                        │                 │
│                            ┌───────────▼──────────┐      │
│                            │ validator.py (v6)    │      │
│                            │ · 变速仅纯BGM(人声安全)│      │
│                            │ · playback_speed字段  │      │
│                            │ · loop_capped→WARNING │      │
│                            └───────────┬──────────┘      │
│                                        │                 │
│                            ┌───────────▼──────────┐      │
│                            │ FFmpeg 渲染           │      │
│                            │ setpts=PTS*playback_speed   │
│                            │ libx264(CRF=18)+aac(192k)   │
│                            └──────────────────────┘      │
│                                        │                 │
│                                   输出 MP4                │
└──────────────────────────────────────────────────────────┘
```

## 二、v6 核心变更

### 2.1 变速补偿仅限纯BGM（人声保护）

**FFmpeg 的 `setpts` 变速会导致音调偏移**——人声变成花栗鼠或低音炮。

v6 硬性规定：
- **只有 `audio_mode="纯BGM"` 的片段允许变速**（原声已静音，无人声可毁）
- `"突出人声"` 和 `"融入BGM"` 的片段**禁止变速**
- auto-fix 在变速前检查相邻片段的 audio_mode：
  - 找到纯BGM候选 → 正常变速补偿
  - 所有候选都含人声 → **拒绝变速**，validator 输出 ERROR

validator 事后也会检查：若 segment 上存在 `playback_speed ≠ 1.0` 且 `audio_mode ≠ "纯BGM"`，直接报 ERROR。

### 2.2 playback_speed 字段（替代 tags 字符串）

v5 将变速比例藏在 `tags: ["speed+1.5%"]` 中。v6 改为独立数值字段：

```json
"playback_speed": 1.04   // 1.0=正常, >1=慢放(拉伸), <1=快放(压缩)
```

渲染引擎直接读取：`setpts=playback_speed*PTS`。**timeline_end 已包含变速后的实际时长**，渲染引擎不再依赖 `source_duration` 计算。

### 2.3 jcut vs gain_fade 约束（正文显式化）

**Planner 强制约束**：`jcut_audio_lead_ms` 自动取 `max(用户设定值, gain_fade_ms)`。

源码位置：`plan.__init__._build_gapless_timeline` 第 260 行附近。此约束保证 jcut 提前进入的音频不会与 gain_fade 渐变区间重叠，杜绝爆音。

### 2.4 loop_capped — max_loops 截断警告

当 `loop` 策略的 `max_loops=3` 不足以填满 BGM 时长时，`render_timeline.json` 顶层写入：

```json
"loop_capped": true
```

validator 检测到此字段后输出 **WARNING**（而非 HINT），明确告知用户素材总时长不足。

渲染引擎默认行为：若 `loop_capped=true`，视频画面结束后 BGM 继续播放至完毕并淡出（不硬切）。

## 三、数据格式

### 3.1 clips.json
```json
{
  "clips": [{
    "clip_id": "a1b2c3d4e5f6",
    "source_path": "/videos/match.mp4",
    "start_sec": 24.83,
    "end_sec": 30.0,
    "duration_sec": 5.17,
    "intensity": 0.9,
    "audio_mode": "融入BGM",
    "tags": ["战斗"],
    "protect_end": true
  }]
}
```

### 3.2 render_timeline.json
```json
{
  "bgm_path": "/music/epic.mp3",
  "bgm_start_offset_sec": 4.5,
  "total_duration_sec": 34.0,
  "output_resolution": "1920x1080",
  "output_fps": 30,
  "scale_mode": "letterbox",
  "audio_sample_rate": 48000,
  "audio_channels": 2,
  "loop_capped": false,
  "segments": [{
    "clip_id": "a1b2c3d4e5f6",
    "source_path": "/videos/match.mp4",
    "source_start": 24.83,
    "source_end": 30.0,
    "source_duration": 5.17,
    "timeline_start": 12.87,
    "timeline_end": 18.02,
    "playback_speed": 1.0,
    "bgm_volume_gain_db": -12.0,
    "gain_fade_ms": 30,
    "source_audio_volume": 0.3,
    "source_audio_fade_in_ms": 100,
    "audio_mode": "融入BGM",
    "intensity": 0.95,
    "tags": ["高能"],
    "jcut_audio_lead_ms": 100,
    "protect_end": false
  }]
}
```

## 四、CLI 用法

```bash
# 1. BGM 分析
python3 audio/analyzer.py song.mp3 outputs/timeline.json

# 2. 标注工作台导出 clips.json (浏览器 localhost:8765)

# 3. 切点规划
python3 planner/__init__.py outputs/clips.json outputs/timeline.json \
    song.mp3 [strategy] [seed] [offset] [--auto-offset] [--scale-mode letterbox]
# strategy:      fit | loop | truncate
# seed:          随机种子 (默认42)
# --auto-offset: 自动使用 suggested_intro_offset
# --scale-mode:  letterbox(默认) | crop | stretch

# 4. 校验
python3 planner/validator.py outputs/render_plan.json
# → 检查 jcut、playback_speed 滥用、loop_capped

# 5. 变速修复（仅纯BGM片段可变速）
python3 planner/validator.py outputs/render_plan.json --auto-fix -o fixed.json

# 6. 渲染
ffmpeg \
  -i bgm.mp3 \
  # ... (按 render_timeline.json 组装) \
  -filter_complex "[v:0]setpts={playback_speed}*PTS[v]" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -ar 48000 -ac 2 \
  output.mp4
```

## 五、变速补偿策略 (v6)

| 间隙 | 策略 | 可变速条件 |
|---|---|---|
| < 500ms | 前一段拉伸（±5%） | **仅 `audio_mode="纯BGM"`** |
| 500ms ~ 2s | 速度波纹（2-3段均摊） | **候选全部为纯BGM** |
| ≥ 2s | 拒绝 | — |
| 相邻全为人声片段 | **拒绝变速** → ERROR | 提示加纯BGM过渡或增素材 |

## 六、修订清单 (v6 vs v5)

| # | 问题 | 修复 |
|---|------|------|
| 1 | 变速导致人声音调失真 | 仅纯BGM片段可变速；含人声片段拒绝变速→ERROR |
| 2a | jcut约束仅存附录 | 正文 2.3 节显式声明：`jcut = max(jcut, gain_fade_ms)` |
| 2b | 变速比例藏在 tags 字符串 | 新增 `playback_speed` 数值字段（渲染器直接读取） |
| 3 | loop截断无用户感知 | `loop_capped: true` → validator WARNING + 渲染自动BGM淡出 |

## 七、依赖

```
librosa numpy scipy ruptures fastapi uvicorn ffmpeg
```
