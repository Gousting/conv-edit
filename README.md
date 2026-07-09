# Conv-Edit 实现文档 (v8)

> 2026-07-11 · v7→v8: 去掉全自动按钮，聊天是唯一入口

## 一、架构

```
浏览器 (localhost:8765)
├── 左侧: 视频上传 → 自动标注 → 多选/批量操作
└── 右侧: ⚙️LLM设置 → 🎵BGM上传+波形 → 🎬风格选择
         → 🤖智能选片 → 🎯生成时间线 → 📊热力图
         → 🤖LLM审查 → 🚀渲染 → 👁️VLM审查 → 📥下载
```

**聊天是唯一入口**——无论是手动选片、半自动对话引导、还是全自动一键出片，都在同一个聊天面板完成。没有独立按钮。

## 二、三条使用路线

### 2.1 手动（零 LLM）
```
丢视频 → 自动标注(强度/模式/标签) → 选风格预设 → 调片筛选
→ 上传BGM(出波形) → 🎯生成时间线 → 🚀渲染下载
```
全程纯算法，不依赖任何 API。适合精确控制每个片段。

### 2.2 半自动（💬 对话引导）
```
丢视频 → 💬聊天面板自动弹出
→ AI: "这是什么内容？想做什么风格？"
→ 你: "永劫无间击杀集锦，快节奏燃的"
→ AI: 自动设风格+智能选片 → "已选12个片段，行吗？"
→ 你: "第3个不要"
→ AI: 调整 → (提示上传BGM) → 自动生成 → 审查 → 渲染
```
像跟剪辑师聊天一样。不需要懂参数，不需要选预设。

### 2.3 全自动（💬 一句话触发）
```
丢视频 → 丢BGM → 在聊天里说"帮我自动规划实现"
→ AI 一口气：智能选片 → 生成时间线 → LLM审查 → VLM审查 → 渲染 → 下载链接
```
不需要点任何按钮。半自动和全自动的区别就是**你说几句**。

| 你在聊天里说 | AI 做什么 |
|-------------|----------|
| "这是永劫无间击杀集锦" | 推风格 → 选片 → 等你确认 |
| "第3个不要" | 调整选中 → 等你下一步 |
| **"帮我自动规划实现"** | 一口气：选片 → 生成 → 渲染 → 给下载链接 |

## 三、剪辑风格预设

| 预设 | 策略 | 强度匹配 | 音频 | 选片偏好 |
|------|------|----------|------|----------|
| 🎮 游戏集锦 | fit | 0.9 | 纯BGM | 短(<4s)+高强度(>0.7) |
| 🎬 电影对白 | fit | 0.3 | 突出人声 | 中长(3-10s)+有对话 |
| 🏔️ 旅行Vlog | fit | 0.5 | 融入BGM | 高低强度交替 |
| 🎵 MV踩点 | loop | 0.8 | 纯BGM | 极短+全选 |
| 📖 叙事短片 | fit | 0.2 | 融入BGM | 保留顺序 |
| 🤖 智能推荐 | fit | 0.6 | LLM决定 | LLM分析后推荐 |

## 四、LLM 配置

打开页面 → 右侧顶部 **⚙️ LLM 设置** → 配置：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| API 地址 | OpenAI 兼容接口 | `http://localhost:11434/v1` |
| API Key | 可选 | — |
| 文本模型 | 审查时间线 | `qwen3.6:27b` |
| 视觉模型 | 智能选片+视觉审查 | `minicpm-v:8b` |

支持 Ollama、vLLM、OpenAI、SiliconFlow 等所有 OpenAI 兼容接口。

## 五、数据格式

### 5.1 clips.json
```json
{
  "clips": [{
    "clip_id": "a1b2c3d4e5f6",
    "source_path": "/videos/match.mp4",
    "start_sec": 24.83,
    "end_sec": 30.0,
    "duration_sec": 5.17,
    "intensity": 0.9,
    "intensity_auto": true,
    "audio_mode": "融入BGM",
    "audio_mode_auto": false,
    "tags": ["高能"],
    "protect_end": true
  }]
}
```

### 5.2 render_timeline.json
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

## 六、API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/analyze` | POST | 上传视频 → 自动切分+标注 |
| `/api/analyze-bgm` | POST | 上传 BGM → 分析 BPM+节拍+波形 |
| `/api/presets` | GET | 获取 6 种剪辑风格预设 |
| `/api/auto-select` | POST | VLM 智能选片+推荐风格 |
| `/api/plan` | POST | 生成时间线+热力图 |
| `/api/llm-review` | POST | LLM 审查时间线 |
| `/api/render` | POST | 渲染视频 |
| `/api/vision-review` | POST | VLM 审查成片画面 |
| `/api/download/{id}` | GET | 下载渲染视频 |
| `/api/llm-config` | GET/POST | 读写 LLM 配置 |
| `/api/export/{id}` | POST | 导出 clips.json |

## 七、核心算法

| 算法 | 实现 |
|------|------|
| 镜头检测 | `ffmpeg scdet` 滤镜 |
| 强度判定 | `ffprobe volumedetect` → RMS→0~1 |
| 音频模式 | `ffmpeg silencedetect` → 人声/音乐/静音 |
| BPM/节拍 | `librosa` beat_track |
| 切点规划 | 强度匹配 + 节拍吸附 + 二次裁剪 + protect_end |
| 变速补偿 | 仅纯BGM片段(±5%)，人声片段禁止变速 |
| 自动修复 | 间隙<500ms拉伸 / 0.5-2s速度波纹 / ≥2s拒绝 |

## 八、约束与保护

- `jcut_audio_lead_ms` 强制 ≥ `gain_fade_ms`（杜绝爆音）
- `playback_speed ≠ 1.0` + 含人声 → ERROR（防止音调失真）
- `protect_end` + 能量容差检查（高能画面不落入低能尾奏）
- `loop_capped`: max_loops=3 硬上限 → WARNING
- 所有音频统一重采样到 48kHz/stereo（`aformat` 滤镜）
- 视频编码 `libx264` CRF=18，音频编码 `aac` 192k

## 九、启动

```bash
cd conv-edit
pip install -r requirements.txt  # fastapi uvicorn librosa scipy ruptures httpx
python3 server.py
# 浏览器打开 http://localhost:8765
```

## 十、修订历史

| 版本 | 变更 |
|------|------|
| v8 | 去掉全自动按钮，聊天是唯一入口；"帮我自动规划实现"一句话全流程 |
| v7 | 6种剪辑风格预设 + LLM智能选片(VLM) + VLM视觉审查 + LLM审查端点 |
| v6 | 人声保护变速 + playback_speed字段 + loop_capped警告 + 默认letterbox |
| v5 | 变速补偿替代黑场 + 负时间戳过滤 + protect_end能量容差 + 多声道下混 |
| v4 | start_sec统一 + 完整offset平移 + jcut约束 + scale_mode + protect_end |
| v3 | gain_fade防爆音 + 二次裁剪 + loop修复 + validator修复建议 |
