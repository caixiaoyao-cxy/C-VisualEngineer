"""
Colab — 测试 TTS 分支的语音合成 + 字幕
"""
# ── 1. 克隆 TTS 分支 ───────────────────────────
import os, sys, json, IPython.display, subprocess
from pathlib import Path

REPO = "https://github.com/Pankeyi88/map2video.git"
if not Path("map2video_tts").exists():
    subprocess.run(["git", "clone", "-b", "TTS", REPO, "map2video_tts"], check=True)
    subprocess.run(["pip", "install", "-e", "map2video_tts", "dashscope", "httpx"], check=True)
os.chdir("map2video_tts")
SRC = Path("src")
if str(SRC.resolve()) not in sys.path:
    sys.path.insert(0, str(SRC.resolve()))

# ── 2. 设置 key ─────────────────────────────────
os.environ["ALIBABA_API_KEY"] = "sk-8b720728530443da80cd9d3cedaa2590"
os.environ["TTS_PROVIDER"] = "dashscope"
os.environ["DASHSCOPE_API_KEY"] = os.environ["ALIBABA_API_KEY"]

# ── 3. TTS 合成 ─────────────────────────────────
from mapgen.media.tts import synthesize_dubbing
from mapgen.media.subtitles import generate_subtitle

text = (
    "欢迎来到杭州，一座融合了千年历史与现代活力的城市。"
    "西湖的晨雾中，断桥残雪诉说着白蛇传说的浪漫；"
    "灵隐寺的钟声里，千年古刹静候着每一位访客。"
)

print("TTS 合成中... (CosyVoice)")
result = synthesize_dubbing(
    text,
    output_path="/content/tts_output.mp3",
    audio_format="mp3",
    voice="longxiaochun",
    model="cosyvoice-v1",
    provider="dashscope",
)
print(json.dumps(result, ensure_ascii=False, indent=2))

print("\n播放音频:")
IPython.display.display(IPython.display.Audio("/content/tts_output.mp3"))

# ── 4. 字幕 ─────────────────────────────────────
print("\n生成字幕:")
srt = generate_subtitle(text, output_path="/content/tts_subtitle.srt", audio_path="/content/tts_output.mp3")
print(Path(srt["path"]).read_text(encoding="utf-8")[:600])
