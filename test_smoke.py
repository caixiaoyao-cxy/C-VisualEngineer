"""
test_smoke.py
冒烟测试 - 验证 GPU + 模型能否正常工作
跑完约 30 秒，出 2 张图即通过
"""

import torch
from diffusers import StableDiffusionPipeline
from config import *

# 1. 检测 GPU
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = DEVICE if torch.cuda.is_available() else "cpu"

# 2. 加载最小模型测试
print(f"\n[加载模型] dreamshaper-8")
pipe = StableDiffusionPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    cache_dir=str(CACHE_DIR),
).to(device)

if ENABLE_ATTENTION_SLICING:
    pipe.enable_attention_slicing()

# 3. 生成 1 张测试图
prompt = f"{STYLE_PREFIX}西湖断桥, 雪景, 二次元{STYLE_SUFFIX}"
print(f"\n[生成] 测试图片...")
print(f"Prompt: {prompt}")

result = pipe(
    prompt=prompt,
    negative_prompt=NEGATIVE_PROMPT,
    width=512,
    height=512,
    num_inference_steps=15,
    guidance_scale=7.5,
).images[0]

out_path = OUTPUT_DIR / "test_smoke.png"
result.save(out_path)
print(f"\n✅ 测试通过！图片已保存: {out_path}")
