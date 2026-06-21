"""生成杭州地图轮廓（填充图）— 所有场景公用"""
import cv2
import numpy as np
from pathlib import Path

out_dir = Path("input/contours")
out_dir.mkdir(parents=True, exist_ok=True)

w, h = 768, 512
img = np.zeros((h, w), dtype=np.uint8)

# 杭州地图简化轮廓
pts = np.array([
    [200, 80], [250, 60], [320, 50], [400, 55], [480, 65],
    [550, 80], [600, 110], [620, 150], [630, 200], [625, 250],
    [610, 290], [580, 320], [540, 350], [500, 370], [450, 390],
    [400, 400], [350, 395], [300, 380], [260, 360], [220, 330],
    [190, 300], [170, 260], [160, 220], [165, 180], [175, 140],
    [190, 110],
], dtype=np.int32)

# 填充白色（内部白色 = 生成区域，外部黑色 = 约束区域外的留白）
cv2.fillPoly(img, [pts], 255)

out_path = out_dir / "hangzhou_map.png"
cv2.imwrite(str(out_path), img)
print(f"生成: {out_path} ({img.shape})")
print(f"  白色 = AI 生成区域 → 画面元素会分布在此范围内")
print(f"  黑色 = 留白区域")
