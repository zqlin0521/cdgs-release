import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# 检查设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 模型配置
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# 使用 'vitb' 模型
encoder = 'vitb'

# 初始化模型
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load('/root/documents/Depth-Anything-V2/depth_anything_v2_vitb.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# 读取输入图像
raw_img = cv2.imread('/root/documents/Depth-Anything-V2/test/00001.jpg')

# 估计深度图
depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

# 归一化深度图
min_depth = np.min(depth)
max_depth = np.max(depth)

normalized_depth_map = np.zeros_like(depth, dtype=np.uint8)
normalized_depth_map = np.uint8(255 * (depth - min_depth) / (max_depth - min_depth))

# 自定义颜色映射函数：从深红色到蓝色的渐变
def custom_colormap(value):
    if value < 64:  # 0-63: 深红到红
        return (255, int(value * 4), 0)
    elif value < 128:  # 64-127: 红到黄
        value -= 64
        return (255, 255, int(value * 4))
    elif value < 192:  # 128-191: 黄到绿
        value -= 128
        return (255 - int(value * 4), 255, 0)
    else:  # 192-255: 绿到蓝
        value -= 192
        return (0, 255 - int(value * 4), 255)

# 应用自定义颜色映射
height, width = normalized_depth_map.shape
colored_depth_map = np.zeros((height, width, 3), dtype=np.uint8)

for i in range(height):
    for j in range(width):
        colored_depth_map[i, j] = custom_colormap(normalized_depth_map[i, j])

# 保存彩色深度图
output_path = '/root/documents/cdgs/Depth-Anything-V2/test/00001_depth_colored1.png'
cv2.imwrite(output_path, colored_depth_map)

print(f"Colored depth map saved to {output_path}")
