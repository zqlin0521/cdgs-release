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
model.load_state_dict(torch.load('/root/documents/cdgs/DepthAnythingV2/depth_anything_v2_vitb.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# 读取输入图像
raw_img = cv2.imread('/root/documents/cdgs/DepthAnythingV2/test/016_gt.png')

# 估计深度图
depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

# 归一化深度图
min_depth = np.min(depth)
max_depth = np.max(depth)
normalized_depth_map = np.uint8(255 * (depth - min_depth) / (max_depth - min_depth))

# 保存灰度深度图
output_path_gray = '/root/documents/cdgs/DepthAnythingV2/test/016_depth_gray.png'
cv2.imwrite(output_path_gray, normalized_depth_map)

print(f"Gray depth map saved to {output_path_gray}")
