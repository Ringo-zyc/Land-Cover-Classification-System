import torch
from cyclegan.models import Generator  # 假设你的模型定义在 cyclegan/models 中

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model_path = 'cyclegan/checkpoints/latest_net_G_A.pth'
netG_A2B = Generator().to(device)
netG_A2B.load_state_dict(torch.load(model_path))
netG_A2B.eval()

print("模型加载成功！")