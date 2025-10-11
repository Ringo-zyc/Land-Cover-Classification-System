import torch
from cyclegan.models.networks import define_G  # 正确导入 define_G

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型参数（需要与训练时一致）
class Opt:
    input_nc = 3            # 输入通道数（例如RGB图像）
    output_nc = 3           # 输出通道数
    ngf = 64                # 生成器滤波器数量
    netG = 'resnet_9blocks' # 生成器架构，与训练时一致
    norm = 'instance'       # 归一化类型
    no_dropout = True       # 是否禁用Dropout
    init_type = 'normal'    # 权重初始化类型
    init_gain = 0.02        # 初始化增益
    gpu_ids = [0]           # GPU ID

opt = Opt()

# 创建生成器模型
model_path = 'cyclegan/checkpoints/latest_net_G_A.pth'  # 模型权重路径
netG_A2B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                    not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

# 加载模型权重
try:
    netG_A2B.load_state_dict(torch.load(model_path, map_location=device))
    print("模型权重加载成功！")
except RuntimeError as e:
    print(f"加载失败：{e}")
    print("可能是键名不匹配，正在调整...")
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 移除 'module.' 前缀
    netG_A2B.load_state_dict(new_state_dict, strict=False)
    print("调整后加载成功！")

# 将模型移到设备上并设置为评估模式
netG_A2B.to(device).eval()
print("模型准备就绪！")

# 测试功能：输入随机张量
input_tensor = torch.randn(1, opt.input_nc, 256, 256).to(device)  # 示例输入：1张256x256图像
output_tensor = netG_A2B(input_tensor)
print("生成器输出形状:", output_tensor.shape)