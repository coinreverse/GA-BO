import torch


# 在__init__或optimize方法开头添加
torch.set_printoptions(
    linewidth=300,  # 每行最大字符数
    threshold=float('inf'),  # 禁用截断
    edgeitems=30,  # 显示首尾各30项
    sci_mode=False,  # 禁用科学计数法
    precision=2  # 设置显示精度为2位小数
)

# 加载 .pt 文件
data = torch.load('results/ga_pareto_front.pt', map_location='cpu')  # 确保路径正确
# data = torch.load('results/hybrid_pareto_front.pt', map_location='cpu')  # 确保路径正确

# 打印文件内容的基本信息
print("文件类型:", type(data))
print("\n文件内容结构:")
if isinstance(data, dict):
    for key, value in data.items():
        print(f"\n键: {key}")
        print(f"值类型: {type(value)}")
        print(f"值内容/形状: {value if isinstance(value, (int, float, str, list)) else value.shape}")
elif isinstance(data, torch.Tensor):
    print("张量形状:", data.shape)
    print("张量内容（前10个元素）:", data.flatten()[:115])  # 避免打印过多数据
else:
    print("内容:", data)



# 格式化打印张量数据
def format_tensor(tensor, decimals=2):
    """格式化张量显示，保留指定位数小数"""
    return torch.round(tensor * 10**decimals) / (10**decimals)

print("\n===== Solutions =====")
print("Shape:", data['solutions'].shape)
print("First 5 solutions:\n", format_tensor(data['solutions']))

print("\n===== Objectives =====")
print("Shape:", data['objectives'].shape)
print("First 5 objective values:\n", format_tensor(data['objectives']))
