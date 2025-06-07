import torch

# 加载 .pt 文件
# data = torch.load('results/ga_pareto_front.pt', map_location='cpu')  # 确保路径正确
data = torch.load('results/pareto_front.pt', map_location='cpu')  # 确保路径正确

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
    print("张量内容（前10个元素）:", data.flatten()[:10])  # 避免打印过多数据
else:
    print("内容:", data)

import torch

data = torch.load('results/pareto_front.pt', map_location='cpu')

print("===== Solutions =====")
print("Shape:", data['solutions'].shape)  # 确认形状
print("First 5 solutions:\n", data['solutions'][:10])  # 打印前5组解

print("\n===== Objectives =====")
print("Shape:", data['objectives'].shape)
print("First 5 objective values:\n", data['objectives'][:10])
