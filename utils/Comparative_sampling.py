import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist


def calculate_spacing(pareto_front):
    """计算Spacing Metric指标"""
    if len(pareto_front) <= 1:
        return 0.0

    # 计算所有解之间的欧氏距离
    distances = cdist(pareto_front, pareto_front)
    np.fill_diagonal(distances, np.inf)  # 将对角线设为无穷大

    # 获取每个解的最近邻距离
    min_distances = np.min(distances, axis=1)

    # 计算指标值
    avg_min = np.mean(min_distances)
    spacing = np.sqrt(np.sum((min_distances - avg_min) ** 2) / (len(pareto_front) - 1))
    return spacing


def plot_ga_results(X_ga, Y_ga, meta_ga, ref_point, title_suffix=""):
    """绘制GA算法的Pareto前沿结果（完全基于PyTorch实现）"""
    device = X_ga.device if torch.is_tensor(X_ga) else 'cpu'

    # 确保所有数据都是Tensor
    if not torch.is_tensor(X_ga):
        X_ga = torch.tensor(X_ga, device=device, dtype=torch.float32)
    if not torch.is_tensor(Y_ga):
        Y_ga = torch.tensor(Y_ga, device=device, dtype=torch.float32)
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, device=device, dtype=torch.float32)

    # 处理元数据
    best_solutions = meta_ga["best_solutions"]
    if not torch.is_tensor(best_solutions):
        best_solutions = torch.tensor(best_solutions, device=device, dtype=torch.float32)

    hv_history = meta_ga["hv_history"]
    if not torch.is_tensor(hv_history):
        hv_history = torch.tensor(hv_history, device=device, dtype=torch.float32)

    # 将数据移动到CPU进行绘图（matplotlib不能在GPU上直接操作）
    X_ga = X_ga.cpu()
    Y_ga = Y_ga.cpu()
    best_solutions = best_solutions.cpu()
    hv_history = hv_history.cpu()
    ref_point = ref_point.cpu()

    # 开始绘图
    plt.figure(figsize=(15, 5))

    # 1. Pareto前沿分布图
    plt.subplot(131)
    # 按第一目标排序
    sorted_idx = torch.argsort(best_solutions[:, 0])
    sorted_front = best_solutions[sorted_idx]

    # 使用Tensor数据绘图
    plt.scatter(Y_ga[:, 0].numpy(), Y_ga[:, 1].numpy(),
                c='gray', alpha=0.4, label="All Solutions")

    plt.plot(sorted_front[:, 0].numpy(), sorted_front[:, 1].numpy(),
             'ro-', markersize=6, linewidth=1.5, label="Pareto Front")

    # 计算边界
    max_val = torch.max(
        torch.cat([best_solutions, ref_point.unsqueeze(0)]), dim=0
    ).values.numpy()
    min_val = torch.min(best_solutions, dim=0).values.numpy()

    plt.xlim(min_val[0] * 0.9, max_val[0] * 1.05)
    plt.ylim(min_val[1] * 0.9, max_val[1] * 1.05)
    plt.xlabel("Objective 1", fontsize=12)
    plt.ylabel("Objective 2", fontsize=12)
    plt.title(f"Pareto Front Distribution {title_suffix}", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend()

    # 2. HV收敛曲线
    plt.subplot(132)
    plt.plot(hv_history.numpy(), 'b-', linewidth=2, label="HV Evolution")

    # 填充区域
    min_hv = hv_history.min().item()
    plt.fill_between(range(len(hv_history)), min_hv, hv_history.numpy(), alpha=0.2)

    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Hypervolume", fontsize=12)
    plt.title(f"Convergence Speed {title_suffix}", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend()

    # 3. 决策空间分布
    plt.subplot(133)
    obj_sum = torch.sum(Y_ga, dim=1).numpy()

    plt.scatter(
        X_ga[:, 0].numpy(),
        X_ga[:, 1].numpy(),
        c=obj_sum,
        cmap='viridis',
        alpha=0.7,
        s=50
    )

    plt.colorbar(label="Objective Sum")
    plt.xlabel("Decision Variable 1", fontsize=12)
    plt.ylabel("Decision Variable 2", fontsize=12)
    plt.title(f"Decision Space {title_suffix}", fontsize=13)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # 在绘图完成后自动保存
    if title_suffix:
        save_path = f"results/ga_plot_{title_suffix.replace('(', '').replace(')', '').replace(' ', '_').lower()}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图像（可选）
    # plt.show()
    plt.close()  # 关闭图像以释放内存
