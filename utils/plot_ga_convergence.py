import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(hv_history, best_solutions, save_path=None):
    """
    绘制GA收敛曲线（超体积和成本）

    Args:
        hv_history: 超体积历史（列表）
        best_solutions: 每代最优解的目标值（列表，形状为n_gen x n_objectives）
        save_path: 图片保存路径（可选）
    """
    plt.figure(figsize=(12, 5))

    # 1. 绘制超体积曲线
    plt.subplot(1, 2, 1)
    plt.plot(hv_history, 'b-', linewidth=2, label="Hypervolume")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume (HV)")
    plt.title("Hypervolume Convergence")
    plt.grid(True)
    plt.legend()

    # 2. 绘制成本曲线（过滤>1000的值）
    best_costs = [sol[0] for sol in best_solutions if sol[0] <= 1000]  # 过滤条件

    # 生成对应的x轴索引（可能比原始数据短）
    valid_generations = [i for i, sol in enumerate(best_solutions) if sol[0] <= 1000]

    plt.subplot(1, 2, 2)
    plt.plot(valid_generations, best_costs, 'r-', linewidth=2, label="Best Cost")

    plt.xlabel("Generation")
    plt.ylabel("Cost (€/MT)")
    plt.title("Best Cost Convergence (Filtered)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
