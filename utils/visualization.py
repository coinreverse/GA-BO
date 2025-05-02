import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from typing import Union


def plot_pareto_front(
        Y: Union[np.ndarray, torch.Tensor],
        title: str = "Pareto Front",
        labels: tuple = ("Cost (¥/kg)", "Lysine (%)", "Energy (MJ/kg)"),
        ref_point: Union[np.ndarray, torch.Tensor, None] = None,
        show_projection: bool = True,
        angle: tuple = (25, 45),
        figsize: tuple = (14, 10)
):
    """
    增强的帕累托前沿可视化

    Args:
        Y: 目标值矩阵 (n_samples, 3) [成本, 赖氨酸, 能量]
        title: 图表标题
        labels: 三个坐标轴的标签
        ref_point: 参考点用于绘制边界
        show_projection: 是否显示2D投影
        angle: 3D视图的仰角和方位角 (elev, azim)
        figsize: 图表尺寸
    """
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    if ref_point is not None and isinstance(ref_point, torch.Tensor):
        ref_point = ref_point.cpu().numpy()

    # 创建图形
    fig = plt.figure(figsize=figsize)

    if show_projection:
        # 创建3D+2D组合视图
        ax3d = fig.add_subplot(221, projection='3d')
        ax_xy = fig.add_subplot(222)
        ax_xz = fig.add_subplot(223)
        ax_yz = fig.add_subplot(224)
        axes_2d = [ax_xy, ax_xz, ax_yz]
    else:
        ax3d = fig.add_subplot(111, projection='3d')

    # 3D绘图
    ax3d.scatter(Y[:, 0], Y[:, 1], Y[:, 2],
                 c='r', marker='o', s=30, alpha=0.8,
                 edgecolors='k', linewidths=0.5)

    # 绘制参考点
    if ref_point is not None:
        ax3d.scatter(*ref_point, c='b', marker='*', s=200, label='Reference Point')

        # 生成三维参考平面
        for i in range(3):
            # 获取轴范围（扩展10%边界）
            max_val = np.max(Y[:, i]) * 1.1
            other_dims = [j for j in range(3) if j != i]
            min_vals = [np.min(Y[:, j]) * 0.9 for j in other_dims]
            max_vals = [np.max(Y[:, j]) * 1.1 for j in other_dims]

            # 生成网格坐标
            u = np.linspace(min_vals[0], max_vals[0], 2)
            v = np.linspace(min_vals[1], max_vals[1], 2)
            u, v = np.meshgrid(u, v)

            # 构建平面坐标
            plane = np.zeros((2, 2, 3))
            plane[:, :, i] = max_val
            plane[:, :, other_dims[0]] = u
            plane[:, :, other_dims[1]] = v

            # 调整坐标顺序并绘制
            ax3d.plot_surface(
                plane[:, :, 0],
                plane[:, :, 1],
                plane[:, :, 2],
                color='gray',
                alpha=0.2,
                linewidth=0,
                antialiased=False
            )

    ax3d.set_xlabel(labels[0])
    ax3d.set_ylabel(labels[1])
    ax3d.set_zlabel(labels[2])
    ax3d.set_title(title)
    ax3d.view_init(elev=angle[0], azim=angle[1])

    if show_projection:
        # 2D投影绘图
        projections = [
            (0, 1, 'Cost vs Lysine'),
            (0, 2, 'Cost vs Energy'),
            (1, 2, 'Lysine vs Energy')
        ]

        for ax, (x_idx, y_idx, sub_title) in zip(axes_2d, projections):
            ax.scatter(Y[:, x_idx], Y[:, y_idx],
                       c='r', marker='o', s=20, alpha=0.6)

            if ref_point is not None:
                ax.axvline(ref_point[x_idx], color='b', linestyle='--', alpha=0.3)
                ax.axhline(ref_point[y_idx], color='b', linestyle='--', alpha=0.3)
                ax.scatter(ref_point[x_idx], ref_point[y_idx],
                           c='b', marker='*', s=100)

            ax.set_xlabel(labels[x_idx])
            ax.set_ylabel(labels[y_idx])
            ax.set_title(sub_title)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    cost = np.random.uniform(2.5, 5.0, 100)
    lysine = np.random.uniform(0.6, 1.0, 100)
    energy = np.random.uniform(12.0, 15.0, 100)
    Y = np.column_stack([cost, lysine, energy])

    # 添加帕累托前沿点
    Y[:7, :] = np.array([
        [2.8, 0.65, 13.0],
        [3.0, 0.72, 13.5],
        [3.2, 0.78, 14.0],
        [3.5, 0.85, 14.2],
        [4.0, 0.92, 14.5],
        [4.5, 0.95, 14.8],
        [5.0, 1.00, 15.0]
    ])

    # 绘制图表
    ref_point = np.array([5.5, 0.5, 11.0])
    fig = plot_pareto_front(
        Y,
        title="Feed Formula Optimization",
        ref_point=ref_point,
        angle=(30, 40)
    )
    plt.show()
