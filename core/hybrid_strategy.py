import numpy as np
import torch
from botorch.utils.multi_objective import pareto
from typing import Tuple, List, Optional
from botorch.utils.multi_objective.hypervolume import Hypervolume


class HybridStrategy:
    def __init__(self, ref_point: torch.Tensor):
        """
        初始化混合策略

        Args:
            ref_point: 用于计算超体积的参考点 (成本, 赖氨酸, 能量)
        """
        self.ref_point = ref_point
        self.hv_calculator = Hypervolume(ref_point=ref_point)

    @staticmethod
    def should_switch_to_bo(
            ga_history: List[float],
            window: int = 10,
            tol: float = 1e-3,
            min_iter: int = 50
    ) -> bool:
        """
        改进的切换决策逻辑

        Args:
            ga_history: 超体积历史记录
            window: 观察窗口大小
            tol: 改进容忍阈值
            min_iter: 最小迭代次数

        Returns:
            bool: 是否切换到BO
        """
        return True

        if len(ga_history) < max(window, min_iter):
            return False

        # 计算滑动窗口内的平均改进
        improvements = np.diff(ga_history[-window:])
        avg_improvement = np.mean(improvements)

        # 双重条件：改进率低于阈值 或 出现退化
        return (avg_improvement < tol) or (avg_improvement < 0)

    def elite_selection(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            n_elites: int = 10,
            diversity_weight: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        改进的精英选择策略（考虑目标值和多样性）

        Args:
            X: 候选解集 (n_samples, n_var)
            Y: 目标值集 (n_samples, n_obj)
            n_elites: 选择的精英数量
            diversity_weight: 多样性权重 (0-1)

        Returns:
            elite_X: 精英解集
            elite_Y: 精英目标值
        """
        # 获取帕累托前沿
        pareto_mask = pareto.is_non_dominated(Y)
        pareto_X, pareto_Y = X[pareto_mask], Y[pareto_mask]

        if len(pareto_X) <= n_elites:
            return pareto_X, pareto_Y

        # 计算目标空间分数 (标准化处理)
        obj_scores = pareto_Y.clone()
        obj_scores[:, 0] = -obj_scores[:, 0]  # 成本最小化
        obj_scores = (obj_scores - obj_scores.min(0).values) / \
                     (obj_scores.max(0).values - obj_scores.min(0).values + 1e-6)
        obj_rank = obj_scores.mean(1)

        # 计算多样性分数 (基于决策空间距离)
        dist_matrix = torch.cdist(pareto_X, pareto_X, p=2)
        diversity_score = dist_matrix.mean(1)
        diversity_rank = diversity_score / diversity_score.max()

        # 综合排名
        combined_rank = (1 - diversity_weight) * obj_rank + diversity_weight * diversity_rank
        elite_indices = torch.argsort(combined_rank, descending=True)[:n_elites]

        return pareto_X[elite_indices], pareto_Y[elite_indices]

    def initialize_bo(
            self,
            ga_results: Tuple[torch.Tensor, torch.Tensor],
            n_samples: int = 50,
            noise_scale: float = 0.05,
            include_perturbations: bool = True,
            include_extreme: bool = True
    ) -> torch.Tensor:
        """
        增强的BO初始化方法

        Args:
            ga_results: (X_ga, Y_ga) 元组
            n_samples: 总样本数
            noise_scale: 扰动幅度
            include_perturbations: 是否包含扰动样本
            include_extreme: 是否包含极端解

        Returns:
            X_init: 初始化样本集 (n_samples, n_var)
        """
        X_ga, Y_ga = ga_results
        elites_X, elites_Y = self.elite_selection(X_ga, Y_ga)

        # 基础样本集
        samples = [elites_X]

        # 添加扰动样本
        if include_perturbations and len(elites_X) > 0:
            n_perturb = min(n_samples // 2, len(elites_X) * 3)
            noise = torch.randn(n_perturb, elites_X.shape[1]) * noise_scale
            perturbed = torch.clamp(
                elites_X.repeat(n_perturb // len(elites_X) + 1, 1)[:n_perturb] + noise,
                0, 1
            )
            samples.append(perturbed)

        # 添加极端解（单目标最优）
        if include_extreme and len(Y_ga) > 0:
            extreme_samples = []
            for obj_idx in range(Y_ga.shape[1]):
                if obj_idx == 0:  # 成本最小化
                    idx = torch.argmin(Y_ga[:, obj_idx])
                else:  # 其他最大化
                    idx = torch.argmax(Y_ga[:, obj_idx])
                extreme_samples.append(X_ga[idx])
            samples.append(torch.stack(extreme_samples))

        # 组合所有样本
        X_init = torch.cat(samples, dim=0)

        # 如果样本不足，补充随机样本
        if len(X_init) < n_samples:
            n_random = n_samples - len(X_init)
            X_init = torch.cat([X_init, torch.rand(n_random, elites_X.shape[1])])

        return X_init[:n_samples]

    def compute_hypervolume(self, Y: torch.Tensor) -> float:
        """计算超体积指标"""
        return self.hv_calculator.compute(Y)
