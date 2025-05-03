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
        self.ref_point = ref_point.float()
        self.hv_calculator = Hypervolume(ref_point=ref_point)

    @staticmethod
    def should_switch_to_bo(
            ga_history: List[float],
            ga_population: torch.Tensor,  # 新增：当前种群解
            window: int = 10,
            tol: float = 1e-3,
            min_iter: int = 50,
            min_std: float = 1e-4  # 新增：解的最小标准差阈值
    ) -> bool:
        """
        改进的切换决策逻辑

        Args:
            ga_history: 超体积历史记录
            ga_population: 当前种群的所有解（目标值）
            window: 观察窗口大小
            tol: 改进容忍阈值
            min_iter: 最小迭代次数
            min_std: 解的最小标准差（避免方差为0时切换）

        Returns:
            bool: 是否切换到BO
        """
        # 基本条件检查
        if len(ga_history) < max(window, min_iter):
            print(
                f"不满足最小迭代次数条件: len(ga_history)={len(ga_history)} < max(window={window}, min_iter={min_iter})")
            return False

        # 条件1：超体积改进是否停滞（增强稳定性）
        recent_hv = ga_history[-window:]
        improvements = np.diff(recent_hv)
        avg_improvement = np.mean(improvements)
        rel_improvement = avg_improvement / (np.mean(recent_hv) + 1e-6)  # 相对改进率
        is_improvement_stagnant = (abs(rel_improvement) < tol)

        # 条件2：增强多样性检查
        if isinstance(ga_population, torch.Tensor):
            population_std = torch.std(ga_population, dim=0).mean().item()
        else:
            population_std = np.std(ga_population, axis=0).mean()
        is_diverse = population_std > min_std

        # 新增条件3：帕累托前沿大小检查
        pareto_mask = pareto.is_non_dominated(ga_population)
        pareto_size = pareto_mask.sum().item()
        has_sufficient_pareto = pareto_size >= 3  # 至少3个帕累托解

        # 调试输出
        print("\n=== 切换条件诊断 ===")
        print(f"超体积改进率: {rel_improvement:.4f} (阈值: {tol})")
        print(f"种群标准差: {population_std:.4f} (阈值: {min_std})")
        print(f"帕累托解数量: {pareto_size} (最小要求: 3)")
        print(
            f"满足条件: 改进停滞={is_improvement_stagnant}, 多样性={is_diverse}, 足够帕累托解={has_sufficient_pareto}")

        return is_improvement_stagnant and is_diverse and has_sufficient_pareto

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

        # 确保所有张量在同一设备上
        device = X.device  # 获取 X 的设备（GPU 或 CPU）
        Y = Y.to(device)
        diversity_weight = torch.tensor(diversity_weight, device=device)  # 将权重转为张量并移到正确设备

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

        # 确保类型正确
        X_ga = X_ga.float() if isinstance(X_ga, torch.Tensor) else torch.tensor(X_ga).float()
        Y_ga = Y_ga.float() if isinstance(Y_ga, torch.Tensor) else torch.tensor(Y_ga).float()

        # 精英选择
        elites_X, _ = self.elite_selection(X_ga, Y_ga)
        n_elites = len(elites_X)

        # 处理空精英集的情况
        if n_elites == 0:
            return torch.rand(n_samples, X_ga.shape[1], device=X_ga.device)

        # 计算扰动样本数量（不超过精英数量）
        n_perturb = min(n_samples - n_elites, n_elites)
        n_random = max(0, n_samples - n_elites - n_perturb)

        # 生成扰动样本
        noise = torch.randn(n_elites, X_ga.shape[1], device=X_ga.device) * float(noise_scale)
        perturbed = elites_X + noise[:n_elites]  # 确保形状匹配
        perturbed = torch.clamp(perturbed, 0, 1)

        # 组合样本（精英+扰动+随机）
        samples = [
            elites_X,
            perturbed[:n_perturb],  # 只取需要的数量
            torch.rand(n_random, X_ga.shape[1], device=X_ga.device)
        ]

        return torch.cat([x for x in samples if len(x) > 0], dim=0)[:n_samples]

    def compute_hypervolume(self, Y: torch.Tensor) -> float:
        """计算超体积指标"""
        # 确保输入是张量
        Y = Y.float() if isinstance(Y, torch.Tensor) else torch.tensor(Y).float()

        # 检查无效值
        if torch.isnan(Y).any() or torch.isinf(Y).any() or len(Y) == 0:
            return 0.0

        # 调整目标方向（假设列顺序：成本、赖氨酸、能量）
        adjusted_front = torch.stack([
            -Y[:, 0],  # 成本最小化→转为最大化
            Y[:, 1],  # 赖氨酸（最大化）
            Y[:, 2]  # 能量（最大化）
        ], dim=1)

        # 动态设置参考点（比当前最差解更差）
        nadir_point = torch.min(adjusted_front, dim=0).values
        ref_point = nadir_point - 0.1 * torch.abs(nadir_point)

        # 确保参考点合理
        ref_point = torch.maximum(ref_point, torch.tensor([-1e6, -1e6, -1e6], device=Y.device))

        # 计算HV
        try:
            hv = Hypervolume(ref_point=ref_point)
            return hv.compute(adjusted_front).item()
        except:
            return 0.0
