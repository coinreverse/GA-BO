import numpy as np
import torch
from botorch.utils.multi_objective import pareto
from typing import Tuple, List
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
        return True
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
            n_samples: int = 30,
            noise_scale: float = 0.05,
    ) -> torch.Tensor:
        """
        混合策略BO初始化：帕累托前沿 + 各目标最优解 + 多样性补充

        Args:
            ga_results: (X_ga, Y_ga) 元组
            n_samples: 总样本数（默认30）
            noise_scale: 扰动幅度

        Returns:
            X_init: 初始化样本集 (n_samples, n_var)
        """
        X_ga, Y_ga = ga_results
        device = X_ga.device

        # 类型转换确保安全
        X_ga = X_ga.float() if isinstance(X_ga, torch.Tensor) else torch.tensor(X_ga, device=device).float()
        Y_ga = Y_ga.float() if isinstance(Y_ga, torch.Tensor) else torch.tensor(Y_ga, device=device).float()

        # === 核心策略 ===
        samples = []

        # 1. 保留所有帕累托前沿解
        pareto_mask = pareto.is_non_dominated(Y_ga)
        pareto_X, pareto_Y = X_ga[pareto_mask], Y_ga[pareto_mask]
        samples.append(pareto_X)
        print(f"保留帕累托解: {len(pareto_X)}个")

        # 2. 添加各目标方向的最优解（即使被支配）
        objective_directions = [
            (0, False),  # 成本最小化
            (1, True),  # 赖氨酸最大化
            (2, True)  # 能量最大化
        ]

        for obj_idx, is_maximize in objective_directions:
            if is_maximize:
                idx = torch.argmax(Y_ga[:, obj_idx])
            else:
                idx = torch.argmin(Y_ga[:, obj_idx])

            # 检查是否已在帕累托解中
            if not pareto_mask[idx]:
                samples.append(X_ga[idx].unsqueeze(0))
                print(f"添加{['成本', '赖氨酸', '能量'][obj_idx]}最优解 (被支配)")

        # 3. 添加被支配解中成本最低的5个解（确保成本敏感）
        non_pareto_mask = ~pareto_mask
        if non_pareto_mask.sum() > 0:
            non_pareto_Y = Y_ga[non_pareto_mask]
            non_pareto_X = X_ga[non_pareto_mask]
            cost_indices = torch.argsort(non_pareto_Y[:, 0])[:5]  # 成本最低的5个
            samples.append(non_pareto_X[cost_indices])
            print(f"添加被支配解中成本最低的{len(cost_indices)}个解")

        # 合并已有样本
        X_combined = torch.cat(samples, dim=0) if samples else torch.rand(n_samples, X_ga.shape[1], device=device)
        n_current = len(X_combined)

        # 4. 补充扰动样本（围绕精英解）
        if n_current < n_samples:
            n_perturb = min(n_samples - n_current, len(pareto_X))
            noise = torch.randn(n_perturb, X_ga.shape[1], device=device) * noise_scale
            perturbed = pareto_X[:n_perturb] + noise
            perturbed = torch.clamp(perturbed, 0, 1)
            X_combined = torch.cat([X_combined, perturbed])
            print(f"添加{len(perturbed)}个扰动样本")

        # 5. 最后用随机样本补足（确保多样性）
        if len(X_combined) < n_samples:
            n_random = n_samples - len(X_combined)
            X_combined = torch.cat([X_combined, torch.rand(n_random, X_ga.shape[1], device=device)])
            print(f"添加{n_random}个随机样本")

        # 去重（避免重复样本影响BO）
        X_init = torch.unique(X_combined, dim=0)[:n_samples]
        print(f"最终初始样本数: {len(X_init)} (去重后)")

        return X_init