import torch
import numpy as np
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


    def should_switch_to_bo(
            self,
            ga_population: torch.Tensor,  # 当前种群所有解的目标值 (n_samples, n_obj)
            ga_hv_history: List[float],  # 超体积历史记录
            min_pareto_points: int = 3,  # 最小帕累托解数量要求
            stagnation_window: int = 5,  # 改进停滞检测窗口
            hv_improve_tol: float = 4e-2  # 超体积改进容忍阈值
    ) -> bool:
        """
        切换决策逻辑 (满足以下所有条件时切换):
        1. 存在至少min_pareto_points个帕累托解
        2. 超体积在最近stagnation_window代内改进小于hv_improve_tol
        3. 种群多样性足够（自动通过帕累托解数量隐含保证）

        Args:
            ga_population: (n_samples, n_obj) 当前种群目标值
            ga_hv_history: 历史超体积记录列表
            min_pareto_points: 最小帕累托解数量
            stagnation_window: 改进停滞检测窗口大小
            hv_improve_tol: 超体积相对改进容忍阈值

        Returns:
            bool: 是否满足切换到BO的条件
        """
        # 条件1：检查帕累托解数量是否足够
        pareto_mask = pareto.is_non_dominated(ga_population)
        pareto_count = pareto_mask.sum().item()
        if pareto_count < min_pareto_points:
            print(f"❌ 帕累托解不足: {pareto_count} < {min_pareto_points}")
            return False

        # 条件2：检查超体积改进是否停滞
        if len(ga_hv_history) < stagnation_window:
            print(f"❌ 历史数据不足: {len(ga_hv_history)} < {stagnation_window}")
            return False

        recent_hv = ga_hv_history[-stagnation_window:]
        hv_improve = (recent_hv[-1] - recent_hv[0]) / (abs(recent_hv[0]) + 1e-6)
        is_stagnant = abs(hv_improve) < hv_improve_tol

        if not is_stagnant:
            print(f"❌ 超体积仍在改进: {hv_improve:.2e} >= {hv_improve_tol:.0e}")
            return False

        # 所有条件满足
        print(f"✅ 可切换到BO: "
              f"帕累托解={pareto_count}, "
              f"超体积改进={hv_improve:.2e}, "
              f"窗口={stagnation_window}代")
        return True

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
            evaluator,
            n_samples: int = 30,
            noise_scale: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合策略BO初始化：帕累托前沿 + 各目标最优解 + 多样性补充

        Args:
            evaluator: 评估扰动样本
            ga_results: (X_ga, Y_ga) 元组
            n_samples: 总样本数（默认30）
            noise_scale: 扰动幅度

        Returns:
            X_init: 初始化样本集 (n_samples, n_var)
        """
        X_ga, Y_ga = ga_results
        device = X_ga.device

        # 类型转换确保安全
        X_ga = X_ga.double() if isinstance(X_ga, torch.Tensor) else torch.tensor(X_ga, device=device).double()
        Y_ga = Y_ga.double() if isinstance(Y_ga, torch.Tensor) else torch.tensor(Y_ga, device=device).double()

        # === 核心策略 ===
        x_samples, y_samples = [], []

        # 1. 保留所有帕累托前沿解
        pareto_mask = pareto.is_non_dominated(Y_ga)
        pareto_X, pareto_Y = X_ga[pareto_mask], Y_ga[pareto_mask]
        x_samples.append(pareto_X)
        y_samples.append(pareto_Y)
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
                x_samples.append(X_ga[idx].unsqueeze(0))
                y_samples.append(Y_ga[idx].unsqueeze(0))
                print(f"添加{['成本', '赖氨酸', '能量'][obj_idx]}最优解 (被支配)")

        # 3. 添加被支配解中成本最低的5个解（确保成本敏感）
        non_pareto_mask = ~pareto_mask
        if non_pareto_mask.sum() > 0:
            non_pareto_Y = Y_ga[non_pareto_mask]
            non_pareto_X = X_ga[non_pareto_mask]
            cost_indices = torch.argsort(non_pareto_Y[:, 0])[:5]  # 成本最低的5个
            x_samples.append(non_pareto_X[cost_indices])
            y_samples.append(non_pareto_Y[cost_indices])
            print(f"添加被支配解中成本最低的{len(cost_indices)}个解")

        # 合并已有样本
        X_combined = torch.cat(x_samples, dim=0) if x_samples else torch.rand(n_samples, X_ga.shape[1], device=device)
        Y_combined = torch.cat(y_samples, dim=0) if y_samples else torch.zeros(n_samples, Y_ga.shape[1], device=device)
        n_current = len(X_combined)

        # 4. 补充扰动样本（围绕精英解）
        if n_current < n_samples:
            n_perturb = int(min(int(n_samples) - int(n_current), int(len(pareto_X))))
            dim = int(X_ga.shape[1])
            noise = torch.randn(int(n_perturb), int(dim), device=device) * float(noise_scale)
            perturbed_X = torch.clamp(pareto_X[:n_perturb] + noise, 0, 1)

            # 评估扰动样本 - 确保输入在正确设备上
            perturbed_Y = evaluator(perturbed_X.to(device))

            # 统一设备处理
            if isinstance(perturbed_Y, (np.ndarray, list)):
                perturbed_Y = torch.tensor(perturbed_Y, device=device, dtype=torch.double)
            elif isinstance(perturbed_Y, torch.Tensor):
                perturbed_Y = perturbed_Y.to(device)
            else:
                raise TypeError("评估器返回类型不支持")

            # 提取需要的3个目标（成本、赖氨酸、能量）
            nutrient_names = evaluator.get_nutrient_names()
            lysin_idx = 1 + nutrient_names.index('L')
            energy_idx = 1 + nutrient_names.index('Energy')

            perturbed_Y = torch.stack([
                perturbed_Y[:, 0],  # 成本（最小化）
                perturbed_Y[:, lysin_idx],  # 赖氨酸（最大化）
                perturbed_Y[:, energy_idx]  # 能量（最大化）
            ], dim=1)

            # 确保合并前的张量在相同设备
            X_combined = X_combined.to(device)
            Y_combined = Y_combined.to(device)

            # 安全合并
            X_combined = torch.cat([X_combined, perturbed_X])
            Y_combined = torch.cat([Y_combined, perturbed_Y])

            print(f"添加{len(perturbed_X)}个扰动样本")

        # 去重（避免重复样本影响BO）
        X_combined, Y_combined = self.deduplicate_rows(X_combined, Y_combined, precision=8)
        X_init = X_combined[:n_samples]
        Y_init = Y_combined[:n_samples]
        print(f"最终初始样本数: {len(X_init)} (去重后)")

        return X_init, Y_init

    def deduplicate_rows(self, X: torch.Tensor, Y: torch.Tensor, precision: int = 8) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """去重方法"""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        X_rounded = np.round(X_np, decimals=precision)
        _, unique_idx = np.unique(X_rounded, axis=0, return_index=True)
        X_unique = X[unique_idx]
        Y_unique = Y[unique_idx]
        return X_unique, Y_unique