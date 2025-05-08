import numpy as np
import torch
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume  # 用于超体积计算
from typing import Tuple

from scipy.stats import pareto

from core.hybrid_strategy import HybridStrategy


class NormalizationRepair:
    def __init__(self, bounds):
        self.bounds = bounds  # 各原料上限

    def __call__(self, problem, pop, **kwargs):
        """pymoo标准调用接口"""
        return self.do(problem, pop, **kwargs)

    def do(self, problem, pop, **kwargs):
        """增强版修复算子"""
        X = pop.get("X")

        # 迭代修复过程
        for _ in range(10):  # 增加迭代次数
            # 边界约束
            X = np.clip(X, 0, self.bounds)

            # 归一化处理
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / (row_sums + 1e-10)

            # 营养约束检查
            if hasattr(problem, 'evaluator'):
                # 明确指定dtype=torch.float32
                X_tensor = torch.tensor(X, device=problem.evaluator.device, dtype=torch.float32)
                nutrients = X_tensor @ problem.evaluator.nutrition
                viol = torch.max(
                    torch.max(problem.evaluator.lower_bounds - nutrients,
                              torch.tensor(0, device=problem.evaluator.device)),
                    torch.max(nutrients - problem.evaluator.upper_bounds,
                              torch.tensor(0, device=problem.evaluator.device))
                )
                if viol.max() <= 1e-6:
                    break

        pop.set("X", X)
        return pop


# 自定义采样器 - 确保初始种群满足总和=1
class DirichletSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        # 生成严格满足总和=1的初始种群
        samples = np.random.dirichlet(np.ones(problem.n_var), size=n_samples)

        # 应用原料用量限制
        samples = samples * problem.xu  # 乘以各原料上限

        # 重新归一化确保总和=1
        row_sums = samples.sum(axis=1, keepdims=True)
        samples = samples / row_sums

        return samples


class FeedProblem(Problem):
    def __init__(self, evaluator, n_var=17, config_path="configs/ga_config.yaml"):
        """
        饲料配方优化问题定义 (与FeedEvaluator完全兼容)

        Args:
            evaluator: FeedEvaluator实例
            n_var: 变量维度（固定为17种原料）
        """
        # 约束总数 = 1(总和) + 17(原料下限) + 17(原料上限) + 10(营养下限) + 10(营养上限) = 55
        super().__init__(
            n_var=n_var,
            n_obj=11,  # 成本、营养
            n_constr=55,
            xl=np.zeros(n_var),
            xu=evaluator.ingredient_upper_bounds.cpu().numpy()
        )
        self.evaluator = evaluator
        self.precision = evaluator.precision
        self.current_gen = 0
        self.nutrient_names = evaluator.get_nutrient_names()
        self.repair = NormalizationRepair(self.xu)
        ref_point = torch.tensor([200.0, 0.0, 0.0])
        self.strategy = HybridStrategy(ref_point)
        self.ga_history = []
        self.best_solutions = []
        self.raw_objectives = None  # 存储原始目标值
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.ref_point = torch.tensor(
            config['ref_point'],
            dtype=torch.float32
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # 转换为PyTorch张量
        pop = self.repair.do(self, Population.new(X=X))
        X_tensor = torch.tensor(pop.get("X"), device=self.evaluator.device, dtype=self.precision)

        # 计算原始目标值（含惩罚值）
        objectives = self.evaluator(X_tensor)
        # 检查解是否有效（是否被施加了惩罚值）
        is_valid = (objectives[:, 0] < self.evaluator.penalty_value * 0.9)  # 假设惩罚值为1e6

        # 计算约束违反量（正值表示违反）
        with torch.no_grad():
            # 原料约束
            ingredient_lower_viol = (self.evaluator.ingredient_lower_bounds - X_tensor).clamp(min=0)
            ingredient_upper_viol = (X_tensor - self.evaluator.ingredient_upper_bounds).clamp(min=0)

            # 营养约束
            nutrient_values = X_tensor @ self.evaluator.nutrition
            nutrient_lower_viol = (self.evaluator.lower_bounds - nutrient_values).clamp(min=0)
            nutrient_upper_viol = (nutrient_values - self.evaluator.upper_bounds).clamp(min=0)

            # 合并所有约束（总和约束 + 原料 + 营养）
            sum_viol = torch.abs(X_tensor.sum(dim=1) - 1.0).unsqueeze(1)
            G = torch.cat([sum_viol, ingredient_lower_viol, ingredient_upper_viol,
                           nutrient_lower_viol, nutrient_upper_viol], dim=1)

            # 对无效解施加额外约束违反惩罚
            F_processed = torch.cat([
                objectives[:, 0].unsqueeze(1),  # 成本
                -objectives[:, 1:11]  # 营养指标（取负）
            ], dim=1)

            # 无效解的目标值设为极大值，约束违反量也设为极大值
            F_processed[~is_valid, :] = 1e6  # 统一惩罚
            G[~is_valid, :] = 1e6  # 替代原来的 += 1e6

            out["F"] = F_processed.cpu().numpy()
            out["G"] = G.cpu().numpy()

        # 保存原始目标值
        self.raw_objectives = objectives
        self._update_best_solutions(out)

        if self.current_gen % 5 == 0:
            self._debug_output(out)

        self.current_gen += 1

    def _update_best_solutions(self, out):
        """更新历史最优解集"""
        G_np = out["G"] if isinstance(out["G"], np.ndarray) else out["G"].cpu().numpy()
        feasible_mask = (G_np <= 1e-6).all(axis=1)  # 可行解需满足所有约束
        if np.any(feasible_mask):
            # 情况1：存在可行解
            if isinstance(out["F"], torch.Tensor):
                # 如果F是PyTorch张量，使用PyTorch操作
                feasible_F = out["F"][feasible_mask, 0]
                best_idx = torch.argmin(feasible_F).item()
                best_solution = self.raw_objectives[feasible_mask][best_idx]
            else:
                # 如果F是NumPy数组，使用NumPy操作
                feasible_F = out["F"][feasible_mask, 0]
                best_idx = np.argmin(feasible_F)
                best_solution = self.raw_objectives[feasible_mask][best_idx]
        else:
            # 情况2：无可行解，选择约束违反最小的
            total_violation = np.sum(np.maximum(G_np, 0), axis=1)
            best_idx = np.argmin(total_violation)
            best_solution = self.raw_objectives[best_idx]

        self.best_solutions.append(best_solution.cpu().numpy())

    def _debug_output(self, out):
        F_np = out["F"] if isinstance(out["F"], np.ndarray) else out["F"].cpu().numpy()

        fronts = NonDominatedSorting().do(F_np)
        if len(fronts[0]) > 0:
            pf_indices = fronts[0]
            pf_objectives = self.raw_objectives[pf_indices].cpu().numpy()

            # 获取营养素名称和对应索引
            nutrient_names = self.evaluator.get_nutrient_names()
            energy_idx = nutrient_names.index('Energy')
            lysine_idx = nutrient_names.index('L')

            print("\n当前帕累托前沿:")
            print(f"- 成本范围: {pf_objectives[:, 0].min():.2f} ~ {pf_objectives[:, 0].max():.2f} €/MT")
            print(
                f"- 赖氨酸范围: {pf_objectives[:, lysine_idx + 1].min():.3f} ~ {pf_objectives[:, lysine_idx + 1].max():.3f}%")
            print(
                f"- 能量范围: {pf_objectives[:, energy_idx + 1].min():.2f} ~ {pf_objectives[:, energy_idx + 1].max():.2f} MJ/kg")

            # 检查约束满足情况
            max_violation = out["G"][pf_indices].max(axis=1)
            feasible_count = (max_violation <= 1e-6).sum()
            print(f"可行解比例: {feasible_count}/{len(pf_indices)} ({feasible_count / len(pf_indices):.1%})")


def run_ga(
        evaluator,
        config_path: str = "configs/ga_config.yaml"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    执行遗传算法优化 (兼容FeedEvaluator)

    Args:
        evaluator: FeedEvaluator实例
        config_path: 遗传算法配置路径

    Returns:
        X: 最优解集 (n_samples, 17)
        F: 目标值集 (n_samples, 3) [成本, 赖氨酸, 能量]
    """
    # 加载配置
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化问题
    problem = FeedProblem(evaluator)

    # 动态加载采样器
    sampling_mapping = {
        "random": FloatRandomSampling(),
        "lhs": LHS(),
        "dirichlet": DirichletSampling()
    }
    sampling_method = config.get("sampling", "dirichlet")  # 默认使用dirichlet
    sampling = sampling_mapping.get(sampling_method.lower())

    # 配置遗传算法
    algorithm = NSGA2(
        pop_size=config['pop_size'],
        sampling=sampling,
        crossover=SBX(
            prob=config['crossover']['prob'],
            eta=config['crossover']['eta']
        ),
        mutation=PM(
            prob=config['mutation']['prob'],
            eta=config['mutation']['eta']
        ),
        eliminate_duplicates=True,
        save_history=True,  # 启用历史记录
        repair=NormalizationRepair(problem.xu),  # 注入修复算子
        constraints_handling="death_penalty"  # 自适应约束处理
    )

    # 运行优化
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', config['n_gen']),
        seed=config.get('seed', 1),
        verbose=True,
    )

    # 获取最优解集
    X = torch.tensor(res.X, dtype=torch.float32, device=evaluator.device)

    F = problem.raw_objectives.cpu()
    # 过滤无效解（惩罚值超过阈值）
    valid_mask = F[:, 0] < evaluator.penalty_value * 0.9  # 假设惩罚值为1e6
    X_valid = X[valid_mask]
    F_valid = F[valid_mask]
    # 如果无有效解，返回约束违反最小的解
    if len(X_valid) == 0:
        print("警告：无完全可行解，返回约束违反最小的解")
        G = torch.tensor(res.G, dtype=torch.float32)
        total_violation = G.sum(dim=1)
        best_idx = torch.argmin(total_violation)
        X_valid = X[best_idx].unsqueeze(0)
        F_valid = F[best_idx].unsqueeze(0)
    # 获取最终种群
    population = torch.tensor(res.pop.get("X"), dtype=torch.float32, device=evaluator.device) if hasattr(res,
                                                                                                         'pop') else torch.tensor(
        [])

    # 检查约束满足情况
    if hasattr(problem, 'raw_objectives'):
        print("\n最终结果验证:")
        best_idx = torch.argmin(F[:, 0])  # 选择成本最低的解
        nutrient_names = evaluator.get_nutrient_names()
        energy_idx = nutrient_names.index('Energy')
        lysine_idx = nutrient_names.index('L')
        print("\n最优解:")
        print(f"- 成本: {F[best_idx, 0].item():.2f} €/MT")
        print(f"- 赖氨酸: {F[best_idx, 1 + lysine_idx].item():.3f}%")
        print(f"- 能量: {F[best_idx, 1 + energy_idx].item():.2f} MJ/kg")

    # 新增: 计算每代的超体积历史
    hv_history = []
    if hasattr(res, 'history'):
        for gen_idx, record in enumerate(res.history):
            if gen_idx % 5 == 0 and hasattr(record, 'pop'):
                # 获取当前代的X值
                X_gen = torch.tensor(record.pop.get("X"), dtype=torch.float32, device=evaluator.device)

                # 重新计算原始目标值（确保使用正确的方向）
                objectives = evaluator(X_gen)
                nutrient_names = evaluator.get_nutrient_names()
                lysin_idx = 1 + nutrient_names.index('L')
                energy_idx = 1 + nutrient_names.index('Energy')

                # 构建原始目标矩阵 [成本, 赖氨酸, 能量]
                F_gen = torch.stack([
                    objectives[:, 0],  # 成本（最小化）
                    objectives[:, lysin_idx],  # 赖氨酸（最大化）
                    objectives[:, energy_idx]  # 能量（最大化）
                ], dim=1)

                def compute_hypervolume(F, ref_point):
                    """修正后的HV计算函数"""
                    # 1. 转换为numpy数组
                    F_np = F.cpu().numpy() if isinstance(F, torch.Tensor) else F.copy()

                    # 2. 统一目标方向（所有目标转为最小化）
                    # 注意：成本已经是最小化，只需转换赖氨酸和能量
                    F_np[:, 1:] = -F_np[:, 1:]  # 最大化目标取负

                    # 3. 调整参考点（与目标方向一致）
                    hv_ref_point = np.array([
                        ref_point[0],  # 成本（最小化）
                        -ref_point[1],  # 赖氨酸（转为最小化）
                        -ref_point[2]  # 能量（转为最小化）
                    ])

                    # 4. 计算非支配前沿
                    nds = NonDominatedSorting()
                    front_indices = nds.do(F_np, only_non_dominated_front=True)

                    # 5. 计算HV（仅使用非支配解）
                    if len(front_indices) > 0:
                        hv_calculator = Hypervolume(ref_point=hv_ref_point)
                        return hv_calculator(F_np[front_indices])
                    return 0.0

                # 使用配置的参考点计算HV
                hv_val = compute_hypervolume(F_gen, problem.ref_point.cpu().numpy())
                hv_history.append((gen_idx, hv_val))

                # 调试输出
                print(f"Generation {gen_idx}: HV = {hv_val:.4f}")

    # 计算HV改进量（基于最近5代的平均值）
    window_size = 5
    hv_improvement = 0.0
    if len(hv_history) >= window_size:
        recent_hv = [hv for _, hv in hv_history[-window_size:]]
        hv_improvement = (recent_hv[-1] - recent_hv[0]) / window_size

    # 提取纯HV值列表（按代数排序）
    sorted_hv_history = sorted(hv_history, key=lambda x: x[0])
    hv_values = [hv for _, hv in sorted_hv_history]

    # 返回结果
    return X_valid, F_valid, population, {
        "best_solutions": problem.best_solutions,
        "hv_history": hv_values,
        "hv_improvement": max(hv_improvement, 0),  # 确保非负
        "population_F": torch.tensor(res.pop.get("F"), dtype=torch.float32)
    }
