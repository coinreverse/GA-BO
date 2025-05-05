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
from typing import Tuple
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
    def __init__(self, evaluator, n_var=17):
        """
        饲料配方优化问题定义 (与FeedEvaluator完全兼容)

        Args:
            evaluator: FeedEvaluator实例
            n_var: 变量维度（固定为17种原料）
        """
        # 约束总数 = 1(总和) + 17(原料下限) + 17(原料上限) + 10(营养下限) + 10(营养上限) = 55
        super().__init__(
            n_var=n_var,
            n_obj=3,  # 成本、赖氨酸、能量
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

    def _evaluate(self, X, out, *args, **kwargs):
        # 转换为PyTorch张量
        pop = self.repair.do(self, Population.new(X=X))
        X_tensor = torch.tensor(pop.get("X"), device=self.evaluator.device, dtype=self.precision,)

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
            out["G"] = torch.cat([
                sum_viol,
                ingredient_lower_viol,
                ingredient_upper_viol,
                nutrient_lower_viol,
                nutrient_upper_viol
            ], dim=1).cpu().numpy()

        # 3. 计算目标函数（统一为最小化）
        objectives = self.evaluator(X_tensor)
        nutrient_names = self.evaluator.get_nutrient_names()
        lysin_idx = 1 + nutrient_names.index('L')
        energy_idx = 1 + nutrient_names.index('Energy')

        # 保存原始目标值
        self.raw_objectives = torch.stack([
            objectives[:, 0],  # 成本（最小化）
            objectives[:, lysin_idx],  # 赖氨酸（最大化）
            objectives[:, energy_idx]  # 能量（最大化）
        ], dim=1)

        # 统一为最小化问题（使用倒数处理最大化目标）
        out["F"] = torch.stack([
            objectives[:, 0],  # 成本（最小化）
            1.0 / (objectives[:, lysin_idx] + 1e-10),  # 赖氨酸最大化转为最小化
            1.0 / (objectives[:, energy_idx] + 1e-10)  # 能量最大化转为最小化
        ], dim=1).cpu().numpy()

        self._update_best_solutions(out)

        if self.current_gen % 5 == 0:
            self._debug_output(X_tensor, out)

        self.current_gen += 1

    def _update_best_solutions(self, out):
        """更新历史最优解集"""
        feasible_mask = (out["G"] <= 1e-6).all(axis=1)  # 可行解需满足所有约束
        if np.any(feasible_mask):
            # 选择可行解中成本最低的
            best_idx = np.argmin(out["F"][feasible_mask, 0])
            self.best_solutions.append(self.raw_objectives[feasible_mask][best_idx].cpu().numpy())
        else:
            # 若无可行解，选择总违反量最小的
            total_violation = np.sum(np.maximum(out["G"], 0), axis=1)
            best_idx = np.argmin(total_violation)
            self.best_solutions.append(self.raw_objectives[best_idx].cpu().numpy())


    def _debug_output(self, X_tensor, out):
        fronts = NonDominatedSorting().do(out["F"])
        if len(fronts[0]) > 0:
            pf_indices = fronts[0]
            pf_objectives = self.raw_objectives[pf_indices].cpu().numpy()

            print("\n当前帕累托前沿:")
            print(f"- 成本范围: {pf_objectives[:, 0].min():.2f} ~ {pf_objectives[:, 0].max():.2f} €/MT")
            print(f"- 赖氨酸范围: {pf_objectives[:, 1].min():.3f} ~ {pf_objectives[:, 1].max():.3f}%")
            print(f"- 能量范围: {pf_objectives[:, 2].min():.2f} ~ {pf_objectives[:, 2].max():.2f} MJ/kg")

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
        repair=NormalizationRepair(problem.xu),  # 注入修复算子
        constraints_handling="adaptive_penalty"  # 自适应约束处理
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

    # 后处理 - 使用原始目标值而不是转换后的值
    if hasattr(problem, 'raw_objectives'):
        F = problem.raw_objectives.cpu()
    else:
        F = torch.tensor(res.F, dtype=torch.float32, device=evaluator.device)
        # 尝试恢复原始值（如果使用了转换）
        F[:, 1] = 1.0 / (F[:, 1] + 1e-10)  # 赖氨酸
        F[:, 2] = 1.0 / (F[:, 2] + 1e-10)  # 能量

    # 确保成本为正
    F[:, 0] = torch.abs(F[:, 0])

    # 获取最终种群
    population = torch.tensor(res.pop.get("X"), dtype=torch.float32, device=evaluator.device) if hasattr(res, 'pop') else torch.tensor([])

    # 检查约束满足情况
    if hasattr(problem, 'raw_objectives'):
        print("\n最终结果验证:")
        best_idx = torch.argmin(F[:, 0])  # 选择成本最低的解
        best_X = population[best_idx] if len(population) > 0 else X[best_idx]

        # 验证约束
        ingredient_lower_viol = (evaluator.ingredient_lower_bounds - best_X).clamp(min=0)
        ingredient_upper_viol = (best_X - evaluator.ingredient_upper_bounds).clamp(min=0)
        nutrient_values = best_X @ evaluator.nutrition
        nutrient_lower_viol = (evaluator.lower_bounds - nutrient_values).clamp(min=0)
        nutrient_upper_viol = (nutrient_values - evaluator.upper_bounds).clamp(min=0)
        sum_viol = torch.abs(best_X.sum() - 1.0)

        print(f"- 原料下限违反量: {ingredient_lower_viol.max().item():.2e}")
        print(f"- 原料上限违反量: {ingredient_upper_viol.max().item():.2e}")
        print(f"- 营养下限违反量: {nutrient_lower_viol.max().item():.2e}")
        print(f"- 营养上限违反量: {nutrient_upper_viol.max().item():.2e}")
        print(f"- 总和约束违反量: {sum_viol.item():.2e}")

        print("\n最优解:")
        print(f"- 成本: {F[best_idx, 0].item():.2f} €/MT")
        print(f"- 赖氨酸: {F[best_idx, 1].item():.3f}%")
        print(f"- 能量: {F[best_idx, 2].item():.2f} MJ/kg")
    # 在run_ga函数最后添加：
    best_idx = torch.argmin(F[:, 0])  # 确保使用相同的选择标准
    best_solution = {
        "Cost": F[best_idx, 0].item(),
        "Lysine": F[best_idx, 1].item(),
        "Energy": F[best_idx, 2].item()
    }
    print("\nFinal Best Solution:")
    print(f"- Cost: {best_solution['Cost']:.2f} €/MT")
    print(f"- Lysine: {best_solution['Lysine']:.3f}%")
    print(f"- Energy: {best_solution['Energy']:.2f} MJ/kg")

    return X, F, population, problem.best_solutions