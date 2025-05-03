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
from botorch.utils.multi_objective import pareto
from typing import Tuple
from core.hybrid_strategy import HybridStrategy


class NormalizationRepair:
    def __init__(self, bounds):
        self.bounds = bounds  # 各原料上限

    def __call__(self, problem, pop, **kwargs):
        """pymoo标准调用接口"""
        return self.do(problem, pop, **kwargs)

    def do(self, problem, pop, **kwargs):
        """使类可被pymoo调用的关键方法"""
        X = pop.get("X")

        # Step 1: 硬约束处理（边界限制）
        X = np.clip(X, 0, self.bounds)

        # Step 2: 智能归一化
        row_sums = X.sum(axis=1, keepdims=True)
        X = X / (row_sums + 1e-10)  # 防止除零

        # Step 3: 按比例缩放以兼容边界
        scale = np.minimum(1.0, self.bounds / X).min(axis=1, keepdims=True)
        X = X * scale

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
            n_constr=54,
            xl=np.zeros(n_var),
            xu=evaluator.ingredient_upper_bounds.cpu().numpy()
        )
        self.evaluator = evaluator
        self.precision = evaluator.precision
        self.current_gen = 0
        self.nutrient_names = evaluator.get_nutrient_names()
        self.repair = NormalizationRepair(self.xu)  # 添加修复算子
        ref_point = torch.tensor([-200.0, 0.0, 0.0])
        self.strategy = HybridStrategy(ref_point)
        self.ga_history = []

    def _evaluate(self, X, out, *args, **kwargs):
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=self.precision, device=self.evaluator.device)
        pop = Population.new(X=X)
        pop = self.repair.do(self, pop)
        X = pop.get("X")

        # 计算营养含量
        nutrient_values = X_tensor @ self.evaluator.nutrition

        constraints = [
            -X_tensor,  # 原料下限
            X_tensor - self.evaluator.ingredient_upper_bounds,
            self.evaluator.lower_bounds - nutrient_values,
            nutrient_values - self.evaluator.upper_bounds
        ]

        # 合并所有约束 (n_samples, 55)
        out["G"] = torch.cat(constraints, dim=1).cpu().numpy()

        # 2. 计算目标函数
        with torch.no_grad():
            objectives = self.evaluator(X_tensor)
            nutrient_names = self.evaluator.get_nutrient_names()
            # 动态获取列索引
            cost_idx = 0  # 成本是第0列
            lysin_idx = 1 + nutrient_names.index('L')  # 赖氨酸在营养指标中的位置 +1（因成本占第0列）
            energy_idx = 1 + nutrient_names.index('Energy')  # 同理
            out["F"] = objectives[:, [cost_idx, lysin_idx, energy_idx]].cpu().numpy()  # 成本、赖氨酸、能量
            out["F"][:, 0] *= -1  # 成本取负以实现最小化

        if self.current_gen % 5 == 0:
            fronts = NonDominatedSorting().do(out["F"])
            pareto_mask = fronts[0]

            # 确保转换为正确的张量格式
            pareto_front = torch.tensor(out["F"][pareto_mask],
                                        dtype=torch.float32,
                                        device=self.evaluator.device)

            # 调整目标方向（如果需要）
            # 假设第一个目标是成本（最小化），其他是最大化
            pareto_front[:, 0] = -pareto_front[:, 0]

            try:
                hv = self.strategy.compute_hypervolume(pareto_front)
                self.ga_history.append(hv)
                print(f"Generation {self.current_gen}: HV = {hv:.4f}")  # 调试输出
            except AttributeError as e:
                print(f"Error computing HV: {str(e)}")
                print("检查 evaluator.strategy 是否存在:", hasattr(self.evaluator, 'strategy'))

        self.current_gen += 1


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

    # 后处理
    X = torch.tensor(res.X, dtype=torch.float32, device=evaluator.device)
    F = torch.tensor(res.F, dtype=torch.float32, device=evaluator.device)

    # 恢复成本原始值 (去掉负号)
    F[:, 0] = -F[:, 0]
    population = torch.tensor(res.pop.get("X"), dtype=torch.float32, device=evaluator.device) if hasattr(res,
                                                                                                         'pop') else torch.tensor(
        [])
    ga_history = problem.ga_history if hasattr(problem, 'ga_history') else []
    return X, F, population, ga_history


def get_pareto_front(X: torch.Tensor, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取帕累托前沿解

    Args:
        X: 解集 (n_samples, 17)
        F: 目标值 [成本, 赖氨酸, 能量]

    Returns:
        pareto_X: 帕累托解
        pareto_F: 帕累托前沿
    """
    # 调整目标方向: 成本最小化，其他最大化
    adjusted_F = torch.stack([
        -F[:, 0],  # 成本取反
        F[:, 1],  # 赖氨酸
        F[:, 2]  # 能量
    ], dim=1)
    pareto_mask = pareto.is_non_dominated(adjusted_F)
    return X[pareto_mask], F[pareto_mask]
