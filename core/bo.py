import gpytorch
import torch
import yaml
import numpy as np
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, Optional, Dict, List


class FeedBO:
    def __init__(
            self,
            input_dim: int,
            objective_dim: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            mc_samples: int = 128
    ):
        """
        饲料配方贝叶斯优化器（基于配置文件初始化）

        Args:
            input_dim: 输入维度（原料数量）
            objective_dim: 目标维度（成本+营养指标）
            device: 计算设备
            mc_samples: 蒙特卡洛采样数
        """
        # 加载配置文件
        self._load_configs()

        # 设备与精度设置
        self.device = torch.device(device)
        self.precision = torch.float64

        # 优化状态初始化
        self.X = None
        self.Y = None
        self.model = None
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([mc_samples]),
            device=self.device
        )

        # 模型配置
        self.input_transform = Normalize(d=input_dim)
        self.outcome_transform = Standardize(m=objective_dim)

        # 边界检查
        self._validate_bounds(input_dim, objective_dim)

    def _load_configs(self):
        """加载配置文件并转换为张量"""
        with open("configs/feed_config.yaml", encoding='utf-8') as f:
            feed_config = yaml.safe_load(f)
        with open("configs/hybrid_config.yaml", encoding='utf-8') as f:
            hybrid_config = yaml.safe_load(f)

        # 营养约束
        self.nutrient_lower = torch.tensor(
            feed_config["nutrient_bounds"]["lower"],
            dtype=self.precision
        )
        self.nutrient_upper = torch.tensor(
            feed_config["nutrient_bounds"]["upper"],
            dtype=self.precision
        )

        # 原料约束
        self.ingredient_lower = torch.tensor(
            feed_config["ingredient_bounds"]["lower"],
            dtype=self.precision
        )
        self.ingredient_upper = torch.tensor(
            feed_config["ingredient_bounds"]["upper"],
            dtype=self.precision
        )

        # 参考点
        self.ref_point = torch.tensor(
            hybrid_config["ref_point"],
            dtype=self.precision
        )

        # 全局边界（用于优化）
        self.bounds = torch.stack([
            self.ingredient_lower,
            self.ingredient_upper
        ]).to(device=self.device)

    def initialize(self, X_init: torch.Tensor, Y_init: torch.Tensor):
        """初始化优化器数据"""
        self.X = X_init.to(device=self.device, dtype=self.precision)
        self.Y = Y_init.to(device=self.device, dtype=self.precision)
        self._update_model()

    def _update_model(self):
        """更新高斯过程模型"""
        self.model = SingleTaskGP(
            self.X,
            self.Y,
            input_transform=self.input_transform,
            outcome_transform=self.outcome_transform
        ).to(self.device)

        # 设置噪声约束
        y_std = self.Y.std(dim=0)
        noise_lb = torch.clamp(y_std * 0.01, min=1e-4)
        self.model.likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.GreaterThan(noise_lb.min().item())
        )

        # 训练模型
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, max_retries=3)

    def optimize_step(self, evaluator) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行单步优化"""
        # 准备约束条件
        constraints = self._prepare_constraints()

        # 定义采集函数
        acq_func = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.ref_point.to(self.device),
            partitioning=FastNondominatedPartitioning(
                ref_point=self.ref_point.to(self.device),
                Y=self._transform_objectives(self.Y)
            ),
            sampler=self.sampler,
            objective=IdentityMCMultiOutputObjective()
        )

        # 优化采集函数
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,  # 串行优化
            num_restarts=10,
            raw_samples=512,
            inequality_constraints=constraints,
            options={"batch_limit": 5}
        )

        # 评估新点
        new_Y = evaluator(candidates)

        # 检查约束并更新数据
        if self._check_constraints(candidates, new_Y):
            self.X = torch.cat([self.X, candidates])
            self.Y = torch.cat([self.Y, new_Y])
            self._update_model()

        return self.X, self.Y

    def _prepare_constraints(self) -> List[Tuple]:
        """准备原料用量约束条件"""
        constraints = []
        for i in range(len(self.ingredient_lower)):
            # 下界约束: x[i] >= lower
            constraints.append((
                torch.tensor([i], device=self.device),
                torch.tensor([1.0], device=self.device),
                self.ingredient_lower[i].to(self.device)
            ))
            # 上界约束: -x[i] >= -upper
            constraints.append((
                torch.tensor([i], device=self.device),
                torch.tensor([-1.0], device=self.device),
                -self.ingredient_upper[i].to(self.device)
            ))
        return constraints

    def _check_constraints(self, X: torch.Tensor, Y: torch.Tensor) -> bool:
        """检查原料和营养约束"""
        # 检查原料比例
        if not (X >= self.ingredient_lower.to(X.device)).all() or \
                not (X <= self.ingredient_upper.to(X.device)).all():
            print("原料比例约束违例")
            return False

        # 检查营养指标
        Y_trans = self._transform_objectives(Y)
        if not (Y_trans[..., 1:] >= self.nutrient_lower.to(Y.device)).all() or \
                not (Y_trans[..., 1:] <= self.nutrient_upper.to(Y.device)).all():
            print("营养指标约束违例")
            return False

        return True


    def _transform_objectives(self, Y: torch.Tensor) -> torch.Tensor:
        """转换目标方向（成本最小化，其他最大化）"""
        Y_trans = Y.clone()
        Y_trans[..., 0] = -Y_trans[..., 0]  # 成本最小化转最大化
        return Y_trans

    def get_pareto_front(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前帕累托前沿"""
        if self.Y is None:
            return None, None

        Y_trans = self._transform_objectives(self.Y)
        pareto_mask = is_non_dominated(Y_trans)
        return self.X[pareto_mask], Y_trans[pareto_mask]


class FeedOptimizationSystem:
    def __init__(
            self,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        完整的饲料配方优化系统

        Args:
            ingredient_names: 原料名称列表 (e.g. ["玉米", "豆粕", ...])
            nutrient_names: 营养指标名称列表 (e.g. ["成本", "粗蛋白", "赖氨酸"])
            device: 计算设备
        """
        self.device = torch.device(device)

        # 子系统初始化
        self.bo = None
        self.current_iter = 0
        self.history = {
            'X': [],
            'Y': [],
            'hv': []
        }

    def initialize(
            self,
            ingredient_bounds: Dict[str, Tuple[float, float]],
            nutrient_bounds: Dict[str, Tuple[float, float]],
            ref_point: Dict[str, float],
            n_init_samples: int = 50
    ):
        """
        初始化优化系统

        Args:
            ingredient_bounds: 原料上下界 {"玉米": (0, 0.4), ...}
            nutrient_bounds: 营养指标上下界 {"粗蛋白": (18, 21), ...}
            ref_point: 参考点 {"成本": -170, "粗蛋白": 0, ...}
            n_init_samples: 初始样本数
        """
        # 转换为tensor格式
        bounds_tensor = self._dict_to_tensor(ingredient_bounds, self.ingredient_names)
        ref_point_tensor = self._dict_to_tensor(ref_point, self.nutrient_names, is_ref=True)

        # 生成初始样本
        X_init = self.sampler.generate_samples(
            n_samples=n_init_samples,
            ingredient_bounds=ingredient_bounds,
            nutrient_bounds=nutrient_bounds
        )

        # 评估初始样本
        Y_init = self.evaluate_formulas(X_init)

        # 初始化BO
        self.bo = FeedBO(
            input_dim=len(self.ingredient_names),
            objective_dim=len(self.nutrient_names),
            bounds={'lower': bounds_tensor[0], 'upper': bounds_tensor[1]},
            ref_point=ref_point_tensor,
            nutrient_bounds=self._prepare_bounds(nutrient_bounds),
            ingredient_bounds=bounds_tensor
        )
        self.bo.initialize(X_init, Y_init)

        # 记录历史
        self._update_history()

    def run_optimization(self, n_iter: int = 20):
        """执行优化循环"""
        for i in range(n_iter):
            self.current_iter += 1
            print(f"\n=== Iteration {self.current_iter} ===")

            # 执行BO单步优化
            X, Y = self.bo.optimize_step(evaluator=self.evaluate_formulas)

            # 记录并打印进展
            self._update_history()
            self._print_current_status()

            # 早期停止检查
            if self._check_convergence():
                print("Optimization converged!")
                break

    def evaluate_formulas(self, X: torch.Tensor) -> torch.Tensor:
        """
        评估配方性能（需要根据实际需求实现）

        Args:
            X: 配方矩阵 (n_samples × n_ingredients)

        Returns:
            Y: 目标值矩阵 (n_samples × n_objectives)
        """
        # 这里应该调用您的实际评估函数
        # 示例实现：
        cost = (X * self._get_ingredient_costs()).sum(1)  # 计算成本
        nutrients = X @ self._get_nutrient_matrix()  # 计算营养指标

        # 组合所有目标 (注意目标方向)
        Y = torch.cat([
            cost.unsqueeze(-1),  # 成本需要最小化
            nutrients[..., :2]  # 示例：只取前两个营养指标
        ], dim=-1)

        return Y.to(device=self.device, dtype=torch.float64)

    def get_optimal_formulas(self, top_k: int = 5) -> Dict:
        """获取当前最优解"""
        pareto_X, pareto_Y = self.bo.get_pareto_front()

        # 按第一个目标（成本）排序
        sorted_idx = torch.argsort(pareto_Y[:, 0])
        top_formulas = []

        for i in range(min(top_k, len(sorted_idx))):
            idx = sorted_idx[i]
            formula = {
                'ingredients': dict(zip(
                    self.ingredient_names,
                    pareto_X[idx].cpu().numpy().round(4)
                )),
                'objectives': dict(zip(
                    self.nutrient_names,
                    pareto_Y[idx].cpu().numpy().round(4)
                ))
            }
            top_formulas.append(formula)

        return {
            'pareto_front': top_formulas,
            'hypervolume': self.history['hv'][-1]
        }

    def _update_history(self):
        """更新优化历史记录"""
        self.history['X'].append(self.bo.X.cpu().clone())
        self.history['Y'].append(self.bo.Y.cpu().clone())

        # 计算当前超体积
        bd = FastNondominatedPartitioning(
            ref_point=self.bo.ref_point,
            Y=self.bo.Y
        )
        self.history['hv'].append(bd.compute_hypervolume().item())

