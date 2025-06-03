import torch
import gpytorch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement, \
    qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective, IdentityMCMultiOutputObjective
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.transforms import normalize
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, Optional, Dict, List


class BOOptimizer:
    def __init__(
            self,
            bounds: Dict[str, List[float]],
            ref_point: Optional[torch.Tensor] = None,
            weights: Optional[torch.Tensor] = None,
            initial_sample_size: Optional[int] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            nutrient_bounds: Optional[Dict[str, List[float]]] = None,
            ingredient_bounds: Optional[Dict[str, List[float]]] = None,
            tol: float = 0.05,
            mc_samples: int = 64
    ):
        """
        Bayesian Optimization 优化器

        Args:
            bounds: 边界字典 {'lower': [..], 'upper': [..]}
            ref_point: 参考点张量
            device: 计算设备
        """
        self.device = torch.device(device)
        print(f"Optimizer initialized on device: {self.device}")  # 调试信息
        self.precision = torch.double
        self.bounds = torch.tensor(
            [bounds['lower'], bounds['upper']],
            device=self.device,
            dtype=self.precision
        )
        self.ref_point = ref_point.to(device=self.device, dtype=self.precision)
        self.weights = weights.to(device=self.device, dtype=self.precision)
        self.initial_sample_size = initial_sample_size

        # 优化状态
        self._X = None
        self._Y = None
        self._model = None
        self.current_iter = 0
        self.history = {
            'X': [],
            'Y': [],
            'hv': [],
            'valid_ratio': []
        }
        self.mc_samples = mc_samples
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))

        self.tol = tol
        if nutrient_bounds:
            self.nutrient_lower = torch.tensor(nutrient_bounds['lower'],
                                               device=self.device,
                                               dtype=self.precision)
            self.nutrient_upper = torch.tensor(nutrient_bounds['upper'],
                                               device=self.device,
                                               dtype=self.precision)
        if ingredient_bounds:
            self.ingredient_lower = torch.tensor(ingredient_bounds['lower'],
                                                 device=self.device,
                                                 dtype=self.precision)
            self.ingredient_upper = torch.tensor(ingredient_bounds['upper'],
                                                 device=self.device,
                                                 dtype=self.precision)

    def _update_model(self):
        """更新高斯过程模型"""
        valid_mask = ~torch.any(self._Y >= 1e5, dim=1)
        X_valid = self._X[valid_mask]
        Y_valid = self._Y[valid_mask]

        if len(X_valid) == 0:
            raise ValueError("No valid observations available for modeling")

        # 使用有效数据继续建模
        outcome_transform = Standardize(m=Y_valid.shape[-1])
        self._model = SingleTaskGP(
            X_valid,
            Y_valid,
            input_transform=Normalize(d=X_valid.shape[-1]),
            outcome_transform=outcome_transform
        )

        # 设置噪声约束
        y_std = self._Y.std()
        noise_lb = max(1e-4, y_std * 1e-3)  # 至少1e-4，或Y标准差的0.1%
        self._model.likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.GreaterThan(noise_lb)
        )

        # 添加模型训练参数
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        fit_gpytorch_mll(
            mll,
            max_retries=3,  # 失败时重试
            max_iter=100,  # 增加迭代次数
        )


    def optimize_step_qehvi(self, evaluator) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行单步优化"""
        with torch.no_grad():
            pred = self._model.posterior(normalize(self._X, self.bounds)).mean
        partitioning = FastNondominatedPartitioning(ref_point=self.ref_point, Y=pred)
        # 添加约束模型
        inequality_constraints, equality_constraints = evaluator.get_acquisition_constraints(dtype=self.precision)
        # 定义采集函数
        acq_func = qExpectedHypervolumeImprovement(
            model=self._model,
            ref_point=self.ref_point.to(self.device),
            partitioning=partitioning,
            sampler=self.sampler,

        )

        # 优化采集函数
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,  # 串行优化
            num_restarts=10,
            raw_samples=256,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

        # 评估新点
        new_Y = evaluator(candidates)
        self._X = torch.cat([self._X, candidates])
        self._Y = torch.cat([self._Y, new_Y])
        self._update_model()

        return self._X, self._Y

    def optimize_step_qnehvi(self, evaluator) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pred = self._model.posterior(normalize(self._X, self.bounds)).mean
        inequality_constraints, equality_constraints = evaluator.get_acquisition_constraints(dtype=self.precision)

        # acq_func = qNoisyExpectedHypervolumeImprovement(
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=self._model,
            ref_point=self.ref_point.to(self.device),
            X_baseline=normalize(self._X, bounds=self.bounds),
            sampler=self.sampler,
        )
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        new_Y = evaluator(candidates)
        self._X = torch.cat([self._X, candidates])
        self._Y = torch.cat([self._Y, new_Y])
        self._update_model()


    def optimize(
            self,
            X_init: torch.Tensor,
            Y_init: torch.Tensor,
            n_iter: int = 20,
            evaluator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行贝叶斯优化

        Args:
            X_init: 初始输入点 (n x d)
            Y_init: 初始目标值 (n x m)
            n_iter: 优化迭代次数
            batch_size: 每批采样点数
            evaluator: 评估函数，用于计算新点的目标值

        Returns:
            优化后的解和目标值
        """
        if self.initial_sample_size is not None and self.initial_sample_size < len(X_init):
            selected_indices = torch.randperm(len(X_init))[:self.initial_sample_size]
            X_init = X_init[selected_indices]
            Y_init = Y_init[selected_indices]
        # 确保数据精度一致并位于正确设备
        self._X = X_init.to(device=self.device, dtype=self.precision)
        self._Y = Y_init.to(device=self.device, dtype=self.precision)
        self.bounds = self.bounds.to(dtype=self.precision)

        # 初始化模型
        self._update_model()

        for i in range(n_iter):
            self.current_iter += 1
            print(f"\n=== Iteration {self.current_iter} ===")

            # 执行BO单步优化
            # self._X, self._Y = self.optimize_step_qehvi(evaluator=evaluator)
            self._X, self._Y = self.optimize_step_qnehvi(evaluator=evaluator)

            # 记录并打印进展
            self._update_history()

        return self._X, self._Y

    def _update_history(self):
        """更新优化历史记录"""
        valid_mask = ~torch.any(self._Y >= 1e5, dim=1)
        valid_ratio = valid_mask.float().mean().item()

        self.history['valid_ratio'].append(valid_ratio)  # 记录有效解比例
        self.history['X'].append(self._X.cpu().clone())
        self.history['Y'].append(self._Y.cpu().clone())

        if valid_ratio > 0:
            # 仅计算有效解的超体积
            valid_Y = self._Y[valid_mask]
            bd = FastNondominatedPartitioning(
                ref_point=self.ref_point,
                Y=valid_Y
            )
            self.history['hv'].append(bd.compute_hypervolume().item())
        else:
            self.history['hv'].append(0.0)
