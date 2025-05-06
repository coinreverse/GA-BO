import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, Optional, Dict, List


class BOOptimizer:
    def __init__(
            self,
            bounds: Dict[str, List[float]],
            ref_point: Optional[torch.Tensor] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Bayesian Optimization 优化器

        Args:
            bounds: 边界字典 {'lower': [..], 'upper': [..]}
            ref_point: 参考点张量
            acqf_config: 采集函数配置
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
        if ref_point is not None:
            self.ref_point = ref_point.to(device=self.device, dtype=torch.double)
        else:
            self.ref_point = None

        # 默认目标权重（可根据实际情况调整）
        self.default_weights = torch.tensor([-1.0, 1.0, 1.0], device=self.device, dtype=torch.double)

        # 优化状态
        self._X = None
        self._Y = None
        self._model = None

    def optimize(
            self,
            X_init: torch.Tensor,
            Y_init: torch.Tensor,
            n_iter: int = 20,
            batch_size: int = 1,
            evaluator=None,
            **kwargs
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

        # 确保数据是双精度并位于正确设备
        self._X = X_init.to(device=self.device, dtype=torch.double)
        self._Y = Y_init.to(device=self.device, dtype=torch.double)

        # 确保边界也是双精度
        self.bounds = self.bounds.to(dtype=torch.double)

        # 初始化模型
        self._update_model()

        X_candidates = []
        Y_candidates = []

        for _ in range(n_iter):
            # 获取下一批候选点
            candidates = self.get_next_candidate(batch_size=batch_size)

            # 评估候选点
            Y_new = evaluator(candidates) if evaluator else self._Y[torch.cdist(candidates, self._X).argmin(dim=1)]

            # 更新数据集
            self.update(candidates, Y_new.to(dtype=torch.double))

            X_candidates.append(candidates)
            Y_candidates.append(Y_new)

        # 合并结果
        X_opt = torch.cat([self._X] + X_candidates)
        Y_opt = torch.cat([self._Y] + Y_candidates)

        return X_opt, Y_opt


    def _update_model(self):
        """更新高斯过程模型"""
        train_Y = self._Y.clone()
        # 创建模型
        self._model = SingleTaskGP(
            self._X,
            train_Y,
            input_transform=Normalize(d=self._X.shape[-1]),
            outcome_transform=Standardize(m=train_Y.shape[-1])  # 添加标准化转换
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

    def get_next_candidate(self, batch_size: int = 1) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        Y_normalized = (self._Y - self._Y.mean(0)) / (self._Y.std(0) + 1e-6)
        # 改进最佳值选择
        if Y_normalized.shape[1] > 1:  # 多目标情况
            # 使用超体积改进作为目标
            partitioning = DominatedPartitioning(
                ref_point=self.ref_point,
                Y=self._Y
            )
            best_f = partitioning.compute_hypervolume()
        else:
            best_f = Y_normalized.max()

        # 根据输出维度自动调整目标函数
        if Y_normalized.shape[1] > 1:  # 多目标情况
            if not hasattr(self, 'default_weights') or len(self.default_weights) != Y_normalized.shape[1]:
                self.default_weights = torch.ones(
                    Y_normalized.shape[1],
                    device=self.device,
                    dtype=torch.double
                )
            objective = LinearMCObjective(self.default_weights)
        else:  # 单目标情况
            objective = None

        acqf = qLogExpectedImprovement(  # 使用改进的LogEI
            model=self._model,
            best_f=best_f,
            objective=objective  # 传入目标函数
        )

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200}
        )
        # 添加边界检查
        candidates = torch.clamp(candidates, min=self.bounds[0], max=self.bounds[1])
        return candidates.detach()

    def update(self, X_new: torch.Tensor, Y_new: torch.Tensor):
        """更新观测数据"""
        self._X = torch.cat([self._X, X_new.to(device=self.device, dtype=torch.double)])
        self._Y = torch.cat([self._Y, Y_new.to(device=self.device, dtype=torch.double)])
        self._update_model()
