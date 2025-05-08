import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
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
            weights: Optional[torch.Tensor] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.ref_point = ref_point.to(device=self.device, dtype=torch.double)
        self.weights = weights.to(device=self.device, dtype=torch.double)


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


        for _ in range(n_iter):
            # 获取下一批候选点
            candidates = self.get_next_candidate(batch_size=batch_size)
            print("-----------------------",candidates)

            # 评估候选点
            Y_new = evaluator(candidates) if evaluator else self._Y[torch.cdist(candidates, self._X).argmin(dim=1)]
            print("=========================",Y_new)

            print(f"Iter {_}: Candidates = {candidates}")
            print(f"Iter {_}: Evaluated Y = {Y_new}")
            with torch.no_grad():
                posterior = self._model.posterior(candidates).mean
                print(f"Iter {_}: GP Predicted Y = {posterior}")
            valid_mask = (Y_new[:, 0] < 1e3)  # 只保留有效解
            if valid_mask.any():  # 如果有有效解才更新
                self.update(candidates[valid_mask], Y_new[valid_mask].to(dtype=torch.double))
            else:
                print(f"Iter {_}: 无有效解")


        print("Final X----------------", self._X)
        print("Final Y=============================", self._Y)
        return self._X, self._Y


    def _update_model(self):
        """更新高斯过程模型"""
        outcome_transform = Standardize(m=self._Y.shape[-1])
        self._model = SingleTaskGP(
            self._X,
            self._Y,
            input_transform=Normalize(d=self._X.shape[-1]),
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

    def get_next_candidate(self, batch_size: int = 1) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

            # 只使用有效解
        valid_mask = (self._Y[:, 0] < 1e5)
        if not valid_mask.any():
            # 无有效解时退回随机采样
            return self._random_sample(batch_size)

        Y_valid = self._Y[valid_mask]

        # 多目标处理
        if Y_valid.shape[1] > 1:
            if self.weights is not None:
                # 修正GenericMCObjective的使用方式
                def objective(samples, X=None):
                    return samples @ self.weights
                objective = GenericMCObjective(objective)
                best_f = (Y_valid * self.weights).sum(dim=1).max()
            else:
                # 使用参考点进行多目标优化
                if self.ref_point is None:
                    raise ValueError("For multi-objective optimization, either weights or ref_point must be provided")
                partitioning = DominatedPartitioning(
                    ref_point=self.ref_point,
                    Y=Y_valid
                )
                best_f = partitioning.compute_hypervolume()
                objective = None
        else:
            best_f = Y_valid.max()
            objective = None

        acqf = qLogExpectedImprovement(
            model=self._model,
            best_f=best_f,
            objective=objective
        )

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5}
        )

        return candidates.detach()

    def _random_sample(self, batch_size: int) -> torch.Tensor:
        """在边界内随机采样"""
        return torch.rand(
            batch_size,
            self.bounds.shape[1],
            device=self.device,
            dtype=self.precision
        ) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]


    def update(self, X_new: torch.Tensor, Y_new: torch.Tensor):
        """更新观测数据"""
        X_all = torch.cat([self._X, X_new.to(device=self.device, dtype=torch.double)])
        Y_all = torch.cat([self._Y, Y_new.to(device=self.device, dtype=torch.double)])

        # 全局过滤有效解
        valid_mask = Y_all[:, 0] < 1e3
        self._X = X_all[valid_mask]
        self._Y = Y_all[valid_mask]
        self._update_model()
