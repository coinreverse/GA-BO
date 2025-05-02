import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, Optional, Dict, List

from scipy.stats._mstats_basic import winsorize


class BOOptimizer:
    def __init__(
            self,
            bounds: Dict[str, List[float]],
            ref_point: Optional[torch.Tensor] = None,
            acqf_config: Optional[Dict] = None,
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
        self.precision = torch.float32
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
        # 允许通过acqf_config覆盖默认权重
        if acqf_config and 'weights' in acqf_config:
            self.default_weights = torch.tensor(
                acqf_config['weights'],
                device=self.device,
                dtype=torch.double
            )

        # 优化状态
        self._X = None
        self._Y = None
        self._model = None
        # 添加Turbo相关参数
        self.tr_hparams = {
            'length_init': 0.8,
            'length_min': 0.01,
            'length_max': 1.6,
            'success_streak': 3,
            'failure_streak': 3,
            'winsor_pct': 5.0,
            'max_tr_size': 2000,
            'min_tr_size': 50,
            'eps': 1e-3
        }

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
        # 确保输入是双精度
        X_init = X_init.to(dtype=self.precision, device=self.device, )
        Y_init = Y_init.to(dtype=self.precision, device=self.device)

        self.initialize(X_init, Y_init)

        X_candidates = []
        Y_candidates = []

        for _ in range(n_iter):
            # 获取下一批候选点
            candidates = self.get_next_candidate(batch_size=batch_size)

            # 评估候选点
            if evaluator is not None:
                Y_new = evaluator(candidates)
            else:
                # 如果没有提供评估器，使用最近邻近似
                _, indices = torch.cdist(candidates, self._X).min(dim=1)
                Y_new = self._Y[indices]

            # 确保新数据是双精度
            Y_new = Y_new.to(dtype=self.precision, device=self.device)

            # 更新数据集
            self.update(candidates, Y_new)

            X_candidates.append(candidates)
            Y_candidates.append(Y_new)

        # 合并结果
        X_opt = torch.cat([self._X] + X_candidates)
        Y_opt = torch.cat([self._Y] + Y_candidates)

        return X_opt, Y_opt

    def initialize(self, X: torch.Tensor, Y: torch.Tensor):
        """初始化观测数据"""
        self._X = X.to(device=self.device, dtype=torch.double)
        self._Y = Y.to(device=self.device, dtype=torch.double)
        if torch.any(torch.isnan(self._Y)) or torch.any(torch.isinf(self._Y)):
            raise ValueError("Y contains invalid values (NaN or Inf)")
        self._update_model()

    def _update_model(self):
        """更新高斯过程模型"""
        # 添加数值检查
        if torch.any(torch.isnan(self._Y)) or torch.any(torch.isinf(self._Y)):
            raise ValueError("Y contains NaN or Inf values")

        # 改进标准化方法，添加winsorize处理异常值
        winsorized_Y = torch.from_numpy(
            winsorize(
                self._Y.cpu().numpy(),
                limits=(0.05, 0.05),  # 截断前后5%的极端值
                axis=0
            )
        ).to(self._Y)

        # 标准化处理
        Y_mean = winsorized_Y.mean(0)
        Y_std = winsorized_Y.std(0)
        Y_std[Y_std < 1e-6] = 1e-6  # 防止除零
        train_Y = (winsorized_Y - Y_mean) / Y_std

        # 创建模型
        self._model = SingleTaskGP(
            self._X,
            train_Y,
            outcome_transform=Standardize(m=winsorized_Y.shape[-1])  # 添加标准化转换
        )

        # 设置更强的噪声约束
        self._model.likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.GreaterThan(1e-4)
        )

        # 添加模型训练参数
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        fit_gpytorch_mll(mll)


    def get_next_candidate(self, batch_size: int = 1) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        Y_normalized = (self._Y - self._Y.mean(0)) / (self._Y.std(0) + 1e-6)
        # 改进最佳值选择
        if Y_normalized.shape[1] > 1:  # 多目标情况
            # 使用超体积改进作为目标
            partitioning = DominatedPartitioning(
                ref_point=self.ref_point,
                Y=Y_normalized
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
