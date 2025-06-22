import time

import torch
import gpytorch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement, \
    qLogNoisyExpectedHypervolumeImprovement, qExpectedHypervolumeImprovement, qLogExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective import Hypervolume
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.transforms import normalize
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, Optional, Dict, List
from typing import Any, Callable, Dict, List, Optional, Union
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


class BOOptimizer:
    class BenchmarkMonitor:
        def __init__(
                self,
                ref_point: Tensor,
                num_outputs: int,
                dim: int,
                tkwargs: Dict[str, Any],
                obj: Optional[Callable] = None,
                constraints: Optional[List[Callable]] = None,
        ):
            self.ref_point = ref_point
            self.num_outputs = num_outputs
            self.dim = dim
            self.tkwargs = tkwargs
            self.obj = obj
            self.constraints = constraints

            # 存储容器
            self.X = torch.zeros(0, dim, **tkwargs)
            self.Y = torch.zeros(0, num_outputs, **tkwargs)
            self.hv_history = []
            self.pareto_X = []
            self.pareto_Y = []
            self.all_hvs = torch.zeros(0, **tkwargs)

        def update(self, X: Tensor, Y: Tensor):
            """更新观测数据"""
            self.X = torch.cat([self.X, X], dim=0)
            self.Y = torch.cat([self.Y, Y], dim=0)

        def record_pf_and_hv(self):
            """记录当前帕累托前沿和超体积"""
            if len(self.Y) == 0:
                self.hv_history.append(0.0)
                return

            if self.obj is not None:
                obj_values = self.obj(self.Y)
            else:
                obj_values = self.Y.clone()

            if self.constraints is not None:
                constraint_values = torch.stack(
                    [c(self.Y) for c in self.constraints], dim=-1
                )
                feas = (constraint_values <= 0.0).all(dim=-1)
                obj_values[~feas] = self.ref_point

            # 确保至少有一个点优于参考点
            if (obj_values < self.ref_point).any():
                pareto_mask = is_non_dominated(obj_values)
                self.pareto_X.append(self.X[pareto_mask].tolist())
                self.pareto_Y.append(obj_values[pareto_mask].tolist())

                partitioning = DominatedPartitioning(
                    ref_point=self.ref_point,
                    Y=obj_values[pareto_mask]
                )
                hv = partitioning.compute_hypervolume().item()
            else:
                hv = 0.0

            self.hv_history.append(hv)
            print(f"Current hypervolume: {hv:.4f}")
            print(f"Pareto front size: {len(self.pareto_Y[-1]) if len(self.pareto_Y) > 0 else 0}")

        def record_all_hvs(self):
            """记录所有观测点的超体积"""
            if self.obj is not None:
                obj_values = self.obj(self.Y)
            else:
                obj_values = self.Y

            if self.constraints is not None:
                constraint_values = torch.stack(
                    [c(self.Y) for c in self.constraints], dim=-1
                )
                feas = (constraint_values <= 0.0).all(dim=-1)
                obj_values[~feas] = self.ref_point

            self.all_hvs = torch.zeros(len(self.Y), **self.tkwargs)
            partitioning = DominatedPartitioning(ref_point=self.ref_point)
            for i in range(len(self.Y)):
                partitioning.update(Y=obj_values[i:i + 1])
                self.all_hvs[i] = partitioning.compute_hypervolume()

        def get_outputs(self) -> Dict:
            """获取监控结果"""
            return {
                "hv_history": self.hv_history,
                "pareto_X": self.pareto_X,
                "pareto_Y": self.pareto_Y,
                "all_hvs": self.all_hvs.tolist(),
            }

    def __init__(
            self,
            bounds: Dict[str, List[float]],
            ref_point: Optional[torch.Tensor] = None,
            weights: Optional[torch.Tensor] = None,
            initial_sample_size: Optional[int] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            tol: float = 0.05,
            mc_samples: int = 64,
            monitor_config: Optional[Dict] = None,
            seed: Optional[int] = None
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
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([mc_samples]),  # 这里合并样本数和维度
                seed=seed
            )
        else:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

        self.tol = tol

        self.monitor = None
        if monitor_config:
            self.monitor = self.BenchmarkMonitor(
                ref_point=ref_point,
                num_outputs=ref_point.shape[-1],
                dim=self.bounds.shape[-1],
                tkwargs={"device": self.device, "dtype": self.precision},
                obj=monitor_config.get("obj"),
                constraints=monitor_config.get("constraints"),
            )

    def _update_model(self):
        """更新高斯过程模型"""
        valid_mask = ~torch.any(self._Y >= 1e5, dim=1)
        X_train = self._X[valid_mask]
        Y_train = self._Y[valid_mask]
        Y_train = Y_train.clamp_min(1e-6)

        # 使用ARD核函数，为每个维度设置不同长度尺度
        covar_module = ScaleKernel(
            RBFKernel(  # 替换MaternKernel
                ard_num_dims=X_train.shape[-1],
                lengthscale_constraint=Interval(0.1, 10.0)  # 调整范围
            ),
            outputscale_constraint=Interval(0.5, 10.0)
        )

        mean_module = gpytorch.means.ConstantMean()
        # 改进的模型配置
        self._model = SingleTaskGP(
            X_train,
            Y_train,
            covar_module=covar_module,
            input_transform=Normalize(d=X_train.shape[-1], bounds=self.bounds),
            outcome_transform=Standardize(m=Y_train.shape[-1]),
            # 添加均值模块
            mean_module=mean_module,
        )
        # 设置噪声约束
        y_std = Y_train.std().item()
        if y_std < 1e-8:
            noise_lb, noise_ub = 1e-5, 1e-4
        else:
            noise_lb = max(1e-5, y_std * 1e-4)
            noise_ub = max(noise_lb + 1e-5, min(0.1, y_std * 0.1))

        self._model.likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.Interval(noise_lb, noise_ub)
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
            # acq_func = qLogExpectedHypervolumeImprovement(
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
        new_Y = evaluator.parent(candidates)
        self._X = torch.cat([self._X, candidates])
        self._Y = torch.cat([self._Y, new_Y])
        self._update_model()

        return self._X, self._Y

    def trust_optimize_step_qehvi(self, evaluator) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于帕累托解实际分布的动态信任域优化"""
        # 1. 计算帕累托解的实际分布范围（而非全局bounds）
        pareto_min = self._X.min(dim=0).values
        pareto_max = self._X.max(dim=0).values
        pareto_range = pareto_max - pareto_min

        # 2. 动态设置信任域（基于解集分布而非固定比例）
        trust_radius = 0.2 * pareto_range  # 在帕累托解实际范围的20%内搜索
        center_X = self._X.mean(dim=0)  # 或使用 hypervolume贡献最大的解

        # 3. 确保信任域不突破物理边界（关键修正！）
        trust_bounds = torch.stack([
            torch.maximum(center_X - trust_radius, self.bounds[0]),  # 不低于物理下限
            torch.minimum(center_X + trust_radius, self.bounds[1])  # 不超过物理上限
        ], dim=0)

        # 4. 带约束的优化（确保配方可行性）
        inequality_constraints, equality_constraints = evaluator.get_acquisition_constraints(dtype=self.precision)

        weighted_Y = self._Y * self.weights
        # 5. 优化执行（增加可行性检查）
        for attempt in range(3):  # 最多尝试3次
            candidates, _ = optimize_acqf(
                # acq_function=qExpectedHypervolumeImprovement(
                acq_function=qLogExpectedHypervolumeImprovement(
                    model=self._model,
                    ref_point=self.ref_point.to(self.device),
                    partitioning=FastNondominatedPartitioning(
                        ref_point=self.ref_point,
                        Y=torch.cat([
                            self._model.posterior(normalize(self._X, self.bounds)).mean * self.weights,
                            weighted_Y
                        ])
                    ),
                    sampler=self.sampler,
                ),
                bounds=trust_bounds,
                q=1,
                num_restarts=5,
                raw_samples=100,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                options={"batch_limit": 3, "maxiter": 100},
                sequential=True,
            )

        # 评估并更新
        new_Y = evaluator(candidates)
        self._X = torch.cat([self._X, candidates])
        self._Y = torch.cat([self._Y, new_Y])
        self._update_model()

        return self._X, self._Y

    def optimize_step_qnehvi(self, evaluator) -> Tuple[torch.Tensor, torch.Tensor]:

        X_baseline = self._X[is_non_dominated(self._Y)] if len(self._X) > 500 else self._X
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=self._model,
            ref_point=self.ref_point,
            sampler=self.sampler,
            X_baseline=X_baseline,  # 添加基线点
            prune_baseline=True,  # 修剪基线
            cache_root=True  # 缓存
        )

        # 优化时使用原始约束
        ineq_constraints, eq_constraints = evaluator.get_acquisition_constraints(dtype=self.precision)

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=4,
            num_restarts=5,
            raw_samples=218,
            inequality_constraints=ineq_constraints,
            equality_constraints=eq_constraints,
            options={"batch_limit": 4, "maxiter": 50, "disp": True},
            sequential=False,
        )

        new_Y = evaluator(candidates)
        self._X = torch.cat([self._X, candidates])
        self._Y = torch.cat([self._Y, new_Y])
        self._update_model()

        return self._X, self._Y

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
            evaluator: 评估函数，用于计算新点的目标值
            use_dynamic_constraints:  动态约束

        Returns:
            优化后的解和目标值
        """
        if self.initial_sample_size is not None and self.initial_sample_size < len(X_init):
            selected_indices_X_init = torch.randperm(len(X_init), device=X_init.device)[:self.initial_sample_size]
            selected_indices_Y_init = torch.randperm(len(Y_init), device=Y_init.device)[:self.initial_sample_size]
            X_init = X_init[selected_indices_X_init]
            Y_init = Y_init[selected_indices_Y_init]

        self._X = X_init.to(device=self.device, dtype=self.precision)
        self._Y = Y_init.to(device=self.device, dtype=self.precision)
        self.bounds = self.bounds.to(dtype=self.precision)

        # 初始化监控器
        if self.monitor:
            self.monitor.update(self._X, self._Y)
            self.monitor.record_pf_and_hv()
            self.monitor.record_all_hvs()

        # 初始化模型
        self._update_model()
        with torch.no_grad():
            pred = self._model.posterior(normalize(self._X, self.bounds)).mean
            print(f"Model prediction range: {pred.min().item():.3f} to {pred.max().item():.3f}")


        for i in range(n_iter):
            self.current_iter += 1
            print(f"\n=== Iteration {self.current_iter} ===")

            # 执行全局优化
            self._X, self._Y = self.optimize_step_qnehvi(evaluator=evaluator)

            if self.monitor:
                self.monitor.update(self._X, self._Y)
                self.monitor.record_pf_and_hv()
            print(f"Model lengthscale: {self._model.covar_module.base_kernel.lengthscale}")
            print(f"Model outputscale: {self._model.covar_module.outputscale}")
            print(f"Model noise: {self._model.likelihood.noise}")

            # 记录并打印进展
            self._update_history(evaluator=evaluator)

        return self._X, self._Y

    def _update_history(self, evaluator):
        """更新优化历史记录"""
        valid_mask = ~torch.any(self._Y >= 1e5, dim=1)
        valid_ratio = valid_mask.float().mean().item()

        self.history['valid_ratio'].append(valid_ratio)
        self.history['X'].append(self._X.cpu().clone())
        self.history['Y'].append(self._Y.cpu().clone())

        if valid_ratio > 0:
            valid_Y = self._Y[valid_mask]
            bd = FastNondominatedPartitioning(ref_point=self.ref_point, Y=valid_Y)
            current_hv = bd.compute_hypervolume().item()
            self.history['hv'].append(current_hv)
        else:
            current_hv = 0.0
            self.history['hv'].append(0.0)

        print(
            f"Iter {self.current_iter:3d} | "
            f"Valid: {valid_ratio * 100:5.1f}% | "
            f"HV: {current_hv:.4f} | "
            f"Total points: {len(self._X)}"
        )

        if hasattr(evaluator, 'current_penalty'):
            print(f" | Penalty: {evaluator.current_penalty:.2f}", end="")
        print()

    def get_monitor_outputs(self) -> Optional[Dict]:
        """获取监控结果"""
        if self.monitor:
            outputs = {
                "hv_history": self.monitor.hv_history,
                "pareto_X": self.monitor.pareto_X,
                "pareto_Y": self.monitor.pareto_Y,
                "all_hvs": self.monitor.all_hvs.tolist() if hasattr(self.monitor, 'all_hvs') else [],
            }
            print("\n=== Monitor Outputs ===")
            print(f"Hypervolume history length: {len(outputs['hv_history'])}")
            print(f"Pareto fronts recorded: {len(outputs['pareto_X'])}")
            return outputs
        return None
