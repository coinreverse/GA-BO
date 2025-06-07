import torch
from typing import Optional
import yaml


class FeedEvaluator:
    def __init__(self, config_path: str = "configs/feed_config.yaml", device: Optional[str] = None,
                 precision: str = 'float32'):
        """
        从 YAML 文件加载所有配置的饲料配方评估器

        Args:
            config_path: YAML 配置文件路径
            device: 可选，覆盖配置文件中的设备设置
        """
        with open(config_path, encoding='utf-8') as f:
            feed_config = yaml.safe_load(f)

        # 设置计算设备（优先使用传入的device参数）
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(feed_config.get("settings", {}).get("device", "cpu"))

        # 加载数据并转换为张量
        self.precision = torch.float32 if precision == 'float32' else torch.float64
        self.costs = torch.tensor(feed_config["costs"], device=self.device, dtype=self.precision)
        self.nutrition = torch.tensor(feed_config["nutrition"], device=self.device, dtype=self.precision)
        self.lower_bounds = torch.tensor(feed_config["nutrient_bounds"]["lower"], device=self.device, dtype=self.precision)
        self.upper_bounds = torch.tensor(feed_config["nutrient_bounds"]["upper"], device=self.device, dtype=self.precision)
        self.ingredient_lower_bounds = torch.tensor(feed_config["ingredient_bounds"]["lower"], device=self.device,
                                                    dtype=self.precision)
        self.ingredient_upper_bounds = torch.tensor(feed_config["ingredient_bounds"]["upper"], device=self.device,
                                                    dtype=self.precision)

        # 加载其他配置
        self.tol = feed_config["settings"]["tol"]
        self.max_iter = feed_config["settings"]["max_iter"]
        self.penalty_value = 1e6

        # 输入验证
        assert self.costs.shape == (17,), "成本向量必须为17维"
        assert self.nutrition.shape == (17, 10), "营养矩阵必须有17种原料和10种营养素"

    def __call__(self, X: torch.Tensor, tol: float = 0.05) -> torch.Tensor:
        """
        评估饲料配方 (兼容NumPy和PyTorch输入)
        修改说明：
        - 不满足约束的配方会返回高惩罚值（1e6）
        - 满足约束的配方正常计算目标值

        Args:
            X: (n_samples, 17) 原料配比矩阵 (NumPy数组或PyTorch张量)
            tol: 配比总和的容差阈值
        """

        # 如果需要，将输入转换为pytorch张量
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.precision, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.precision)

        # 确保我们处于评估模式（没有梯度）
        with torch.no_grad():
            # Calculate basic values
            cost = X @ self.costs
            nutrients = X @ self.nutrition

            # 将任何 NaN/inf 值替换为大数字
            nutrients = torch.where(
                torch.isfinite(nutrients),
                nutrients,
                torch.tensor(1e4, device=self.device)
            )

            # 检查约束
            valid = self._check_constraints(X, nutrients, tol)

            # 初始化全惩罚值的输出（形状=(n_samples, 11)）
            output = torch.full(
                (X.shape[0], 11),
                self.penalty_value,  # 例如1e6
                device=self.device,
                dtype=self.precision
            )

            # 对有效解填充实际计算值
            if valid.any():
                output[valid] = torch.cat([
                    cost[valid].unsqueeze(1),
                    nutrients[valid]
                ], dim=1)

            return output


    def _check_constraints(self, X: torch.Tensor, nutrients: torch.Tensor, tol: float) -> torch.Tensor:
        """
        检查给定配方的所有约束

        Args:
            X: 成分比例 (n_samples, 17)
            nutrients: 计算的营养素 (n_samples, 10)
            tol: 比率总和的容差

        Returns:
            布尔张量指示哪些配方满足所有限制
        """
        # 检查成分总和是否约为 1（在误差范围内）
        sum_check = torch.abs(torch.sum(X, dim=1) - 1.0) <= tol

        # 检查成分边界
        ingredient_lower_check = torch.all(X >= self.ingredient_lower_bounds, dim=1)
        ingredient_upper_check = torch.all(X <= self.ingredient_upper_bounds, dim=1)

        # 检查营养界
        nutrient_lower_check = torch.all(nutrients >= self.lower_bounds, dim=1)
        nutrient_upper_check = torch.all(nutrients <= self.upper_bounds, dim=1)


        # 必须满足所有约束
        return sum_check & ingredient_lower_check & ingredient_upper_check & nutrient_lower_check & nutrient_upper_check

    @staticmethod
    def get_nutrient_names() -> list:
        """获取营养素名称"""
        return ['CF', 'Ca', 'AP', 'DM', 'CP', 'MC', 'T', 'Tp', 'Energy', 'L']

    def get_acquisition_constraints(self, dtype=None):
        """获取用于采集函数的约束条件（不等式和等式约束）

        返回:
            tuple: (ineq_constraints, eq_constraints) 包含不等式和等式约束的元组
        """
        # 转换所有预存张量的类型
        nutrition = self.nutrition.to(dtype=dtype)
        costs = self.costs.to(dtype=dtype)
        ingredient_lower_bounds = self.ingredient_lower_bounds.to(dtype=dtype)
        ingredient_upper_bounds = self.ingredient_upper_bounds.to(dtype=dtype)
        lower_bounds = self.lower_bounds.to(dtype=dtype)
        upper_bounds = self.upper_bounds.to(dtype=dtype)

        # 创建全1向量用于等式约束
        ones = torch.ones(17, dtype=dtype, device=self.device)

        # 不等式约束列表
        ineq_constraints = []

        # 1. 用料比例下限约束
        for i in range(17):
            indices = torch.tensor([i], device=self.device)
            coefficients = torch.tensor([1.0], device=self.device, dtype=dtype)
            rhs = float(ingredient_lower_bounds[i].item())
            ineq_constraints.append((indices, coefficients, rhs))

        # 2. 用料比例上限约束
        for i in range(17):
            indices = torch.tensor([i], device=self.device)
            coefficients = torch.tensor([-1.0], device=self.device, dtype=dtype)
            rhs = -float(ingredient_upper_bounds[i].item())
            ineq_constraints.append((indices, coefficients, rhs))

        # 3. 营养下限约束
        for j in range(10):
            indices = torch.arange(17, device=self.device)
            coefficients = nutrition[:, j].clone()  # 确保是独立张量
            rhs = float(lower_bounds[j].item())
            ineq_constraints.append((indices, coefficients, rhs))

        # 4. 营养上限约束
        for j in range(10):
            indices = torch.arange(17, device=self.device)
            coefficients = -nutrition[:, j].clone()
            rhs = -float(upper_bounds[j].item())
            ineq_constraints.append((indices, coefficients, rhs))

        # 5. 成本上限约束
        indices = torch.arange(17, device=self.device)
        coefficients = costs.clone()
        rhs = 143.0
        ineq_constraints.append((indices, coefficients, rhs))

        # 等式约束
        eq_constraints = [
            (torch.arange(17, device=self.device), ones, 1.0)
        ]

        return ineq_constraints, eq_constraints

