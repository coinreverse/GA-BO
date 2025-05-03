import torch
from typing import Tuple, Optional
import pandas as pd
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
            config = yaml.safe_load(f)

        # 设置计算设备（优先使用传入的device参数）
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(config.get("settings", {}).get("device", "cpu"))

        # 加载数据并转换为张量
        self.precision = torch.float32 if precision == 'float32' else torch.float64
        self.costs = torch.tensor(config["costs"], device=self.device, dtype=self.precision)
        self.nutrition = torch.tensor(config["nutrition"], device=self.device, dtype=self.precision)
        self.lower_bounds = torch.tensor(config["nutrient_bounds"]["lower"], device=self.device, dtype=self.precision)
        self.upper_bounds = torch.tensor(config["nutrient_bounds"]["upper"], device=self.device, dtype=self.precision)
        self.ingredient_lower_bounds = torch.tensor(config["ingredient_bounds"]["lower"], device=self.device,
                                                    dtype=self.precision)
        self.ingredient_upper_bounds = torch.tensor(config["ingredient_bounds"]["upper"], device=self.device,
                                                    dtype=self.precision)

        # 加载其他配置
        self.tol = config["settings"]["tol"]
        self.max_iter = config["settings"]["max_iter"]

        # 输入验证
        assert self.costs.shape == (17,), "成本向量必须为17维"
        assert self.nutrition.shape == (17, 10), "营养矩阵必须有17种原料和10种营养素"

    def __call__(self, X: torch.Tensor, tol: float = 0.01) -> torch.Tensor:
        """
        评估饲料配方 (兼容NumPy和PyTorch输入)
        修改说明：
        - 不满足约束的配方会返回高惩罚值（1e6）
        - 满足约束的配方正常计算目标值

        Args:
            X: (n_samples, 17) 原料配比矩阵 (NumPy数组或PyTorch张量)
            tol: 配比总和的容差阈值
        """

        # 转换输入为PyTorch张量
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.precision, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.precision)

        # 初始化结果张量（默认设为高惩罚值）
        batch_size = X.shape[0]
        results = torch.full((batch_size, 11), 1e6, device=self.device, dtype=self.precision)  # 11个输出指标（成本+10种营养素）

        # 检查约束条件（增强版）
        sum_deviation = torch.abs(X.sum(dim=1) - 1)
        ingredient_violation = ((X < self.ingredient_lower_bounds) | (X > self.ingredient_upper_bounds)).any(dim=1)
        valid_mask = (sum_deviation < tol) & ~ingredient_violation

        # 只对有效样本进行计算
        if valid_mask.any():
            X_valid = X[valid_mask]

            # 计算目标指标（增加数值保护）
            with torch.no_grad():
                cost = X_valid @ self.costs
                nutrients = X_valid @ self.nutrition

                # 添加可控的随机扰动
                noise_scale = 1e-6 * cost.abs().mean().item()  # 自适应噪声比例
                cost = cost + torch.randn_like(cost) * noise_scale
                nutrients = nutrients + torch.randn_like(nutrients) * noise_scale

                # 确保数值有效
                cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
                nutrients = torch.nan_to_num(nutrients, nan=1e6, posinf=1e6, neginf=1e6)

                # 填充有效结果
                results[valid_mask] = torch.cat([cost.unsqueeze(1), nutrients], dim=1)

        return results


    def check_constraints(self, X: torch.Tensor) -> Tuple[bool, str]:
        """
        检查配方是否满足所有营养约束

        Args:
            X: (17,) 单个配方向量

        Returns:
            Tuple[是否满足约束, 错误信息]
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        # 检查配方总和是否为1
        if abs(X.sum() - 1) > 1e-4:
            return False, f"配方总和应为1 (当前: {X.sum().item():.4f})"

        # 检查原料用量限制
        violation_mask = (X > self.ingredient_upper_bounds)
        if violation_mask.any():
            violated_indices = torch.where(violation_mask)[0]
            return False, f"原料 {violated_indices.tolist()} 超出最大用量限制"

        # 计算营养含量
        nutrient_values = X @ self.nutrition

        # 检查营养下限
        lower_violations = nutrient_values < self.lower_bounds
        if lower_violations.any():
            violated_nutrients = torch.where(lower_violations)[0]
            return False, f"营养素 {violated_nutrients.tolist()} 低于下限"

        # 检查营养上限
        upper_violations = nutrient_values > self.upper_bounds
        if upper_violations.any():
            violated_nutrients = torch.where(upper_violations)[0]
            return False, f"营养素 {violated_nutrients.tolist()} 超过上限"

        return True, "配方有效"


    def generate_random_formula(self, n_samples: int = 1) -> torch.Tensor:
        """
        生成随机配方 (满足总和=1和原料用量限制)

        Args:
            n_samples: 生成样本数

        Returns:
            (n_samples, 17) 随机配方矩阵
        """
        samples = []
        for _ in range(n_samples):
            while True:
                # 在原料用量限制内随机生成
                sample = torch.rand(17, device=self.device) * self.ingredient_upper_bounds
                sample = sample / sample.sum()  # 归一化

                # 验证是否满足所有约束
                valid, _ = self.check_constraints(sample)
                if valid:
                    samples.append(sample)
                    break

        return torch.stack(samples)


    def optimize_cost(self, max_iter: int = 1000) -> Tuple[torch.Tensor, float]:
        """
        优化配方成本 (模拟第一份代码的优化过程)

        Returns:
            Tuple[最优配方, 最小成本]
        """
        best_cost = float('inf')
        best_formula = None

        for _ in range(max_iter):
            formula = self.generate_random_formula(1).squeeze(0)
            current_cost = (formula @ self.costs).item()

            if current_cost < best_cost:
                best_cost = current_cost
                best_formula = formula.clone()

        return best_formula, best_cost


    def optimize_nutrient(self, target_nutrient: int, maximize: bool = True,
                          max_iter: int = 1000) -> Tuple[torch.Tensor, float]:
        """
        优化特定营养素含量

        Args:
            target_nutrient: 目标营养素索引
            maximize: 是否最大化
            max_iter: 最大迭代次数

        Returns:
            Tuple[最优配方, 营养素值]
        """
        best_value = -float('inf') if maximize else float('inf')
        best_formula = None

        for _ in range(max_iter):
            formula = self.generate_random_formula(1).squeeze(0)
            nutrient_value = (formula @ self.nutrition[:, target_nutrient]).item()

            if (maximize and nutrient_value > best_value) or \
                    (not maximize and nutrient_value < best_value):
                best_value = nutrient_value
                best_formula = formula.clone()

        return best_formula, best_value


    @staticmethod
    def get_nutrient_names() -> list:
        """获取营养素名称"""
        return ['CF', 'Ca', 'AP', 'DM', 'CP', 'MC', 'T', 'Tp', 'Energy', 'L']


    def to_dataframe(self, X: torch.Tensor) -> pd.DataFrame:
        """
        将配方和营养信息转换为DataFrame

        Args:
            X: (n_samples, 17) 配方矩阵

        Returns:
            包含配方和营养信息的DataFrame
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        # 计算成本和营养含量
        costs = X @ self.costs
        nutrients = X @ self.nutrition

        # 创建DataFrame
        df = pd.DataFrame(X.cpu().numpy(),
                          columns=[f'原料{i + 1}' for i in range(17)])

        # 添加成本和营养列
        df['成本'] = costs.cpu().numpy()
        for i, name in enumerate(self.get_nutrient_names()):
            df[name] = nutrients[:, i].cpu().numpy()

        return df
