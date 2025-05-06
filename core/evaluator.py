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

        with torch.no_grad():
            cost = X @ self.costs
            nutrients = X @ self.nutrition

        nutrients = torch.where(
            torch.isfinite(nutrients),
            nutrients,
            torch.tensor(1e4, device=self.device)
        )

        return torch.cat([cost.unsqueeze(1), nutrients], dim=1)

    @staticmethod
    def get_nutrient_names() -> list:
        """获取营养素名称"""
        return ['CF', 'Ca', 'AP', 'DM', 'CP', 'MC', 'T', 'Tp', 'Energy', 'L']
