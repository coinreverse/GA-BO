import logging
from pathlib import Path
from datetime import datetime
import torch


class OptimizationLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._setup_logger()

    def _setup_logger(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"optimization_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def log_metrics(metrics: dict):
        """记录关键指标"""
        for k, v in metrics.items():
            if torch.is_tensor(v):
                v = v.cpu().numpy().tolist()
            logging.info(f"{k}: {v}")

    @staticmethod
    def log_pareto(Y: torch.Tensor):
        """记录帕累托前沿统计信息"""
        if len(Y) > 0:
            stats = {
                "min_cost": Y[:, 0].min(),
                "max_lysine": Y[:, 1].max(),
                "max_energy": Y[:, 2].max(),
                "num_solutions": len(Y)
            }
            logging.info("Pareto Front Stats: " + ", ".join(
                f"{k}={v:.3f}" for k, v in stats.items()
            ))
