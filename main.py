import random

import numpy as np
import torch
import time
from typing import Dict, Any
import yaml
from matplotlib import pyplot as plt

from core.genetic_algorithm import run_ga
from core.bayesian_optim import BOOptimizer
from core.evaluator import FeedEvaluator
from core.hybrid_strategy import HybridStrategy
from utils.Comparative_sampling import plot_ga_results
from utils.plot_ga_convergence import plot_convergence

torch.set_default_tensor_type(torch.cuda.DoubleTensor)  # 全局默认 CUDA

def load_configs() -> Dict[str, Any]:
    """加载所有配置文件"""
    with open("configs/ga_config.yaml", encoding='utf-8') as f:
        ga_config = yaml.safe_load(f)
    with open("configs/bo_config.yaml", encoding='utf-8') as f:
        bo_config = yaml.safe_load(f)
    with open("configs/hybrid_config.yaml", encoding='utf-8') as f:
        hybrid_config = yaml.safe_load(f)
    with open("configs/feed_config.yaml", encoding='utf-8') as f:
        feed_config = yaml.safe_load(f)
    return ga_config, bo_config, hybrid_config, feed_config


def save_results(X: torch.Tensor, Y: torch.Tensor, filename: str = "results/pareto_front.pt"):
    """保存优化结果"""
    torch.save({
        'solutions': X,
        'objectives': Y
    }, filename)
    print(f"Results saved to {filename}")


def main():
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    # 记录总开始时间
    start_time = time.time()

    # 加载配置
    ga_config, bo_config, hybrid_config, feed_config = load_configs()

    # 初始化组件
    evaluator = FeedEvaluator(config_path="configs/feed_config.yaml",
                              device='cuda' if torch.cuda.is_available() else 'cpu', precision='float32')
    ref_point = torch.tensor(hybrid_config['ref_point'])
    strategy = HybridStrategy(ref_point)

    # 阶段1: GA全局探索
    print("\n=== Phase 1: Genetic Algorithm Exploration ===")

    X_ga, Y_ga, ga_population, ga_metadata = run_ga(
        evaluator=evaluator,
        ref_point=ref_point,
    )

    ga_time = time.time() - start_time  # 计算GA耗时
    print(f"GA阶段耗时: {ga_time:.2f}秒")

    # 提取切换判断所需数据
    ga_population = ga_metadata["population_F"]  # 目标值矩阵
    ga_hv_history = ga_metadata["hv_history"]  # 超体积历史列表

    # 4. 绘制收敛曲线
    plot_convergence(
        hv_history=ga_metadata["hv_history"],
        best_solutions=ga_metadata["best_solutions"],
        save_path="results/ga_convergence.png"  # 可选保存路径
    )
    # 绘制解集散布图
    strategy_name="random"
    plot_ga_results(
        X_ga=X_ga,
        Y_ga=Y_ga,
        meta_ga={
            "best_solutions": ga_metadata["best_solutions"],
            "hv_history": ga_hv_history
        },
        ref_point=ref_point,
        title_suffix=f"({strategy_name})"
    )

    # 最终结果处理
    elite_ga_X, elite_ga_Y = strategy.elite_selection(
        X_ga, Y_ga,
        n_elites=hybrid_config['n_elites'],
        diversity_weight=hybrid_config['diversity_weight']
    )

    # 保存结果
    save_results(elite_ga_X, elite_ga_Y, filename="results/ga_pareto_front.pt")


    # 自适应切换决策
    # if False:
    if strategy.should_switch_to_bo(
            ga_hv_history=ga_hv_history,
            ga_population=ga_population,
    ):
        # 阶段2: BO局部开发
        print("\n=== Phase 2: Bayesian Optimization Refinement ===")

        # 转换为张量（如果需要）
        if not isinstance(X_ga, torch.Tensor):
            X_ga = torch.tensor(X_ga, dtype=torch.double)
        if not isinstance(Y_ga, torch.Tensor):
            Y_ga = torch.tensor(Y_ga, dtype=torch.double)

        # 准备BO初始样本
        X_init, Y_init = strategy.initialize_bo(
            ga_results=(X_ga, Y_ga),
            n_samples=bo_config['raw_samples'],
        )

        print("\n=== BO 初始样本检查 ===")
        print("X_init shape:", X_init.shape)  # 应和 GA 的决策变量维度一致
        print("Y_init shape:", Y_init.shape)  # 应和 GA 的目标值维度一致
        print("X_init 范围 (min/max):", X_init.min(dim=0).values, X_init.max(dim=0).values)
        print("Y_init 范围 (min/max):", Y_init.min(dim=0).values, Y_init.max(dim=0).values)
        # 运行BO优化
        bo = BOOptimizer(
            bounds=feed_config['ingredient_bounds'],  # 直接传入边界字典
            ref_point=ref_point,
            initial_sample_size=bo_config['initial_sample_size'],
            monitor_config={  # 可选：监控配置
                "obj": None,  # 可替换为自定义目标转换函数
                "constraints": None  # 可添加约束函数列表
            },
            seed=42
        )

        print("目标方向验证 - 成本样本值:", Y_init[:, 0].min(), Y_init[:, 0].max())


        print("\n=== BO 配置检查 ===")
        print("参考点:", ref_point)
        print("初始样本目标值范围:", Y_init.min(dim=0).values, Y_init.max(dim=0).values)
        X_hybrid, Y_hybrid = bo.optimize(
            X_init=X_init,
            Y_init=Y_init,
            n_iter=bo_config['n_iter'],
            evaluator=evaluator,
        )
        monitor_results = bo.get_monitor_outputs()
        if monitor_results:
            print("\n=== Optimization Monitoring ===")
            print(f"Final Hypervolume: {monitor_results['hv_history'][-1]:.4f}")
            print(f"Best Solution Found X: {monitor_results['pareto_X'][-1][0]}")
            print(f"Best Solution Found Y: {monitor_results['pareto_Y'][-1][0]}")

    else:
        print("GA optimization sufficient, skipping BO phase")
        X_hybrid, Y_hybrid = X_ga, Y_ga

    # 最终结果处理
    elite_X, elite_Y = strategy.elite_selection(
        X_hybrid, Y_hybrid,
        n_elites=hybrid_config['n_elites'],
        diversity_weight=hybrid_config['diversity_weight']
    )

    # 保存结果
    save_results(elite_X, elite_Y, filename="results/hybrid_pareto_front.pt")

    # 输出最佳解
    nutrient_names = evaluator.get_nutrient_names()
    energy_idx = nutrient_names.index('Energy')
    lysine_idx = nutrient_names.index('L')
    min_cost_idx = torch.argmin(elite_Y[:, 0])
    print("\nBest Solution Summary:")
    print(f"- Cost: {elite_Y[min_cost_idx, 0]:.2f} €/MT")
    print(f"- Lysine: {elite_Y[min_cost_idx, 1 + lysine_idx]:.3f}%")
    print(f"- Energy: {elite_Y[min_cost_idx, 1 + energy_idx]:.2f} MJ/kg")
    print("\nOptimization completed!")

    total_time = time.time() - start_time
    bo_time = total_time - ga_time
    print("\n=== 性能统计 ===")
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"- GA阶段: {ga_time:.2f}秒 ({ga_time / total_time * 100:.1f}%)")
    print(f"- BO阶段: {bo_time:.2f}秒 ({bo_time / total_time * 100:.1f}%)")



if __name__ == "__main__":
    main()
