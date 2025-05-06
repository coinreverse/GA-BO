import torch
from typing import Dict, Any
import yaml
from core.genetic_algorithm import run_ga
from core.bayesian_optim import BOOptimizer
from core.evaluator import FeedEvaluator
from core.hybrid_strategy import HybridStrategy
from utils.visualization import plot_pareto_front
import matplotlib.pyplot as plt


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
        config_path="configs/ga_config.yaml",
    )

    # 提取切换判断所需数据
    ga_population = ga_metadata["population_F"]  # (n_samples, 3) 目标值矩阵
    ga_hv_history = ga_metadata["hv_history"]  # 超体积历史列表
    # 可视化GA结果
    fig_ga = plot_pareto_front(
        Y_ga,
        title="GA Pareto Front",
        ref_point=ref_point,
        angle=(25, 45)
    )
    plt.savefig("results/ga_pareto.png")
    plt.close()

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
            evaluator=evaluator,
            n_samples=bo_config['raw_samples'],
            noise_scale=hybrid_config['noise_scale'],
        )
        # 运行BO优化
        bo = BOOptimizer(
            bounds=feed_config['ingredient_bounds'],  # 直接传入边界字典
            ref_point=ref_point,
        )
        nutrient_names = evaluator.get_nutrient_names()
        X_hybrid, Y_hybrid = bo.optimize(
            X_init=X_init,
            Y_init=Y_init,
            n_iter=bo_config['n_iter'],
            batch_size=bo_config['batch_size'],
            evaluator=lambda x: evaluator(x)[:, [0, 1 + nutrient_names.index('L'), 1 + nutrient_names.index('Energy')]]
        )
        #
        # # 合并GA和BO结果
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_hybrid = torch.cat([X_ga, X_hybrid])
        Y_hybrid = torch.cat([Y_ga.to(device), Y_hybrid.to(device)])
    else:
        print("GA optimization sufficient, skipping BO phase")
        X_hybrid, Y_hybrid = X_ga, Y_ga

    # 最终结果处理
    elite_X, elite_Y = strategy.elite_selection(
        X_hybrid, Y_hybrid,
        n_elites=hybrid_config['n_elites'],
        diversity_weight=hybrid_config['diversity_weight']
    )

    # 可视化最终结果
    fig_final = plot_pareto_front(
        elite_Y,
        title="Final Pareto Front",
        ref_point=ref_point,
        angle=(30, 50),
        figsize=(16, 12)
    )
    plt.savefig("results/final_pareto.png")
    plt.show()

    # 保存结果
    save_results(elite_X, elite_Y)

    # 输出最佳解
    min_cost_idx = torch.argmin(elite_Y[:, 0])
    print("\nBest Solution Summary:")
    print(f"- Cost: {elite_Y[min_cost_idx, 0]:.2f} €/MT")
    print(f"- Lysine: {elite_Y[min_cost_idx, 1]:.3f}%")
    print(f"- Energy: {elite_Y[min_cost_idx, 2]:.2f} MJ/kg")
    print("\nOptimization completed!")


if __name__ == "__main__":
    main()
