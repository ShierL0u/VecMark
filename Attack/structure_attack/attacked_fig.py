import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_attack_results(result_dir):
    """
    从结果目录加载攻击测试结果

    Args:
        result_dir: 结果文件目录

    Returns:
        dict: 按攻击类型和攻击率组织的结果数据
    """
    attack_results = defaultdict(dict)

    if not os.path.exists(result_dir):
        print(f"Directory {result_dir} not found")
        return attack_results

    # 遍历结果目录中的所有文件
    for filename in os.listdir(result_dir):
        if filename.endswith('.json') and filename != 'attack_test_summary.json':
            filepath = os.path.join(result_dir, filename)

            try:
                # 解析文件名，提取攻击类型和攻击率
                parts = filename.replace('.json', '').split('_')
                if len(parts) >= 2:
                    # 处理文件名中的多个下划线
                    if 'Deleted_Entities' in filename:
                        attack_type = 'Deleted Entities'
                        rate_str = parts[-1]
                    elif 'Inserted_Entities' in filename:
                        attack_type = 'Inserted Entities'
                        rate_str = parts[-1]
                    elif 'Modified_Entities' in filename:
                        attack_type = 'Modified Entities'
                        rate_str = parts[-1]
                    elif 'Modified_Attributes' in filename:
                        attack_type = 'Modified Attributes'
                        rate_str = parts[-1]
                    else:
                        continue

                    # 转换攻击率为浮点数
                    try:
                        attack_rate = float(rate_str)
                    except ValueError:
                        continue

                    # 读取JSON文件
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 提取匹配率
                    match_rate = data.get('match_rate', 0.0)

                    attack_results[attack_type][attack_rate] = match_rate

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    return attack_results

def plot_attack_comparison(dataset_results, rag_results, attack_type, fig_dir):
    """
    为指定的攻击类型绘制VecMark(on Dataset)和VecMark(on RAG)的对比图

    Args:
        dataset_results: 数据集攻击结果 {attack_rate: match_rate}
        rag_results: RAG攻击结果 {attack_rate: match_rate}
        attack_type: 攻击类型
        fig_dir: 图片保存目录
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8))

    # 准备数据
    if attack_type == "Modified Attributes":
        # Modified Attributes使用低攻击率
        rates_to_plot = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    else:
        # 其他攻击类型使用高攻击率
        rates_to_plot = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # 提取数据集结果
    dataset_rates = []
    dataset_match_rates = []
    for rate in rates_to_plot:
        if rate in dataset_results:
            dataset_rates.append(rate)
            dataset_match_rates.append(dataset_results[rate])

    # 提取RAG结果
    rag_rates = []
    rag_match_rates = []
    for rate in rates_to_plot:
        if rate in rag_results:
            rag_rates.append(rate)
            rag_match_rates.append(rag_results[rate])

    # 定义颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # 绘制数据集曲线
    if dataset_rates:
        plt.plot(dataset_rates, dataset_match_rates,
                color=colors[0], marker='o', linewidth=2, markersize=8,
                label='VecMark(on Dataset)', linestyle='-')

    # 绘制RAG曲线
    if rag_rates:
        plt.plot(rag_rates, rag_match_rates,
                color=colors[1], marker='o', linewidth=2, markersize=8,
                label='VecMark(on RAG)', linestyle='-')

    # 设置坐标轴
    if attack_type != "Modified Attributes":
        plt.xlim(0, 1)
        if dataset_rates:
            plt.xticks(dataset_rates)
    else:
        if dataset_rates:
            x_min, x_max = min(dataset_rates), max(dataset_rates)
            plt.xlim(x_min, x_max)

    # 设置纵坐标
    all_match_rates = dataset_match_rates + rag_match_rates
    if all_match_rates:
        y_min, y_max = min(all_match_rates), max(all_match_rates)
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        plt.ylim(0, 1.2)

    plt.xlabel('Attack Rate', fontsize=14)
    plt.ylabel('WER(%)', fontsize=14)
    plt.title(f'{attack_type} - VecMark Detection Rate Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, alpha=0.3)

    # 保存图片
    safe_name = attack_type.replace(' ', '_')
    fig_path = os.path.join(fig_dir, f'{safe_name}_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved comparison chart for {attack_type}: {fig_path}")

def create_comparison_figures():
    """
    创建所有攻击类型的对比图表
    """
    # 定义目录路径
    base_dir = "/root/autodl-tmp/vectormark/data/attacked_eva"
    dataset_result_dir = os.path.join(base_dir, "detDataset_result")
    rag_result_dir = os.path.join(base_dir, "detRAG_result")
    fig_dir = os.path.join(base_dir, "fig")

    # 创建图片保存目录
    os.makedirs(fig_dir, exist_ok=True)

    # 定义攻击类型
    attack_types = ["Deleted Entities", "Inserted Entities", "Modified Entities", "Modified Attributes"]

    # 加载结果数据
    print("Loading dataset attack results...")
    dataset_results = load_attack_results(dataset_result_dir)

    print("Loading RAG attack results...")
    rag_results = load_attack_results(rag_result_dir)

    # 为每种攻击类型生成对比图
    print("\nGenerating comparison figures...")
    for attack_type in attack_types:
        if attack_type in dataset_results or attack_type in rag_results:
            dataset_attack_data = dataset_results.get(attack_type, {})
            rag_attack_data = rag_results.get(attack_type, {})

            if dataset_attack_data or rag_attack_data:
                plot_attack_comparison(dataset_attack_data, rag_attack_data, attack_type, fig_dir)
            else:
                print(f"No data found for {attack_type}")
        else:
            print(f"No results found for {attack_type}")

    print("All comparison figures generated successfully!")
    print(f"Figures saved to: {fig_dir}")

def create_summary_comparison():
    """
    创建所有攻击类型的汇总对比图
    """
    # 定义目录路径
    base_dir = "/root/autodl-tmp/vectormark/data/attacked_eva"
    dataset_result_dir = os.path.join(base_dir, "detDataset_result")
    rag_result_dir = os.path.join(base_dir, "detRAG_result")
    fig_dir = os.path.join(base_dir, "fig")

    # 加载结果数据
    dataset_results = load_attack_results(dataset_result_dir)
    rag_results = load_attack_results(rag_result_dir)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(16, 12))

    # 定义攻击类型和对应的颜色/标记
    attack_types = ["Deleted Entities", "Inserted Entities", "Modified Entities", "Modified Attributes"]
    # 使用不同的颜色方案
    dataset_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝色、橙色、绿色、红色
    rag_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']       # 对应的浅色版本

    # 收集所有数据用于设置坐标轴范围
    all_rates = []
    all_match_rates = []

    # 为每个攻击类型绘制两条曲线
    for i, attack_type in enumerate(attack_types):
        if attack_type == "Modified Attributes":
            rates_to_plot = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        else:
            rates_to_plot = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        # 绘制数据集曲线
        dataset_attack_data = dataset_results.get(attack_type, {})
        dataset_rates = [rate for rate in rates_to_plot if rate in dataset_attack_data]
        dataset_match_rates = [dataset_attack_data[rate] for rate in dataset_rates]

        if dataset_rates:
            plt.plot(dataset_rates, dataset_match_rates,
                    color=dataset_colors[i % len(dataset_colors)],
                    marker='o', linewidth=2, markersize=6,
                    label=f'{attack_type} - VecMark(on Dataset)',
                    linestyle='-')

        # 绘制RAG曲线
        rag_attack_data = rag_results.get(attack_type, {})
        rag_rates = [rate for rate in rates_to_plot if rate in rag_attack_data]
        rag_match_rates = [rag_attack_data[rate] for rate in rag_rates]

        if rag_rates:
            plt.plot(rag_rates, rag_match_rates,
                    color=rag_colors[i % len(rag_colors)],
                    marker='o', linewidth=2, markersize=6,
                    label=f'{attack_type} - VecMark(on RAG)',
                    linestyle='-')

        # 收集数据
        all_rates.extend(dataset_rates)
        all_rates.extend(rag_rates)
        all_match_rates.extend(dataset_match_rates)
        all_match_rates.extend(rag_match_rates)

    # 设置坐标轴
    if all_rates:
        plt.xlim(0, 1)
        unique_rates = sorted(set(all_rates))
        plt.xticks(unique_rates)

    if all_match_rates:
        y_min, y_max = min(all_match_rates), max(all_match_rates)
        y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        plt.ylim(0, 1.2)

    plt.xlabel('Attack Rate', fontsize=14)
    plt.ylabel('WER(%)', fontsize=14)
    plt.title('All Attack Types - VecMark Detection Rate Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='lower left', bbox_to_anchor=(0.02, 0.02))
    plt.grid(True, alpha=0.3)

    # 调整布局以适应图例
    plt.tight_layout()

    # 保存汇总图
    summary_path = os.path.join(fig_dir, 'all_attacks_summary_comparison.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved summary comparison chart: {summary_path}")

if __name__ == "__main__":
    print("Starting attacked figure generation...")

    # 创建单个攻击类型的对比图
    create_comparison_figures()

    # 创建汇总对比图
    create_summary_comparison()

    print("\nFigure generation completed!")
