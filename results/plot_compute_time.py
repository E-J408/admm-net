from pathlib import Path
from utils.plotUtils import *
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns


def plot_compute_time(datafile_list, output_dir="time/res", algorithm_names=None):
    # 判断数据文件是否存在
    for datafile in datafile_list:
        if not Path(datafile).exists():
            # 抛出异常
            raise FileNotFoundError(f"数据文件{datafile}不存在")
    matlab_style_plots()
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取txt文件，每个文件1000行，代表1000次运行时间
    time_data = []
    time_means = []
    time_stds = []
    time_medians = []
    for datafile in datafile_list:
        with open(datafile, "r") as f:
            data = [float(line.strip()) for line in f.readlines()]
            time_data.append(data)
            time_means.append(np.mean(data))
            time_stds.append(np.std(data))
            time_medians.append(np.median(data))
    # 设置算法名称（如果未提供，从文件名提取）
    if algorithm_names is None:
        algorithm_names = []
        for f in datafile_list:
            name = Path(f).stem
            if name.startswith('time_'):
                name = name[5:]  # 移除 'time_' 前缀
            if name == 'time':
                name = 'Baseline'
            algorithm_names.append(name.replace('_', ' ').title())
        # 5. 按平均时间排序
        sorted_indices = np.argsort(time_means)
        sorted_means = [time_means[i] for i in sorted_indices]
        sorted_stds = [time_stds[i] for i in sorted_indices]
        sorted_names = [algorithm_names[i] for i in sorted_indices]

        # 获取对应的排序后数据
        sorted_data = [time_data[i] for i in sorted_indices]

        # 开始绘图
        # 颜色方案
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithm_names)))
        # ============================
        # 图1: 平均运行时间柱状图
        # ============================
        print("正在生成: 平均运行时间柱状图...")
        fig1 = plt.figure(figsize=(10, 7), dpi=300)

        # 创建柱状图
        bars = plt.bar(range(len(sorted_means)), sorted_means,
                       color=colors, edgecolor='black',
                       alpha=0.85, linewidth=1.2, width=0.6)

        # 在柱子上方添加数值标签
        # 在柱子上方添加平均时间标签
        for i, (bar, mean_val) in enumerate(zip(bars, sorted_means)):
            height = bar.get_height()
            # 将时间转换为毫秒，如果小于0.001秒
            if mean_val < 0.001:
                time_str = f"{mean_val * 1000:.2f}ms"
            elif mean_val < 1:
                time_str = f"{mean_val * 1000:.0f}ms"
            else:
                time_str = f"{mean_val:.3f}s"

            plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     time_str,
                     ha='center', va='bottom', fontsize=12, fontweight='normal',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='gray', alpha=0.9))

        # 设置图表属性
        plt.xlabel('算法名称', fontsize=14, fontweight='bold', labelpad=10)
        plt.ylabel('平均运行时间/s', fontsize=14, fontweight='bold', labelpad=10)
        plt.title('算法平均运行时间对比',
                  fontsize=16, fontweight='bold', pad=20)

        # 设置x轴刻度
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=0,
                   ha='center', fontsize=12)
        plt.yticks(fontsize=12)

        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 添加图例说明
        # plt.text(0.02, 0.98, '误差线表示标准差 (±1σ)',
        #          transform=plt.gca().transAxes, fontsize=10,
        #          bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
        #                    edgecolor='gray', alpha=0.8),
        #          verticalalignment='top')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        output_path1 = output_dir / "平均运行时间对比.png"
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path1}")
        plt.savefig(output_dir / "平均运行时间对比.pdf", bbox_inches='tight')
        print(f"✓ 已保存: {output_dir}/平均运行时间对比.pdf")
        plt.close()


        # ============================
        # 图4: 加速比对比图
        # ============================
        print("\n正在生成: 加速比对比图...")
        fig4 = plt.figure(figsize=(10, 7), dpi=300)

        # 计算加速比（相对于最慢算法）
        slowest_time = max(time_means)
        speedup_ratios = [slowest_time / mean for mean in sorted_means]

        # 创建水平条形图
        bars = plt.barh(range(len(speedup_ratios)), speedup_ratios,
                        color=colors, edgecolor='black',
                        alpha=0.85, linewidth=1.2, height=0.5)

        # 在条形上添加标签
        for i, (bar, ratio) in enumerate(zip(bars, speedup_ratios)):
            width = bar.get_width()
            plt.text(width + 0.05, bar.get_y() + bar.get_height() / 2.,
                     f'{ratio:.2f}倍', va='center', fontsize=11, fontweight='normal',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='gray', alpha=0.9))

        # 设置图表属性
        plt.xlabel('加速比', fontsize=14, fontweight='bold', labelpad=10)
        plt.ylabel('算法名称', fontsize=14, fontweight='bold', labelpad=10)
        plt.title(f'算法加速比对比\n(基准: {sorted_names[-1]})',
                  fontsize=16, fontweight='bold', pad=20)
        plt.yticks(range(len(sorted_names)), sorted_names, fontsize=12)
        plt.xticks(fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--', axis='x')

        # 添加参考线
        plt.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='基准 (1倍)')
        plt.axvline(x=2.0, color='green', linestyle=':', linewidth=1, alpha=0.5, label='2倍加速')
        plt.axvline(x=5.0, color='blue', linestyle=':', linewidth=1, alpha=0.5, label='5倍加速')

        # 添加图例
        plt.legend(fontsize=11, loc='upper right', frameon=True, framealpha=0.9)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        output_path4 = output_dir / "加速比对比图.png"
        plt.savefig(output_path4, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path4}")
        plt.savefig(output_dir / "加速比对比图.pdf", bbox_inches='tight')
        print(f"✓ 已保存: {output_dir}/加速比对比图.pdf")
        plt.close()

        # ============================
        # 图5: 运行时间概率密度图
        # ============================
        print("\n正在生成: 运行时间概率密度图...")
        fig5 = plt.figure(figsize=(10, 7), dpi=300)

        # 绘制每个算法的KDE曲线
        for i, idx in enumerate(sorted_indices):
            data = time_data[idx]
            sns.kdeplot(data, label=sorted_names[i], color=colors[i],
                        linewidth=2.5, fill=True, alpha=0.2)

        plt.xlabel('运行时间 (秒)', fontsize=14, fontweight='bold', labelpad=10)
        plt.ylabel('概率密度', fontsize=14, fontweight='bold', labelpad=10)
        plt.title('算法运行时间概率密度分布 (1000次运行)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, title='算法', title_fontsize=13,
                   frameon=True, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 设置坐标轴格式
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

        # 调整布局
        plt.tight_layout()

        # 保存图片
        output_path5 = output_dir / "运行时间概率密度分布.png"
        plt.savefig(output_path5, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path5}")
        plt.savefig(output_dir / "运行时间概率密度分布.pdf", bbox_inches='tight')
        print(f"✓ 已保存: {output_dir}/运行时间概率密度分布.pdf")
        plt.close()

        # ============================
        # 图6: 累积分布函数图 (新增)
        # ============================
        print("\n正在生成: 运行时间累积分布函数图...")
        fig6 = plt.figure(figsize=(10, 7), dpi=300)

        # 绘制每个算法的累积分布函数
        for i, idx in enumerate(sorted_indices):
            data = np.sort(time_data[idx])
            y = np.arange(1, len(data) + 1) / len(data)
            plt.plot(data, y, label=sorted_names[i], color=colors[i], linewidth=2.5)

        plt.xlabel('运行时间 (秒)', fontsize=14, fontweight='bold', labelpad=10)
        plt.ylabel('累积概率', fontsize=14, fontweight='bold', labelpad=10)
        plt.title('算法运行时间累积分布函数 (CDF)', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 设置坐标轴格式
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

        # 添加参考线
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='中位数')
        plt.axhline(y=0.9, color='green', linestyle=':', linewidth=1, alpha=0.5, label='90%分位点')

        # 添加图例
        plt.legend(fontsize=11, frameon=True, framealpha=0.9)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        output_path6 = output_dir / "运行时间累积分布函数.png"
        plt.savefig(output_path6, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path6}")
        plt.savefig(output_dir / "运行时间累积分布函数.pdf", bbox_inches='tight')
        print(f"✓ 已保存: {output_dir}/运行时间累积分布函数.pdf")
        plt.close()

        # ============================
        # 输出统计报告
        # ============================
        print("\n" + "=" * 80)
        print("算法性能分析报告")
        print("=" * 80)

        # 创建统计报告文件
        report_path = output_dir / "性能分析报告.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("算法性能分析报告\n")
            f.write("=" * 80 + "\n\n")

            for i, idx in enumerate(sorted_indices):
                f.write(f"{sorted_names[i]}:\n")
                f.write(f"  平均运行时间: {time_means[idx]:.6f} 秒\n")
                f.write(f"  标准差: {time_stds[idx]:.6f} 秒\n")
                f.write(f"  中位数: {time_medians[idx]:.6f} 秒\n")
                f.write(f"  最小值: {min(time_data[idx]):.6f} 秒\n")
                f.write(f"  最大值: {max(time_data[idx]):.6f} 秒\n")
                f.write(f"  极差: {max(time_data[idx]) - min(time_data[idx]):.6f} 秒\n")
                f.write(f"  变异系数: {(time_stds[idx] / time_means[idx] * 100):.2f}%\n")
                f.write(f"  Q1 (25%分位数): {np.percentile(time_data[idx], 25):.6f} 秒\n")
                f.write(f"  Q3 (75%分位数): {np.percentile(time_data[idx], 75):.6f} 秒\n")
                f.write(f"  四分位距: {np.percentile(time_data[idx], 75) - np.percentile(time_data[idx], 25):.6f} 秒\n")
                f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("相对性能比较:\n")
            f.write("-" * 80 + "\n")

            # 以最快的算法为基准
            baseline_idx = sorted_indices[0]
            baseline_time = time_means[baseline_idx]
            baseline_name = sorted_names[0]

            f.write(f"\n参考算法: {baseline_name} (最快)\n\n")

            for i, idx in enumerate(sorted_indices[1:], 1):
                speedup = baseline_time / time_means[idx]
                if speedup < 1:
                    slowdown = 1 / speedup
                    f.write(f"  {sorted_names[i]} 比 {baseline_name} 慢 {slowdown:.2f} 倍\n")
                else:
                    f.write(f"  {sorted_names[i]} 比 {baseline_name} 快 {speedup:.2f} 倍\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("总结:\n")
            f.write("=" * 80 + "\n")
            f.write(f"最快算法: {sorted_names[0]} ({sorted_means[0]:.6f} 秒)\n")
            f.write(f"最慢算法: {sorted_names[-1]} ({sorted_means[-1]:.6f} 秒)\n")
            f.write(f"性能范围: {sorted_means[-1] / sorted_means[0]:.2f} 倍差异\n")

        print(f"✓ 已保存统计报告: {report_path}")
        print("=" * 80)

        # 在控制台也输出摘要信息
        print("\n" + "-" * 80)
        print("总结:")
        print("-" * 80)
        print(f"最快算法: {sorted_names[0]} ({sorted_means[0]:.6f} 秒)")
        print(f"最慢算法: {sorted_names[-1]} ({sorted_means[-1]:.6f} 秒)")
        print(f"性能范围: {sorted_means[-1] / sorted_means[0]:.2f} 倍差异")
        print("-" * 80)

        # 列出生成的所有文件
        print("\n" + "=" * 80)
        print("生成的文件列表:")
        print("=" * 80)
        print("图片文件:")
        print(f"1. 平均运行时间对比图: {output_path1}")
        print(f"4. 加速比对比图: {output_path4}")
        print(f"5. 运行时间概率密度分布图: {output_path5}")
        print(f"6. 运行时间累积分布函数图: {output_path6}")
        print(f"\n文本文件:")
        print(f"7. 性能分析报告: {report_path}")
        print("=" * 80)
if __name__ == '__main__':
    file_list = []
    file_list.append("time/time.txt")
    file_list.append("time/time_net.txt")
    file_list.append("time/time_net_5.txt")
    plot_compute_time(file_list)
