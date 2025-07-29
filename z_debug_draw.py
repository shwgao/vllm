import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def clean_dataframe(df, numeric_columns=None):
    """
    清理DataFrame：将空字符串转为NaN，并将指定列转换为数值型。
    """
    df = df.replace('', np.nan)
    if numeric_columns:
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def print_df_info(df):
    """
    打印DataFrame的基本信息和缺失值统计。
    """
    print("数据概览:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)
    print("\n缺失值统计:")
    print(df.isnull().sum())
    print("\n数值统计:")
    print(df.describe())

def plot_line_chart(df, x, y, hue, title, xlabel, ylabel, save_path):
    """
    通用折线图绘制函数。
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        marker="o",
        linewidth=2,
        markersize=6
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title=hue, title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")

def plot_compare_two_csv(
    csv1, csv2, x, y, hue, title, xlabel, ylabel, save_path,
    label1="tp=2", label2="tp=4"
):
    """
    对比两个csv文件，将结果画在同一张图上。
    """
    numeric_columns = [y]
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df1 = clean_dataframe(df1, numeric_columns)
    df2 = clean_dataframe(df2, numeric_columns)
    df1['tp_label'] = label1
    df2['tp_label'] = label2
    df_all = pd.concat([df1, df2], ignore_index=True)
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_all,
        x=x,
        y=y,
        hue=hue,
        style='tp_label',
        markers=True,
        dashes=False,
        linewidth=2,
        markersize=6
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title=f"{hue} / tp", title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")

if __name__ == "__main__":
    # 设置中文字体（如有需要）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取和清理数据
    numeric_columns = ['throughput_requests', 'throughput_total_tokens', 'throughput_output_tokens']
    df = pd.read_csv("benchmark_results_1tp_4pp.csv")
    df = clean_dataframe(df, numeric_columns)
    print_df_info(df)

    # 检查数据有效性
    if df['throughput_total_tokens'].notna().sum() == 0:
        print("错误: 没有有效的 throughput_total_tokens 数据!")
    if df['throughput_output_tokens'].notna().sum() == 0:
        print("错误: 没有有效的 throughput_output_tokens 数据!")

    # 画 total tokens/s 随 input_len 变化的曲线
    plot_line_chart(
        df=df,
        x="input_len",
        y="throughput_total_tokens",
        hue="num_prompts",
        title="vLLM Throughput (total tokens/s) vs Input Length",
        xlabel="Input Length",
        ylabel="Throughput (total tokens/s)",
        save_path="throughput_total_tokens_vs_input_len.png"
    )

    # 画 total tokens/s 随 num_prompts*input_len 变化的曲线
    if 'num_prompts' in df.columns and 'input_len' in df.columns:
        df['num_prompts_times_input_len'] = df['num_prompts'] * df['input_len']
        plot_line_chart(
            df=df,
            x="num_prompts_times_input_len",
            y="throughput_total_tokens",
            hue="num_prompts",
            title="vLLM Throughput (total tokens/s) vs num_prompts * input_len",
            xlabel="num_prompts * input_len",
            ylabel="Throughput (total tokens/s)",
            save_path="throughput_total_tokens_vs_num_prompts_times_input_len.png"
        )

    # 画 output tokens/s
    plot_line_chart(
        df=df,
        x="input_len",
        y="throughput_output_tokens",
        hue="num_prompts",
        title="vLLM Throughput (output tokens/s) vs Input Length",
        xlabel="Input Length",
        ylabel="Throughput (output tokens/s)",
        save_path="throughput_output_tokens_vs_input_len.png"
    )

    # 画 requests/s
    plot_line_chart(
        df=df,
        x="input_len",
        y="throughput_requests",
        hue="num_prompts",
        title="vLLM Throughput (requests/s) vs Input Length",
        xlabel="Input Length",
        ylabel="Throughput (requests/s)",
        save_path="throughput_requests_vs_input_len.png"
    )

    # # 对比两组csv的画图用法示例
    # plot_compare_two_csv(
    #     csv1="benchmark_results_2tp.csv",
    #     csv2="benchmark_results.csv",
    #     x="input_len",
    #     y="throughput_total_tokens",
    #     hue="num_prompts",
    #     title="vLLM Throughput (total tokens/s) vs Input Length (tp=2 vs tp=4)",
    #     xlabel="Input Length",
    #     ylabel="Throughput (total tokens/s)",
    #     save_path="compare_total_tokens_vs_input_len_tp2_tp4.png",
    #     label1="tp=2",
    #     label2="tp=4"
    # )

    print("可视化完成!")
