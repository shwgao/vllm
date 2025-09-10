#!/usr/bin/env python3
# 统计LLM数据集输入和输出的长度分布
# 使用tokenizer统计token数量

import json
import matplotlib.pyplot as plt
import numpy as np
import os
# 设置环境变量避免并行化问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer
    USE_TRANSFORMERS = True
except ImportError:
    USE_TRANSFORMERS = False
    print("transformers not available, using simple character-based tokenization")

def simple_tokenize(text):
    """简单的基于空格的tokenization作为fallback"""
    return text.split()

def plot_length_distribution(data_path, model_name="gpt2"):
    """
    统计并绘制对话数据集中输入和输出的token长度分布
    
    Args:
        data_path: 数据文件路径
        model_name: 用于tokenization的模型名称
    """
    tokenizer = None
    if USE_TRANSFORMERS:
        print(f"Loading tokenizer for {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Successfully loaded {model_name} tokenizer")
        except Exception as e:
            print(f"Failed to load tokenizer {model_name}: {e}")
            print("Using simple word-based tokenization...")
            tokenizer = None
    else:
        print("Using simple word-based tokenization...")
        tokenizer = None
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    input_lengths = []
    output_lengths = []
    
    print("Processing conversations...")
    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(data)} items")
            
        conversations = item.get('conversations', [])
        
        # 提取human和gpt的对话
        for conv in conversations:
            if conv.get('from') == 'human':
                text = conv.get('value', '')
                if text:
                    if tokenizer:
                        try:
                            tokens = tokenizer.encode(text, add_special_tokens=False)
                            input_lengths.append(len(tokens))
                        except Exception as e:
                            # fallback to simple tokenization
                            tokens = simple_tokenize(text)
                            input_lengths.append(len(tokens))
                    else:
                        tokens = simple_tokenize(text)
                        input_lengths.append(len(tokens))
            elif conv.get('from') == 'gpt':
                text = conv.get('value', '')
                if text:
                    if tokenizer:
                        try:
                            tokens = tokenizer.encode(text, add_special_tokens=False)
                            output_lengths.append(len(tokens))
                        except Exception as e:
                            # fallback to simple tokenization
                            tokens = simple_tokenize(text)
                            output_lengths.append(len(tokens))
                    else:
                        tokens = simple_tokenize(text)
                        output_lengths.append(len(tokens))
    
    print(f"Found {len(input_lengths)} input texts and {len(output_lengths)} output texts")
    
    # 计算统计信息
    print("\n=== Input Length Statistics ===")
    if input_lengths:
        print(f"Mean: {np.mean(input_lengths):.2f} tokens")
        print(f"Median: {np.median(input_lengths):.2f} tokens")
        print(f"Min: {min(input_lengths)} tokens")
        print(f"Max: {max(input_lengths)} tokens")
        print(f"95th percentile: {np.percentile(input_lengths, 95):.2f} tokens")
        print(f"99th percentile: {np.percentile(input_lengths, 99):.2f} tokens")
    
    print("\n=== Output Length Statistics ===")
    if output_lengths:
        print(f"Mean: {np.mean(output_lengths):.2f} tokens")
        print(f"Median: {np.median(output_lengths):.2f} tokens")
        print(f"Min: {min(output_lengths)} tokens")
        print(f"Max: {max(output_lengths)} tokens")
        print(f"95th percentile: {np.percentile(output_lengths, 95):.2f} tokens")
        print(f"99th percentile: {np.percentile(output_lengths, 99):.2f} tokens")
    
    # 绘制分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 输入长度分布
    if input_lengths:
        ax1.hist(input_lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Input Length (tokens)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Input Length Distribution')
        ax1.axvline(np.mean(input_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(input_lengths):.1f}')
        ax1.axvline(np.median(input_lengths), color='green', linestyle='--', label=f'Median: {np.median(input_lengths):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 输出长度分布
    if output_lengths:
        ax2.hist(output_lengths, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Output Length (tokens)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Output Length Distribution')
        ax2.axvline(np.mean(output_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(output_lengths):.1f}')
        ax2.axvline(np.median(output_lengths), color='green', linestyle='--', label=f'Median: {np.median(output_lengths):.1f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'length_distribution.png'")

if __name__ == "__main__":
    # 使用gpt2 tokenizer进行统计，处理前10000条数据
    import sys
    
    data_path = '/nfs/stak/users/gaosho/hpc-share/projects/data/ShareGPT_V3_unfiltered_cleaned_split.json'
    model_name = "gpt2"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--sample':
            # 创建一个采样版本来快速测试
            print("Creating sample data for quick testing...")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 只取前1000条数据
            sample_data = data[:1000]
            sample_path = '/tmp/sample_data.json'
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            print(f"Processing sample data with {len(sample_data)} items...")
            plot_length_distribution(sample_path, model_name)
        else:
            print("Usage: python statistic.py [--sample]")
    else:
        print("Processing full dataset...")
        plot_length_distribution(data_path, model_name)
