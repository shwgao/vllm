#!/usr/bin/env python3
"""
示例: 如何使用提取的查询从Prometheus获取数据并绘图
"""

from extract_and_plot_grafana import GrafanaDataExtractor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 配置
JSON_FILE = '/ccsopen/home/shouwei/projects/vllm-newest/vllm/z_tests/vLLM-1764960880818.json'
PROMETHEUS_URL = 'http://localhost:9090'  # 修改为你的Prometheus地址

def example_1_extract_queries():
    """示例1: 提取并查看所有查询表达式"""
    print("="*80)
    print("示例1: 提取查询表达式")
    print("="*80)
    
    extractor = GrafanaDataExtractor(JSON_FILE)
    extractor.load_dashboard()
    extractor.extract_queries()
    extractor.print_queries()
    extractor.export_queries_to_file('prometheus_queries.txt')

def example_2_plot_from_prometheus():
    """示例2: 从Prometheus获取数据并绘图"""
    print("\n" + "="*80)
    print("示例2: 从Prometheus获取数据并绘图")
    print("="*80)
    
    extractor = GrafanaDataExtractor(JSON_FILE, PROMETHEUS_URL)
    extractor.load_dashboard()
    extractor.extract_queries()
    
    # 绘制E2E Request Latency
    extractor.plot_panel('E2E Request Latency', save_path='e2e_latency.png')
    
    # 绘制其他panels
    # extractor.plot_panel('Token Throughput', save_path='token_throughput.png')
    # extractor.plot_panel('Inter Token Latency', save_path='inter_token_latency.png')

def example_3_manual_plot():
    """示例3: 手动查询并绘图（更灵活）"""
    print("\n" + "="*80)
    print("示例3: 手动查询并绘图")
    print("="*80)
    
    import requests
    
    # 设置时间范围
    end = datetime.now()
    start = end - timedelta(hours=2)
    
    # 查询表达式（从提取的查询中选择）
    queries = {
        'P99': 'histogram_quantile(0.99, sum by(le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))',
        'P95': 'histogram_quantile(0.95, sum by(le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))',
        'Average': 'rate(vllm:e2e_request_latency_seconds_sum[5m]) / rate(vllm:e2e_request_latency_seconds_count[5m])'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, expr in queries.items():
        url = f"{PROMETHEUS_URL}/api/v1/query_range"
        params = {
            'query': expr,
            'start': start.timestamp(),
            'end': end.timestamp(),
            'step': '15s'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                result = data['data']['result'][0]
                timestamps = [datetime.fromtimestamp(float(ts)) for ts, _ in result['values']]
                values = [float(val) for _, val in result['values']]
                
                ax.plot(timestamps, values, label=label, linewidth=2)
        except Exception as e:
            print(f"查询失败 {label}: {e}")
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('E2E Request Latency', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('manual_plot.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存到: manual_plot.png")

if __name__ == '__main__':
    # 运行示例1: 提取查询
    # example_1_extract_queries()
    
    # 如果需要从Prometheus获取数据，取消下面的注释并设置正确的PROMETHEUS_URL
    example_2_plot_from_prometheus()
    # example_3_manual_plot()
