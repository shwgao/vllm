# Grafana数据提取和绘图指南

这个工具集可以帮助你从Grafana导出的JSON文件中提取Prometheus查询表达式，获取数据并绘制图表。

## 文件说明

- `extract_and_plot_grafana.py`: 主要的数据提取和绘图类
- `plot_example.py`: 使用示例
- `vLLM-1764960880818.json`: Grafana导出的dashboard配置

## 安装依赖

```bash
pip install pandas numpy matplotlib requests
```

## 使用方法

### 方法1: 提取查询表达式（不需要Prometheus连接）

如果你只想查看或导出Prometheus查询表达式：

```python
from extract_and_plot_grafana import GrafanaDataExtractor

extractor = GrafanaDataExtractor('vLLM-1764960880818.json')
extractor.load_dashboard()
extractor.extract_queries()
extractor.print_queries()  # 打印到控制台
extractor.export_queries_to_file('queries.txt')  # 导出到文件
```

### 方法2: 从Prometheus获取数据并绘图

如果你有Prometheus访问权限：

```python
from extract_and_plot_grafana import GrafanaDataExtractor

# 设置Prometheus URL
extractor = GrafanaDataExtractor(
    'vLLM-1764960880818.json',
    prometheus_url='http://localhost:9090'  # 修改为你的Prometheus地址
)

extractor.load_dashboard()
extractor.extract_queries()

# 绘制特定panel的图表
extractor.plot_panel('E2E Request Latency', save_path='e2e_latency.png')
extractor.plot_panel('Token Throughput', save_path='token_throughput.png')
```

### 方法3: 手动查询Prometheus API

如果你想要更多控制，可以直接使用Prometheus API：

```python
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

prometheus_url = 'http://localhost:9090'
end = datetime.now()
start = end - timedelta(hours=2)

# 查询表达式
query = 'histogram_quantile(0.99, sum by(le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))'

url = f"{prometheus_url}/api/v1/query_range"
params = {
    'query': query,
    'start': start.timestamp(),
    'end': end.timestamp(),
    'step': '15s'
}

response = requests.get(url, params=params)
data = response.json()

# 处理数据并绘图
if data['status'] == 'success' and data['data']['result']:
    result = data['data']['result'][0]
    timestamps = [datetime.fromtimestamp(float(ts)) for ts, _ in result['values']]
    values = [float(val) for _, val in result['values']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, values)
    plt.xlabel('Time')
    plt.ylabel('Latency (seconds)')
    plt.title('P99 E2E Request Latency')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('latency.png', dpi=300)
```

## 快速开始

### 激活环境

首先激活你的Python环境（例如 `.ven-defiant`）：

```bash
# 如果是conda环境
conda activate .ven-defiant

# 如果是venv/virtualenv
source ~/.ven-defiant/bin/activate
# 或
source .ven-defiant/bin/activate
```

### 安装依赖（如果需要）

```bash
pip install pandas numpy matplotlib requests
```

### 运行脚本

```bash
# 方法1: 使用启动脚本（会自动尝试激活环境）
./run_extract.sh

# 方法2: 手动激活环境后运行
conda activate .ven-defiant  # 或 source ~/.ven-defiant/bin/activate
python extract_and_plot_grafana.py

# 运行完整示例
python plot_example.py
```

## 从JSON中提取的查询类型

从你的dashboard中，主要包含以下类型的监控指标：

1. **E2E Request Latency** - 端到端请求延迟（P99, P95, P90, P50, Average）
2. **Token Throughput** - Token吞吐量（Prompt Tokens/Sec, Generation Tokens/Sec）
3. **Inter Token Latency** - Token间延迟
4. **Request Status** - 请求状态（运行中、等待中）
5. **Time to First Token** - 首Token时间
6. **GPU Cache Usage** - GPU缓存使用率
7. **Request Queue Time** - 请求队列时间
8. **Request Processing Time** - 请求处理时间（Prefill, Decode）

## 注意事项

1. **Grafana导出的是配置，不是数据**: Grafana导出的JSON文件只包含dashboard配置和查询表达式，不包含实际的时间序列数据。你需要从Prometheus获取数据。

2. **变量替换**: 查询中可能包含变量（如`$model_name`, `$__rate_interval`），需要根据实际情况替换：
   - `$model_name` → 具体的模型名称或使用正则表达式
   - `$__rate_interval` → 时间间隔（如`5m`）

3. **时间范围**: Dashboard中的时间范围是相对时间（如`now-2h`），需要转换为绝对时间戳才能查询Prometheus。

4. **Prometheus访问**: 确保你的Prometheus服务可访问，并且有相应的数据。

## 如果无法访问Prometheus

如果你无法直接访问Prometheus，可以：

1. **导出查询表达式**: 使用`export_queries_to_file()`导出所有查询
2. **手动在Grafana中查询**: 在Grafana UI中执行这些查询并导出CSV
3. **使用Grafana API**: 如果有Grafana访问权限，可以使用Grafana API获取数据

## 示例输出

运行脚本后，你会得到：

- `prometheus_queries.txt`: 所有提取的Prometheus查询表达式
- `*.png`: 绘制的图表（如果设置了Prometheus URL）
