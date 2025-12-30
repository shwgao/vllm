#!/usr/bin/env python3
"""
从Grafana导出的JSON文件中提取Prometheus查询表达式，获取数据并保存为CSV
并支持对比不同baseline的结果

使用示例：
1. 对比baseline结果：
   python extract_and_plot_grafana.py --compare-baselines --metrics E2E_Request_Latency_P99_ Inter_Token_Latency_P99_ --aggregation-method mean

2. 提取Grafana查询：
   python extract_and_plot_grafana.py --export-queries-only
"""

import json
import re
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import requests

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use('Agg')  # 使用非交互式后端
except ImportError:
    print("错误: 需要安装 pandas matplotlib numpy")
    print("运行: pip install pandas matplotlib numpy requests")
    exit(1)

# 尝试导入scipy（可选，用于更高级的平滑方法）
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class GrafanaDataExtractor:
    """从Grafana JSON中提取查询并获取Prometheus数据"""
    
    def __init__(self, json_file: str, prometheus_url: Optional[str] = None, 
                 model_path: Optional[str] = None, model_name: Optional[str] = None):
        self.json_file = json_file
        self.prometheus_url = prometheus_url or 'http://localhost:9090'
        self.model_path = model_path or '/ccsopen/home/shouwei/model/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/50fd307e57011801c7833c87efa1984ddf2db42f'
        self.model_name = model_name
        self.dashboard_data = None
        self.queries = []
        
    def load_dashboard(self):
        """加载Grafana dashboard JSON"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.dashboard_data = json.load(f)
        print(f"✓ 已加载dashboard: {self.dashboard_data.get('title', 'Unknown')}")
        
    def extract_queries(self) -> List[Dict]:
        """从dashboard中提取所有Prometheus查询表达式"""
        if not self.dashboard_data:
            self.load_dashboard()
    
        queries = []
        for panel in self.dashboard_data.get('panels', []):
            for target in panel.get('targets', []):
                if target.get('datasource', {}).get('type') == 'prometheus':
                    expr = target.get('expr', '')
                    if expr:
                        queries.append({
                            'panel_title': panel.get('title', 'Unknown Panel'),
                            'expr': expr,
                            'legendFormat': target.get('legendFormat', ''),
                            'refId': target.get('refId', ''),
                            'hide': target.get('hide', False)
                        })
        
        self.queries = queries
        print(f"✓ 提取了 {len(queries)} 个查询表达式")
        return queries
    
    def _get_model_name(self) -> str:
        """获取用于查询的model_name值"""
        if self.model_name:
            return self.model_name
        
        # 从Prometheus获取实际的model_name值
        if self.prometheus_url:
            try:
                url = f"{self.prometheus_url}/api/v1/label/model_name/values"
                data = requests.get(url, timeout=10).json()
                if data.get('status') == 'success' and data.get('data'):
                    return data['data'][0]  # 使用第一个值
            except Exception:
                pass
        
        # 从路径提取
        for part in self.model_path.split('/'):
            if '--' in part and 'models' in part:
                return part.replace('models--', '').replace('--', '-')
        return '.*'
    
    def _sanitize_filename(self, text: str, max_len: int = 50) -> str:
        """清理文件名并限制长度"""
        text = re.sub(r'[<>:"/\\|?*\s]', '_', text).strip('._')
        return text[:max_len] if len(text) > max_len else text
    
    def _format_metric_for_filename(self, metric: Dict) -> str:
        """格式化metric标签用于文件名"""
        parts = []
        for k, v in sorted(metric.items()):
            if k == '__name__':
                continue
            v_str = str(v)
            # 处理长值（特别是model_name）
            if k == 'model_name' and len(v_str) > 50:
                for part in v_str.split('/'):
                    if '--' in part and 'models' in part:
                        v_str = part.replace('models--', '').replace('--', '-')
                        break
                if len(v_str) > 50:
                    v_str = hashlib.md5(v_str.encode()).hexdigest()[:8]
            elif len(v_str) > 30:
                v_str = v_str[:27] + "..."
            parts.append(f"{k}_{v_str}")
        result = '_'.join(parts)
        return result[:100] if len(result) > 100 else result
    
    def check_prometheus_connection(self) -> bool:
        """检查Prometheus连接是否可用"""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/status/config", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False
    
    def query_prometheus(self, expr: str, start: datetime, end: datetime, 
                        step: str = '15s', debug: bool = False) -> Optional[List[Dict]]:
        """从Prometheus查询数据"""
        # 替换变量
        model_name = self._get_model_name()
        is_regex = '=~"' in expr or "=~'" in expr
        escaped = re.escape(model_name) if is_regex else model_name.replace('"', '\\"')
        expr = expr.replace('$model_name', escaped).replace('$__rate_interval', '1m0s')
        
        if debug:
            print(f"  [DEBUG] model_name: {model_name}")
            print(f"  [DEBUG] 替换后的表达式: {expr}")
        
        # 查询
        try:
            params = {
                'query': expr,
                'start': start.timestamp(),
                'end': end.timestamp(),
                'step': step
            }
            if debug:
                print(f"  [DEBUG] 请求URL: {self.prometheus_url}/api/v1/query_range")
                print(f"  [DEBUG] 请求参数: {params}")
            
            response = requests.get(f"{self.prometheus_url}/api/v1/query_range", 
                                  params=params, timeout=30)
            data = response.json()
            
            if debug:
                print(f"  [DEBUG] 响应状态: {data.get('status')}")
                if data.get('status') != 'success':
                    print(f"  [DEBUG] 错误信息: {data.get('error', 'Unknown error')}")
                if data.get('data', {}).get('result'):
                    print(f"  [DEBUG] 返回结果数量: {len(data['data']['result'])}")
                else:
                    print(f"  [DEBUG] 无结果数据")
            
            if data.get('status') != 'success':
                error_msg = data.get('error', 'Unknown error')
                if debug:
                    print(f"  [DEBUG] Prometheus错误: {error_msg}")
                return None
            
            if not data.get('data', {}).get('result'):
                if debug:
                    print(f"  [DEBUG] 查询成功但无数据")
                return None
            
            results = []
            for result in data['data']['result']:
                results.append({
                    'metric': result.get('metric', {}),
                    'values': [(float(ts), float(val)) for ts, val in result.get('values', [])]
                })
            return results
        except requests.exceptions.ConnectionError as e:
            error_msg = str(e)
            if 'Connection refused' in error_msg or 'Failed to establish' in error_msg:
                print(f"  ✗ 无法连接到Prometheus: {self.prometheus_url}")
                print(f"    请检查Prometheus服务是否运行，或使用 --prometheus-url 指定正确的地址")
            else:
                print(f"  ✗ 连接错误: {error_msg}")
            return None
        except requests.exceptions.Timeout:
            print(f"  ✗ 查询超时（>30秒）")
            return None
        except Exception as e:
            print(f"  ✗ 查询失败: {type(e).__name__}: {e}")
            return None
    
    def parse_time_range(self, time_range_json: str) -> Tuple[datetime, datetime]:
        """解析时间范围JSON字符串"""
        tr = json.loads(time_range_json)
        
        def parse(t: str) -> datetime:
            if t.endswith('Z'):
                return datetime.fromisoformat(t.replace('Z', '+00:00'))
            elif ' ' in t:
                try:
                    return datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    return datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
            return datetime.fromisoformat(t)
        
        return parse(tr['from']), parse(tr['to'])
    
    def export_queries_to_file(self, output_file: str = 'prometheus_queries.txt'):
        """导出所有查询表达式到文本文件"""
        if not self.queries:
            self.extract_queries()
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Prometheus查询表达式\n" + "="*80 + "\n\n")
            current_panel = None
            for query in self.queries:
                if query['panel_title'] != current_panel:
                    current_panel = query['panel_title']
                    f.write(f"\n【{current_panel}】\n" + "-"*80 + "\n")
                f.write(f"Legend: {query['legendFormat']}\n")
                f.write(f"Expression:\n{query['expr']}\n\n")
        print(f"✓ 查询表达式已导出到: {output_file}")
    
    def query_all_and_save_csv(self, time_range_json: str, 
                              output_base_dir: str = './z_tests/results',
                              folder_name: str = 'ours',
                              debug: bool = False,
                              overwrite: bool = False):
        """查询所有Prometheus查询并将结果保存为CSV文件"""
        if not self.queries:
            self.extract_queries()
        
        start, end = self.parse_time_range(time_range_json)
        output_dir = Path(output_base_dir) / folder_name
        if output_dir.exists():
            if overwrite:
                import shutil
                print(f"  ⚠ 文件夹已存在，删除旧文件夹: {output_dir}")
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"  ⚠ 文件夹已存在，将继续使用: {output_dir}")
                print(f"  (使用 --overwrite 参数可删除旧文件夹)")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"开始查询数据并保存到: {output_dir}")
        print(f"时间范围: {start} 到 {end}")
        print(f"共 {len(self.queries)} 个查询")
        print(f"Prometheus URL: {self.prometheus_url}")
        print(f"{'='*80}\n")
        
        # 检查Prometheus连接
        print("检查Prometheus连接...")
        if not self.check_prometheus_connection():
            print(f"\n✗ 错误: 无法连接到Prometheus服务器: {self.prometheus_url}")
            print(f"   请检查:")
            print(f"   1. Prometheus服务是否正在运行")
            print(f"   2. URL是否正确（当前: {self.prometheus_url}）")
            print(f"   3. 网络连接是否正常")
            print(f"   4. 使用 --prometheus-url 参数指定正确的Prometheus地址")
            print(f"\n   例如: --prometheus-url http://your-prometheus-host:9090")
            return
        print("✓ Prometheus连接正常\n")
        
        success, failed = 0, 0
        
        for i, query in enumerate(self.queries, 1):
            if query.get('hide', False):
                continue
                
            panel_title = query['panel_title']
            legend = query['legendFormat'] or query['refId'] or 'query'
            print(f"[{i}/{len(self.queries)}] 查询: {panel_title} - {legend}")
            
            if debug:
                print(f"  [DEBUG] 查询表达式: {query['expr']}")
            
            results = self.query_prometheus(query['expr'], start, end, debug=debug)
            if not results:
                print(f"  ⚠ 无数据，跳过\n")
                failed += 1
                continue
            
            for series_idx, series_data in enumerate(results):
                values = series_data['values']
                if not values:
                    continue
                
                # 创建DataFrame
                df = pd.DataFrame({
                    'timestamp': [datetime.fromtimestamp(ts) for ts, _ in values],
                    'value': [val for _, val in values]
                }).dropna()
                
                if df.empty:
                    continue
                
                # 生成文件名
                panel = self._sanitize_filename(panel_title, 40)
                legend_part = self._sanitize_filename(legend, 30)
                metric = self._format_metric_for_filename(series_data['metric'])
                filename = f"{panel}_{legend_part}_{metric}.csv"
                if len(results) > 1:
                    filename = f"{panel}_{legend_part}_{metric}_series{series_idx}.csv"
                
                # 确保文件名不超过200字符
                if len(filename) > 200:
                    max_metric = 200 - len(panel) - len(legend_part) - 20
                    metric = metric[:max_metric-3] + "..."
                    filename = f"{panel}_{legend_part}_{metric}.csv"
                
                filepath = output_dir / filename
                df.to_csv(filepath, index=False, encoding='utf-8')
                print(f"  ✓ 已保存: {filename} ({len(df)} 行数据)")
                success += 1
            
            print()
        
        print(f"\n{'='*80}")
        print(f"完成! 成功: {success}, 失败: {failed}")
        print(f"数据保存在: {output_dir}")
        print(f"{'='*80}\n")


class BaselineComparator:
    """对比不同baseline的Grafana结果"""
    
    def __init__(self, results_dir: str = './z_tests/grafana-results', 
                 smooth_window: Optional[int] = None,
                 smooth_method: str = 'polyfit'):
        self.results_dir = Path(results_dir)
        self.baselines = ['ours', 'TP', 'DP']
        self.colors = {'ours': '#1f77b4', 'TP': '#ff7f0e', 'DP': '#2ca02c', 
                       'shift_parallel': '#9467bd', 'Ours': '#1f77b4'}
        self.font_size = 28
        self.line_width = 3
        self.alpha = 0.8
        self.markers = 'o'
        self.smooth_window = smooth_window  # 平滑窗口大小，None表示不平滑
        self.smooth_method = smooth_method  # 平滑方法：'polyfit'（多项式拟合）或'savgol'（Savitzky-Golay）
    
    def _sanitize_filename(self, text: str, max_len: int = 50) -> str:
        """清理文件名并限制长度"""
        text = re.sub(r'[<>:"/\\|?*\s]', '_', text).strip('._')
        return text[:max_len] if len(text) > max_len else text
    
    def _smooth_data(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """对数据进行平滑处理（使用拟合方法）
        
        Args:
            df: 包含'relative_time'和'value'列的DataFrame
            window: 平滑窗口大小（数据点数量）或多项式阶数
        
        Returns:
            平滑后的DataFrame
        """
        if df.empty or len(df) < 3:
            return df.copy()
        
        # 按时间排序
        df_sorted = df.sort_values('relative_time').copy()
        
        x = df_sorted['relative_time'].values
        y = df_sorted['value'].values
        
        if self.smooth_method == 'savgol' and HAS_SCIPY:
            # 使用Savitzky-Golay滤波器（需要scipy）
            # window必须是奇数，且小于数据点数量
            window_size = min(window if window % 2 == 1 else window - 1, len(y) - 1)
            if window_size < 3:
                window_size = 3 if len(y) >= 3 else len(y)
            poly_order = min(3, window_size - 1)  # 多项式阶数通常为3
            try:
                y_smooth = savgol_filter(y, window_length=window_size, polyorder=poly_order)
                df_sorted['value'] = y_smooth
            except Exception as e:
                print(f"  ⚠ Savitzky-Golay平滑失败，回退到多项式拟合: {e}")
                # 回退到多项式拟合
                self._smooth_data_polyfit(df_sorted, x, y, window)
        else:
            # 使用多项式拟合（默认方法）
            self._smooth_data_polyfit(df_sorted, x, y, window)
        
        return df_sorted
    
    def _smooth_data_polyfit(self, df_sorted: pd.DataFrame, x: np.ndarray, y: np.ndarray, degree: int):
        """使用多项式拟合进行平滑
        
        Args:
            df_sorted: 要修改的DataFrame
            x: 时间值数组
            y: 原始值数组
            degree: 多项式阶数
        """
        # 确保阶数合理
        max_degree = min(degree, len(y) - 1, 10)  # 最多10阶，且不能超过数据点数-1
        if max_degree < 1:
            max_degree = 1
        
        try:
            # 多项式拟合
            coeffs = np.polyfit(x, y, max_degree)
            # 计算拟合值
            y_smooth = np.polyval(coeffs, x)
            df_sorted['value'] = y_smooth
        except Exception as e:
            print(f"  ⚠ 多项式拟合失败: {e}，使用原始数据")
            # 如果拟合失败，保持原始数据
        
    def find_matching_files(self, baseline: str, metric_prefix: str, note_folder: Optional[str] = None) -> List[Path]:
        """查找匹配指定指标前缀的所有文件
        
        Args:
            baseline: baseline名称（对应lookup_table的key）
            metric_prefix: 指标前缀
            note_folder: 可选的note文件夹路径（用于批量比较模式）
        """
        if note_folder:
            # 批量比较模式：从 <baseline>/<note_folder>/ 查找
            baseline_dir = self.results_dir / baseline / note_folder
        else:
            # 传统模式：从 <baseline>/ 查找
            baseline_dir = self.results_dir / baseline
        
        if not baseline_dir.exists():
            return []
        
        matching_files = []
        for file_path in baseline_dir.glob('*.csv'):
            if file_path.name.startswith(metric_prefix):
                matching_files.append(file_path)
        
        return sorted(matching_files)
    
    def load_csv_with_relative_time(self, file_path: Path) -> Optional[pd.DataFrame]:
        """加载CSV文件并将时间转换为相对时间"""
        try:
            df = pd.read_csv(file_path)
            if 'timestamp' not in df.columns or 'value' not in df.columns:
                return None
            
            # 解析时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            if df.empty:
                return None
            
            # 转换为相对时间（秒）
            start_time = df['timestamp'].min()
            df['relative_time'] = (df['timestamp'] - start_time).dt.total_seconds()
            
            return df[['relative_time', 'value']].copy()
        except Exception as e:
            print(f"  ⚠ 加载文件失败 {file_path}: {e}")
            return None
    
    def aggregate_multiple_files(self, dataframes: List[pd.DataFrame], 
                                 method: str = 'mean') -> Optional[pd.DataFrame]:
        """聚合多个DataFrame（平均或求和）"""
        if not dataframes:
            return None
        
        if len(dataframes) == 1:
            return dataframes[0].copy()
        
        # 找到所有时间点的并集
        all_times = set()
        for df in dataframes:
            all_times.update(df['relative_time'].values)
        all_times = sorted(all_times)
        
        # 对每个时间点进行插值并聚合
        result_data = []
        for time in all_times:
            values = []
            for df in dataframes:
                # 找到最接近的时间点
                time_diff = (df['relative_time'] - time).abs()
                closest_idx = time_diff.idxmin()
                closest_time = df.loc[closest_idx, 'relative_time']
                # 如果时间差小于2秒，使用该值（允许一定的时间偏差）
                if abs(closest_time - time) < 2.0:
                    values.append(df.loc[closest_idx, 'value'])
            
            if values:
                if method == 'mean':
                    agg_value = sum(values) / len(values)
                elif method == 'sum':
                    agg_value = sum(values)
                elif method == 'min':
                    agg_value = min(values)
                elif method == 'max':
                    agg_value = max(values)
                else:
                    agg_value = sum(values) / len(values)  # 默认平均
                result_data.append({'relative_time': time, 'value': agg_value})
        
        if not result_data:
            return None
        
        result_df = pd.DataFrame(result_data)
        result_df = result_df.sort_values('relative_time').reset_index(drop=True)
        return result_df
    
    def plot_metric_comparison(self, metric_prefix: str, 
                               scale: Optional[str] = 'linear',
                               output_file: Optional[str] = None,
                               aggregation_method: str = 'mean',
                               title: Optional[str] = None,
                               baselines: Optional[List[str]] = None,
                               note_folder: Optional[str] = None):
        """绘制单个指标的对比图
        
        Args:
            metric_prefix: 指标前缀
            scale: 坐标轴刻度类型 ('linear' 或 'log')
            output_file: 输出文件路径
            aggregation_method: 聚合方法 ('mean' 或 'sum')
            title: 图表标题
            baselines: 要比较的baseline列表（如果为None，使用self.baselines）
            note_folder: 可选的note文件夹路径（用于批量比较模式）
        """
        print(f"\n处理指标: {metric_prefix}")
        
        if baselines is None:
            baselines = self.baselines
        
        baseline_data = {}
        
        for baseline in baselines:
            files = self.find_matching_files(baseline, metric_prefix, note_folder)
            if not files:
                print(f"  ⚠ {baseline}: 未找到匹配文件，跳过")
                continue
            
            print(f"  {baseline}: 找到 {len(files)} 个文件")
            
            # 加载所有文件
            dataframes = []
            for file_path in files:
                df = self.load_csv_with_relative_time(file_path)
                if df is not None and not df.empty:
                    dataframes.append(df)
            
            if not dataframes:
                print(f"  ⚠ {baseline}: 所有文件加载失败，跳过")
                continue
            
            # 聚合多个文件
            if len(dataframes) > 1:
                aggregated_df = self.aggregate_multiple_files(dataframes, aggregation_method)
                if aggregated_df is not None:
                    baseline_data[baseline] = aggregated_df
                    print(f"  ✓ {baseline}: 聚合了 {len(dataframes)} 个文件")
            else:
                baseline_data[baseline] = dataframes[0]
                print(f"  ✓ {baseline}: 使用单个文件")
                
            if metric_prefix == 'Time_To_First_Token_Latency_P90_':
                # baseline_data[baseline] = baseline_data[baseline] * 0.9
                # # nemotron
                # offset = {'Ours': 45, 'ours': 45, 'TP': 45, 'DP': 30, 'shift_parallel': 35}
                # llama-3-70b
                offset = {'Ours': 35, 'ours': 25, 'TP': 45, 'DP': 30, 'shift_parallel': 35}
                # gpt-oss-120b
                # offset = {'Ours': 30, 'ours': 30, 'TP': 15, 'DP': 35, 'shift_parallel': 15}
                # # 使用.get()方法避免KeyError，如果baseline不在offset中，默认使用0
                time_offset = offset.get(baseline, 0)
                baseline_data[baseline]['relative_time'] = baseline_data[baseline]['relative_time'] + time_offset
            
            if metric_prefix == 'Queue_Time_':
                # nemotron
                # offset = {'Ours': 0, 'ours': 0, 'TP': -30, 'DP': 25, 'shift_parallel': 15}
                # llama-3-70b
                # offset = {'Ours': 0, 'ours': 0, 'TP': 20, 'DP': 25, 'shift_parallel': 15}
                # gpt-oss-120b
                offset = {'Ours': -10, 'ours': -10, 'TP': 15, 'DP': 5, 'shift_parallel': 15}
                time_offset = offset.get(baseline, 0)
                baseline_data[baseline]['relative_time'] = baseline_data[baseline]['relative_time'] + time_offset
                
            if metric_prefix == 'Request_Prompt_Length_':
                # nemotron
                # offset = {'Ours': 0, 'ours': 0, 'TP': -45, 'DP': 25, 'shift_parallel': 15}
                # llama-3-70b
                # offset = {'Ours': 0, 'ours': 0, 'TP': 15, 'DP': 25, 'shift_parallel': 15}
                # gpt-oss-120b
                offset = {'Ours': 0, 'ours': 0, 'TP': 15, 'DP': 5, 'shift_parallel': 15}
                time_offset = offset.get(baseline, 0)
                baseline_data[baseline]['relative_time'] = baseline_data[baseline]['relative_time'] + time_offset
            # if baseline == 'TP' and metric_prefix == 'Queue_Time_':
            #     baseline_data[baseline]['relative_time'] = baseline_data[baseline]['relative_time'] + 30
                
            # if baseline == 'TP' and metric_prefix == 'Request_Prompt_Length_':
            #     baseline_data[baseline]['relative_time'] = baseline_data[baseline]['relative_time'] + 30
        
        if not baseline_data:
            print(f"  ✗ 没有可用的数据，跳过绘图")
            return
        
        # 如果只有一个baseline有数据，也跳过（无法对比）
        if len(baseline_data) < 2:
            print(f"  ⚠ 只有 {len(baseline_data)} 个baseline有数据，跳过对比")
            return
        
        # 应用平滑处理
        if self.smooth_window:
            method_name = 'Savitzky-Golay' if (self.smooth_method == 'savgol' and HAS_SCIPY) else '多项式拟合'
            print(f"  📊 应用平滑处理（方法: {method_name}, 参数: {self.smooth_window}）")
            for baseline in baseline_data.keys():
                baseline_data[baseline] = self._smooth_data(baseline_data[baseline], self.smooth_window)
        
        # 绘制对比图
        plt.figure(figsize=(10, 4))
        
        colors = {'ours': '#1f77b4', 'TP': '#ff7f0e', 'DP': '#2ca02c', 
                  'shift_parallel': '#9467bd', 'Ours': '#1f77b4'}
        linestyles = {'ours': '-', 'TP': '--', 'DP': '-.', 
                      'shift_parallel': ':', 'Ours': '-'}
        
        # # 截取数据，保留数据relative_time在100到400秒之间的数据， 然后重新计算relative_time
        # for baseline in baseline_data.keys():
        #     df = baseline_data[baseline]
        #     if metric_prefix == 'Time_To_First_Token_Latency_P90_':
        #         df = df[(df['relative_time'] >= 100) & (df['relative_time'] <= 400)].copy()
        #     else:
        #         df = df[(df['relative_time'] >= 165) & (df['relative_time'] <= 455)].copy()
        #     df['relative_time'] = df['relative_time'] - df['relative_time'].min()
        #     baseline_data[baseline] = df
            
        for baseline in baseline_data.keys():
            df = baseline_data[baseline]
            df = df[(df['relative_time'] >= 150) & (df['relative_time'] <= 500)].copy()
            df['relative_time'] = df['relative_time'] - df['relative_time'].min()
            baseline_data[baseline] = df
            
        for baseline, df in baseline_data.items():
            plt.plot(
                df['relative_time'],
                df['value'],
                label=baseline,
                color=colors.get(baseline, 'gray'),
                linestyle=linestyles.get(baseline, '-'),
                linewidth=self.line_width,
                alpha=self.alpha,
                marker=self.markers if isinstance(self.markers, str) else None
            )
        
        plt.xlabel('Relative Time (seconds)', fontsize=self.font_size)
        # plt.ylabel('Value', fontsize=12)
        
        # y轴使用log scale
        if scale == 'log':
            plt.yscale('log')
        #     if metric_prefix == 'Queue_Time_':
        #         plt.ylim(0, 120)
        # plt.xlim(0, 300)
        
        # y轴ticks字体大小, 格式为10^x
        # from matplotlib.ticker import LogFormatter
        plt.yticks(fontsize=self.font_size-4)
        # plt.gca().yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
        plt.xticks(fontsize=self.font_size-4)
        # plt.gca().xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
        
        # 生成友好的标题
        if not title:
            title = metric_prefix.rstrip('_').replace('_', ' ')
        # plt.title(title, fontsize=self.font_size, fontweight='bold')
        
        # only show legend for TTFT
        if metric_prefix == 'Request_Prompt_Length_':
            # legend shows on top of the plot box and lies horizontally
            # Optimized: smaller font, better spacing, frame styling
            plt.legend(fontsize=self.font_size-2, loc='lower center', 
                      bbox_to_anchor=(0.5, 0.95), ncol=len(baselines),
                      frameon=True, framealpha=0.9, fancybox=True, 
                      columnspacing=1, handletextpad=0.5,
                      borderpad=0.3, handlelength=1.5)

        # plt.legend(fontsize=self.font_size, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ 图表已保存: {output_file}")
        else:
            # 清理文件名
            safe_name = metric_prefix.rstrip('_').replace(' ', '_')
            output_file = self.results_dir / f"{safe_name}_comparison.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ 图表已保存: {output_file}")
        
        plt.close()
    
    def plot_multiple_metrics(self, metrics: List[Tuple], 
                             output_dir: Optional[str] = None,
                             aggregation_method: str = 'mean',
                             baselines: Optional[List[str]] = None,
                             note_folder: Optional[str] = None):
        """绘制多个指标的对比图
        
        Args:
            metrics: 指标列表，每个元素可以是：
                - (metric_prefix, scale) - 二元组，使用默认aggregation_method
                - (metric_prefix, scale, aggregation_method) - 三元组
            output_dir: 输出目录
            aggregation_method: 默认聚合方法（当metrics为二元组时使用）
            baselines: 要比较的baseline列表
            note_folder: 可选的note文件夹路径（用于批量比较模式）
        """
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.results_dir / 'comparisons'
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"开始绘制 {len(metrics)} 个指标的对比图")
        print(f"输出目录: {output_path}")
        print(f"聚合方法: {aggregation_method}")
        if note_folder:
            print(f"Note文件夹: {note_folder}")
        if baselines:
            print(f"Baselines: {', '.join(baselines)}")
        print(f"{'='*80}\n")
        
        success_count = 0
        skip_count = 0
        
        for metric_item in metrics:
            # 解析metrics格式：支持二元组或三元组
            if len(metric_item) == 2:
                metric_prefix, scale = metric_item
                agg_method = aggregation_method
            elif len(metric_item) == 3:
                metric_prefix, scale, agg_method = metric_item
            else:
                print(f"  ⚠ 跳过无效的metric格式: {metric_item}")
                skip_count += 1
                continue
            
            # 清理文件名，移除末尾的下划线
            safe_name = metric_prefix.rstrip('_').replace(' ', '_')
            # 文件名不再包含note信息（因为note已经在目录名中了）
            output_file = output_path / f"{safe_name}_comparison.pdf"
            
            # 保存当前baseline_data数量，用于判断是否成功绘制
            before_count = len([b for b in (baselines or self.baselines) 
                               if (self.results_dir / b / (note_folder or '')).exists()])
            
            self.plot_metric_comparison(
                metric_prefix, scale, str(output_file), agg_method,
                baselines=baselines, note_folder=note_folder
            )
            
            # 检查文件是否生成（简单判断）
            if output_file.exists():
                success_count += 1
            else:
                skip_count += 1
        
        print(f"\n{'='*80}")
        print(f"完成! 成功绘制: {success_count}, 跳过: {skip_count}")
        print(f"图表保存在: {output_path}")
        print(f"{'='*80}\n")
    
    def _get_metric_data(self, metric_prefix: str, 
                        aggregation_method: str = 'mean',
                        baselines: Optional[List[str]] = None,
                        note_folder: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """获取指定metric的所有baseline数据（内部辅助方法）
        
        Args:
            metric_prefix: 指标前缀
            aggregation_method: 聚合方法
            baselines: baseline列表
            note_folder: note文件夹路径
            
        Returns:
            字典，key为baseline名称，value为DataFrame
        """
        if baselines is None:
            baselines = self.baselines
        
        baseline_data = {}
        
        for baseline in baselines:
            files = self.find_matching_files(baseline, metric_prefix, note_folder)
            if not files:
                continue
            
            # 加载所有文件
            dataframes = []
            for file_path in files:
                df = self.load_csv_with_relative_time(file_path)
                if df is not None and not df.empty:
                    dataframes.append(df)
            
            if not dataframes:
                continue
            
            # 聚合多个文件
            if len(dataframes) > 1:
                aggregated_df = self.aggregate_multiple_files(dataframes, aggregation_method)
                if aggregated_df is not None:
                    baseline_data[baseline] = aggregated_df
            else:
                baseline_data[baseline] = dataframes[0]
            
            # 应用时间偏移（复用原有逻辑）
            if metric_prefix == 'Time_To_First_Token_Latency_P90_':
                offset = {'Ours': 45, 'ours': 45, 'TP': 45, 'DP': 30, 'shift_parallel': 55}
                time_offset = offset.get(baseline, 0)
                baseline_data[baseline]['relative_time'] = baseline_data[baseline]['relative_time'] + time_offset
            
            if metric_prefix == 'Queue_Time_':
                offset = {'Ours': 0, 'ours': 0, 'TP': -30, 'DP': 25, 'shift_parallel': 30}
                time_offset = offset.get(baseline, 0)
                baseline_data[baseline]['relative_time'] = baseline_data[baseline]['relative_time'] + time_offset
                
            if metric_prefix == 'Request_Prompt_Length_':
                offset = {'Ours': 0, 'ours': 0, 'TP': -45, 'DP': 25, 'shift_parallel': 30}
                time_offset = offset.get(baseline, 0)
                baseline_data[baseline]['relative_time'] = baseline_data[baseline]['relative_time'] + time_offset
            
            # 数据裁剪和平滑
            df = baseline_data[baseline]
            df = df[(df['relative_time'] >= 150) & (df['relative_time'] <= 500)].copy()
            df['relative_time'] = df['relative_time'] - df['relative_time'].min()
            baseline_data[baseline] = df
            
            # 应用平滑处理
            if self.smooth_window:
                baseline_data[baseline] = self._smooth_data(baseline_data[baseline], self.smooth_window)
        
        return baseline_data
    
    def plot_metrics_together(self, metrics: List[Tuple],
                             output_file: Optional[str] = None,
                             aggregation_method: str = 'mean',
                             baselines: Optional[List[str]] = None,
                             note_folder: Optional[str] = None,
                             shared_x_range: bool = True):
        """将多个指标绘制在同一个figure中，纵向排列
        
        Args:
            metrics: 指标列表，每个元素可以是：
                - (metric_prefix, scale) - 二元组，使用默认aggregation_method
                - (metric_prefix, scale, aggregation_method) - 三元组
            output_file: 输出文件路径
            aggregation_method: 默认聚合方法
            baselines: 要比较的baseline列表
            note_folder: 可选的note文件夹路径
            shared_x_range: 是否所有subplot共享相同的x轴范围
        """
        if baselines is None:
            baselines = self.baselines
        
        print(f"\n{'='*80}")
        print(f"开始绘制组合图，包含 {len(metrics)} 个指标")
        if note_folder:
            print(f"Note文件夹: {note_folder}")
        print(f"Baselines: {', '.join(baselines)}")
        print(f"{'='*80}\n")
        
        # 解析metrics并加载数据
        metric_configs = []
        all_baseline_data = {}
        
        for metric_item in metrics:
            # 解析metrics格式
            if len(metric_item) == 2:
                metric_prefix, scale = metric_item
                agg_method = aggregation_method
            elif len(metric_item) == 3:
                metric_prefix, scale, agg_method = metric_item
            else:
                print(f"  ⚠ 跳过无效的metric格式: {metric_item}")
                continue
            
            metric_configs.append((metric_prefix, scale, agg_method))
            
            # 加载数据
            print(f"  加载指标: {metric_prefix}")
            baseline_data = self._get_metric_data(metric_prefix, agg_method, baselines, note_folder)
            
            # 过滤掉数据不足的baselines
            if len(baseline_data) < 2:
                print(f"  ⚠ {metric_prefix}: 只有 {len(baseline_data)} 个baseline有数据，跳过")
                continue
            
            all_baseline_data[metric_prefix] = baseline_data
            print(f"  ✓ {metric_prefix}: 加载了 {len(baseline_data)} 个baseline的数据")
        
        if not metric_configs or not all_baseline_data:
            print(f"  ✗ 没有可用的数据，跳过绘图")
            return
        
        # 确定共享的x轴范围
        if shared_x_range:
            all_x_min, all_x_max = float('inf'), float('-inf')
            for baseline_data in all_baseline_data.values():
                for df in baseline_data.values():
                    if not df.empty:
                        all_x_min = min(all_x_min, df['relative_time'].min())
                        all_x_max = max(all_x_max, df['relative_time'].max())
            # 如果没有找到有效范围，使用默认值
            if all_x_min == float('inf') or all_x_max == float('-inf'):
                all_x_min, all_x_max = 0, 350
        
        # 创建figure和subplots
        n_metrics = len(metric_configs)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
        
        # 如果只有一个subplot，axes不是数组，需要转换
        if n_metrics == 1:
            axes = [axes]
        
        colors = {'ours': '#1f77b4', 'TP': '#ff7f0e', 'DP': '#2ca02c', 
                  'shift_parallel': '#9467bd', 'Ours': '#1f77b4'}
        linestyles = {'ours': '-', 'TP': '--', 'DP': '-.', 
                      'shift_parallel': ':', 'Ours': '-'}
        
        # 绘制每个metric
        for idx, (metric_prefix, scale, agg_method) in enumerate(metric_configs):
            if metric_prefix not in all_baseline_data:
                continue
            
            ax = axes[idx]
            baseline_data = all_baseline_data[metric_prefix]
            
            # 绘制每个baseline
            for baseline, df in baseline_data.items():
                ax.plot(
                    df['relative_time'],
                    df['value'],
                    label=baseline,
                    color=colors.get(baseline, 'gray'),
                    linestyle=linestyles.get(baseline, '-'),
                    linewidth=self.line_width,
                    alpha=self.alpha,
                    marker=self.markers if isinstance(self.markers, str) else None
                )
            
            # 设置y轴scale
            if scale == 'log':
                ax.set_yscale('log')
            
            # 设置标题（使用metric名称）
            title = metric_prefix.rstrip('_').replace('_', ' ')
            ax.set_ylabel(title, fontsize=self.font_size)
            ax.tick_params(axis='both', labelsize=self.font_size-4)
            ax.grid(True, alpha=0.3)
            
            # 设置x轴范围（如果共享）
            if shared_x_range:
                ax.set_xlim(all_x_min, all_x_max)
            
            # 只在第一个subplot显示legend
            if idx == 0:
                ax.legend(fontsize=self.font_size-2, loc='upper right', 
                         frameon=True, framealpha=0.9, fancybox=True)
        
        # 只在最下面的subplot显示x轴标签
        axes[-1].set_xlabel('Relative Time (seconds)', fontsize=self.font_size)
        
        plt.tight_layout()
        
        # 保存图表
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ 组合图表已保存: {output_file}")
        else:
            # 生成默认文件名
            safe_names = '_'.join([m[0].rstrip('_').replace(' ', '_')[:20] 
                                  for m in metric_configs[:3]])  # 只取前3个名称
            output_file = self.results_dir / f"{safe_names}_combined_comparison.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ 组合图表已保存: {output_file}")
        
        plt.close()


def run_batch_queries_from_lookup(extractor: GrafanaDataExtractor,
                                  lookup_table,
                                  output_dir: str,
                                  debug: bool = False,
                                  overwrite: bool = False) -> None:
    """根据lookup_table批量执行查询并按key/note分类保存结果"""
    if not lookup_table:
        print("✗ lookup_table为空，跳过批量查询")
        return

    for key, entry in lookup_table.items():
        time_ranges = entry.get("time_ranges", []) or []
        notes = entry.get("note", []) or []

        if len(time_ranges) != len(notes):
            print(f"⚠ 跳过 {key}: time_ranges数量({len(time_ranges)}) 与 note数量({len(notes)}) 不匹配")
            continue

        key_folder = extractor._sanitize_filename(str(key), max_len=80)
        for idx, (time_range_json, note_text) in enumerate(zip(time_ranges, notes), start=1):
            note_folder = extractor._sanitize_filename(str(note_text), max_len=120)
            folder_name = str(Path(key_folder) / note_folder)
            print(f"\n=== [{key}] ({idx}/{len(time_ranges)}) note: {note_text}")
            extractor.query_all_and_save_csv(
                time_range_json,
                output_dir,
                folder_name,
                debug=debug,
                overwrite=overwrite,
            )


def run_batch_combined_plots_from_lookup(comparator: BaselineComparator,
                                         lookup_table: Dict,
                                         combined_metrics: List[Tuple],
                                         output_dir: str,
                                         aggregation_method: str = 'mean',
                                         model_name: Optional[str] = None) -> None:
    """根据lookup_table批量执行组合绘图，按note分组
    
    Args:
        comparator: BaselineComparator实例
        lookup_table: lookup_table字典，包含key和对应的note列表
        combined_metrics: 要组合在一起的指标列表，每个元素可以是(metric_prefix, scale)或(metric_prefix, scale, aggregation_method)
        output_dir: 输出目录
        aggregation_method: 默认聚合方法
        model_name: 模型名称，用于创建子目录
    """
    if not lookup_table:
        print("✗ lookup_table为空，跳过批量组合绘图")
        return
    
    # 收集所有唯一的note
    all_notes = set()
    for entry in lookup_table.values():
        notes = entry.get("note", []) or []
        all_notes.update(notes)
    
    if not all_notes:
        print("✗ 未找到任何note，跳过批量组合绘图")
        return
    
    # 获取所有keys（baselines）
    all_keys = list(lookup_table.keys())
    
    print(f"\n{'='*80}")
    print(f"开始批量组合绘图")
    print(f"Baselines: {', '.join(all_keys)}")
    print(f"Notes: {len(all_notes)} 个")
    print(f"组合指标: {len(combined_metrics)} 个")
    print(f"{'='*80}\n")
    
    # 按note分组进行组合绘图
    for note_idx, note in enumerate(sorted(all_notes), 1):
        print(f"\n{'='*80}")
        print(f"[{note_idx}/{len(all_notes)}] 处理 note: {note}")
        print(f"{'='*80}")
        
        # 清理note名称用于文件夹路径
        note_folder_safe = comparator._sanitize_filename(str(note), max_len=120)
        
        # 检查哪些keys有这个note的数据
        available_keys = []
        for key in all_keys:
            entry = lookup_table.get(key, {})
            notes_list = entry.get("note", []) or []
            if note in notes_list:
                # 检查对应的文件夹是否存在
                key_folder = comparator._sanitize_filename(str(key), max_len=80)
                note_path = comparator.results_dir / key_folder / note_folder_safe
                if note_path.exists() and any(note_path.glob('*.csv')):
                    available_keys.append(key)
        
        if len(available_keys) < 2:
            print(f"  ⚠ 只有 {len(available_keys)} 个baseline有数据，跳过此note的组合绘图")
            continue
        
        print(f"  ✓ 找到 {len(available_keys)} 个baseline的数据: {', '.join(available_keys)}")
        
        # 为这个note创建输出目录
        if model_name:
            note_output_dir = Path(output_dir) / model_name / note_folder_safe
        else:
            note_output_dir = Path(output_dir) / note_folder_safe
        note_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        safe_names = '_'.join([m[0].rstrip('_').replace(' ', '_')[:20] 
                              for m in combined_metrics[:3]])
        output_file = note_output_dir / f"{safe_names}_combined_comparison.pdf"
        
        # 执行组合绘图
        comparator.plot_metrics_together(
            combined_metrics,
            output_file=str(output_file),
            aggregation_method=aggregation_method,
            baselines=available_keys,
            note_folder=note_folder_safe
        )
    
    print(f"\n{'='*80}")
    print(f"批量组合绘图完成!")
    print(f"{'='*80}\n")


def run_batch_comparisons_from_lookup(comparator: BaselineComparator,
                                      lookup_table: Dict,
                                      metrics: List[Tuple],
                                      output_dir: str,
                                      aggregation_method: str = 'mean',
                                      model_name: Optional[str] = None) -> None:
    """根据lookup_table批量执行数据比较，按note分组
    
    Args:
        comparator: BaselineComparator实例
        lookup_table: lookup_table字典，包含key和对应的note列表
        metrics: 指标列表，每个元素可以是(metric_prefix, scale)或(metric_prefix, scale, aggregation_method)
        output_dir: 输出目录
        aggregation_method: 默认聚合方法
        model_name: 模型名称，用于创建子目录（如 'Llama3-70B-Instruct'）
    """
    if not lookup_table:
        print("✗ lookup_table为空，跳过批量比较")
        return
    
    # 收集所有唯一的note
    all_notes = set()
    for entry in lookup_table.values():
        notes = entry.get("note", []) or []
        all_notes.update(notes)
    
    if not all_notes:
        print("✗ 未找到任何note，跳过批量比较")
        return
    
    # 获取所有keys（baselines）
    all_keys = list(lookup_table.keys())
    
    print(f"\n{'='*80}")
    print(f"开始批量比较")
    print(f"Baselines: {', '.join(all_keys)}")
    print(f"Notes: {len(all_notes)} 个")
    print(f"Metrics: {len(metrics)} 个")
    print(f"{'='*80}\n")
    
    # 按note分组进行比较
    for note_idx, note in enumerate(sorted(all_notes), 1):
        print(f"\n{'='*80}")
        print(f"[{note_idx}/{len(all_notes)}] 处理 note: {note}")
        print(f"{'='*80}")
        
        # 清理note名称用于文件夹路径
        note_folder_safe = comparator._sanitize_filename(str(note), max_len=120)
        
        # 检查哪些keys有这个note的数据
        available_keys = []
        for key in all_keys:
            entry = lookup_table.get(key, {})
            notes_list = entry.get("note", []) or []
            if note in notes_list:
                # 检查对应的文件夹是否存在
                key_folder = comparator._sanitize_filename(str(key), max_len=80)
                note_path = comparator.results_dir / key_folder / note_folder_safe
                if note_path.exists() and any(note_path.glob('*.csv')):
                    available_keys.append(key)
        
        if len(available_keys) < 2:
            print(f"  ⚠ 只有 {len(available_keys)} 个baseline有数据，跳过此note的比较")
            continue
        
        print(f"  ✓ 找到 {len(available_keys)} 个baseline的数据: {', '.join(available_keys)}")
        
        # 为这个note创建输出目录
        # 如果提供了model_name，在输出目录下创建model_name子目录
        if model_name:
            note_output_dir = Path(output_dir) / model_name / note_folder_safe
        else:
            note_output_dir = Path(output_dir) / note_folder_safe
        note_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行比较
        comparator.plot_multiple_metrics(
            metrics,
            output_dir=str(note_output_dir),
            aggregation_method=aggregation_method,
            baselines=available_keys,
            note_folder=note_folder_safe
        )
    
    print(f"\n{'='*80}")
    print(f"批量比较完成!")
    print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从Grafana JSON提取查询并获取Prometheus数据')
    parser.add_argument('--json-file', type=str, 
                       default='/ccsopen/home/shouwei/projects/vllm-newest/vllm/z_tests/vLLM-1764960880818.json',
                       help='Grafana导出的JSON文件路径')
    parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090',
                       help='Prometheus API地址')
    parser.add_argument('--model-path', type=str,
                       default='/ccsopen/home/shouwei/model/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a',
                       help='模型路径')
    parser.add_argument('--model-name', type=str,
                       help='直接指定model_name值')
    parser.add_argument('--lookup-table', type=str,
                       default={"Ours": 
                           {'time_ranges':'{"from":"2025-12-09T00:58:35.063Z","to":"2025-12-09T01:06:12.140Z"}',
                            'note':['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0']
                            }})
    parser.add_argument('--output-dir', type=str, default='./z_tests/grafana-results',
                       help='输出目录')
    parser.add_argument('--folder-name', type=str, default='ours-gpt-oss-120b',
                       help='文件夹名称')
    parser.add_argument('--export-queries-only', action='store_true',
                       help='仅导出查询表达式')
    parser.add_argument('--compare-baselines', default=True,
                       help='对比baseline结果')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=[('E2E_Request_Latency_P90_', 'linear', 'mean'), 
                                ('E2E_Request_Latency_P95_', 'linear', 'mean'),
                                ('E2E_Request_Latency_P99_', 'linear', 'mean'),
                                ('E2E_Request_Latency_P50_', 'linear', 'mean'),
                                ('E2E_Request_Latency_Average_', 'linear', 'mean'),
                                ('Inter_Token_Latency_P90_', 'linear', 'mean'), 
                                ('Inter_Token_Latency_P95_', 'linear', 'mean'),
                                ('Inter_Token_Latency_P99_', 'linear', 'mean'),
                                ('Inter_Token_Latency_P50_', 'linear', 'mean'),
                                ('Inter_Token_Latency_Average_', 'linear', 'mean'),
                                ('Time_To_First_Token_Latency_P90_', 'linear', 'mean'),
                                ('Time_To_First_Token_Latency_P95_', 'linear', 'mean'),
                                ('Time_To_First_Token_Latency_P99_', 'linear', 'mean'),
                                ('Time_To_First_Token_Latency_P50_', 'linear', 'mean'),
                                ('Time_To_First_Token_Latency_Average_', 'linear', 'mean'),
                                ('Token_Throughput_', 'linear', 'sum'),
                                ('Token_Throughput_Generation_', 'linear', 'sum'),
                                ('Token_Throughput_Prompt_', 'linear', 'sum'),
                                ('Queue_Time_', 'linear', 'mean'),
                                ('Request_Prompt_Length_', 'linear', 'max'),
                                ('Scheduler_State_Num_Running_', 'linear', 'sum'),
                                ('Scheduler_State_Num_Waiting_', 'linear', 'sum'),
                               ],
                       help='要对比的指标列表（使用前缀匹配）')
    parser.add_argument('--aggregation-method', type=str, default='mean',
                       choices=['mean', 'sum'],
                       help='多个文件时的聚合方法：mean（平均）或sum（求和）')
    parser.add_argument('--comparison-output-dir', type=str,
                       default='./z_tests/grafana-results/comparisons',
                       help='对比图输出目录')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式，显示详细的查询信息')
    parser.add_argument('--overwrite', default=True,
                       help='如果输出文件夹已存在，删除旧文件夹后重新查询')
    parser.add_argument('--time-range', type=str,
                       help='单次查询模式下使用的时间范围JSON字符串')
    parser.add_argument('--smooth-window', type=int, default=None,
                       help='数据平滑参数：对于多项式拟合表示阶数（建议3-5），对于Savitzky-Golay表示窗口大小（奇数）。None表示不平滑')
    parser.add_argument('--smooth-method', type=str, default='polyfit',
                       choices=['polyfit', 'savgol'],
                       help='平滑方法：polyfit（多项式拟合，默认）或savgol（Savitzky-Golay滤波器，需要scipy）')
    parser.add_argument('--combined-metrics', type=str, nargs='+',
                       help='要组合在一起绘制的指标列表（使用前缀匹配），例如：Queue_Time_ Time_To_First_Token_Latency_P90_ Request_Prompt_Length_')
    
    args = parser.parse_args()
    models = {
        'Llama3-70B-Instruct': '/ccsopen/home/shouwei/model/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/50fd307e57011801c7833c87efa1984ddf2db42f',
        'gpt-oss-120b': '/ccsopen/home/shouwei/model/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a',
        'nemotron': '/ccsopen/home/shouwei/model/hub/models--nvidia--Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct/snapshots/02ae0431f885eb8f4994112a7f132a38451abe52',
    }
    operating_model = 'Llama3-70B-Instruct'
    
    # for llama-3-70b
    if operating_model == 'Llama3-70B-Instruct':
        args.lookup_table = {
        "shift_parallel": 
            {"time_ranges": 
                ['{"from":"2025-12-11T02:10:24.798Z","to":"2025-12-11T02:20:06.413Z"}',
                '{"from":"2025-12-11T02:23:42.869Z","to":"2025-12-11T02:33:27.679Z"}',
                '{"from":"2025-12-11T02:33:29.891Z","to":"2025-12-11T02:45:55.567Z"}'],
            'note':
                ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0',
                'input=2000-4000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0',
                'input=2000-4000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps10.0']
            },
        "Ours":
            {"time_ranges":
                ['{"from":"2025-12-11T02:56:04.120Z","to":"2025-12-11T03:06:06.652Z"}',
                '{"from":"2025-12-11T04:16:56.987Z","to":"2025-12-11T04:26:35.355Z"}',
                '{"from":"2025-12-11T03:57:19.448Z","to":"2025-12-11T04:10:08.357Z"}'],
            'note':
                ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0',
                'input=2000-4000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0',
                'input=2000-4000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps10.0']
            },
        "TP":
            {"time_ranges":
                ['{"from":"2025-12-11T05:31:54.801Z","to":"2025-12-11T05:41:41.430Z"}',
                '{"from":"2025-12-11T05:06:09.287Z","to":"2025-12-11T05:15:51.517Z"}',
                '{"from":"2025-12-11T05:17:50.123Z","to":"2025-12-11T05:31:01.567Z"}'],
            'note':
                ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0',
                'input=2000-4000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0',
                'input=2000-4000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps10.0']
            },
        "DP":
            {"time_ranges":
                ['{"from":"2025-12-11T05:46:14.360Z","to":"2025-12-11T05:55:47.885Z"}',
                '{"from":"2025-12-11T05:56:48.149Z","to":"2025-12-11T06:06:35.794Z"}',
                '{"from":"2025-12-11T06:08:16.879Z","to":"2025-12-11T06:21:04.340Z"}'],
            'note':
                ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0',
                'input=2000-4000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0',
                'input=2000-4000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps10.0']
            },
        }   
    elif operating_model == 'gpt-oss-120b':
    # for gpt-oss-120b
        args.lookup_table = {
            "Ours":
                {"time_ranges":
                    ['{"from":"2025-12-12T20:02:22.106Z","to":"2025-12-12T20:12:31.454Z"}'],
                'note':
                    ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0']
                },
            "TP":
                {"time_ranges":
                    ['{"from":"2025-12-12T19:43:24.477Z","to":"2025-12-12T19:52:50.669Z"}'],
                'note':
                    ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0']
                },
            "DP":
                {"time_ranges":
                    ['{"from":"2025-12-12T20:19:36.795Z","to":"2025-12-12T20:29:42.945Z"}'],
                'note':
                    ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0']
                },
            }   
    elif operating_model == 'nemotron':
    # for nemotron
        args.lookup_table = {
            "Ours":
                {"time_ranges":
                    ['{"from":"2025-12-14T02:27:56.745Z","to":"2025-12-14T02:38:39.829Z"}'],
                'note':
                    ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0']
                },
            "DP":
                {"time_ranges":
                    ['{"from":"2025-12-14T02:54:42.197Z","to":"2025-12-14T03:04:17.540Z"}'],
                'note':
                    ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0']
                },
            "TP":
                {"time_ranges":
                    ['{"from":"2025-12-14T02:06:30.577Z","to":"2025-12-14T02:17:13.661Z"}'],
                'note':
                    ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0']
                },
            "shift_parallel":
                {"time_ranges":
                    ['{"from":"2025-12-14T20:18:05.001Z","to":"2025-12-14T20:28:15.016Z"}'],
                'note':
                    ['input=1000-2000num=2000lowrps=2start_ratio=0.4end_ratio=0.5peak_rps20.0']
                },
        }
    else:
        print(f"Unsupported model: {args.model_name}")
        return
    # just for testing on a single metric
    # args.metrics = [('Time_To_First_Token_Latency_P90_', 'log', 'mean')]
    
    args.model_path = models[operating_model]
    args.output_dir = f'./z_tests/grafana-results/{operating_model}'
    
    # 如果是对比模式
    if args.compare_baselines:
        comparator = BaselineComparator(
            args.output_dir, 
            smooth_window=args.smooth_window,
            smooth_method=args.smooth_method
        )
        
        # 处理组合绘图（如果有指定）
        if args.combined_metrics:
            # 将combined_metrics转换为元组列表格式
            combined_metrics_list = []
            for metric_str in args.combined_metrics:
                # 尝试从args.metrics中找到对应的配置
                metric_config = None
                for m in args.metrics:
                    if isinstance(m, (tuple, list)) and len(m) >= 1:
                        if m[0] == metric_str:
                            metric_config = m
                            break
                
                if metric_config:
                    if len(metric_config) == 2:
                        combined_metrics_list.append((metric_config[0], metric_config[1], args.aggregation_method))
                    elif len(metric_config) == 3:
                        combined_metrics_list.append(metric_config)
                    else:
                        combined_metrics_list.append((metric_config[0], 'linear', args.aggregation_method))
                else:
                    # 如果没有找到配置，使用默认值
                    combined_metrics_list.append((metric_str, 'linear', args.aggregation_method))
            
            print(f"组合绘图指标: {[m[0] for m in combined_metrics_list]}")
            
            if args.lookup_table:
                # 批量组合绘图
                run_batch_combined_plots_from_lookup(
                    comparator,
                    args.lookup_table,
                    combined_metrics_list,
                    args.comparison_output_dir,
                    args.aggregation_method,
                    model_name=operating_model
                )
            else:
                # 单次组合绘图
                safe_names = '_'.join([m[0].rstrip('_').replace(' ', '_')[:20] 
                                      for m in combined_metrics_list[:3]])
                output_file = Path(args.comparison_output_dir) / f"{safe_names}_combined_comparison.pdf"
                Path(args.comparison_output_dir).mkdir(parents=True, exist_ok=True)
                comparator.plot_metrics_together(
                    combined_metrics_list,
                    output_file=str(output_file),
                    aggregation_method=args.aggregation_method
                )
        
        # 如果提供了lookup_table，执行批量比较（单独绘图）
        if args.lookup_table:
            run_batch_comparisons_from_lookup(
                comparator,
                args.lookup_table,
                args.metrics,
                args.comparison_output_dir,
                args.aggregation_method,
                model_name=operating_model
            )
        elif not args.combined_metrics:
            # 传统模式：直接比较所有baselines（只在没有组合绘图时执行）
            comparator.plot_multiple_metrics(
                args.metrics, 
                args.comparison_output_dir,
                args.aggregation_method
            )
        return
    
    # 原有的提取和查询功能
    extractor = GrafanaDataExtractor(args.json_file, args.prometheus_url,
                                     args.model_path, args.model_path)
    extractor.load_dashboard()
    extractor.extract_queries()
    extractor.export_queries_to_file('./z_tests/prometheus_monitoring/prometheus_queries.txt')
    
    if args.export_queries_only:
        print("\n✓ 仅导出查询表达式完成")
        return
    
    # 批量模式：按lookup_table的key和note分类保存
    if args.lookup_table:
        run_batch_queries_from_lookup(
            extractor,
            args.lookup_table,
            args.output_dir,
            debug=args.debug,
            overwrite=args.overwrite,
        )
    # 兼容单次模式：如果提供了time_range则执行一次查询
    elif args.time_range:
        extractor.query_all_and_save_csv(
            args.time_range,
            args.output_dir,
            args.folder_name,
            debug=args.debug,
            overwrite=args.overwrite,
        )
    else:
        print("⚠ 未提供time_range且lookup_table为空，未执行任何查询。")


if __name__ == '__main__':
    main()