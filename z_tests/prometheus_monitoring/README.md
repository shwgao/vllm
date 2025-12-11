# Prometheus 和 Grafana 监控

## 启动服务

在 defiant-compute 节点上：

```bash
cd /ccsopen/home/shouwei/projects/vllm-newest/vllm/z_tests/prometheus_monitoring
bash 快速启动步骤.sh
```

## 访问

从本地机器创建 SSH 隧道：

```bash
ssh -L 9090:localhost:9090 -L 3000:localhost:3000 defiant-compute
```

浏览器访问：
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## 首次使用

如果容器未构建，在登录节点运行：

```bash
bash build_containers.sh
```

## 故障排除

### 问题：删除 Prometheus 数据后，Grafana 仍显示旧数据

**原因：**
1. Prometheus 服务仍在运行，使用旧的内存缓存
2. Grafana 缓存了旧的查询结果
3. Prometheus 没有正确连接到 vLLM，无法获取新数据
4. 数据删除不彻底，仍有残留文件

**解决方法：**

运行修复脚本：

```bash
cd /ccsopen/home/shouwei/projects/vllm-newest/vllm/z_tests/prometheus_monitoring
bash fix_prometheus_data.sh
```

该脚本会：
1. 停止所有运行中的 Prometheus 和 Grafana 进程
2. 彻底清理 Prometheus 数据目录 (`~/prometheus_data/data`)
3. 清理 Grafana 缓存（可选）
4. 验证 vLLM 连接
5. 验证并修复 Prometheus 配置
6. 提供重启服务的选项

**手动步骤：**

如果脚本无法解决问题，可以手动执行：

```bash
# 1. 停止服务
kill $(cat prometheus.pid grafana.pid 2>/dev/null) 2>/dev/null || true

# 2. 清理数据
rm -rf ~/prometheus_data/data/*
rm -f ~/prometheus_data/data/lock

# 3. 清理 Grafana 缓存（可选）
rm -rf ~/grafana_data/data/sessions ~/grafana_data/data/png

# 4. 重启服务
bash startup.sh
```

**验证新数据：**

1. 在 Prometheus UI (http://localhost:9090) 中查询：
   ```
   up{job="vllm"}
   ```
   应该返回 `1`（表示 vLLM 可访问）

2. 在 Grafana 中：
   - 刷新页面（Ctrl+F5 强制刷新）
   - 检查时间范围设置为 "Last 5 minutes" 或 "now"
   - 检查数据源连接状态

3. 检查日志：
   ```bash
   tail -f prometheus.log
   tail -f grafana.log
   ```

### 其他常见问题

**Prometheus 无法连接 vLLM：**
- 确认 vLLM 正在运行：`curl http://localhost:8000/health`
- 确认 vLLM 启用了 metrics：检查启动参数中是否有 `--enable-metrics`
- 检查 Prometheus 配置中的 vLLM 地址和端口是否正确

**Grafana 显示 "No data"：**
- 检查 Prometheus 数据源配置
- 确认 Prometheus 正在运行：`curl http://localhost:9090/-/healthy`
- 检查时间范围设置
