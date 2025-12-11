#!/bin/bash
# 修复 Prometheus 数据问题脚本
# 用于解决删除数据后 Grafana 仍显示旧数据的问题

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "修复 Prometheus 数据问题"
echo "=========================================="
echo ""

PROMETHEUS_DIR="$HOME/prometheus_data"
GRAFANA_DIR="$HOME/grafana_data"

# 1. 停止所有运行中的 Prometheus 和 Grafana 进程
echo "步骤 1: 停止运行中的服务..."
if [ -f "$SCRIPT_DIR/prometheus.pid" ]; then
    OLD_PID=$(cat "$SCRIPT_DIR/prometheus.pid" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "  停止 Prometheus (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$SCRIPT_DIR/prometheus.pid"
fi

if [ -f "$SCRIPT_DIR/grafana.pid" ]; then
    OLD_PID=$(cat "$SCRIPT_DIR/grafana.pid" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "  停止 Grafana (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$SCRIPT_DIR/grafana.pid"
fi

# 检查是否有其他 Prometheus/Grafana 进程
PROMETHEUS_PIDS=$(pgrep -f "prometheus" || true)
if [ -n "$PROMETHEUS_PIDS" ]; then
    echo "  发现其他 Prometheus 进程: $PROMETHEUS_PIDS"
    echo "  请手动停止这些进程"
fi

GRAFANA_PIDS=$(pgrep -f "grafana" || true)
if [ -n "$GRAFANA_PIDS" ]; then
    echo "  发现其他 Grafana 进程: $GRAFANA_PIDS"
    echo "  请手动停止这些进程"
fi

echo "✓ 服务已停止"
echo ""

# 2. 彻底清理 Prometheus 数据目录
echo "步骤 2: 清理 Prometheus 数据..."
if [ -d "$PROMETHEUS_DIR/data" ]; then
    echo "  删除 $PROMETHEUS_DIR/data 下的所有数据..."
    rm -rf "$PROMETHEUS_DIR/data"/*
    rm -rf "$PROMETHEUS_DIR/data"/.[!.]* 2>/dev/null || true  # 删除隐藏文件
    echo "✓ Prometheus 数据已清理"
else
    echo "  Prometheus 数据目录不存在，创建新目录..."
    mkdir -p "$PROMETHEUS_DIR/data"
fi

# 清理锁文件
if [ -f "$PROMETHEUS_DIR/data/lock" ]; then
    echo "  删除残留锁文件..."
    rm -f "$PROMETHEUS_DIR/data/lock"
fi

echo ""

# 3. 清理 Grafana 缓存（可选，但建议）
echo "步骤 3: 清理 Grafana 缓存..."
read -p "是否清理 Grafana 缓存？这不会删除仪表板配置 (y/n) [y]: " CLEAN_GRAFANA
CLEAN_GRAFANA=${CLEAN_GRAFANA:-y}

if [ "$CLEAN_GRAFANA" = "y" ]; then
    if [ -d "$GRAFANA_DIR/data" ]; then
        # 只清理缓存，保留配置
        echo "  清理 Grafana 缓存目录..."
        # Grafana 的缓存通常在 sessions 和 png 目录
        rm -rf "$GRAFANA_DIR/data/sessions" 2>/dev/null || true
        rm -rf "$GRAFANA_DIR/data/png" 2>/dev/null || true
        # 清理临时文件
        find "$GRAFANA_DIR/data" -name "*.tmp" -delete 2>/dev/null || true
        echo "✓ Grafana 缓存已清理"
    fi
fi

echo ""

# 4. 验证 vLLM 连接
echo "步骤 4: 检查 vLLM 连接..."
VLLM_HOST=${VLLM_HOST:-localhost}
VLLM_PORT=${VLLM_PORT:-8000}

echo "  检查 http://${VLLM_HOST}:${VLLM_PORT}/metrics ..."
if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/metrics" > /dev/null 2>&1; then
    echo "✓ vLLM metrics 端点可访问"
    # 显示一些指标确认
    echo "  示例指标:"
    curl -s "http://${VLLM_HOST}:${VLLM_PORT}/metrics" | head -5
else
    echo "✗ 警告: 无法连接到 vLLM metrics 端点"
    echo "  请确认:"
    echo "    1. vLLM 服务正在运行"
    echo "    2. vLLM 启用了 metrics 端点 (--enable-metrics)"
    echo "    3. 地址和端口正确: ${VLLM_HOST}:${VLLM_PORT}"
    echo ""
    read -p "是否继续？(y/n) [y]: " CONTINUE
    CONTINUE=${CONTINUE:-y}
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

echo ""

# 5. 验证 Prometheus 配置
echo "步骤 5: 验证 Prometheus 配置..."
if [ ! -f "$PROMETHEUS_DIR/config/prometheus.yml" ]; then
    echo "  创建 Prometheus 配置文件..."
    mkdir -p "$PROMETHEUS_DIR/config"
    cat > "$PROMETHEUS_DIR/config/prometheus.yml" << EOF
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets:
          - '${VLLM_HOST}:${VLLM_PORT}'
        labels:
          instance: vllm-server
          job: vllm
EOF
    echo "✓ 配置文件已创建"
else
    echo "✓ 配置文件已存在"
    # 检查配置是否正确
    if grep -q "${VLLM_HOST}:${VLLM_PORT}" "$PROMETHEUS_DIR/config/prometheus.yml"; then
        echo "✓ 配置中的 vLLM 地址正确"
    else
        echo "⚠ 警告: 配置文件中的 vLLM 地址可能不正确"
        echo "  当前配置:"
        grep -A 2 "targets:" "$PROMETHEUS_DIR/config/prometheus.yml" || true
    fi
fi

echo ""

# 6. 提供重启服务的选项
echo "步骤 6: 准备重启服务..."
echo ""
echo "现在可以重启 Prometheus 和 Grafana 服务了。"
echo ""
read -p "是否现在启动服务？(y/n) [y]: " START_SERVICES
START_SERVICES=${START_SERVICES:-y}

if [ "$START_SERVICES" = "y" ]; then
    echo ""
    echo "启动服务..."
    bash "$SCRIPT_DIR/startup.sh"
else
    echo ""
    echo "手动启动服务，运行:"
    echo "  cd $SCRIPT_DIR"
    echo "  bash startup.sh"
fi

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "重要提示:"
echo "1. 如果 Grafana 仍显示旧数据，请:"
echo "   - 刷新浏览器页面 (Ctrl+F5 强制刷新)"
echo "   - 在 Grafana 中检查数据源连接"
echo "   - 确认时间范围设置为 'Last 5 minutes' 或 'now'"
echo ""
echo "2. 验证新数据:"
echo "   - 访问 Prometheus: http://localhost:9090"
echo "   - 查询: up{job=\"vllm\"}"
echo "   - 应该返回 1 (表示 vLLM 可访问)"
echo ""
echo "3. 如果问题仍然存在:"
echo "   - 检查 Prometheus 日志: tail -f $SCRIPT_DIR/prometheus.log"
echo "   - 检查 Grafana 日志: tail -f $SCRIPT_DIR/grafana.log"
echo "   - 确认 vLLM 正在生成新的 metrics"
echo ""
