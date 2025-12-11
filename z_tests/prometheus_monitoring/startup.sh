#!/bin/bash
# 快速启动 Prometheus 和 Grafana - 适配你的配置
# 在 defiant-compute 节点上运行此脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "启动 Prometheus 和 Grafana 服务"
echo "=========================================="
echo ""

# 检查是否在计算节点上
if [[ $(hostname) != *"defiant"* ]]; then
    echo "警告: 看起来你不在 Defiant 计算节点上"
    echo "当前主机名: $(hostname)"
    read -p "是否继续？(y/n) [y]: " CONTINUE
    CONTINUE=${CONTINUE:-y}
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

# 配置
VLLM_HOST=${VLLM_HOST:-localhost}
VLLM_PORT=${VLLM_PORT:-8007}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-9090}
GRAFANA_PORT=${GRAFANA_PORT:-3000}

echo "配置信息:"
echo "  vLLM 地址: ${VLLM_HOST}:${VLLM_PORT}"
echo "  Prometheus 端口: ${PROMETHEUS_PORT}"
echo "  Grafana 端口: ${GRAFANA_PORT}"
echo ""

# 检查容器是否存在
if [ ! -f "$SCRIPT_DIR/prometheus.sif" ]; then
    echo "错误: prometheus.sif 不存在"
    echo "正在构建容器..."
    bash "$SCRIPT_DIR/build_containers.sh"
    if [ $? -ne 0 ]; then
        echo "容器构建失败，请检查错误信息"
        exit 1
    fi
fi

if [ ! -f "$SCRIPT_DIR/grafana.sif" ]; then
    echo "错误: grafana.sif 不存在"
    echo "正在构建容器..."
    bash "$SCRIPT_DIR/build_containers.sh"
    if [ $? -ne 0 ]; then
        echo "容器构建失败，请检查错误信息"
        exit 1
    fi
fi

echo "✓ 容器镜像已就绪"
echo ""

# 创建目录
PROMETHEUS_DIR="$HOME/prometheus_data"
GRAFANA_DIR="$HOME/grafana_data"

mkdir -p "$PROMETHEUS_DIR/data" "$PROMETHEUS_DIR/config"
mkdir -p "$GRAFANA_DIR/data" "$GRAFANA_DIR/logs"

echo "✓ 目录已创建"
echo ""

# 清理残留的进程和锁文件
echo "检查并清理残留进程..."
if [ -f "$SCRIPT_DIR/prometheus.pid" ]; then
    OLD_PID=$(cat "$SCRIPT_DIR/prometheus.pid" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ! ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "  清理残留的 Prometheus PID 文件"
        rm -f "$SCRIPT_DIR/prometheus.pid"
    elif [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "  警告: Prometheus 进程 (PID: $OLD_PID) 仍在运行"
        read -p "  是否停止它？(y/n) [y]: " KILL_OLD
        KILL_OLD=${KILL_OLD:-y}
        if [ "$KILL_OLD" = "y" ]; then
            kill "$OLD_PID" 2>/dev/null || true
            sleep 2
            rm -f "$SCRIPT_DIR/prometheus.pid"
        fi
    fi
fi

if [ -f "$SCRIPT_DIR/grafana.pid" ]; then
    OLD_PID=$(cat "$SCRIPT_DIR/grafana.pid" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ! ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "  清理残留的 Grafana PID 文件"
        rm -f "$SCRIPT_DIR/grafana.pid"
    elif [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "  警告: Grafana 进程 (PID: $OLD_PID) 仍在运行"
        read -p "  是否停止它？(y/n) [y]: " KILL_OLD
        KILL_OLD=${KILL_OLD:-y}
        if [ "$KILL_OLD" = "y" ]; then
            kill "$OLD_PID" 2>/dev/null || true
            sleep 2
            rm -f "$SCRIPT_DIR/grafana.pid"
        fi
    fi
fi

# 清理 Prometheus 残留锁文件
LOCK_FILE="$PROMETHEUS_DIR/data/lock"
if [ -f "$LOCK_FILE" ]; then
    # 检查是否有进程持有锁文件
    if ! lsof "$LOCK_FILE" > /dev/null 2>&1; then
        echo "  清理 Prometheus 残留锁文件"
        rm -f "$LOCK_FILE"
    else
        echo "  警告: Prometheus 锁文件被进程占用，请手动检查"
    fi
fi

echo ""

# 创建 Prometheus 配置（使用 .yml 扩展名，因为 Prometheus 期望这个）
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

echo "✓ Prometheus 配置已创建"
echo ""

# 检查 vLLM 是否可访问
echo "检查 vLLM 连接..."
if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "✓ vLLM 可访问"
else
    echo "⚠ 警告: 无法连接到 vLLM，请确认它正在运行"
    echo "  尝试: curl http://${VLLM_HOST}:${VLLM_PORT}/health"
    read -p "是否继续？(y/n) [y]: " CONTINUE
    CONTINUE=${CONTINUE:-y}
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

echo ""

# 启动 Prometheus
echo "启动 Prometheus..."
# 注意：移除 --net 选项，因为计算节点上普通用户无法使用网络桥接
# 容器将使用主机网络命名空间
apptainer exec \
    --bind "$PROMETHEUS_DIR/config:/etc/prometheus" \
    --bind "$PROMETHEUS_DIR/data:/prometheus" \
    "$SCRIPT_DIR/prometheus.sif" \
    prometheus \
        --config.file=/etc/prometheus/prometheus.yml \
        --storage.tsdb.path=/prometheus \
        --web.enable-lifecycle \
        --web.listen-address=0.0.0.0:9090 > prometheus.log 2>&1 &

PROMETHEUS_PID=$!
echo "Prometheus 已启动 (PID: $PROMETHEUS_PID)"
echo $PROMETHEUS_PID > prometheus.pid

# 等待并检查 Prometheus 是否成功启动
sleep 3
if ! ps -p "$PROMETHEUS_PID" > /dev/null 2>&1; then
    echo "✗ 错误: Prometheus 进程已退出，查看 prometheus.log 了解详情"
    tail -20 prometheus.log
    exit 1
fi

# 启动 Grafana
echo "启动 Grafana..."
export GF_PATHS_DATA=/var/lib/grafana
export GF_PATHS_LOGS=/var/log/grafana
export GF_SERVER_HTTP_PORT=3000
export GF_SERVER_HTTP_ADDRESS=0.0.0.0

# 注意：移除 --net 选项，因为计算节点上普通用户无法使用网络桥接
# 容器将使用主机网络命名空间
apptainer exec \
    --bind "$GRAFANA_DIR/data:/var/lib/grafana" \
    --bind "$GRAFANA_DIR/logs:/var/log/grafana" \
    "$SCRIPT_DIR/grafana.sif" \
    /run.sh > grafana.log 2>&1 &

GRAFANA_PID=$!
echo "Grafana 已启动 (PID: $GRAFANA_PID)"
echo $GRAFANA_PID > grafana.pid

# 等待并检查 Grafana 是否成功启动
sleep 3
if ! ps -p "$GRAFANA_PID" > /dev/null 2>&1; then
    echo "✗ 错误: Grafana 进程已退出，查看 grafana.log 了解详情"
    tail -20 grafana.log
    exit 1
fi

echo ""
echo "=========================================="
echo "服务启动完成！"
echo "=========================================="
echo ""

NODE_HOSTNAME=$(hostname)
echo "节点: $NODE_HOSTNAME"
echo ""
echo "服务地址:"
echo "  Prometheus: http://${NODE_HOSTNAME}:${PROMETHEUS_PORT}"
echo "  Grafana:    http://${NODE_HOSTNAME}:${GRAFANA_PORT}"
echo ""

echo "从本地机器访问，创建 SSH 隧道:"
echo "  ssh -L ${PROMETHEUS_PORT}:localhost:${PROMETHEUS_PORT} \\"
echo "      -L ${GRAFANA_PORT}:localhost:${GRAFANA_PORT} \\"
echo "      defiant-compute"
echo ""

echo "然后在浏览器打开:"
echo "  Prometheus: http://localhost:${PROMETHEUS_PORT}"
echo "  Grafana:    http://localhost:${GRAFANA_PORT} (admin/admin)"
echo ""

echo "查看日志:"
echo "  tail -f prometheus.log"
echo "  tail -f grafana.log"
echo ""

echo "停止服务:"
echo "  kill \$(cat prometheus.pid) \$(cat grafana.pid)"
echo ""

echo "等待服务初始化..."
sleep 5

# 检查服务状态
echo ""
echo "检查服务状态..."
if curl -s http://localhost:${PROMETHEUS_PORT}/-/healthy > /dev/null 2>&1; then
    echo "✓ Prometheus 运行正常"
else
    echo "✗ Prometheus 健康检查失败，查看 prometheus.log"
fi

if curl -s http://localhost:${GRAFANA_PORT}/api/health > /dev/null 2>&1; then
    echo "✓ Grafana 运行正常"
else
    echo "✗ Grafana 健康检查失败，查看 grafana.log"
fi

echo ""
echo "=========================================="
echo "完成！服务在后台运行"
echo "=========================================="

