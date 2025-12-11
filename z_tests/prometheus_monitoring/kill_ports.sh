#!/bin/bash
# 杀死占用 9090 和 3000 端口的进程

echo "查找占用 9090 和 3000 端口的进程..."

# 查找占用端口的进程
PIDS_9090=$(ss -tlnp 2>/dev/null | grep ':9090' | grep -oP 'pid=\K\d+' | sort -u)
PIDS_3000=$(ss -tlnp 2>/dev/null | grep ':3000' | grep -oP 'pid=\K\d+' | sort -u)

echo "占用 9090 端口的进程: ${PIDS_9090:-无}"
echo "占用 3000 端口的进程: ${PIDS_3000:-无}"

# 查找所有 prometheus 和 grafana 进程
PROMETHEUS_PIDS=$(pgrep -f prometheus 2>/dev/null | sort -u)
GRAFANA_PIDS=$(pgrep -f grafana 2>/dev/null | sort -u)

echo ""
echo "所有 Prometheus 进程: ${PROMETHEUS_PIDS:-无}"
echo "所有 Grafana 进程: ${GRAFANA_PIDS:-无}"

# 合并所有需要杀死的进程
ALL_PIDS=$(echo "$PIDS_9090 $PIDS_3000 $PROMETHEUS_PIDS $GRAFANA_PIDS" | tr ' ' '\n' | sort -u | tr '\n' ' ')

if [ -z "$ALL_PIDS" ] || [ "$ALL_PIDS" = " " ]; then
    echo ""
    echo "没有找到需要杀死的进程"
    exit 0
fi

echo ""
echo "准备杀死以下进程: $ALL_PIDS"
read -p "确认杀死这些进程？(y/n) [y]: " CONFIRM
CONFIRM=${CONFIRM:-y}

if [ "$CONFIRM" = "y" ]; then
    for pid in $ALL_PIDS; do
        if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
            echo "杀死进程 $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 2
    
    echo ""
    echo "验证端口是否已释放..."
    REMAINING_9090=$(ss -tlnp 2>/dev/null | grep ':9090' | grep LISTEN || true)
    REMAINING_3000=$(ss -tlnp 2>/dev/null | grep ':3000' | grep LISTEN || true)
    
    if [ -z "$REMAINING_9090" ] && [ -z "$REMAINING_3000" ]; then
        echo "✓ 端口 9090 和 3000 已释放"
    else
        echo "⚠ 仍有进程占用端口:"
        [ -n "$REMAINING_9090" ] && echo "  9090: $REMAINING_9090"
        [ -n "$REMAINING_3000" ] && echo "  3000: $REMAINING_3000"
    fi
    
    # 再次检查进程
    REMAINING_PROM=$(pgrep -f prometheus 2>/dev/null || true)
    REMAINING_GRAF=$(pgrep -f grafana 2>/dev/null || true)
    
    if [ -n "$REMAINING_PROM" ] || [ -n "$REMAINING_GRAF" ]; then
        echo ""
        echo "⚠ 仍有残留进程:"
        [ -n "$REMAINING_PROM" ] && echo "  Prometheus: $REMAINING_PROM"
        [ -n "$REMAINING_GRAF" ] && echo "  Grafana: $REMAINING_GRAF"
        echo "  可能需要检查是否有 supervisor 或 systemd 服务自动重启了这些进程"
    fi
else
    echo "操作已取消"
fi

