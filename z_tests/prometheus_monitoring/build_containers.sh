#!/bin/bash
# Script to build Prometheus and Grafana Apptainer containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}构建 Prometheus 和 Grafana 容器${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 显示当前节点信息
HOSTNAME=$(hostname)
echo -e "${YELLOW}当前节点: ${HOSTNAME}${NC}"

# 检查是否在计算节点
if [[ "$HOSTNAME" == *"defiant-nv"* ]] || [[ "$HOSTNAME" == *"compute"* ]]; then
    echo -e "${RED}⚠️  警告：你当前在计算节点上${NC}"
    echo ""
    echo "计算节点通常无法访问外部网络，无法构建容器。"
    echo "请切换到登录节点运行此脚本："
    echo "  ssh defiant"
    echo "  cd $SCRIPT_DIR"
    echo "  bash build_containers.sh"
    exit 1
fi

echo ""

# 检查网络连接
echo -e "${YELLOW}检查网络连接...${NC}"
if timeout 5 curl -s -I https://registry-1.docker.io/v2/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 可以访问 Docker Hub${NC}"
else
    echo -e "${YELLOW}⚠️  无法访问 Docker Hub（可能正常，继续尝试构建）${NC}"
fi

echo ""

# 检查定义文件
if [ ! -f "prometheus.def" ]; then
    echo -e "${RED}✗ prometheus.def 不存在${NC}"
    exit 1
fi
if [ ! -f "grafana.def" ]; then
    echo -e "${RED}✗ grafana.def 不存在${NC}"
    exit 1
fi

# 构建 Prometheus
echo -e "${YELLOW}[1/2] 构建 Prometheus 容器...${NC}"
if [ -f "prometheus.sif" ]; then
    echo -e "${YELLOW}⚠️  prometheus.sif 已存在${NC}"
    read -p "是否重新构建？(y/n) [n]: " REBUILD
    REBUILD=${REBUILD:-n}
    if [ "$REBUILD" = "y" ]; then
        rm -f prometheus.sif
    else
        echo -e "${GREEN}跳过 Prometheus 构建（使用现有镜像）${NC}"
        SKIP_PROMETHEUS=true
    fi
fi

if [ "$SKIP_PROMETHEUS" != "true" ]; then
    echo "这可能需要几分钟时间，请耐心等待..."
    echo ""
    
    if apptainer build prometheus.sif prometheus.def; then
        SIZE=$(du -h prometheus.sif 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓ Prometheus 容器构建成功 (大小: ${SIZE})${NC}"
    else
        echo -e "${RED}✗ Prometheus 容器构建失败${NC}"
        echo ""
        echo "可能的原因："
        echo "1. 网络连接问题（无法访问 Docker Hub）"
        echo "2. 磁盘空间不足"
        echo "3. 权限问题"
        exit 1
    fi
fi

echo ""

# 构建 Grafana
echo -e "${YELLOW}[2/2] 构建 Grafana 容器...${NC}"
if [ -f "grafana.sif" ]; then
    echo -e "${YELLOW}⚠️  grafana.sif 已存在${NC}"
    read -p "是否重新构建？(y/n) [n]: " REBUILD
    REBUILD=${REBUILD:-n}
    if [ "$REBUILD" = "y" ]; then
        rm -f grafana.sif
    else
        echo -e "${GREEN}跳过 Grafana 构建（使用现有镜像）${NC}"
        SKIP_GRAFANA=true
    fi
fi

if [ "$SKIP_GRAFANA" != "true" ]; then
    echo "这可能需要几分钟时间，请耐心等待..."
    echo ""
    
    if apptainer build grafana.sif grafana.def; then
        SIZE=$(du -h grafana.sif 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓ Grafana 容器构建成功 (大小: ${SIZE})${NC}"
    else
        echo -e "${RED}✗ Grafana 容器构建失败${NC}"
        echo ""
        echo "可能的原因："
        echo "1. 网络连接问题（无法访问 Docker Hub）"
        echo "2. 磁盘空间不足"
        echo "3. 权限问题"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}所有容器构建成功！${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 显示文件信息
echo "容器文件："
ls -lh *.sif 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo -e "${YELLOW}下一步：${NC}"
echo "1. 容器文件已保存在: $SCRIPT_DIR"
echo "2. 可以在计算节点上使用这些容器"
echo "3. 运行: bash 快速启动步骤.sh"
echo ""

