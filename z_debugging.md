# vLLM Benchmark 代理问题调试指南

## 问题描述
在使用 `vllm bench serve` 命令时，服务一直显示 "Waiting for endpoint to become up" 并无限等待，无法正常进行基准测试。

## 问题原因
系统环境设置了代理（`http_proxy`, `https_proxy`, `ftp_proxy`），导致 vLLM benchmark 工具对 localhost 的请求也被代理服务器处理，造成连接失败。

## 解决方案（不修改源码）

### 方法1：使用 no_proxy 环境变量（推荐）

```bash
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
vllm bench serve --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --host 0.0.0.0 --port 8000 --random-input-len 32 --random-output-len 4 --num-prompts 5
```

**优点：**
- ✅ 不需要修改源码
- ✅ 保持代理设置，其他网络请求仍使用代理
- ✅ 只对 localhost 禁用代理，精确解决问题
- ✅ 简单易用

### 方法2：创建包装脚本

创建 `run_benchmark.sh`：
```bash
#!/bin/bash
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
cd /ccsopen/home/shouwei/projects/vllm-newest/vllm
source .venv/bin/activate
vllm bench serve "$@"
```

使用方法：
```bash
chmod +x run_benchmark.sh
./run_benchmark.sh --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --host 0.0.0.0 --port 8000 --random-input-len 32 --random-output-len 4 --num-prompts 5
```

### 方法3：使用 alias

在 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
alias vllm-bench='no_proxy="localhost,127.0.0.1,0.0.0.0" NO_PROXY="localhost,127.0.0.1,0.0.0.0" vllm bench'
```

使用方法：
```bash
vllm-bench serve --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --host 0.0.0.0 --port 8000 --random-input-len 32 --random-output-len 4 --num-prompts 5
```

### 方法4：使用环境变量文件

创建 `.env` 文件：
```bash
no_proxy=localhost,127.0.0.1,0.0.0.0
NO_PROXY=localhost,127.0.0.1,0.0.0.0
```

使用方法：
```bash
source .env && vllm bench serve --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --host 0.0.0.0 --port 8000 --random-input-len 32 --random-output-len 4 --num-prompts 5
```

## 验证解决方案

成功运行后应该看到类似输出：
```
============ Serving Benchmark Result ============
Successful requests:                     5         
Failed requests:                         0         
Request throughput (req/s):              12.65     
Output token throughput (tok/s):         50.60     
Mean TTFT (ms):                          217.50    
Mean TPOT (ms):                          58.09     
==================================================
```

## 推荐使用方法1

**方法1（no_proxy 环境变量）**是最优雅的解决方案，因为：
- 不需要修改源码
- 保持代理设置
- 只对 localhost 禁用代理
- 简单易用
- 兼容性好

## 注意事项

1. 确保 vLLM API 服务器正在运行（端口 8000）
2. 如果使用不同的主机或端口，需要相应调整 `no_proxy` 设置
3. 某些网络环境可能需要额外的代理配置
