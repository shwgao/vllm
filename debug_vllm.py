#!/usr/bin/env python3
"""
Debug script for vLLM serving
This script bypasses the debugger compatibility issues by running vLLM directly
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Set up environment
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': str(Path(__file__).parent),
        'CUDA_VISIBLE_DEVICES': '2,3',
        'PYTHONNOUSERSITE': '1',
        'PYTHONBREAKPOINT': '0'  # Disable breakpoint debugging
    })
    
    # vLLM command arguments
    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.cli.main',
        'serve',
        './models/gradientai-Llama-3-8B-Instruct-Gradient-1048k',
        '--disable-log-requests',
        '--tensor-parallel-size', '2',
        '--max-model-len', '65000',
        '--enforce-eager'
    ]
    
    print("Starting vLLM server with debug-friendly configuration...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment: {env}")
    
    try:
        # Run vLLM server
        subprocess.run(cmd, env=env, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error running vLLM server: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
