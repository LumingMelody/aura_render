#!/bin/bash
# 生产环境启动脚本 - 不会重复加载

PORT=${PORT:-8000}

echo "🚀 启动Aura Render生产环境..."
echo "📍 端口: $PORT"

# 不使用 --reload，避免重复加载
python3 -m uvicorn app:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 1 \
  --log-level info

# 如果需要多worker并发处理：
# --workers 4
