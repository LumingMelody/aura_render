#!/usr/bin/env python3
"""
测试日志系统
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置日志
import logging
from utils.logger import setup_logging, get_logger, LogCategory

log_dir = Path(__file__).parent / "logs"
setup_logging(
    log_dir=log_dir,
    log_level=logging.INFO,
    enable_console=True,
    enable_json=False,
    enable_performance=False,
    max_file_size=100 * 1024 * 1024
)

# 测试 app.py 的 logger
logger_app = get_logger("aura_render.app").with_context(category=LogCategory.SYSTEM)

# 测试 coze 模块的 logger
logger_coze = get_logger("coze.image_search").with_context(category=LogCategory.SYSTEM)

print("\n" + "="*80)
print("测试日志系统")
print("="*80 + "\n")

# 测试不同级别的日志
logger_app.info("✅ 这是来自 app 的 INFO 日志")
logger_app.warning("⚠️ 这是来自 app 的 WARNING 日志")
logger_app.error("❌ 这是来自 app 的 ERROR 日志")

print()

logger_coze.info("🔍 这是来自 coze 模块的 INFO 日志")
logger_coze.info("✅ Coze 搜索到图片: https://example.com/image.jpg")
logger_coze.warning("⚠️ 这是来自 coze 模块的 WARNING 日志")

print("\n" + "="*80)
print("测试完成，请检查：")
print(f"1. 控制台输出（上方）")
print(f"2. 日志文件: {log_dir}/aura_render.log")
print("="*80 + "\n")
