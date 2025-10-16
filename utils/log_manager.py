"""
日志管理工具 - 自动压缩和清理日志文件
"""
import os
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LogManager:
    """日志文件管理 - 压缩和清理"""

    @staticmethod
    def compress_old_logs(log_dir: Path, days: int = 7) -> int:
        """
        压缩旧日志文件

        Args:
            log_dir: 日志目录
            days: 压缩多少天前的日志

        Returns:
            压缩的文件数量
        """
        if not log_dir.exists():
            logger.warning(f"日志目录不存在: {log_dir}")
            return 0

        cutoff_time = datetime.now() - timedelta(days=days)
        compressed_count = 0

        # 只处理 .log 和 .jsonl 文件，跳过已压缩的
        for log_file in log_dir.glob("*.log*"):
            # 跳过已压缩文件
            if log_file.suffix == '.gz':
                continue

            # 检查文件修改时间
            if log_file.stat().st_mtime < cutoff_time.timestamp():
                try:
                    # 压缩文件
                    gz_path = Path(f'{log_file}.gz')
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(gz_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # 删除原文件
                    log_file.unlink()
                    compressed_count += 1
                    logger.info(f"已压缩日志文件: {log_file.name} -> {gz_path.name}")

                except Exception as e:
                    logger.error(f"压缩日志文件失败 {log_file}: {e}")

        return compressed_count

    @staticmethod
    def clean_old_logs(log_dir: Path, days: int = 30) -> int:
        """
        删除过期的压缩日志

        Args:
            log_dir: 日志目录
            days: 删除多少天前的压缩日志

        Returns:
            删除的文件数量
        """
        if not log_dir.exists():
            logger.warning(f"日志目录不存在: {log_dir}")
            return 0

        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0

        # 只删除 .gz 压缩文件
        for log_file in log_dir.glob("*.gz"):
            if log_file.stat().st_mtime < cutoff_time.timestamp():
                try:
                    log_file.unlink()
                    deleted_count += 1
                    logger.info(f"已删除过期日志: {log_file.name}")

                except Exception as e:
                    logger.error(f"删除日志文件失败 {log_file}: {e}")

        return deleted_count

    @staticmethod
    def get_total_size(log_dir: Path) -> int:
        """
        获取日志目录总大小（字节）

        Args:
            log_dir: 日志目录

        Returns:
            总大小（字节）
        """
        if not log_dir.exists():
            return 0

        total_size = 0
        for log_file in log_dir.glob("**/*"):
            if log_file.is_file():
                total_size += log_file.stat().st_size

        return total_size

    @staticmethod
    def cleanup_logs(log_dir: Path, compress_days: int = 7, delete_days: int = 30):
        """
        执行日志清理（压缩 + 删除）

        Args:
            log_dir: 日志目录
            compress_days: 压缩多少天前的日志
            delete_days: 删除多少天前的日志
        """
        logger.info(f"开始日志清理任务 - 目录: {log_dir}")

        # 获取清理前的大小
        size_before = LogManager.get_total_size(log_dir)

        # 压缩旧日志
        compressed = LogManager.compress_old_logs(log_dir, compress_days)
        logger.info(f"已压缩 {compressed} 个日志文件")

        # 删除过期日志
        deleted = LogManager.clean_old_logs(log_dir, delete_days)
        logger.info(f"已删除 {deleted} 个过期日志")

        # 获取清理后的大小
        size_after = LogManager.get_total_size(log_dir)
        saved_mb = (size_before - size_after) / (1024 * 1024)

        logger.info(f"日志清理完成 - 释放空间: {saved_mb:.2f} MB")


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    log_dir = Path(__file__).parent.parent / "logs"
    LogManager.cleanup_logs(log_dir, compress_days=7, delete_days=30)
