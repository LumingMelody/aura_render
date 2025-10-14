# file_manager.py

import os
import hashlib
import requests
import time
from typing import Optional, Union
from pathlib import Path

class DownloadManager:
    def __init__(self, cache_dir: str = "./cache", user_agent: str = None):
        """
        初始化下载管理器
        
        :param cache_dir: 缓存目录
        :param user_agent: 自定义 User-Agent
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.user_agent = user_agent or "DownloadManager/1.0"

    def _url_to_hash(self, url: str) -> str:
        """将 URL 转为 SHA256 哈希作为文件名"""
        return hashlib.sha256(url.encode()).hexdigest()

    def _get_cache_path(self, url: str) -> Path:
        """获取该 URL 对应的缓存文件路径"""
        filename = self._url_to_hash(url)
        return self.cache_dir / filename

    def _is_file_expired(self, filepath: Path, max_age_seconds: int) -> bool:
        """判断文件是否过期"""
        if not filepath.exists():
            return True
        file_mtime = filepath.stat().st_mtime
        return (time.time() - file_mtime) > max_age_seconds

    def download_or_load(
        self,
        url: str,
        max_age_days: int = 7,
        timeout: int = 30,
        return_path: bool = False
    ) -> Optional[Union[bytes, str, Path]]:
        """
        下载或加载缓存文件
        
        :param url: 下载链接
        :param max_age_days: 缓存最大有效天数（默认7天）
        :param timeout: 下载超时时间
        :param return_path: 是否返回文件路径（True），否则返回文件内容（bytes）
        :return: 文件内容（bytes）或文件路径（Path），失败返回 None
        """
        cache_path = self._get_cache_path(url)
        max_age_seconds = max_age_days * 24 * 3600

        # 检查缓存是否存在且未过期
        if cache_path.exists() and not self._is_file_expired(cache_path, max_age_seconds):
            print(f"[缓存命中] {url}")
            if return_path:
                return cache_path
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                print(f"[错误] 读取缓存失败: {e}")
                return None
        else:
            print(f"[缓存未命中] 正在下载: {url}")
            try:
                headers = {'User-Agent': self.user_agent}
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()

                # 写入缓存
                cache_path.write_bytes(response.content)

                if return_path:
                    return cache_path
                return response.content

            except requests.RequestException as e:
                print(f"[下载失败] {url}: {e}")
                return None

    def clear_cache(self, expired_only: bool = True, max_age_days: int = 7):
        """
        清理缓存
        
        :param expired_only: 是否只清理过期文件
        :param max_age_days: 判断过期的时间阈值（仅当 expired_only=True 时有效）
        """
        max_age_seconds = max_age_days * 24 * 3600
        removed_count = 0
        total_count = 0

        for cache_file in self.cache_dir.iterdir():
            if cache_file.is_file():
                total_count += 1
                should_remove = False
                if expired_only:
                    if self._is_file_expired(cache_file, max_age_seconds):
                        should_remove = True
                else:
                    should_remove = True

                if should_remove:
                    try:
                        cache_file.unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"[清理失败] {cache_file}: {e}")

        print(f"缓存清理完成: 共 {total_count} 个文件，删除 {removed_count} 个")

    def cache_info(self) -> dict:
        """
        获取缓存信息
        """
        files = list(self.cache_dir.iterdir())
        file_count = len([f for f in files if f.is_file()])
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        return {
            "cache_dir": str(self.cache_dir),
            "file_count": file_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }


# ======================
# 使用示例
# ======================

if __name__ == "__main__":
    dm = DownloadManager(cache_dir="./my_cache")

    # 示例 URL（可替换为真实链接）
    url = "https://httpbin.org/uuid"  # 每次返回不同 UUID，便于测试

    # 第一次：下载
    content1 = dm.download_or_load(url)
    print("内容:", content1)

    # 第二次：从缓存读取
    content2 = dm.download_or_load(url)
    print("内容（缓存）:", content2)

    # 查看缓存信息
    print("缓存信息:", dm.cache_info())

    # 清理过期缓存（超过7天的）
    dm.clear_cache(expired_only=True, max_age_days=7)

    # 强制清理所有缓存
    # dm.clear_cache(expired_only=False)