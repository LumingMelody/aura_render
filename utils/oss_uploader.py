"""
阿里云OSS图片上传工具
"""
import os
import oss2
from pathlib import Path
from typing import Optional
import uuid
from datetime import datetime


class OSSUploader:
    """阿里云OSS上传器"""

    def __init__(self):
        """从环境变量初始化OSS配置"""
        self.access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.endpoint = os.getenv("OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com")
        self.bucket_name = os.getenv("OSS_BUCKET_NAME")
        self.domain = os.getenv("OSS_DOMAIN")

        if not all([self.access_key_id, self.access_key_secret, self.bucket_name]):
            raise ValueError("OSS配置不完整，请检查环境变量: OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET_NAME")

        # 初始化OSS认证和Bucket
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

        # 如果没有设置自定义域名，使用默认的OSS域名
        if not self.domain:
            self.domain = f"https://{self.bucket_name}.{self.endpoint.replace('https://', '')}"

    def upload_image(self, local_path: str, remote_path: Optional[str] = None) -> str:
        """
        上传图片到OSS

        参数:
            local_path: 本地图片路径
            remote_path: OSS远程路径（可选），如果不指定则自动生成

        返回:
            图片的公网URL
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {local_path}")

        # 如果没有指定远程路径，自动生成一个
        if not remote_path:
            # 生成格式: keyframes/2025-09-30/uuid_filename.jpg
            date_folder = datetime.now().strftime("%Y-%m-%d")
            unique_id = str(uuid.uuid4())[:8]
            filename = local_path.name
            remote_path = f"keyframes/{date_folder}/{unique_id}_{filename}"

        # 上传文件
        with open(local_path, 'rb') as f:
            result = self.bucket.put_object(remote_path, f)

        # 检查上传是否成功
        if result.status != 200:
            raise Exception(f"OSS上传失败，状态码: {result.status}")

        # 返回完整的公网URL
        url = f"{self.domain}/{remote_path}"
        return url

    def upload_video(self, local_path: str, remote_path: Optional[str] = None) -> str:
        """
        上传视频到OSS

        参数:
            local_path: 本地视频路径
            remote_path: OSS远程路径（可选），如果不指定则自动生成

        返回:
            视频的公网URL
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {local_path}")

        # 如果没有指定远程路径，自动生成一个
        if not remote_path:
            # 生成格式: video_clips/2025-09-30/uuid_filename.mp4
            date_folder = datetime.now().strftime("%Y-%m-%d")
            unique_id = str(uuid.uuid4())[:8]
            filename = local_path.name
            remote_path = f"video_clips/{date_folder}/{unique_id}_{filename}"

        # 上传文件
        with open(local_path, 'rb') as f:
            result = self.bucket.put_object(remote_path, f)

        # 检查上传是否成功
        if result.status != 200:
            raise Exception(f"OSS上传失败，状态码: {result.status}")

        # 返回完整的公网URL
        url = f"{self.domain}/{remote_path}"
        return url

    def upload_images_batch(self, local_paths: list[str]) -> list[str]:
        """
        批量上传图片

        参数:
            local_paths: 本地图片路径列表

        返回:
            图片公网URL列表
        """
        urls = []
        for local_path in local_paths:
            try:
                url = self.upload_image(local_path)
                urls.append(url)
                print(f"✅ 上传成功: {Path(local_path).name} -> {url}")
            except Exception as e:
                print(f"❌ 上传失败: {local_path}, 错误: {e}")
                urls.append(None)
        return urls

    def delete_image(self, remote_path: str) -> bool:
        """
        删除OSS上的图片

        参数:
            remote_path: OSS远程路径

        返回:
            是否删除成功
        """
        try:
            self.bucket.delete_object(remote_path)
            return True
        except Exception as e:
            print(f"❌ 删除失败: {remote_path}, 错误: {e}")
            return False


# 全局单例
_uploader_instance = None


def get_oss_uploader() -> OSSUploader:
    """获取OSS上传器单例"""
    global _uploader_instance
    if _uploader_instance is None:
        _uploader_instance = OSSUploader()
    return _uploader_instance


# 示例用法
if __name__ == "__main__":
    # 测试上传
    uploader = get_oss_uploader()

    # 上传单个文件
    # url = uploader.upload_image("/path/to/local/image.jpg")
    # print(f"图片URL: {url}")

    # 批量上传
    # urls = uploader.upload_images_batch([
    #     "/path/to/image1.jpg",
    #     "/path/to/image2.jpg"
    # ])
    # print(f"上传完成，共 {len(urls)} 个文件")