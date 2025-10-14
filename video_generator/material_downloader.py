"""
素材下载器 - 从免费资源库下载视频、图片和音频素材
"""

import os
import json
import random
import requests
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)

# 素材缓存目录
CACHE_DIR = Path("/tmp/aura_render_materials")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class MaterialDownloader:
    """素材下载和管理器"""

    def __init__(self):
        """初始化下载器"""
        self.cache_dir = CACHE_DIR
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        # 免费资源库配置
        self.free_resources = {
            "videos": [
                # Pexels免费视频
                "https://www.pexels.com/video/digital-projection-of-abstract-geometrical-lines-3129671/download/",
                "https://www.pexels.com/video/close-up-video-of-codes-3129957/download/",
                # Pixabay免费视频（需要转换为直接链接）
            ],
            "images": [
                # Unsplash免费图片
                "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb",
                "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5",
                "https://images.unsplash.com/photo-1535223289827-42f1e9919769",
                "https://images.unsplash.com/photo-1550751827-4bd374c3f58b",
                "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b"
            ],
            "music": [
                # FreeMusicArchive等免费音乐资源
                # 暂时使用本地生成
            ]
        }

    def download_material(self, material_type: str, keywords: List[str] = None) -> Optional[str]:
        """
        下载素材

        Args:
            material_type: 素材类型 (video/image/music)
            keywords: 关键词（用于搜索相关素材）

        Returns:
            下载的文件路径
        """
        try:
            # 检查缓存
            cached_file = self._check_cache(material_type, keywords)
            if cached_file:
                logger.info(f"使用缓存素材: {cached_file}")
                return cached_file

            # 根据类型下载素材
            if material_type == "video":
                return self._download_video(keywords)
            elif material_type == "image":
                return self._download_image(keywords)
            elif material_type == "music":
                return self._download_music(keywords)
            else:
                logger.warning(f"不支持的素材类型: {material_type}")
                return None

        except Exception as e:
            logger.error(f"素材下载失败: {e}")
            return None

    def _check_cache(self, material_type: str, keywords: List[str]) -> Optional[str]:
        """检查缓存中是否有可用素材"""
        cache_key = f"{material_type}_{'_'.join(keywords or ['default'])}"
        cache_path = self.cache_dir / f"{cache_key}.cache"

        if cache_path.exists():
            # 检查缓存是否过期（7天）
            if time.time() - cache_path.stat().st_mtime < 7 * 24 * 3600:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    file_path = cached_data.get('file_path')
                    if file_path and Path(file_path).exists():
                        return file_path
        return None

    def _save_to_cache(self, material_type: str, keywords: List[str], file_path: str):
        """保存到缓存"""
        cache_key = f"{material_type}_{'_'.join(keywords or ['default'])}"
        cache_path = self.cache_dir / f"{cache_key}.cache"

        with open(cache_path, 'w') as f:
            json.dump({
                'file_path': file_path,
                'material_type': material_type,
                'keywords': keywords,
                'timestamp': time.time()
            }, f)

    def _download_video(self, keywords: List[str]) -> Optional[str]:
        """下载视频素材"""
        # 使用本地生成代替真实下载
        return self._generate_placeholder_video(keywords)

    def _download_image(self, keywords: List[str]) -> Optional[str]:
        """下载图片素材"""
        try:
            # 从Unsplash下载免费图片
            if keywords:
                # 搜索相关图片
                search_url = f"https://source.unsplash.com/1920x1080/?{','.join(keywords)}"
            else:
                # 随机选择一个预设图片
                search_url = random.choice(self.free_resources["images"])
                # 添加尺寸参数
                if "unsplash" in search_url:
                    search_url += "?w=1920&h=1080"

            # 下载图片
            response = self.session.get(search_url, timeout=30)
            if response.status_code == 200:
                # 保存图片
                file_ext = self._get_file_extension(response.headers.get('content-type', ''))
                file_name = f"image_{int(time.time())}{file_ext or '.jpg'}"
                file_path = self.cache_dir / file_name

                with open(file_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"✅ 图片下载成功: {file_path}")
                self._save_to_cache("image", keywords, str(file_path))
                return str(file_path)

        except Exception as e:
            logger.error(f"图片下载失败: {e}")

        # 如果下载失败，生成占位图
        return self._generate_placeholder_image(keywords)

    def _download_music(self, keywords: List[str]) -> Optional[str]:
        """下载音乐素材"""
        # 暂时返回None，使用生成器内置的音乐生成
        return None

    def _generate_placeholder_video(self, keywords: List[str]) -> str:
        """生成占位视频"""
        from moviepy import ColorClip, TextClip, CompositeVideoClip

        # 创建彩色背景
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        bg_color = random.choice(colors)

        # 创建视频片段
        clip = ColorClip(size=(1920, 1080), color=bg_color, duration=5)

        # 添加文字
        if keywords:
            text = " | ".join(keywords)
            txt_clip = TextClip(
                text=text,
                font_size=50,
                color='white',
                font='PingFang-SC-Regular'
            )
            txt_clip = txt_clip.set_position('center').set_duration(5)
            clip = CompositeVideoClip([clip, txt_clip])

        # 保存视频
        file_path = self.cache_dir / f"placeholder_video_{int(time.time())}.mp4"
        clip.write_videofile(str(file_path), fps=24, logger=None)
        clip.close()

        return str(file_path)

    def _generate_placeholder_image(self, keywords: List[str]) -> str:
        """生成占位图片"""
        from PIL import Image, ImageDraw, ImageFont
        import random

        # 创建图片
        colors = [(255, 107, 107), (78, 205, 196), (69, 183, 209), (150, 206, 180)]
        bg_color = random.choice(colors)

        img = Image.new('RGB', (1920, 1080), color=bg_color)
        draw = ImageDraw.Draw(img)

        # 添加文字
        if keywords:
            text = " | ".join(keywords)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 60)
            except:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = (1920 - text_width) // 2
            y = (1080 - text_height) // 2

            draw.text((x, y), text, fill=(255, 255, 255), font=font)

        # 保存图片
        file_path = self.cache_dir / f"placeholder_image_{int(time.time())}.jpg"
        img.save(file_path, 'JPEG')

        return str(file_path)

    def _get_file_extension(self, content_type: str) -> str:
        """根据content-type获取文件扩展名"""
        extensions = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'video/mp4': '.mp4',
            'video/quicktime': '.mov',
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav'
        }
        return extensions.get(content_type, '')

    def get_stock_footage(self, theme: str, count: int = 5) -> List[str]:
        """获取库存素材"""
        footage = []
        for i in range(count):
            if theme == "tech":
                keywords = ["technology", "digital", "innovation"]
            elif theme == "nature":
                keywords = ["nature", "landscape", "outdoor"]
            elif theme == "business":
                keywords = ["business", "office", "professional"]
            else:
                keywords = ["abstract", "creative", "art"]

            # 下载或生成素材
            material = self.download_material("image", keywords)
            if material:
                footage.append(material)

        return footage


# 单例模式
_downloader_instance = None

def get_material_downloader() -> MaterialDownloader:
    """获取素材下载器实例"""
    global _downloader_instance
    if _downloader_instance is None:
        _downloader_instance = MaterialDownloader()
    return _downloader_instance