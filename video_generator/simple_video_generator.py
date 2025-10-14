#!/usr/bin/env python3
"""
简化版视频生成器 - 生成简单但真实的视频文件
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

logger = logging.getLogger(__name__)

# 确保输出目录存在 - 改为项目目录下的outputs文件夹
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SimpleVideoGenerator:
    """简化版视频生成器 - 使用OpenCV和FFmpeg"""

    def __init__(self):
        """初始化视频生成器"""
        self.default_resolution = (1920, 1080)
        self.default_fps = 30
        self.default_duration = 30

        # 颜色主题
        self.themes = {
            "励志": {"primary": (255, 107, 107), "secondary": (78, 205, 196), "bg": (149, 225, 211)},
            "专业": {"primary": (44, 62, 80), "secondary": (52, 152, 219), "bg": (236, 240, 241)},
            "创新": {"primary": (155, 89, 182), "secondary": (231, 76, 60), "bg": (243, 156, 18)},
            "科技": {"primary": (0, 180, 216), "secondary": (0, 119, 182), "bg": (202, 240, 248)},
            "温馨": {"primary": (244, 162, 97), "secondary": (231, 111, 81), "bg": (249, 220, 196)}
        }

    def generate_video(self,
                       task_id: str,
                       description: str,
                       keywords: List[str],
                       duration: int = 30,
                       emotion: str = "励志") -> Dict[str, Any]:
        """
        生成真实视频

        Args:
            task_id: 任务ID
            description: 视频描述
            keywords: 关键词列表
            duration: 视频时长（秒）
            emotion: 情感基调

        Returns:
            视频生成结果
        """
        try:
            logger.info(f"🎬 开始生成简化视频 - Task ID: {task_id}")

            # 选择颜色主题
            theme = self.themes.get(emotion, self.themes["专业"])

            # 生成视频帧
            output_path = OUTPUT_DIR / f"video_{task_id}.mp4"

            # 使用OpenCV创建视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.default_fps, self.default_resolution)

            total_frames = duration * self.default_fps
            segment_frames = total_frames // (len(keywords) + 2)

            frame_count = 0

            # 1. 生成开场帧
            logger.info("🎨 生成开场动画...")
            for i in range(segment_frames):
                frame = self._create_intro_frame(i, segment_frames, description, theme)
                out.write(frame)
                frame_count += 1

            # 2. 为每个关键词生成帧
            for keyword_idx, keyword in enumerate(keywords):
                logger.info(f"🎨 生成关键词帧: {keyword}")
                for i in range(segment_frames):
                    frame = self._create_keyword_frame(i, segment_frames, keyword, keyword_idx + 1, theme)
                    out.write(frame)
                    frame_count += 1

            # 3. 生成结尾帧
            logger.info("🎨 生成结尾动画...")
            remaining_frames = total_frames - frame_count
            for i in range(remaining_frames):
                frame = self._create_outro_frame(i, remaining_frames, theme)
                out.write(frame)

            out.release()

            # 检查文件是否生成成功
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB

                result = {
                    "success": True,
                    "output_path": str(output_path),
                    "duration": duration,
                    "resolution": f"{self.default_resolution[0]}x{self.default_resolution[1]}",
                    "file_size_mb": round(file_size, 2),
                    "frames": total_frames,
                    "fps": self.default_fps,
                    "emotion": emotion,
                    "keywords": keywords,
                    "timestamp": datetime.now().isoformat()
                }

                logger.info(f"✅ 简化视频生成成功: {output_path} ({file_size:.2f} MB)")
                return result
            else:
                raise Exception("Video file was not created")

        except Exception as e:
            logger.error(f"❌ 简化视频生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }

    def _create_intro_frame(self, frame_idx: int, total_frames: int, title: str, theme: Dict) -> np.ndarray:
        """创建开场帧"""
        # 创建画布
        img = np.full((self.default_resolution[1], self.default_resolution[0], 3), theme["bg"], dtype=np.uint8)

        # 转换为PIL进行文字绘制
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 动画效果 - 淡入
        progress = frame_idx / total_frames
        alpha = min(1.0, progress * 2)

        # 绘制标题
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 80)
        except:
            font = ImageFont.load_default()

        # 限制标题长度
        display_title = title[:30] + "..." if len(title) > 30 else title

        # 获取文字尺寸
        bbox = draw.textbbox((0, 0), display_title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.default_resolution[0] - text_width) // 2
        y = (self.default_resolution[1] - text_height) // 2

        # 绘制主文字
        draw.text((x, y), display_title, fill=theme["primary"], font=font)

        # 添加装饰线条
        if progress > 0.5:
            line_y = y + text_height + 50
            line_width = int(text_width * min(1.0, (progress - 0.5) * 2))
            line_x = x + (text_width - line_width) // 2
            draw.rectangle([line_x, line_y, line_x + line_width, line_y + 5], fill=theme["secondary"])

        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_keyword_frame(self, frame_idx: int, total_frames: int, keyword: str, index: int, theme: Dict) -> np.ndarray:
        """创建关键词帧"""
        # 创建画布
        img = np.full((self.default_resolution[1], self.default_resolution[0], 3), theme["bg"], dtype=np.uint8)

        # 转换为PIL进行文字绘制
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 动画效果 - 缩放
        progress = frame_idx / total_frames
        scale = 0.8 + 0.2 * np.sin(progress * 2 * np.pi)

        try:
            font_size = int(120 * scale)
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
            small_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 40)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # 绘制序号
        number_text = f"#{index}"
        draw.text((100, 100), number_text, fill=theme["secondary"], font=small_font)

        # 绘制关键词
        bbox = draw.textbbox((0, 0), keyword, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.default_resolution[0] - text_width) // 2
        y = (self.default_resolution[1] - text_height) // 2

        # 绘制背景圆形
        padding = 50
        center_x = x + text_width // 2
        center_y = y + text_height // 2
        radius = max(text_width, text_height) // 2 + padding

        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            fill=(*theme["primary"], 100)
        )

        # 绘制关键词
        draw.text((x, y), keyword, fill=(255, 255, 255), font=font)

        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_outro_frame(self, frame_idx: int, total_frames: int, theme: Dict) -> np.ndarray:
        """创建结尾帧"""
        # 创建画布
        img = np.full((self.default_resolution[1], self.default_resolution[0], 3), theme["bg"], dtype=np.uint8)

        # 转换为PIL进行文字绘制
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 60)
            small_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 30)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # 绘制感谢文字
        thanks_text = "谢谢观看"
        bbox = draw.textbbox((0, 0), thanks_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.default_resolution[0] - text_width) // 2
        y = (self.default_resolution[1] - text_height) // 2 - 50

        draw.text((x, y), thanks_text, fill=theme["primary"], font=font)

        # 绘制副标题
        subtitle = "Created with Aura Render"
        sub_bbox = draw.textbbox((0, 0), subtitle, font=small_font)
        sub_width = sub_bbox[2] - sub_bbox[0]

        sub_x = (self.default_resolution[0] - sub_width) // 2
        sub_y = y + text_height + 30

        draw.text((sub_x, sub_y), subtitle, fill=theme["secondary"], font=small_font)

        # 添加Logo（简单的圆形）
        logo_y = sub_y + 80
        logo_size = 60
        logo_x = (self.default_resolution[0] - logo_size) // 2
        draw.ellipse(
            [logo_x, logo_y, logo_x + logo_size, logo_y + logo_size],
            fill=theme["primary"]
        )

        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# 单例模式
_simple_generator_instance = None

def get_simple_video_generator() -> SimpleVideoGenerator:
    """获取简化视频生成器实例"""
    global _simple_generator_instance
    if _simple_generator_instance is None:
        _simple_generator_instance = SimpleVideoGenerator()
    return _simple_generator_instance


if __name__ == "__main__":
    # 测试视频生成
    generator = get_simple_video_generator()
    result = generator.generate_video(
        task_id="simple_test_001",
        description="这是一个测试视频，展示AI技术的创新力量",
        keywords=["人工智能", "创新", "未来"],
        duration=10,
        emotion="科技"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))