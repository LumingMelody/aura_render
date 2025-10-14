#!/usr/bin/env python3
"""
真实视频生成器 - 使用MoviePy生成实际视频文件
"""

import os
import json
import random
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import cv2
from moviepy import VideoClip, TextClip, CompositeVideoClip, AudioClip, AudioFileClip, concatenate_videoclips, ColorClip
import requests
from gtts import gTTS

logger = logging.getLogger(__name__)

# 确保输出目录存在
OUTPUT_DIR = Path("/tmp/aura_render_outputs")
ASSETS_DIR = Path("/tmp/aura_render_assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


class RealVideoGenerator:
    """真实视频生成器"""

    def __init__(self):
        """初始化视频生成器"""
        self.temp_dir = Path(tempfile.gettempdir()) / "aura_render_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 视频参数
        self.default_resolution = (1920, 1080)
        self.default_fps = 30
        self.default_duration = 30

        # 颜色主题
        self.themes = {
            "励志": {"primary": "#FF6B6B", "secondary": "#4ECDC4", "bg": "#95E1D3"},
            "专业": {"primary": "#2C3E50", "secondary": "#3498DB", "bg": "#ECF0F1"},
            "创新": {"primary": "#9B59B6", "secondary": "#E74C3C", "bg": "#F39C12"},
            "科技": {"primary": "#00B4D8", "secondary": "#0077B6", "bg": "#CAF0F8"},
            "温馨": {"primary": "#F4A261", "secondary": "#E76F51", "bg": "#F9DCC4"}
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
            logger.info(f"🎬 开始生成真实视频 - Task ID: {task_id}")

            # 选择颜色主题
            theme = self.themes.get(emotion, self.themes["专业"])

            # 生成视频片段
            clips = []
            segment_duration = duration / (len(keywords) + 2)  # +2 for intro and outro

            # 1. 创建开场片段
            intro_clip = self._create_intro_clip(
                title=description[:50],
                duration=segment_duration,
                theme=theme
            )
            clips.append(intro_clip)

            # 2. 为每个关键词创建片段
            for i, keyword in enumerate(keywords):
                keyword_clip = self._create_keyword_clip(
                    keyword=keyword,
                    index=i + 1,
                    duration=segment_duration,
                    theme=theme
                )
                clips.append(keyword_clip)

            # 3. 创建结尾片段
            outro_clip = self._create_outro_clip(
                duration=segment_duration,
                theme=theme
            )
            clips.append(outro_clip)

            # 4. 合并所有片段
            final_video = concatenate_videoclips(clips, method="compose")

            # 5. 添加背景音乐
            audio_path = self._generate_background_music(duration)
            if audio_path and os.path.exists(audio_path):
                background_audio = AudioFileClip(audio_path)
                background_audio = background_audio.subclipped(0, min(duration, background_audio.duration))
                background_audio = background_audio.volumex(0.3)  # 降低音量
                final_video = final_video.with_audio(background_audio)

            # 6. 生成字幕
            subtitles = self._generate_subtitles(description, keywords, duration)
            if subtitles:
                final_video = self._add_subtitles(final_video, subtitles)

            # 7. 输出视频
            output_path = OUTPUT_DIR / f"video_{task_id}.mp4"
            final_video.write_videofile(
                str(output_path),
                fps=self.default_fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / f"temp_audio_{task_id}.m4a"),
                remove_temp=True
            )

            # 清理临时文件
            for clip in clips:
                clip.close()
            final_video.close()

            # 获取文件信息
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB

            result = {
                "success": True,
                "output_path": str(output_path),
                "duration": duration,
                "resolution": f"{self.default_resolution[0]}x{self.default_resolution[1]}",
                "file_size_mb": round(file_size, 2),
                "segments": len(clips),
                "emotion": emotion,
                "keywords": keywords,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ 视频生成成功: {output_path}")
            return result

        except Exception as e:
            logger.error(f"❌ 视频生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }

    def _create_intro_clip(self, title: str, duration: float, theme: Dict) -> VideoClip:
        """创建开场动画"""
        def make_frame(t):
            """生成每一帧"""
            img = Image.new('RGB', self.default_resolution, color=theme["bg"])
            draw = ImageDraw.Draw(img)

            # 动画效果 - 淡入
            alpha = min(1.0, t / 2.0)

            # 绘制标题
            try:
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 80)
            except:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = (self.default_resolution[0] - text_width) // 2
            y = (self.default_resolution[1] - text_height) // 2

            # 绘制阴影
            shadow_offset = 5
            draw.text((x + shadow_offset, y + shadow_offset), title,
                     fill=(0, 0, 0, int(128 * alpha)), font=font)

            # 绘制主文字
            draw.text((x, y), title, fill=theme["primary"], font=font)

            # 添加装饰元素
            if t > 1:
                # 绘制动态线条
                line_y = int(y + text_height + 50)
                line_width = int(text_width * min(1.0, (t - 1) / 1.0))
                line_x = x + (text_width - line_width) // 2
                draw.rectangle(
                    [line_x, line_y, line_x + line_width, line_y + 5],
                    fill=theme["secondary"]
                )

            return np.array(img)

        return VideoClip(make_frame, duration=duration)

    def _create_keyword_clip(self, keyword: str, index: int, duration: float, theme: Dict) -> VideoClip:
        """创建关键词展示片段"""
        def make_frame(t):
            """生成每一帧"""
            img = Image.new('RGB', self.default_resolution, color=theme["bg"])
            draw = ImageDraw.Draw(img)

            # 动画效果 - 缩放
            scale = 0.8 + 0.2 * np.sin(t * 2 * np.pi / 3)

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
            text_bbox = draw.textbbox((0, 0), keyword, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = (self.default_resolution[0] - text_width) // 2
            y = (self.default_resolution[1] - text_height) // 2

            # 绘制背景形状
            padding = 50
            shape_alpha = int(255 * 0.3)
            overlay = Image.new('RGBA', self.default_resolution, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.ellipse(
                [x - padding, y - padding,
                 x + text_width + padding, y + text_height + padding],
                fill=(*ImageColor.getrgb(theme["primary"]), shape_alpha)
            )
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)

            # 绘制关键词
            draw.text((x, y), keyword, fill="white", font=font)

            # 添加动态粒子效果
            if t > 0.5:
                for _ in range(5):
                    px = random.randint(0, self.default_resolution[0])
                    py = random.randint(0, self.default_resolution[1])
                    draw.ellipse([px, py, px + 5, px + 5], fill=theme["secondary"])

            return np.array(img)

        return VideoClip(make_frame, duration=duration)

    def _create_outro_clip(self, duration: float, theme: Dict) -> VideoClip:
        """创建结尾动画"""
        def make_frame(t):
            """生成每一帧"""
            img = Image.new('RGB', self.default_resolution, color=theme["bg"])
            draw = ImageDraw.Draw(img)

            # 淡出效果
            alpha = max(0, 1.0 - t / duration)

            try:
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 60)
                small_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 30)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()

            # 绘制感谢文字
            thanks_text = "Thanks for Watching"
            text_bbox = draw.textbbox((0, 0), thanks_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = (self.default_resolution[0] - text_width) // 2
            y = (self.default_resolution[1] - text_height) // 2 - 50

            draw.text((x, y), thanks_text,
                     fill=theme["primary"], font=font)

            # 绘制副标题
            subtitle = "Created with Aura Render"
            sub_bbox = draw.textbbox((0, 0), subtitle, font=small_font)
            sub_width = sub_bbox[2] - sub_bbox[0]

            sub_x = (self.default_resolution[0] - sub_width) // 2
            sub_y = y + text_height + 30

            draw.text((sub_x, sub_y), subtitle,
                     fill=theme["secondary"], font=small_font)

            # 添加Logo或图标（简单的圆形）
            logo_y = sub_y + 80
            logo_size = 60
            logo_x = (self.default_resolution[0] - logo_size) // 2
            draw.ellipse(
                [logo_x, logo_y, logo_x + logo_size, logo_y + logo_size],
                fill=theme["primary"]
            )

            return np.array(img)

        return VideoClip(make_frame, duration=duration)

    def _generate_background_music(self, duration: int) -> Optional[str]:
        """生成或获取背景音乐"""
        try:
            # 这里可以集成音乐生成API或使用预设音乐
            # 暂时使用一个简单的正弦波作为示例
            audio_path = self.temp_dir / f"bgm_{datetime.now().timestamp()}.mp3"

            # 生成简单的音调
            from moviepy import AudioClip

            def make_audio(t):
                """生成音频波形"""
                # 创建和谐的音调
                frequency1 = 440  # A4
                frequency2 = 554  # C#5
                frequency3 = 659  # E5

                signal = (np.sin(2 * np.pi * frequency1 * t) * 0.3 +
                         np.sin(2 * np.pi * frequency2 * t) * 0.2 +
                         np.sin(2 * np.pi * frequency3 * t) * 0.1)

                # 添加淡入淡出
                if isinstance(t, (int, float)):
                    if t < 2:
                        signal *= t / 2
                    elif t > duration - 2:
                        signal *= (duration - t) / 2
                else:
                    # Handle numpy arrays
                    fade_in_mask = t < 2
                    fade_out_mask = t > duration - 2
                    signal = np.where(fade_in_mask, signal * t / 2, signal)
                    signal = np.where(fade_out_mask, signal * (duration - t) / 2, signal)

                return signal

            audio_clip = AudioClip(make_audio, duration=duration, fps=44100)
            audio_clip.write_audiofile(str(audio_path), logger=None)
            audio_clip.close()

            return str(audio_path)

        except Exception as e:
            logger.warning(f"背景音乐生成失败: {e}")
            return None

    def _generate_subtitles(self, description: str, keywords: List[str], duration: int) -> List[Dict]:
        """生成字幕数据"""
        subtitles = []
        segment_duration = duration / (len(keywords) + 2)

        # 开场字幕
        subtitles.append({
            "start": 0,
            "end": segment_duration,
            "text": description[:50]
        })

        # 关键词字幕
        for i, keyword in enumerate(keywords):
            start_time = segment_duration * (i + 1)
            subtitles.append({
                "start": start_time,
                "end": start_time + segment_duration,
                "text": f"关键词: {keyword}"
            })

        # 结尾字幕
        subtitles.append({
            "start": duration - segment_duration,
            "end": duration,
            "text": "感谢观看"
        })

        return subtitles

    def _add_subtitles(self, video: VideoClip, subtitles: List[Dict]) -> VideoClip:
        """添加字幕到视频"""
        subtitle_clips = []

        for subtitle in subtitles:
            # 创建文字片段
            # Create text clip with font fallback
            try:
                txt_clip = TextClip(
                    text=subtitle["text"],
                    font_size=40,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    font='Arial',  # Use system default font
                    method='caption',
                    size=(self.default_resolution[0] - 100, None)
                )
            except Exception as e:
                self.logger.warning(f"Failed to create text clip: {e}")
                # Skip this subtitle if it fails
                continue

            # 设置位置和时长
            txt_clip = txt_clip.set_position(('center', 'bottom'))
            txt_clip = txt_clip.set_start(subtitle["start"])
            txt_clip = txt_clip.set_duration(subtitle["end"] - subtitle["start"])

            subtitle_clips.append(txt_clip)

        # 合并字幕和视频
        return CompositeVideoClip([video] + subtitle_clips)

    def generate_from_template(self, template: str, task_id: str, **kwargs) -> Dict[str, Any]:
        """基于模板生成视频"""
        templates = {
            "product_demo": {
                "description": "产品演示视频",
                "keywords": ["创新", "品质", "专业"],
                "duration": 30,
                "emotion": "专业"
            },
            "birthday": {
                "description": "生日祝福视频",
                "keywords": ["祝福", "快乐", "美好"],
                "duration": 20,
                "emotion": "温馨"
            },
            "tech_intro": {
                "description": "科技介绍视频",
                "keywords": ["AI", "未来", "智能"],
                "duration": 30,
                "emotion": "科技"
            }
        }

        if template in templates:
            params = templates[template]
            params.update(kwargs)
            return self.generate_video(task_id, **params)
        else:
            return {"success": False, "error": f"Unknown template: {template}"}


# 单例模式
_generator_instance = None

def get_video_generator() -> RealVideoGenerator:
    """获取视频生成器实例"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = RealVideoGenerator()
    return _generator_instance


if __name__ == "__main__":
    # 测试视频生成
    generator = get_video_generator()
    result = generator.generate_video(
        task_id="test_001",
        description="这是一个测试视频，展示AI技术的创新力量",
        keywords=["人工智能", "创新", "未来"],
        duration=15,
        emotion="科技"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))