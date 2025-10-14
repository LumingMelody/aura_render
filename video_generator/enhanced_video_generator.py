#!/usr/bin/env python3
"""
增强版视频生成器 - 生成带有场景和特效的视频
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import subprocess
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

logger = logging.getLogger(__name__)

# 确保输出目录存在
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class EnhancedVideoGenerator:
    """增强版视频生成器 - 带场景和特效"""

    def __init__(self):
        """初始化视频生成器"""
        self.default_resolution = (1920, 1080)
        self.default_fps = 30

        # 场景模板
        self.scene_templates = {
            "现代城市": self._create_city_scene,
            "科技元素": self._create_tech_scene,
            "创新": self._create_innovation_scene,
            "未来": self._create_future_scene,
            "default": self._create_default_scene
        }

        # 颜色方案
        self.color_schemes = {
            "科技创新": {
                "primary": (0, 180, 216),
                "secondary": (144, 19, 254),
                "accent": (255, 0, 255),
                "bg_gradient": [(0, 10, 40), (0, 50, 100)],
                "text": (255, 255, 255),
                "glow": (0, 255, 255)
            },
            "励志": {
                "primary": (255, 107, 107),
                "secondary": (78, 205, 196),
                "accent": (255, 215, 0),
                "bg_gradient": [(40, 60, 90), (100, 150, 200)],
                "text": (255, 255, 255),
                "glow": (255, 200, 100)
            }
        }

    def generate_video(self,
                       task_id: str,
                       description: str,
                       keywords: List[str],
                       duration: int = 30,
                       theme: str = "科技创新") -> Dict[str, Any]:
        """
        生成增强版视频
        """
        try:
            logger.info(f"🎬 开始生成增强视频 - Task ID: {task_id}")
            logger.info(f"📋 主题: {theme}, 时长: {duration}秒")
            logger.info(f"🔑 关键词: {keywords}")

            # 选择颜色方案
            colors = self.color_schemes.get(theme, self.color_schemes["科技创新"])

            # 输出路径
            output_path = OUTPUT_DIR / f"video_{task_id}.mp4"

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.default_fps, self.default_resolution)

            total_frames = duration * self.default_fps

            # 计算每个部分的帧数
            intro_frames = int(total_frames * 0.1)  # 10% 开场
            outro_frames = int(total_frames * 0.1)  # 10% 结尾
            content_frames = total_frames - intro_frames - outro_frames  # 80% 内容

            # 每个关键词的帧数
            frames_per_keyword = content_frames // len(keywords) if keywords else content_frames

            current_frame = 0

            # 1. 生成开场动画（带科技感）
            logger.info("🎨 生成开场动画...")
            for i in range(intro_frames):
                frame = self._create_intro_frame(i, intro_frames, description, colors, theme)
                out.write(frame)
                current_frame += 1

            # 2. 为每个关键词生成场景
            for idx, keyword in enumerate(keywords):
                logger.info(f"🎨 生成场景: {keyword}")
                scene_func = self.scene_templates.get(keyword, self.scene_templates["default"])

                for i in range(frames_per_keyword):
                    frame = scene_func(i, frames_per_keyword, keyword, colors, idx)
                    out.write(frame)
                    current_frame += 1

            # 3. 生成结尾动画
            logger.info("🎨 生成结尾动画...")
            remaining_frames = total_frames - current_frame
            for i in range(remaining_frames):
                frame = self._create_outro_frame(i, remaining_frames, colors, theme)
                out.write(frame)

            out.release()

            # 验证视频文件
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
                    "theme": theme,
                    "keywords": keywords,
                    "timestamp": datetime.now().isoformat()
                }

                logger.info(f"✅ 增强视频生成成功: {output_path} ({file_size:.2f} MB)")
                return result
            else:
                raise Exception("视频文件未能创建")

        except Exception as e:
            logger.error(f"❌ 增强视频生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }

    def _create_gradient_background(self, colors: List[tuple]) -> np.ndarray:
        """创建渐变背景"""
        img = np.zeros((self.default_resolution[1], self.default_resolution[0], 3), dtype=np.uint8)

        start_color = np.array(colors[0])
        end_color = np.array(colors[1])

        for y in range(self.default_resolution[1]):
            ratio = y / self.default_resolution[1]
            color = start_color + (end_color - start_color) * ratio
            img[y, :] = color.astype(np.uint8)

        return img

    def _add_particle_effects(self, img: np.ndarray, num_particles: int = 50, color: tuple = (255, 255, 255)) -> np.ndarray:
        """添加粒子效果"""
        for _ in range(num_particles):
            x = random.randint(0, self.default_resolution[0])
            y = random.randint(0, self.default_resolution[1])
            radius = random.randint(1, 3)
            cv2.circle(img, (x, y), radius, color, -1)
        return img

    def _create_intro_frame(self, frame_idx: int, total_frames: int, title: str, colors: Dict, theme: str) -> np.ndarray:
        """创建科技感开场帧"""
        # 渐变背景
        img = self._create_gradient_background(colors["bg_gradient"])

        # 添加网格效果
        grid_spacing = 50
        grid_color = (*colors["secondary"], 30)
        for x in range(0, self.default_resolution[0], grid_spacing):
            cv2.line(img, (x, 0), (x, self.default_resolution[1]), grid_color, 1)
        for y in range(0, self.default_resolution[1], grid_spacing):
            cv2.line(img, (0, y), (self.default_resolution[0], y), grid_color, 1)

        # 添加粒子效果
        progress = frame_idx / total_frames
        num_particles = int(20 + 30 * progress)
        img = self._add_particle_effects(img, num_particles, colors["glow"])

        # 转换为PIL进行文字绘制
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 动画效果 - 淡入 + 缩放
        alpha = min(1.0, progress * 2)
        scale = 0.8 + 0.2 * alpha

        try:
            font_size = int(80 * scale)
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
            small_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 30)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # 主标题
        display_title = title[:40] + "..." if len(title) > 40 else title
        bbox = draw.textbbox((0, 0), display_title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.default_resolution[0] - text_width) // 2
        y = (self.default_resolution[1] - text_height) // 2

        # 绘制发光效果
        for offset in range(3, 0, -1):
            glow_color = (*colors["glow"], int(50 * alpha))
            draw.text((x-offset, y-offset), display_title, fill=glow_color, font=font)
            draw.text((x+offset, y+offset), display_title, fill=glow_color, font=font)

        # 绘制主文字
        draw.text((x, y), display_title, fill=colors["text"], font=font)

        # 添加主题标签
        if progress > 0.5:
            tag_text = f"#{theme}"
            tag_bbox = draw.textbbox((0, 0), tag_text, font=small_font)
            tag_width = tag_bbox[2] - tag_bbox[0]
            tag_x = (self.default_resolution[0] - tag_width) // 2
            tag_y = y + text_height + 50
            draw.text((tag_x, tag_y), tag_text, fill=colors["accent"], font=small_font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_city_scene(self, frame_idx: int, total_frames: int, keyword: str, colors: Dict, scene_idx: int) -> np.ndarray:
        """创建现代城市场景"""
        # 深色背景（夜景）
        img = np.full((self.default_resolution[1], self.default_resolution[0], 3), (10, 20, 40), dtype=np.uint8)

        # 绘制建筑物剪影
        building_count = 8
        building_width = self.default_resolution[0] // building_count

        for i in range(building_count):
            height = random.randint(300, 700)
            x = i * building_width
            y = self.default_resolution[1] - height

            # 建筑物主体
            cv2.rectangle(img, (x, y), (x + building_width - 10, self.default_resolution[1]),
                         (30, 40, 60), -1)

            # 窗户灯光
            window_rows = height // 40
            window_cols = (building_width - 20) // 30
            for row in range(window_rows):
                for col in range(window_cols):
                    if random.random() > 0.3:  # 70%的窗户亮着
                        wx = x + 10 + col * 30
                        wy = y + 10 + row * 40
                        window_color = colors["glow"] if random.random() > 0.8 else (255, 200, 100)
                        cv2.rectangle(img, (wx, wy), (wx + 20, wy + 25), window_color, -1)

        # 添加动态光线效果
        progress = frame_idx / total_frames
        for i in range(3):
            light_x = int((self.default_resolution[0] * (progress + i/3)) % self.default_resolution[0])
            cv2.line(img, (light_x, 0), (light_x, self.default_resolution[1]), colors["accent"], 2)

        # 添加文字
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 100)
        except:
            font = ImageFont.load_default()

        # 绘制关键词
        bbox = draw.textbbox((0, 0), keyword, font=font)
        text_width = bbox[2] - bbox[0]
        x = (self.default_resolution[0] - text_width) // 2
        y = 200

        # 发光效果
        for offset in range(5, 0, -1):
            draw.text((x, y-offset), keyword, fill=(*colors["glow"], 100), font=font)
        draw.text((x, y), keyword, fill=(255, 255, 255), font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_tech_scene(self, frame_idx: int, total_frames: int, keyword: str, colors: Dict, scene_idx: int) -> np.ndarray:
        """创建科技元素场景"""
        # 深色科技背景
        img = self._create_gradient_background([(0, 0, 20), (0, 50, 100)])

        # 添加电路板图案
        progress = frame_idx / total_frames

        # 绘制电路线
        for i in range(10):
            start_x = random.randint(0, self.default_resolution[0])
            start_y = random.randint(0, self.default_resolution[1])
            end_x = random.randint(0, self.default_resolution[0])
            end_y = random.randint(0, self.default_resolution[1])

            cv2.line(img, (start_x, start_y), (end_x, end_y), colors["secondary"], 1)

            # 在线的端点添加节点
            cv2.circle(img, (start_x, start_y), 5, colors["glow"], -1)
            cv2.circle(img, (end_x, end_y), 5, colors["glow"], -1)

        # 添加数据流效果
        for i in range(20):
            x = int((self.default_resolution[0] * (progress * 2 + i/20)) % self.default_resolution[0])
            y = int(self.default_resolution[1] / 2 + 200 * math.sin(x / 100 + progress * 10))
            cv2.circle(img, (x, y), 3, colors["accent"], -1)

        # 添加HUD元素
        hud_elements = [
            (100, 100, 300, 200),
            (self.default_resolution[0] - 400, 100, 300, 200),
            (100, self.default_resolution[1] - 300, 300, 200)
        ]

        for x, y, w, h in hud_elements:
            cv2.rectangle(img, (x, y), (x + w, y + h), colors["secondary"], 2)
            # 添加内部线条
            for i in range(1, 4):
                line_y = y + (h // 4) * i
                cv2.line(img, (x + 10, line_y), (x + w - 10, line_y), colors["glow"], 1)

        # 添加文字
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 120)
            small_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 30)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # 主关键词
        bbox = draw.textbbox((0, 0), keyword, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (self.default_resolution[0] - text_width) // 2
        text_y = (self.default_resolution[1] - 120) // 2

        # 绘制科技感文字
        draw.text((text_x, text_y), keyword, fill=colors["text"], font=font)

        # 添加扫描线效果
        scan_y = int((self.default_resolution[1] * progress) % self.default_resolution[1])
        cv2.line(img, (0, scan_y), (self.default_resolution[0], scan_y), colors["glow"], 2)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_innovation_scene(self, frame_idx: int, total_frames: int, keyword: str, colors: Dict, scene_idx: int) -> np.ndarray:
        """创建创新主题场景"""
        # 渐变背景
        img = self._create_gradient_background(colors["bg_gradient"])

        # 创建创新元素 - 灯泡图形
        center_x = self.default_resolution[0] // 2
        center_y = self.default_resolution[1] // 2 - 100

        progress = frame_idx / total_frames
        pulse = 1 + 0.2 * math.sin(progress * 4 * math.pi)

        # 绘制灯泡轮廓
        bulb_radius = int(150 * pulse)
        cv2.circle(img, (center_x, center_y), bulb_radius, colors["accent"], 3)

        # 绘制灯泡底部
        cv2.rectangle(img,
                     (center_x - 50, center_y + bulb_radius - 20),
                     (center_x + 50, center_y + bulb_radius + 50),
                     colors["secondary"], -1)

        # 添加光线效果
        num_rays = 12
        for i in range(num_rays):
            angle = (2 * math.pi * i / num_rays) + progress * 2
            ray_length = 200 + 50 * math.sin(progress * 4 * math.pi)
            end_x = int(center_x + ray_length * math.cos(angle))
            end_y = int(center_y + ray_length * math.sin(angle))
            cv2.line(img, (center_x, center_y), (end_x, end_y), colors["glow"], 2)

        # 添加创意点子（小圆点）
        for i in range(30):
            angle = random.random() * 2 * math.pi
            distance = 250 + random.randint(0, 200)
            x = int(center_x + distance * math.cos(angle + progress))
            y = int(center_y + distance * math.sin(angle + progress))
            cv2.circle(img, (x, y), 5, colors["accent"], -1)

        # 添加文字
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 100)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), keyword, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (self.default_resolution[0] - text_width) // 2
        text_y = center_y + bulb_radius + 150

        draw.text((text_x, text_y), keyword, fill=colors["text"], font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_future_scene(self, frame_idx: int, total_frames: int, keyword: str, colors: Dict, scene_idx: int) -> np.ndarray:
        """创建未来主题场景"""
        # 太空背景
        img = np.full((self.default_resolution[1], self.default_resolution[0], 3), (0, 0, 30), dtype=np.uint8)

        # 添加星星
        for _ in range(200):
            x = random.randint(0, self.default_resolution[0])
            y = random.randint(0, self.default_resolution[1])
            brightness = random.randint(100, 255)
            cv2.circle(img, (x, y), 1, (brightness, brightness, brightness), -1)

        # 绘制地球或行星
        progress = frame_idx / total_frames
        planet_x = int(self.default_resolution[0] * 0.8)
        planet_y = int(self.default_resolution[1] * 0.3)
        planet_radius = 200

        # 行星主体
        cv2.circle(img, (planet_x, planet_y), planet_radius, (50, 100, 150), -1)
        cv2.circle(img, (planet_x, planet_y), planet_radius, colors["secondary"], 3)

        # 添加轨道环
        cv2.ellipse(img, (planet_x, planet_y), (planet_radius + 50, 30),
                   -20, 0, 360, colors["glow"], 2)

        # 添加飞行器或卫星
        ship_x = int(self.default_resolution[0] * progress)
        ship_y = int(self.default_resolution[1] * 0.6 + 50 * math.sin(progress * 4 * math.pi))

        # 飞行器主体
        points = np.array([
            [ship_x, ship_y],
            [ship_x - 40, ship_y + 20],
            [ship_x - 30, ship_y],
            [ship_x - 40, ship_y - 20]
        ])
        cv2.fillPoly(img, [points], colors["accent"])

        # 推进器火焰
        flame_length = int(20 + 10 * math.sin(progress * 20))
        cv2.line(img, (ship_x - 40, ship_y),
                (ship_x - 40 - flame_length, ship_y), colors["glow"], 5)

        # 添加网格透视效果
        for i in range(10):
            y = self.default_resolution[1] - i * 50
            cv2.line(img, (0, y), (self.default_resolution[0], y),
                    (*colors["secondary"], 50), 1)

        # 添加文字
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 120)
            small_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 40)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # 主关键词
        bbox = draw.textbbox((0, 0), keyword, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (self.default_resolution[0] - text_width) // 2
        text_y = 100

        # 绘制未来感文字
        for offset in range(3, 0, -1):
            draw.text((text_x, text_y + offset * 2), keyword,
                     fill=(*colors["glow"], 100), font=font)
        draw.text((text_x, text_y), keyword, fill=colors["text"], font=font)

        # 添加年份
        year_text = "2050"
        year_bbox = draw.textbbox((0, 0), year_text, font=small_font)
        year_width = year_bbox[2] - year_bbox[0]
        year_x = (self.default_resolution[0] - year_width) // 2
        year_y = text_y + 150
        draw.text((year_x, year_y), year_text, fill=colors["accent"], font=small_font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_default_scene(self, frame_idx: int, total_frames: int, keyword: str, colors: Dict, scene_idx: int) -> np.ndarray:
        """创建默认场景"""
        # 使用科技场景作为默认
        return self._create_tech_scene(frame_idx, total_frames, keyword, colors, scene_idx)

    def _create_outro_frame(self, frame_idx: int, total_frames: int, colors: Dict, theme: str) -> np.ndarray:
        """创建结尾帧"""
        # 渐变背景
        img = self._create_gradient_background(colors["bg_gradient"])

        # 添加粒子淡出效果
        progress = frame_idx / total_frames
        fade_alpha = 1 - progress
        num_particles = int(50 * fade_alpha)
        img = self._add_particle_effects(img, num_particles, colors["glow"])

        # 转换为PIL
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 80)
            small_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 40)
            tiny_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 25)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            tiny_font = ImageFont.load_default()

        # 感谢文字
        thanks_text = "感谢观看"
        bbox = draw.textbbox((0, 0), thanks_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.default_resolution[0] - text_width) // 2
        y = (self.default_resolution[1] - text_height) // 2 - 100

        # 发光效果
        for offset in range(5, 0, -1):
            alpha = int(255 * fade_alpha * (1 - offset/5))
            draw.text((x, y-offset), thanks_text,
                     fill=(*colors["glow"], alpha), font=font)

        draw.text((x, y), thanks_text, fill=colors["text"], font=font)

        # 主题标签
        theme_text = f"#{theme}"
        theme_bbox = draw.textbbox((0, 0), theme_text, font=small_font)
        theme_width = theme_bbox[2] - theme_bbox[0]
        theme_x = (self.default_resolution[0] - theme_width) // 2
        theme_y = y + text_height + 50
        draw.text((theme_x, theme_y), theme_text, fill=colors["accent"], font=small_font)

        # 制作信息
        info_text = "Created with Aura Render"
        info_bbox = draw.textbbox((0, 0), info_text, font=tiny_font)
        info_width = info_bbox[2] - info_bbox[0]
        info_x = (self.default_resolution[0] - info_width) // 2
        info_y = theme_y + 60
        draw.text((info_x, info_y), info_text, fill=colors["secondary"], font=tiny_font)

        # 添加Logo效果
        logo_y = info_y + 50
        logo_size = int(80 * (1 + 0.2 * math.sin(progress * 4 * math.pi)))
        logo_x = (self.default_resolution[0] - logo_size) // 2

        # 绘制动态Logo
        cv2.circle(img,
                  (self.default_resolution[0] // 2, logo_y + logo_size // 2),
                  logo_size // 2, colors["primary"], 3)
        cv2.circle(img,
                  (self.default_resolution[0] // 2, logo_y + logo_size // 2),
                  logo_size // 3, colors["accent"], 2)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# 单例模式
_enhanced_generator_instance = None

def get_enhanced_video_generator() -> EnhancedVideoGenerator:
    """获取增强视频生成器实例"""
    global _enhanced_generator_instance
    if _enhanced_generator_instance is None:
        _enhanced_generator_instance = EnhancedVideoGenerator()
    return _enhanced_generator_instance


if __name__ == "__main__":
    # 测试视频生成
    generator = get_enhanced_video_generator()
    result = generator.generate_video(
        task_id="enhanced_test_001",
        description="制作一个关于科技创新的30秒宣传视频，包含现代城市场景和科技元素",
        keywords=["现代城市", "科技元素", "创新", "未来"],
        duration=30,
        theme="科技创新"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))