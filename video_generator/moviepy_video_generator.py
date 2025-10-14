#!/usr/bin/env python3
"""
MoviePy视频生成器 - 使用真实素材和MoviePy处理
"""

import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from moviepy import (
        VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
        TextClip, concatenate_videoclips, ColorClip
    )
    # 不使用特效，先实现基础功能
    # from moviepy.video.fx import fadeout, fadein
    # from moviepy.audio.fx.volumex import volumex
except ImportError:
    print("MoviePy not installed. Install with: pip install moviepy")
    raise

from materials_supplies.mock_materials_api import get_recommended_materials_for_vgp
from video_generator.aliyun_text_to_video import get_aliyun_text_to_video_client

logger = logging.getLogger(__name__)

class MoviePyVideoGenerator:
    """基于MoviePy的真实视频生成器"""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "aura_render"
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def generate_video(self,
                      task_id: str,
                      description: str,
                      keywords: List[str],
                      duration: int = 30,
                      theme: str = "科技创新",
                      vgp_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成真实视频 - 使用阿里云文生视频降级方案"""
        try:
            logger.info(f"🎬 开始生成MoviePy视频 - Task ID: {task_id}")

            # 获取VGP分析数据
            video_type = "商业类"
            emotions = {}

            if vgp_analysis:
                video_type = vgp_analysis.get("video_type", "商业类")
                emotions = vgp_analysis.get("emotions", {})

            # 尝试使用阿里云文生视频降级方案
            return self._generate_with_aliyun_fallback(
                task_id=task_id,
                description=description,
                keywords=keywords,
                duration=duration,
                theme=theme,
                video_type=video_type,
                emotions=emotions
            )

        except Exception as e:
            logger.error(f"❌ MoviePy视频生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id,
                "generator": "MoviePy"
            }

    def _generate_with_aliyun_fallback(self,
                                     task_id: str,
                                     description: str,
                                     keywords: List[str],
                                     duration: int,
                                     theme: str,
                                     video_type: str,
                                     emotions: Dict[str, float]) -> Dict[str, Any]:
        """使用阿里云文生视频降级方案生成视频"""
        try:
            logger.info(f"🚀 使用阿里云文生视频降级方案生成 {duration}秒 视频")

            # 获取阿里云文生视频客户端
            aliyun_client = get_aliyun_text_to_video_client()

            # 构建文本提示词
            text_prompts = self._build_text_prompts(description, keywords, theme, video_type, emotions)

            # 生成多个5秒视频片段
            video_segments = aliyun_client.generate_multi_segment_video(
                text_prompts=text_prompts,
                target_duration=duration
            )

            logger.info(f"📹 生成了 {len(video_segments)} 个视频片段")

            # 检查是否有成功生成的片段
            successful_segments = [seg for seg in video_segments if seg.get("success", False)]

            if successful_segments:
                # 使用MoviePy拼接多个片段
                return self._concatenate_video_segments(
                    segments=successful_segments,
                    task_id=task_id,
                    target_duration=duration,
                    theme=theme,
                    keywords=keywords,
                    video_type=video_type
                )
            else:
                # 所有片段都失败，回退到程序化视频
                logger.warning("⚠️ 阿里云文生视频全部失败，回退到程序化视频")
                return self._generate_programmatic_video(
                    task_id=task_id,
                    description=description,
                    keywords=keywords,
                    duration=duration,
                    theme=theme,
                    video_type=video_type
                )

        except Exception as e:
            logger.error(f"❌ 阿里云文生视频降级方案失败: {str(e)}")
            # 最后回退到程序化视频
            return self._generate_programmatic_video(
                task_id=task_id,
                description=description,
                keywords=keywords,
                duration=duration,
                theme=theme,
                video_type=video_type
            )

    def _build_text_prompts(self,
                          description: str,
                          keywords: List[str],
                          theme: str,
                          video_type: str,
                          emotions: Dict[str, float]) -> List[str]:
        """构建文生视频的文本提示词"""
        prompts = []

        # 基于描述和关键词生成提示词
        base_prompt = f"{description}，主题：{theme}"

        # 为每个关键词生成专门的提示词
        for keyword in keywords:
            if "科技" in keyword or "AI" in keyword or "人工智能" in keyword:
                prompt = f"现代化科技场景，{keyword}相关的创新技术展示，高科技感，专业画面"
            elif "城市" in keyword or "未来" in keyword:
                prompt = f"未来城市景观，{keyword}元素突出，科技感强烈，现代化建筑"
            elif "创新" in keyword or "发展" in keyword:
                prompt = f"创新科技实验室场景，{keyword}概念可视化，专业技术展示"
            else:
                prompt = f"{keyword}相关的专业场景，现代化环境，高质量画面"

            prompts.append(prompt)

        # 如果关键词不够，添加通用提示词
        if len(prompts) < 6:  # 30秒需要6个片段
            generic_prompts = [
                f"高科技办公环境，展示{theme}相关内容",
                f"现代化研发中心，{theme}技术展示",
                f"科技感数据可视化场景，{theme}应用展示",
                f"专业团队工作场景，{theme}项目推进",
                f"未来科技实验室，{theme}创新研发",
                f"现代化展示大厅，{theme}成果展示"
            ]

            for prompt in generic_prompts:
                if len(prompts) < 6:
                    prompts.append(prompt)

        return prompts[:6]  # 最多6个片段

    def _concatenate_video_segments(self,
                                  segments: List[Dict[str, Any]],
                                  task_id: str,
                                  target_duration: int,
                                  theme: str,
                                  keywords: List[str],
                                  video_type: str) -> Dict[str, Any]:
        """使用MoviePy拼接多个视频片段"""
        try:
            logger.info(f"🔗 开始拼接 {len(segments)} 个视频片段")

            # 加载所有视频片段
            video_clips = []
            total_size_mb = 0

            for i, segment in enumerate(segments):
                if segment.get("local_path") and Path(segment["local_path"]).exists():
                    try:
                        clip = VideoFileClip(segment["local_path"])
                        video_clips.append(clip)
                        total_size_mb += segment.get("file_size_mb", 0)
                        logger.info(f"✅ 加载片段 {i+1}: {segment['local_path']}")
                    except Exception as e:
                        logger.warning(f"⚠️ 无法加载片段 {i+1}: {str(e)}")

            if not video_clips:
                raise Exception("没有可用的视频片段进行拼接")

            # 拼接视频
            final_video = concatenate_videoclips(video_clips, method="compose")

            # 调整到目标时长
            if final_video.duration > target_duration:
                final_video = final_video.subclip(0, target_duration)
            elif final_video.duration < target_duration:
                # 如果时长不够，循环播放
                loops_needed = int(target_duration / final_video.duration) + 1
                repeated_clips = [final_video] * loops_needed
                final_video = concatenate_videoclips(repeated_clips, method="compose")
                final_video = final_video.subclip(0, target_duration)

            # 输出最终视频
            output_path = self.output_dir / f"aliyun_video_{task_id}.mp4"

            logger.info(f"🎬 正在输出拼接视频到: {output_path}")
            final_video.write_videofile(
                str(output_path),
                fps=24,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / f"temp_audio_{task_id}.m4a"),
                remove_temp=True,
                logger=None
            )

            # 清理资源
            for clip in video_clips:
                clip.close()
            final_video.close()

            # 验证输出
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)

                result = {
                    "success": True,
                    "output_path": str(output_path),
                    "duration": target_duration,
                    "resolution": "1280x720",  # 阿里云文生视频分辨率
                    "file_size_mb": round(file_size, 2),
                    "fps": 24,
                    "theme": theme,
                    "keywords": keywords,
                    "video_type": video_type,
                    "generator": "AliyunTextToVideo + MoviePy",
                    "segments_count": len(segments),
                    "segments_total_size_mb": round(total_size_mb, 2),
                    "timestamp": datetime.now().isoformat()
                }

                logger.info(f"✅ 阿里云文生视频拼接成功: {output_path} ({file_size:.2f} MB)")
                return result
            else:
                raise Exception("拼接视频文件未能创建")

        except Exception as e:
            logger.error(f"❌ 视频片段拼接失败: {str(e)}")
            raise

    def _generate_programmatic_video(self,
                                   task_id: str,
                                   description: str,
                                   keywords: List[str],
                                   duration: int,
                                   theme: str,
                                   video_type: str) -> Dict[str, Any]:
        """生成程序化视频（最后的回退方案）"""
        try:
            logger.info(f"🎨 生成程序化视频作为最后回退方案")

            # 创建视频片段
            clips = []
            segment_duration = duration / max(len(keywords) + 2, 1)

            # 开场片段
            intro_clip = self._create_intro_clip(description, segment_duration)
            clips.append(intro_clip)

            # 关键词片段
            for i, keyword in enumerate(keywords):
                keyword_clip = self._create_keyword_clip(keyword, segment_duration, i)
                clips.append(keyword_clip)

            # 结尾片段
            outro_clip = self._create_outro_clip(segment_duration)
            clips.append(outro_clip)

            # 拼接视频
            final_video = concatenate_videoclips(clips, method="compose")

            # 确保时长
            if final_video.duration > duration:
                final_video = final_video.subclip(0, duration)

            # 添加静音音频
            final_video = self._add_background_audio(final_video, duration)

            # 输出视频
            output_path = self.output_dir / f"programmatic_video_{task_id}.mp4"

            logger.info(f"🎬 正在输出程序化视频到: {output_path}")
            final_video.write_videofile(
                str(output_path),
                fps=30,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / f"temp_audio_{task_id}.m4a"),
                remove_temp=True,
                logger=None
            )

            # 清理资源
            for clip in clips:
                clip.close()
            final_video.close()

            # 验证输出
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)

                result = {
                    "success": True,
                    "output_path": str(output_path),
                    "duration": duration,
                    "resolution": "1920x1080",
                    "file_size_mb": round(file_size, 2),
                    "fps": 30,
                    "theme": theme,
                    "keywords": keywords,
                    "video_type": video_type,
                    "generator": "MoviePy Programmatic",
                    "timestamp": datetime.now().isoformat()
                }

                logger.info(f"✅ 程序化视频生成成功: {output_path} ({file_size:.2f} MB)")
                return result
            else:
                raise Exception("程序化视频文件未能创建")

        except Exception as e:
            logger.error(f"❌ 程序化视频生成失败: {str(e)}")
            raise

    def _create_intro_clip(self, title: str, duration: float) -> VideoFileClip:
        """创建开场片段"""
        # 创建蓝色渐变背景
        clip = ColorClip(size=(1920, 1080), color=(30, 100, 200), duration=duration)

        # 添加标题文字
        try:
            title_text = title[:50] if len(title) > 50 else title
            text_clip = TextClip(
                text=title_text,
                font_size=60,
                color='white',
                font='Arial',
                duration=duration
            )
            # Note: set_position not available in MoviePy 2.2.1

            # 合成视频
            clip = CompositeVideoClip([clip, text_clip])

        except Exception as e:
            logger.warning(f"添加标题文字失败: {e}")

        # 暂时不使用淡入效果，直接返回clip
        return clip

    def _create_keyword_clip(self, keyword: str, duration: float, index: int) -> VideoFileClip:
        """创建关键词片段"""
        # 创建不同颜色的背景
        colors = [
            (50, 150, 200),   # 蓝色
            (200, 100, 50),   # 橙色
            (100, 200, 50),   # 绿色
            (200, 50, 150),   # 紫色
            (150, 200, 100)   # 青色
        ]
        color = colors[index % len(colors)]

        clip = ColorClip(size=(1920, 1080), color=color, duration=duration)

        # 添加关键词文字
        try:
            text_clip = TextClip(
                text=keyword,
                font_size=80,
                color='white',
                font='Arial',
                duration=duration
            )
            # Note: set_position not available in MoviePy 2.2.1

            # 添加序号
            number_clip = TextClip(
                text=f"#{index + 1}",
                font_size=40,
                color='white',
                font='Arial',
                duration=duration
            )
            # Note: set_position not available in MoviePy 2.2.1

            # 合成视频
            clip = CompositeVideoClip([clip, text_clip, number_clip])

        except Exception as e:
            logger.warning(f"添加关键词文字失败: {e}")

        return clip

    def _create_outro_clip(self, duration: float) -> VideoFileClip:
        """创建结尾片段"""
        # 创建深色背景
        clip = ColorClip(size=(1920, 1080), color=(20, 20, 20), duration=duration)

        # 添加结尾文字
        try:
            thanks_clip = TextClip(
                text="感谢观看",
                font_size=60,
                color='white',
                font='Arial',
                duration=duration
            )
            # Note: set_position not available in MoviePy 2.2.1

            subtitle_clip = TextClip(
                text="Created with Aura Render",
                font_size=30,
                color='gray',
                font='Arial',
                duration=duration
            )
            # Note: set_position not available in MoviePy 2.2.1

            # 合成视频
            clip = CompositeVideoClip([clip, thanks_clip, subtitle_clip])

        except Exception as e:
            logger.warning(f"添加结尾文字失败: {e}")

        # 暂时不使用淡出效果，直接返回clip
        return clip

    def _add_background_audio(self, video: VideoFileClip, duration: int) -> VideoFileClip:
        """添加背景音频（静音占位符）"""
        try:
            from moviepy import AudioClip

            # 创建静音音频
            def make_frame(t):
                return [0, 0]  # 立体声静音

            audio_clip = AudioClip(make_frame, duration=duration, fps=44100)
            return video.with_audio(audio_clip)

        except Exception as e:
            logger.warning(f"添加背景音频失败: {e}")
            return video

# 全局实例
_moviepy_generator_instance = None

def get_moviepy_video_generator() -> MoviePyVideoGenerator:
    """获取MoviePy视频生成器实例"""
    global _moviepy_generator_instance
    if _moviepy_generator_instance is None:
        _moviepy_generator_instance = MoviePyVideoGenerator()
    return _moviepy_generator_instance

if __name__ == "__main__":
    # 测试生成器
    generator = get_moviepy_video_generator()

    vgp_analysis = {
        "video_type": "商业类",
        "emotions": {"励志": 0.8, "科技": 0.6}
    }

    result = generator.generate_video(
        task_id="moviepy_test_001",
        description="科技创新宣传视频测试",
        keywords=["科技", "创新", "未来"],
        duration=20,
        theme="科技创新",
        vgp_analysis=vgp_analysis
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))