"""
Coze 图片搜索模块

根据用户描述从 Coze 工作流搜索相关图片
"""

import json
import random
import logging
import re
from typing import Optional, List, Dict, Any

from cozepy import Coze, TokenAuth, COZE_CN_BASE_URL

# 使用项目的日志系统
try:
    from utils.logger import get_logger, LogCategory
    logger = get_logger("coze.image_search").with_context(category=LogCategory.SYSTEM)
except ImportError:
    # Fallback 到标准 logger
    logger = logging.getLogger(__name__)


async def extract_search_keywords(description: str) -> str:
    """
    从用户描述中提取搜索关键词

    使用千问 API 提取核心关键词，如果 API 不可用则使用规则提取

    Args:
        description: 用户描述文本

    Returns:
        提取的关键词字符串

    Examples:
        >>> await extract_search_keywords("制作一个苹果手机宣传视频")
        "苹果手机"
        >>> await extract_search_keywords("生成60秒的科技产品介绍短视频，重点展示AI功能")
        "科技产品 AI功能"
    """
    try:
        # 尝试使用千问 API 提取关键词
        import os
        dashscope_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('AI__DASHSCOPE_API_KEY')

        if dashscope_key:
            try:
                import dashscope
                from dashscope import Generation

                dashscope.api_key = dashscope_key

                prompt = f"""从以下视频描述中提取核心搜索关键词，只返回最关键的2-3个词，用空格分隔。
不要返回"制作"、"生成"、"视频"等动作词和介质词，只返回主题名词。

描述：{description}

关键词："""

                response = Generation.call(
                    model='qwen-turbo',
                    prompt=prompt,
                    max_tokens=50
                )

                if response.status_code == 200:
                    keywords = response.output.text.strip()
                    # 清理可能的额外字符
                    keywords = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', keywords)
                    logger.info(f"✅ AI 提取关键词: '{description}' -> '{keywords}'")
                    return keywords

            except Exception as e:
                logger.warning(f"⚠️ AI 关键词提取失败，使用规则提取: {e}")

        # Fallback: 使用规则提取关键词
        keywords = _extract_keywords_by_rules(description)
        logger.info(f"📝 规则提取关键词: '{description}' -> '{keywords}'")
        return keywords

    except Exception as e:
        logger.error(f"❌ 关键词提取失败: {e}")
        # 最后的 fallback：返回前30个字符
        return description[:30]


def _extract_keywords_by_rules(description: str) -> str:
    """
    使用规则提取关键词（Fallback 方法）

    规则：
    1. 移除常见动作词（制作、生成、创建等）
    2. 移除时长描述（60秒、一分钟等）
    3. 移除介质词（视频、短视频、宣传片等）
    4. 保留核心名词和形容词

    Args:
        description: 用户描述

    Returns:
        提取的关键词
    """
    # 移除常见的无用词
    remove_patterns = [
        r'制作(一个|一段)?',
        r'生成(一个|一段)?',
        r'创建(一个|一段)?',
        r'帮我',
        r'我想',
        r'\d+秒(的)?',
        r'一分钟(的)?',
        r'(宣传)?视频',
        r'短视频',
        r'宣传片',
        r'广告片',
        r'的',
        r'，',
        r'。',
    ]

    text = description
    for pattern in remove_patterns:
        text = re.sub(pattern, ' ', text)

    # 清理多余空格
    text = ' '.join(text.split())

    # 取前20个字符作为关键词
    keywords = text[:20].strip()

    return keywords if keywords else description[:20]


class CozeImageSearcher:
    """Coze 图片搜索器"""

    def __init__(
        self,
        token: str = "pat_cwIbrVcSP2ac6oTaCCdyVZ1qvc5tIse5fyGaCtZsftPIyNyippcQy4rzlEuFc85G",
        base_url: str = COZE_CN_BASE_URL,
        workflow_id: str = "7561281578149642279"
    ):
        """
        初始化 Coze 图片搜索器

        Args:
            token: Coze API token
            base_url: Coze API base URL
            workflow_id: 图片搜索工作流ID
        """
        self.coze = Coze(auth=TokenAuth(token=token), base_url=base_url)
        self.workflow_id = workflow_id

    async def search_images(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        根据查询搜索图片

        Args:
            query: 搜索查询（通常是用户描述）
            max_results: 最多返回的图片数量

        Returns:
            图片信息列表，每个元素包含 display_url, title, size 等

        Example:
            >>> searcher = CozeImageSearcher()
            >>> images = await searcher.search_images("产品展示视频")
            >>> print(images[0]['display_url'])
        """
        try:
            logger.info(f"🔍 开始从 Coze 搜索图片，查询: {query}")

            # 调用 Coze 工作流
            # 注意：参数名应该是 "input" 而不是 "query"
            workflow = self.coze.workflows.runs.create(
                workflow_id=self.workflow_id,
                parameters={"input": query}
            )

            # 解析返回结果
            response = json.loads(workflow.data)

            # 提取图片列表（注意：响应的键名是 "output" 而不是 "result"）
            result = response.get('output', [])

            if not result:
                logger.warning("⚠️ Coze 未返回任何图片")
                return []

            # 解析图片信息
            images = []
            for item in result[:max_results]:
                picture_info = item.get('picture_info', {})
                if picture_info and 'display_url' in picture_info:
                    images.append({
                        'display_url': picture_info.get('display_url'),
                        'title': picture_info.get('title', ''),
                        'size': picture_info.get('size', {}),
                        'right_protect': picture_info.get('right_protect', '')
                    })

            logger.info(f"✅ Coze 搜索到 {len(images)} 张图片")
            return images

        except Exception as e:
            logger.error(f"❌ Coze 图片搜索失败: {e}")
            return []

    async def search_random_image(self, query: str, extract_keywords: bool = True) -> Optional[str]:
        """
        搜索并随机返回一张图片的 URL

        Args:
            query: 搜索查询（可能是完整的描述）
            extract_keywords: 是否先提取关键词（默认True）

        Returns:
            随机选择的图片 URL，如果搜索失败则返回 None

        Example:
            >>> searcher = CozeImageSearcher()
            >>> image_url = await searcher.search_random_image("制作一个苹果手机宣传视频")
            >>> print(image_url)
        """
        # 提取关键词优化搜索
        search_query = query
        if extract_keywords:
            search_query = await extract_search_keywords(query)
            logger.info(f"🔑 搜索关键词: '{search_query}' (原始: '{query[:50]}...')")

        images = await self.search_images(search_query, max_results=10)

        if not images:
            return None

        # 随机选择一张图片
        selected_image = random.choice(images)
        display_url = selected_image['display_url']

        logger.info(f"🎲 随机选择图片: {selected_image.get('title', 'Untitled')}")
        logger.info(f"📸 图片URL: {display_url}")

        return display_url


# 全局实例（单例模式）
_coze_image_searcher = None


def get_coze_image_searcher() -> CozeImageSearcher:
    """获取全局 Coze 图片搜索器实例"""
    global _coze_image_searcher
    if _coze_image_searcher is None:
        _coze_image_searcher = CozeImageSearcher()
    return _coze_image_searcher


async def search_reference_image_from_coze(query: str, extract_keywords: bool = True) -> Optional[str]:
    """
    便捷函数：从 Coze 搜索参考图片

    Args:
        query: 搜索查询（用户描述，可能很长）
        extract_keywords: 是否先提取关键词优化搜索（默认True）

    Returns:
        随机选择的图片 URL，如果搜索失败则返回 None

    Examples:
        >>> await search_reference_image_from_coze("制作一个苹果手机宣传视频")
        # 会先提取"苹果手机"作为关键词搜索
    """
    searcher = get_coze_image_searcher()
    return await searcher.search_random_image(query, extract_keywords=extract_keywords)
