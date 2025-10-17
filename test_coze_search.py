#!/usr/bin/env python3
"""
测试 Coze 图片搜索功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.cliptemplate.coze.image_search import (
    CozeImageSearcher,
    search_reference_image_from_coze,
    extract_search_keywords
)


async def test_keyword_extraction():
    """测试关键词提取功能"""
    print("=" * 80)
    print("测试关键词提取功能")
    print("=" * 80)

    test_cases = [
        "制作一个苹果手机宣传视频",
        "生成60秒的科技产品介绍短视频，重点展示AI功能",
        "帮我创建一个关于环保的公益广告",
        "我想要一段展示公司文化的企业宣传片",
    ]

    for description in test_cases:
        keywords = await extract_search_keywords(description)
        print(f"📝 描述: {description}")
        print(f"🔑 关键词: {keywords}")
        print()

    print("=" * 80 + "\n")


async def test_coze_image_search():
    """测试 Coze 图片搜索"""

    # 先测试关键词提取
    await test_keyword_extraction()

    print("=" * 80)
    print("测试 Coze 图片搜索功能（带关键词提取）")
    print("=" * 80)

    # 测试查询（使用完整描述）
    test_queries = [
        "制作一个苹果手机宣传视频",
        "生成科技创新产品的介绍视频",
        "帮我创建dota2游戏高光时刻剪辑",
    ]

    for query in test_queries:
        print(f"\n🔍 原始描述: {query}")
        print("-" * 80)

        try:
            # 测试搜索并返回随机图片（会自动提取关键词）
            image_url = await search_reference_image_from_coze(query, extract_keywords=True)

            if image_url:
                print(f"✅ 搜索成功")
                print(f"📸 图片URL: {image_url}")
            else:
                print(f"⚠️ 未搜索到图片")

        except Exception as e:
            print(f"❌ 搜索失败: {e}")

    print("\n" + "=" * 80)

    # 测试搜索多张图片
    print("\n测试搜索多张图片")
    print("=" * 80)

    try:
        searcher = CozeImageSearcher()
        images = await searcher.search_images("产品展示", max_results=5)

        if images:
            print(f"✅ 搜索到 {len(images)} 张图片:")
            for idx, img in enumerate(images, 1):
                print(f"{idx}. {img['title'][:50]}...")
                print(f"   URL: {img['display_url']}")
        else:
            print("⚠️ 未搜索到图片")

    except Exception as e:
        print(f"❌ 搜索失败: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_coze_image_search())
