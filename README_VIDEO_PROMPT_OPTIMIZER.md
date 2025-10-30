# 视频生成提示词优化系统

## 概述

本系统实现了`视频生成.md`文档中描述的完整12步提示词优化流程，将简单的产品名称转换为专业的、结构化的视频生成提示词。

## 功能特点

### ✅ 已实现的12步优化流程

1. **全局产品描述** - 提取产品核心特性和用途
2. **宣传偏好分析** - 分析宣传雷点和目标受众偏好
3. **产品时代偏好** - 判断适合的时代背景（现代/传统/复古/未来）
4. **故事线分镜设计** - 基于「惊鸿一瞥」高端品牌片规范设计分镜
5. **全局要素统一** - 建立视觉基因（色彩、光影、材质、运动规则）
6. **片段分割** - 判断镜头连续性
7. **首帧和中间过程描述** - 分离静态画面和动态过程
8. **首帧细化** - 60字限定，结构化描述（主体+环境+构图+光影+色彩+风格+情绪）
9. **去括号优化** - 清理冗余描述
10. **一致性检查** - 判断图生图策略（产品图/前一帧/独立生成）
11. **中间过程细化** - 专业运镜描述（推拉摇移升降）
12. **去括号优化** - 最终清理

### 🎨 提升点对比

| 项目 | 旧方式（简单拼接） | 新方式（12步优化） |
|------|------------------|------------------|
| **提示词长度** | 一句话 | 详细的多段式描述 |
| **风格控制** | 仅风格标签（如cinematic） | 视觉基因、色彩调色板、光影规则、材质语言 |
| **运镜描述** | 无 | 推拉摇移升降等专业术语 |
| **构图指导** | 无 | 特写/中景/全景、俯仰角、黄金分割 |
| **光影描述** | 无 | 硬光/柔光/体积光/伦勃朗光 |
| **连续性保证** | 每次独立生成 | 图生图判断、场景一致性检查 |
| **首帧细化** | 无 | 结构化模板（60字限定） |

## 快速开始

### 1. 安装依赖

```bash
pip install asyncio aiohttp
```

### 2. 基础使用

#### 方式1: 仅生成优化的提示词（不调用视频API）

```python
import asyncio
from video_generate_protocol.prompt_optimizer import VideoPromptOptimizer

async def main():
    # 初始化优化器
    optimizer = VideoPromptOptimizer()

    # 优化提示词
    result = await optimizer.optimize(
        product_name="智能手表",
        user_input="强调科技感和运动场景"  # 可选
    )

    # 查看结果
    print(f"产品描述: {result.product_description}")
    print(f"视觉风格: {result.visual_style.target_style}")
    print(f"分镜数量: {len(result.storyboard)}")

    # 查看每个分镜的细化提示词
    for shot in result.storyboard:
        print(f"\n镜头{shot.shot_index}:")
        print(f"  首帧: {shot.first_frame_clean}")
        print(f"  运动: {shot.middle_process_clean}")
        print(f"  策略: {shot.generation_strategy}")

asyncio.run(main())
```

#### 方式2: 完整流程（包括视频生成）

```python
import asyncio
import os
from video_generate_protocol.nodes.optimized_video_generation_node import OptimizedVideoGenerationNode

async def main():
    # 初始化节点（需要API密钥）
    node = OptimizedVideoGenerationNode(
        qwen_api_key=os.getenv("DASHSCOPE_API_KEY")
    )

    # 准备上下文
    context = {
        "product_name": "有机茶叶",
        "product_image_url": "https://example.com/tea.jpg",  # 可选
        "user_requirements": "强调自然、传统工艺",  # 可选
        "output_dir": "/tmp/video_output",
        "merge_clips": True  # 是否合并视频片段
    }

    # 执行生成
    result = await node.execute(context)

    if result["success"]:
        print(f"✅ 生成成功！")
        print(f"视频片段: {result['clips_count']}个")
        print(f"最终视频: {result.get('final_video_url')}")

asyncio.run(main())
```

### 3. 运行测试

```bash
# 测试提示词优化器（不需要API密钥）
python test_optimized_video_generation.py

# 完整测试（需要设置API密钥）
export DASHSCOPE_API_KEY=your_api_key_here
python test_optimized_video_generation.py
```

## 核心模块说明

### 1. VideoPromptOptimizer (`prompt_optimizer.py`)

提示词优化器，负责执行12步优化流程。

**主要方法:**
- `optimize(product_name, user_input)` - 执行完整优化
- 内部方法: `_step1_...` 到 `_step11_...` 对应12个步骤

**返回数据结构:**
```python
OptimizedPromptResult(
    product_description: str,        # 产品描述
    marketing_analysis: Dict,         # 宣传分析
    era_preference: str,              # 时代偏好
    visual_style: VisualStyle,        # 视觉风格
    storyboard: List[StoryboardShot], # 分镜列表
    total_duration: float             # 总时长
)
```

### 2. OptimizedVideoGenerationNode (`optimized_video_generation_node.py`)

完整的视频生成节点，集成提示词优化和视频生成。

**执行流程:**
1. 调用`VideoPromptOptimizer`生成优化提示词
2. 转换为视频生成参数
3. 调用`StoryboardToVideoProcessor`生成视频
4. 可选：合并视频片段

**输入参数:**
- `product_name` (必需): 产品名称
- `product_image_url` (可选): 产品图片URL
- `user_requirements` (可选): 用户需求描述
- `output_dir` (可选): 输出目录
- `merge_clips` (可选): 是否合并视频

### 3. StoryboardToVideoProcessor (`qwen_integration.py`)

视频生成处理器，已增强支持`video_prompt`参数。

**改进点:**
- 支持单独的`video_prompt`字段（运动描述）
- 优先使用`video_prompt`，否则使用`refined_prompt`

## 数据流示意图

```
产品名称 "智能手表"
    ↓
[步骤1-3] 产品分析
    ↓
产品描述 + 宣传偏好 + 时代背景
    ↓
[步骤4] 分镜设计
    ↓
6-8个原始分镜
    ↓
[步骤5] 视觉统一
    ↓
全局视觉基因（色彩/光影/材质/运动）
    ↓
[步骤6] 连续性判断
    ↓
标记连续/独立镜头
    ↓
[步骤7-9] 首帧细化
    ↓
60字结构化首帧描述
    ↓
[步骤10] 一致性检查
    ↓
图生图策略（产品图/前帧/独立）
    ↓
[步骤11-12] 中间过程细化
    ↓
专业运镜描述
    ↓
完整的优化提示词
    ↓
视频生成API
    ↓
最终视频
```

## 输出示例

### 优化前（旧方式）
```
[风格要求: realistic] 请生成一个连续的视频片段: 智能手表特写
```

### 优化后（新方式）

**首帧提示词:**
```
智能手表表盘特写，OLED屏幕发出柔和蓝光，中景构图，浅景深，侧光从右侧照射形成轮廓光，主色调为深空灰与科技蓝，玻璃表面有细腻的反射，极简主义风格
```

**运动提示词:**
```
镜头缓慢推进，从中景过渡到特写，焦点从表盘边缘平滑移动到屏幕中心，光影随镜头移动产生微妙变化，运动节奏舒缓平稳，强调产品细节质感
```

**视觉风格约束:**
- 核心主题: 科技与人性的无缝连接
- 核心情绪: 静谧的期待
- 主色调: 深空灰、科技蓝
- 光源: 定向侧光 + 屏幕补光
- 风格: 极简主义

## 测试结果

运行`test_optimized_video_generation.py`会执行3个测试：

1. **提示词优化器测试** - 验证12步流程是否正确执行
2. **完整集成测试** - 验证从提示词到视频的完整流程（需要API）
3. **对比测试** - 展示优化前后的提示词差异

## 注意事项

1. **LLM调用频率**: 12步流程会多次调用LLM，建议控制并发
2. **API限流**: 视频生成API有限流，代码已添加2秒延迟
3. **提示词长度**: 首帧限制60字，确保API兼容性
4. **图生图策略**: 第一个镜头使用产品图，后续镜头参考前帧
5. **错误处理**: 每步都有fallback机制，失败时使用默认值

## 目录结构

```
video_generate_protocol/
├── prompt_optimizer.py                    # 提示词优化器（12步流程）
├── nodes/
│   ├── optimized_video_generation_node.py # 完整的视频生成节点
│   └── qwen_integration.py                # 视频生成处理器（已增强）
test_optimized_video_generation.py         # 测试脚本
README_VIDEO_PROMPT_OPTIMIZER.md           # 本文档
```

## 下一步改进

- [ ] 支持批量产品生成
- [ ] 添加提示词缓存机制
- [ ] 支持自定义视觉风格模板
- [ ] 优化LLM调用次数（合并某些步骤）
- [ ] 添加提示词评分和质量检测
- [ ] 支持多语言提示词生成

## 常见问题

**Q: 为什么不直接使用简单的提示词？**
A: 简单提示词缺乏视觉一致性、运镜指导和构图细节，生成的视频质量和连贯性较差。12步优化提供了结构化、专业化的指导。

**Q: 生成一个视频需要多长时间？**
A: 提示词优化约1-2分钟（取决于LLM速度），视频生成取决于片段数量和API速度，通常5-10分钟。

**Q: 可以跳过某些优化步骤吗？**
A: 不建议。12步是相互关联的，跳过会影响最终质量。如果需要加速，可以考虑缓存某些步骤的结果。

**Q: 支持哪些产品类型？**
A: 理论上支持所有产品，优化器会自动分析产品类别并调整宣传策略。

## 技术支持

如有问题，请查看：
1. 日志输出（包含详细的执行信息）
2. 测试脚本（`test_optimized_video_generation.py`）
3. 原始文档（`视频生成.md`）
