# Coze 图片搜索模块

## 功能介绍

从 Coze 工作流自动搜索参考图片，用于视频生成。

## 核心特性

### 1. 智能关键词提取

自动从用户描述中提取核心关键词，提升搜索准确度。

**示例：**
```
输入: "制作一个苹果手机宣传视频"
提取: "苹果手机"

输入: "生成60秒的科技产品介绍短视频，重点展示AI功能"
提取: "科技产品 AI功能"
```

**提取策略：**
- **首选方案：** 使用千问 API 智能提取（准确度高）
- **备用方案：** 规则匹配提取（移除动作词、时长、介质词等）
- **最终兜底：** 返回前30个字符

### 2. 随机图片选择

从搜索结果中随机选择一张图片，增加多样性。

### 3. 完整错误处理

- Coze API 调用失败不会阻断视频生成流程
- 多层 fallback 机制保证系统稳定性

## 使用方法

### 基础用法

```python
from core.cliptemplate.coze.image_search import search_reference_image_from_coze

# 自动提取关键词并搜索
image_url = await search_reference_image_from_coze("制作一个苹果手机宣传视频")

# 不提取关键词，直接搜索
image_url = await search_reference_image_from_coze("苹果手机", extract_keywords=False)
```

### 高级用法

```python
from core.cliptemplate.coze.image_search import CozeImageSearcher, extract_search_keywords

# 单独提取关键词
keywords = await extract_search_keywords("制作一个苹果手机宣传视频")
print(keywords)  # "苹果手机"

# 搜索多张图片
searcher = CozeImageSearcher()
images = await searcher.search_images("苹果手机", max_results=10)
for img in images:
    print(img['display_url'], img['title'])
```

## 配置说明

### 环境变量

```bash
# 千问 API（用于关键词提取）
DASHSCOPE_API_KEY=your_api_key
```

### Coze 配置

在代码中配置（也可以改为环境变量）：
- Token: `pat_cwIbrVcSP2ac6oTaCCdyVZ1qvc5tIse5fyGaCtZsftPIyNyippcQy4rzlEuFc85G`
- Workflow ID: `7561281578149642279`
- Base URL: `https://api.coze.cn`

## 集成说明

该模块已集成到 `app.py` 的视频生成流程中：

1. 检查用户是否上传产品图片
2. 如果没有，自动从 Coze 搜索
3. 使用 `user_description_id` 提取关键词
4. 调用 Coze 工作流搜索图片
5. 随机选择一张添加到 `reference_media`
6. 后续 VGP 节点自动使用该参考图片

## 测试

```bash
# 运行测试脚本
python3 test_coze_search.py
```

测试内容包括：
- 关键词提取测试
- 单张图片搜索测试
- 多张图片搜索测试

## 数据格式

### Coze 返回格式

```json
{
  "result": [
    {
      "picture_info": {
        "display_url": "https://...",
        "title": "图片标题",
        "size": {"height": "720", "width": "1280"},
        "right_protect": "版权信息"
      }
    }
  ]
}
```

### 提取的图片信息

```python
{
    'display_url': 'https://...',
    'title': '图片标题',
    'size': {'height': '720', 'width': '1280'},
    'right_protect': '版权信息'
}
```

## 性能说明

- 关键词提取（千问 API）：~1-2秒
- Coze 图片搜索：~1-2秒
- 总计增加延迟：~2-4秒

## 注意事项

1. **关键词长度：** 提取的关键词控制在 20 个字符以内
2. **图片数量：** 默认搜索 10 张，随机选 1 张
3. **错误处理：** 搜索失败不影响主流程，只是不添加参考图片
4. **API 限制：** 注意 Coze 和千问的 API 调用频率限制

## 优化建议

1. 可以根据视频类型调整关键词提取策略
2. 可以添加图片质量过滤（尺寸、格式等）
3. 可以缓存热门搜索词的结果
4. 可以支持多张图片组合参考
