# 素材库集成 - 快速开始指南

## 🚀 快速开始

### 1. 配置素材库认证

编辑 `.env` 文件，添加素材库认证信息：

```bash
# 素材库API认证Token (从素材库管理后台获取)
MATERIAL_LIBRARY_AUTH=你的Authorization_Token

# 素材库API配置 (已预配置，通常不需要修改)
MATERIAL_LIBRARY_HOST=agent.cstlanbaai.com
MATERIAL_LIBRARY_BASE_PATH=/gateway/admin-api/agent/resource/page
```

### 2. 测试素材库连接

运行测试脚本验证配置：

```bash
python test_material_library.py
```

测试将执行：
- ✅ 素材库客户端连接测试
- ✅ 音频素材搜索测试
- ✅ 视频素材搜索测试
- ✅ BGM匹配功能测试
- ✅ 完整工作流集成测试

### 3. 启动服务

```bash
# 开发模式
python3 app.py

# 或使用uvicorn
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### 4. 生成视频

发送请求到 `/vgp/generate` 接口：

```bash
curl -X POST "http://localhost:8001/vgp/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "1",
    "id": "test_001",
    "theme_id": "产品展示",
    "user_description_id": "智能投影仪产品展示",
    "target_duration_id": 10,
    "keywords_id": ["智能投影仪", "4K", "便携"],
    "reference_media": {
      "product_images": [
        {
          "url": "https://example.com/product.jpg",
          "type": "product"
        }
      ]
    }
  }'
```

**关键字段说明：**
- `tenant_id`: **必填**，用于素材库认证
- `id`: 业务ID，用于关联
- 其他字段参考VGP接口文档

## 📋 工作流程

### BGM匹配流程

```
用户请求 (/vgp/generate)
    ↓
提取 tenant_id
    ↓
初始化素材库客户端
    ↓
执行VGP工作流
    ↓
Node 5: BGM合成节点
    ├─ 分析情绪: "冷静"
    ├─ 分析风格: "极简电子"
    ├─ 构造搜索策略
    └─ 调用素材库API
        ├─ 策略1: tag=极简电子 ✅ 找到3个
        ├─ 获取音频时长 (ffprobe)
        ├─ 随机裁剪片段
        └─ 返回BGM URL
    ↓
IMS转换器
    ├─ 检查BGM URL有效性
    ├─ 过滤占位符URL
    └─ 生成最终Timeline
    ↓
视频渲染完成
```

## 🔍 日志监控

### 查看BGM匹配日志

```bash
# 实时监控
tail -f logs/aura_render.log | grep "🎵"

# 查看素材库相关日志
tail -f logs/aura_render.log | grep "素材库"
```

### 关键日志标记

| 标记 | 含义 | 示例 |
|------|------|------|
| 🎵 | BGM匹配 | `🎵 开始匹配BGM: category=极简电子...` |
| ✅ | 成功 | `✅ 策略 1 找到 3 个候选音频` |
| ⚠️ | 警告 | `⚠️ 所有BGM搜索策略都失败` |
| ❌ | 错误 | `❌ 素材库API调用失败` |

## 💡 使用示例

### 示例1: 科技产品视频

```json
{
  "tenant_id": "1",
  "theme_id": "产品展示",
  "user_description_id": "智能手表展示，科技感",
  "target_duration_id": 15,
  "keywords_id": ["智能手表", "健康", "运动"],
  "reference_media": {
    "product_images": [
      {"url": "https://...", "type": "product"}
    ]
  }
}
```

**预期BGM：**
- 风格：电子、科技
- 情绪：冷静、专业
- 节奏：中等 (80-100 BPM)

### 示例2: 温馨生活视频

```json
{
  "tenant_id": "1",
  "theme_id": "生活记录",
  "user_description_id": "家庭温馨时刻，轻松氛围",
  "target_duration_id": 20,
  "keywords_id": ["家庭", "温馨", "日常"]
}
```

**预期BGM：**
- 风格：轻音乐、钢琴曲
- 情绪：温馨、放松
- 节奏：慢速 (60-80 BPM)

### 示例3: 励志视频

```json
{
  "tenant_id": "1",
  "theme_id": "励志短片",
  "user_description_id": "个人成长历程，激励人心",
  "target_duration_id": 30,
  "keywords_id": ["成长", "梦想", "努力"]
}
```

**预期BGM：**
- 风格：流行、管弦乐
- 情绪：励志、激昂
- 节奏：快速 (100-120 BPM)

## 🛠️ 故障排查

### 问题1: BGM没有声音

**症状：** 生成的视频没有背景音乐

**排查步骤：**

```bash
# 1. 检查日志
tail -100 logs/aura_render.log | grep "BGM"

# 2. 查找关键信息
# - "⚠️ 所有BGM搜索策略都失败" → 素材库中没有匹配的音频
# - "⚠️ 跳过无效的BGM URL" → 匹配到的是占位符URL
# - "❌ 素材库API调用失败" → 认证或网络问题
```

**解决方案：**

1. **认证问题:**
   ```bash
   # 检查 .env 配置
   grep MATERIAL_LIBRARY_AUTH .env

   # 确认token有效性
   python test_material_library.py
   ```

2. **标签匹配问题:**
   - 检查素材库中是否有对应的tag
   - 尝试添加更多通用的音频标签（如"背景音乐"）

3. **网络问题:**
   ```bash
   # 测试网络连通性
   curl -v https://agent.cstlanbaai.com
   ```

### 问题2: 花字太大

**症状：** 视频中的文字遮挡画面

**解决：** 已在代码中调整，花字大小为原来的一半

如需进一步调整，修改 `ims_converter/utils.py:361-370`

### 问题3: 素材库API返回空

**症状：** API调用成功但返回结果为空

**可能原因：**
1. 使用的tag在素材库中不存在
2. tenant_id权限不足

**解决：**
```python
# 测试不同的tag
python test_material_library.py

# 查看API原始返回
# 在 material_library_client.py 中启用debug日志
logger.setLevel(logging.DEBUG)
```

## 📊 性能优化建议

### 1. 音频时长缓存

当前每次都用ffprobe获取时长，可以添加缓存：

```python
# 在 bgm_matcher.py 中
_duration_cache = {}

async def _get_audio_duration(audio_url: str) -> float:
    if audio_url in _duration_cache:
        return _duration_cache[audio_url]

    duration = await _fetch_duration(audio_url)
    _duration_cache[audio_url] = duration
    return duration
```

### 2. 素材搜索结果缓存

对于相同的搜索条件，可以缓存结果：

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def search_audios_cached(tag: str, keyword: str):
    return client.search_audios(tag=tag, keyword=keyword)
```

### 3. 并发处理

如果需要匹配多个BGM片段，可以并发请求：

```python
tasks = [match_bgm(request) for request in bgm_requests]
results = await asyncio.gather(*tasks)
```

## 🔐 安全建议

1. **不要提交 .env 到版本控制**
   ```bash
   # 确认 .env 在 .gitignore 中
   echo ".env" >> .gitignore
   ```

2. **使用环境变量管理敏感信息**
   ```bash
   # 生产环境
   export MATERIAL_LIBRARY_AUTH="production_token"
   ```

3. **定期更新认证Token**
   - 建议每月更换一次
   - 发现泄露立即更换

## 📚 相关文档

- [素材库配置说明](MATERIAL_LIBRARY_SETUP.md)
- [VGP工作流文档](../vgp_documents/)
- [IMS转换器文档](../ims_converter/)

## 🆘 获取帮助

遇到问题？

1. 查看日志: `logs/aura_render.log`
2. 运行测试: `python test_material_library.py`
3. 查看文档: `docs/MATERIAL_LIBRARY_SETUP.md`
4. 检查配置: `.env` 文件
