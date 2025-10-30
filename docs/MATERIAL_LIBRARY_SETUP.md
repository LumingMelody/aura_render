# 素材库集成配置说明

## 概述

已集成真实的素材库API，用于BGM和视频素材的智能匹配。

## 接口信息

- **域名**: `https://agent.cstlanbaai.com`
- **端点**: `/gateway/admin-api/agent/resource/page`
- **请求方式**: GET

## 认证配置

### 1. Authorization Token

在 `.env` 文件中添加：

```bash
MATERIAL_LIBRARY_AUTH=你的固定Authorization值
```

### 2. Tenant ID

Tenant ID 会自动从 `/vgp/generate` 请求中提取：

```json
{
  "tenant_id": "1",
  "id": "404",
  ...
}
```

## 素材类型

| type | 说明 | 用途 |
|------|------|------|
| 1 | 视频素材库 | 视频片段匹配 |
| 2 | 音频素材库 | BGM音乐匹配 |

## BGM匹配策略

BGM匹配使用多级fallback策略：

1. **优先**: 精确风格匹配 (`tag=极简电子`)
2. **次选**: 情绪匹配 (`tag=冷静`)
3. **组合**: 风格+情绪 (`tag=极简电子&name=冷静`)
4. **兜底**: 任意BGM (`name=背景音乐`)

### 示例流程

```
输入: mood="冷静", genre="极简电子 / Lo-fi", duration=5秒

策略1: tag="极简电子"
  → 找到3个候选 ✅
  → 使用ffprobe获取时长
  → 随机裁剪5秒片段
  → 返回BGM URL

如果策略1失败 → 尝试策略2 (tag="冷静")
如果策略2失败 → 尝试策略3 (组合搜索)
如果策略3失败 → 尝试策略4 (兜底)
如果全部失败 → 返回空列表（视频无BGM）
```

## 音频时长获取

使用 `ffprobe` 获取真实音频时长：

```bash
ffprobe -v error -show_entries format=duration -of json <audio_url>
```

如果 `ffprobe` 失败，使用默认时长 120秒。

## 修改的文件

### 1. 新增文件

- `materials_supplies/material_library_client.py` - 素材库API客户端

### 2. 修改文件

- `materials_supplies/matcher/bgm_matcher.py` - BGM匹配逻辑
- `vgp_api.py` - 初始化素材库客户端
- `ims_converter/converter.py` - 过滤无效BGM URL
- `ims_converter/utils.py` - 花字大小调整

## 测试素材库连接

### 手动测试

```python
from materials_supplies.material_library_client import MaterialLibraryClient

# 初始化客户端
client = MaterialLibraryClient(
    tenant_id="1",
    authorization="你的token"
)

# 搜索BGM
audios = client.search_audios(tag="冷静", page_size=5)
print(f"找到 {len(audios)} 个音频")
for audio in audios:
    print(f"  - {audio['name']}: {audio['url']}")

# 搜索视频
videos = client.search_videos(tag="科技", page_size=5)
print(f"找到 {len(videos)} 个视频")
```

## 日志监控

查看素材匹配日志：

```bash
tail -f logs/aura_render.log | grep "🎵\|素材库"
```

关键日志标记：
- `🎵` - BGM匹配相关
- `✅` - 成功
- `⚠️` - 警告（如搜索失败、URL无效）
- `❌` - 错误

## 故障排查

### 1. BGM没有声音

检查日志：
```
⚠️ 所有BGM搜索策略都失败，返回空列表
```

**原因**:
- 素材库中没有匹配的标签
- Authorization未配置或无效

**解决**:
- 检查素材库中的tag是否包含所需标签
- 确认 `.env` 中的 `MATERIAL_LIBRARY_AUTH` 配置正确

### 2. 花字太大

已在 `ims_converter/utils.py` 中调整花字大小：
- 小字: 20
- 中等字: 28 (默认)
- 大字: 35
- 超大字: 43

### 3. 素材库连接失败

检查日志：
```
❌ 素材库API调用失败: [error message]
```

**排查**:
1. 确认网络可以访问 `agent.cstlanbaai.com`
2. 确认 `tenant_id` 和 `Authorization` 正确
3. 检查接口返回的 code 和 msg

## 环境变量总结

```bash
# 素材库认证
MATERIAL_LIBRARY_AUTH=你的token

# 千问API (现有)
DASHSCOPE_API_KEY=你的密钥

# OSS配置 (现有)
OSS_ACCESS_KEY_ID=xxx
OSS_ACCESS_KEY_SECRET=xxx
```

## 未来扩展

### 1. 视频素材匹配

当前视频素材通过万相AI生成。如需使用素材库：

修改 `materials_supplies/matcher/intelligent_video_matcher.py`，在AI生成前先调用素材库搜索。

### 2. 增强搜索

- 添加更多搜索策略（BPM匹配、乐器匹配等）
- 使用AI评分选择最佳候选
- 缓存搜索结果提升性能

### 3. 音频预览

- 添加音频裁剪预览功能
- 支持用户手动选择BGM

## 相关文档

- [VGP工作流文档](vgp_documents/)
- [IMS转换器文档](../ims_converter/)
- [API接口文档](https://agent.cstlanbaai.com/doc.html)
