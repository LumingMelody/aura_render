# IMS转换器使用指南

VGP到阿里云IMS的完整集成方案

---

## 📋 目录

1. [快速开始](#快速开始)
2. [API端点](#api端点)
3. [使用示例](#使用示例)
4. [完整流程](#完整流程)
5. [故障排查](#故障排查)

---

## 🚀 快速开始

### 1. 启动服务

```bash
# 启动FastAPI服务
python app.py

# 服务将运行在: http://localhost:8001
```

### 2. 测试IMS转换

```bash
# 运行集成测试
python test_ims_integration.py
```

---

## 📡 API端点

### 1. POST /api/ims/convert

将VGP输出转换为IMS Timeline格式

**请求体:**

```json
{
  "vgp_result": {
    "effects_sequence_id": [...],
    "text_overlay_track_id": {...},
    "auxiliary_track_id": {...}
  },
  "use_filter_preset": true,
  "output_config": {
    "MediaURL": "oss://bucket/output.mp4",
    "Width": 1920,
    "Height": 1080,
    "VideoCodec": "H.264",
    "AudioCodec": "AAC"
  }
}
```

**响应:**

```json
{
  "success": true,
  "timeline": {
    "VideoTracks": [...],
    "EffectTracks": [...],
    "TextTracks": [...]
  },
  "ims_request": {
    "Timeline": {...},
    "OutputMediaConfig": {...}
  },
  "summary": {
    "total_clips": 10,
    "transitions": 9,
    "filters": 10,
    "effects": 5,
    "texts": 3,
    "overlays": 2
  }
}
```

### 2. GET /api/ims/mappings

获取所有VGP到IMS的映射配置

**响应:**

```json
{
  "transitions": {
    "cross_dissolve": "fade",
    "zoom_transition": "simplezoom",
    ...
  },
  "filters": {
    "presets": {
      "cinematic": "m1",
      "vibrant": "pl3",
      ...
    },
    "categories": {...}
  },
  "effects": {...},
  "flower_styles": {...}
}
```

### 3. POST /api/ims/preview

预览IMS转换结果(不实际提交)

**请求体:**

```json
{
  "vgp_result": {...}
}
```

**响应:**

```json
{
  "success": true,
  "summary": {
    "total_clips": 5,
    "transitions": 4,
    ...
  },
  "timeline_preview": {...},
  "recommendations": {
    "use_filter_preset": true,
    "estimated_processing_time": 10,
    "warnings": []
  }
}
```

---

## 💡 使用示例

### 示例1: Python客户端

```python
import requests
import json

# 1. 准备VGP输出
vgp_result = {
    "effects_sequence_id": [
        {
            "source_url": "oss://bucket/video1.mp4",
            "start": 0.0,
            "end": 5.0,
            "transition_out": {
                "type": "cross_dissolve",
                "duration": 1.0
            },
            "color_filter": {
                "preset": "cinematic",
                "intensity": 0.8
            }
        }
    ]
}

# 2. 调用转换API
response = requests.post(
    "http://localhost:8001/api/ims/convert",
    json={
        "vgp_result": vgp_result,
        "use_filter_preset": True,
        "output_config": {
            "MediaURL": "oss://my-bucket/output/video.mp4",
            "Width": 1920,
            "Height": 1080
        }
    }
)

# 3. 获取IMS Timeline
result = response.json()
if result["success"]:
    ims_timeline = result["timeline"]
    summary = result["summary"]

    print(f"转换成功! 共{summary['total_clips']}个片段")
    print(f"IMS Timeline: {json.dumps(ims_timeline, indent=2)}")
else:
    print(f"转换失败: {result['error']}")
```

### 示例2: cURL命令

```bash
# 转换VGP到IMS
curl -X POST http://localhost:8001/api/ims/convert \
  -H "Content-Type: application/json" \
  -d '{
    "vgp_result": {
      "filter_sequence_id": [
        {
          "source_url": "oss://bucket/video.mp4",
          "start": 0.0,
          "end": 10.0,
          "color_filter": {"preset": "cinematic"}
        }
      ]
    },
    "use_filter_preset": true
  }'

# 获取映射配置
curl http://localhost:8001/api/ims/mappings

# 预览转换
curl -X POST http://localhost:8001/api/ims/preview \
  -H "Content-Type: application/json" \
  -d '{"vgp_result": {...}}'
```

### 示例3: JavaScript/TypeScript

```typescript
// 转换VGP到IMS
async function convertToIMS(vgpResult: any) {
  const response = await fetch('http://localhost:8001/api/ims/convert', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      vgp_result: vgpResult,
      use_filter_preset: true,
      output_config: {
        MediaURL: 'oss://bucket/output.mp4',
        Width: 1920,
        Height: 1080,
      },
    }),
  });

  const result = await response.json();

  if (result.success) {
    console.log('转换成功!', result.summary);
    return result.timeline;
  } else {
    throw new Error(result.error);
  }
}
```

---

## 🔄 完整流程

### 方案A: VGP生成 → IMS转换 → 视频合成

```python
import requests

BASE_URL = "http://localhost:8001"

# Step 1: 生成VGP (调用你的VGP生成接口)
vgp_response = requests.post(
    f"{BASE_URL}/api/vgp/generate",  # 你的VGP生成接口
    json={
        "theme": "旅行vlog",
        "duration": 60,
        "style": "cinematic"
    }
)
vgp_result = vgp_response.json()

# Step 2: 转换为IMS Timeline
ims_response = requests.post(
    f"{BASE_URL}/api/ims/convert",
    json={
        "vgp_result": vgp_result,
        "use_filter_preset": True,
        "output_config": {
            "MediaURL": "oss://my-bucket/output/final_video.mp4",
            "Width": 1920,
            "Height": 1080,
            "VideoCodec": "H.264",
            "AudioCodec": "AAC"
        }
    }
)
ims_timeline = ims_response.json()

# Step 3: 提交到阿里云IMS进行视频合成
# (需要集成阿里云IMS SDK)
from alibabacloud_ice20201109 import client as ice_client

ims_client = ice_client.Client(config)
result = ims_client.submit_media_producing_job(
    ims_timeline["ims_request"]
)

print(f"视频合成任务ID: {result.job_id}")
```

### 方案B: 只使用转换器(不调用IMS)

```python
from ims_converter import IMSConverter

# 创建转换器
converter = IMSConverter(use_filter_preset=True)

# 转换VGP输出
ims_timeline = converter.convert(vgp_result)

# 生成IMS请求
ims_request = converter.convert_to_ims_request(
    vgp_result,
    output_config={
        "MediaURL": "oss://bucket/output.mp4",
        "Width": 1920,
        "Height": 1080
    }
)

# 保存为JSON文件
import json
with open('ims_timeline.json', 'w') as f:
    json.dump(ims_request, f, indent=2)
```

---

## 🔧 配置选项

### 滤镜模式选择

**预设模式** (推荐):
```python
{
  "use_filter_preset": true
}
```
- 优点: 简单快速，效果稳定
- 映射: VGP `cinematic` → IMS `m1`

**精确参数模式**:
```python
{
  "use_filter_preset": false
}
```
- 优点: 精确控制色彩参数
- 转换: VGP倍数制 → IMS偏移制

### 输出配置

```python
{
  "output_config": {
    "MediaURL": "oss://bucket/path/video.mp4",  # 必填
    "Width": 1920,                              # 必填
    "Height": 1080,                             # 必填
    "VideoCodec": "H.264",                      # 可选
    "AudioCodec": "AAC",                        # 可选
    "FrameRate": 30,                            # 可选
    "VideoBitrate": "5000",                     # 可选
    "AudioBitrate": "128"                       # 可选
  }
}
```

---

## 🐛 故障排查

### 问题1: 无法连接到服务器

```bash
❌ 错误: requests.exceptions.ConnectionError
```

**解决方案:**
```bash
# 1. 确认服务是否运行
curl http://localhost:8001/health

# 2. 检查端口是否被占用
lsof -i :8001

# 3. 重启服务
python app.py
```

### 问题2: 转换失败

```json
{
  "success": false,
  "error": "KeyError: 'filter_sequence_id'"
}
```

**解决方案:**
- 确保VGP输出包含必要的字段
- 至少需要以下之一:
  - `filter_sequence_id`
  - `effects_sequence_id`
  - `transition_sequence_id`

### 问题3: IMS参数不支持

```
⚠️ 警告: 特效类型 'border_glow' 在IMS中不支持
```

**解决方案:**
- 查看映射表: `GET /api/ims/mappings`
- 使用支持的特效类型
- 或者忽略不支持的特效

### 问题4: 导入错误

```python
ModuleNotFoundError: No module named 'ims_converter'
```

**解决方案:**
```bash
# 确保ims_converter在项目根目录
ls -la ims_converter/

# 应该看到:
# ims_converter/
# ├── __init__.py
# ├── converter.py
# ├── utils.py
# └── configs/
```

---

## 📚 更多资源

- **IMS转换器文档**: `ims_converter/README.md`
- **映射配置**: `ims_converter/configs/mappings.py`
- **使用示例**: `ims_converter_examples.py`
- **测试脚本**: `test_ims_converter.py`
- **集成测试**: `test_ims_integration.py`

---

## ✅ 检查清单

部署前确认:

- [ ] FastAPI服务正常运行
- [ ] IMS转换器已正确安装
- [ ] 所有映射配置已加载
- [ ] 测试用例全部通过
- [ ] 阿里云OSS配置正确
- [ ] IMS SDK已配置 (如需直接调用IMS)

---

## 🎯 下一步

1. **集成IMS SDK**: 实现自动���交到阿里云IMS
2. **添加缓存**: 缓存频繁使用的转换结果
3. **批量转换**: 支持批量VGP输出转换
4. **实时预览**: 添加WebSocket实时预览功能
5. **错误恢复**: 实现转换失败的自动重试机制

---

**需���帮助?**

- 查看文档: `ims_converter/README.md`
- 运行测试: `python test_ims_integration.py`
- 查看日志: `logs/aura_render.log`
