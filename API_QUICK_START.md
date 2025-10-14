# VGP新工作流API - 快速开始

## 🎯 问题解决

之前 `/generate` 接口**不支持新工作流**，现在有两个解决方案：

### ✅ 方案1：使用新的 `/vgp/generate` 接口（推荐）

这是专门为新工作流创建的API接口。

### ⚠️ 方案2：使用旧的 `/generate` 接口（不支持新流程）

旧接口使用固定逻辑，不支持模板参数。

---

## 🚀 使用新API接口

### 接口地址

```
POST /vgp/generate
```

### 正确的请求格式

```json
{
    "theme_id": "产品展示",
    "user_description_id": "展示智能投影仪的完整功能演示。首先展示投影仪的外观设计，黑色磨砂质感的机身。然后展示开机投影，自动对焦。接着演示在白墙上投射4K画面。展示多种使用场景：客厅观影、办公演示。最后展示智能功能：语音控制、无线投屏。",
    "target_duration_id": 20,
    "keywords_id": [
        "智能投影仪",
        "4K高清",
        "便携",
        "语音控制"
    ],
    "reference_media": {
        "product_images": [
            {
                "url": "https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/%E6%B5%8B%E8%AF%95%E5%95%86%E5%93%81.jpg",
                "type": "product",
                "weight": 1.0
            }
        ]
    },
    "template": "vgp_new_pipeline",
    "max_parallel_nodes": 5,
    "total_timeout": 3600.0,
    "auto_retry": true,
    "enable_monitoring": true,
    "session_id": "session_001",
    "user_id": "user_001"
}
```

### 响应示例

```json
{
    "success": true,
    "instance_id": "workflow_abc123",
    "task_id": "task_xyz789",
    "message": "视频生成任务已提交，使用模板: vgp_new_pipeline",
    "status": "processing",
    "estimated_time": 40.0
}
```

---

## 📊 查询任务状态

### 接口

```
GET /vgp/status/{instance_id}
```

### 请求示例

```bash
curl http://localhost:8000/vgp/status/workflow_abc123
```

### 响应示例

```json
{
    "instance_id": "workflow_abc123",
    "status": "processing",
    "progress": 65.5,
    "current_node": "node_9_bgm_composition",
    "execution_time": 25.3,
    "result": null,
    "error_message": null
}
```

### 状态值

- `submitted` - 已提交
- `processing` - 处理中
- `completed` - 已完成
- `failed` - 失败
- `cancelled` - 已取消

---

## 🛠 完整的cURL示例

### 1. 提交任务

```bash
curl -X POST http://localhost:8000/vgp/generate \
  -H "Content-Type: application/json" \
  -d '{
    "theme_id": "产品展示",
    "user_description_id": "展示智能投影仪的完整功能演示...",
    "target_duration_id": 20,
    "keywords_id": ["智能投影仪", "4K高清", "便携", "语音控制"],
    "reference_media": {
        "product_images": [{
            "url": "https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/%E6%B5%8B%E8%AF%95%E5%95%86%E5%93%81.jpg",
            "type": "product",
            "weight": 1.0
        }]
    },
    "template": "vgp_new_pipeline"
}'
```

### 2. 查询状态

```bash
curl http://localhost:8000/vgp/status/workflow_abc123
```

### 3. 取消任务

```bash
curl -X POST http://localhost:8000/vgp/cancel/workflow_abc123
```

---

## 🐍 Python示例

```python
import requests
import time

# 1. 提交任务
url = "http://localhost:8000/vgp/generate"

payload = {
    "theme_id": "产品展示",
    "user_description_id": "展示智能投影仪的完整功能演示...",
    "target_duration_id": 20,
    "keywords_id": ["智能投影仪", "4K高清", "便携", "语音控制"],
    "reference_media": {
        "product_images": [{
            "url": "https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/%E6%B5%8B%E8%AF%95%E5%95%86%E5%93%81.jpg",
            "type": "product",
            "weight": 1.0
        }]
    },
    "template": "vgp_new_pipeline",
    "max_parallel_nodes": 5
}

response = requests.post(url, json=payload)
result = response.json()

if result["success"]:
    instance_id = result["instance_id"]
    print(f"✅ 任务提交成功: {instance_id}")

    # 2. 轮询状态
    status_url = f"http://localhost:8000/vgp/status/{instance_id}"

    while True:
        status_response = requests.get(status_url)
        status = status_response.json()

        print(f"📊 状态: {status['status']}, 进度: {status.get('progress', 0):.1f}%")

        if status['status'] in ['completed', 'failed', 'cancelled']:
            break

        time.sleep(5)  # 每5秒查询一次

    # 3. 处理结果
    if status['status'] == 'completed':
        print("✅ 视频生成成功!")
        print(f"结果: {status['result']}")
    else:
        print(f"❌ 任务失败: {status.get('error_message')}")
else:
    print(f"❌ 提交失败: {result}")
```

---

## 🔧 其他API

### 查看可用模板

```bash
GET /vgp/templates
```

响应：
```json
{
    "templates": [...],
    "recommended": "vgp_new_pipeline",
    "description": {
        "vgp_new_pipeline": "新版VGP工作流，优化的16节点架构，素材生成集中化",
        "vgp_full_pipeline": "旧版VGP工作流，保留用于兼容"
    }
}
```

### 查看活跃任务

```bash
GET /vgp/active-tasks
```

### 健康检查

```bash
GET /vgp/system/health
```

---

## ⚙️ 参数说明

### 必填参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `theme_id` | string | 主题ID（如：产品展示、教学视频） |
| `user_description_id` | string | 详细描述 |
| `target_duration_id` | integer | 目标时长（秒），5-300 |
| `keywords_id` | array | 关键词列表 |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `template` | string | `vgp_new_pipeline` | 工作流模板 |
| `max_parallel_nodes` | integer | 5 | 最大并行节点数 |
| `total_timeout` | float | 3600.0 | 总超时时间（秒） |
| `auto_retry` | boolean | true | 自动重试 |
| `enable_monitoring` | boolean | true | 启用监控 |
| `reference_media` | object | null | 参考媒体 |
| `session_id` | string | 自动生成 | 会话ID |
| `user_id` | string | "anonymous" | 用户ID |

---

## 📝 与旧API的对比

| 特性 | `/generate`（旧） | `/vgp/generate`（新） |
|------|-------------------|----------------------|
| 支持工作流模板 | ❌ 不支持 | ✅ 支持 |
| 字段位置 | 顶层 | 顶层 |
| 新工作流架构 | ❌ | ✅ |
| Node 5素材集中化 | ❌ | ✅ |
| 并行优化 | 一般 | ✅ 优化 |
| 状态查询 | `/task/{id}/status` | `/vgp/status/{id}` |

---

## 🎉 总结

1. **新接口**: 使用 `/vgp/generate` 而不是 `/generate`
2. **字段位置**: 所有字段在顶层，不需要嵌套在 `input` 内
3. **模板参数**: 设置 `"template": "vgp_new_pipeline"` 使用新工作流
4. **状态查询**: 使用 `/vgp/status/{instance_id}` 查询任务状态

**您的新请求应该这样写**：
```json
{
    "theme_id": "产品展示",
    "keywords_id": ["智能投影仪", "4K高清", "便携", "语音控制"],
    "target_duration_id": 20,
    "user_description_id": "展示智能投影仪的完整功能演示...",
    "reference_media": {
        "product_images": [{
            "url": "图片URL",
            "type": "product",
            "weight": 1.0
        }]
    },
    "template": "vgp_new_pipeline"
}
```

现在可以正常使用了！🚀
