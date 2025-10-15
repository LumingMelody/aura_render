# VGP API 参数说明文档

## 接口概览

**接口地址**: `POST /vgp/generate`

**功能**: 使用VGP新工作流生成视频（16节点DAG架构）

**响应模式**: 异步处理，立即返回任务ID，通过状态接口查询进度

---

## 请求参数说明

### 📌 必需参数

这些参数必须提供，否则接口会返回验证错误。

| 参数名 | 类型 | 说明 | 示例 |
|-------|------|------|------|
| `theme_id` | `string` | 视频主题类型 | `"产品展示"`, `"教学视频"`, `"品牌宣传"` |
| `user_description_id` | `string` | 详细的视频内容描述，包括镜头、场景、动作等 | `"智能投影仪产品展示。黑色磨砂机身特写，镜头环绕产品，展示投影功能启动"` |

### ✅ 可选参数（有默认值）

这些参数可以不提供，系统会使用默认值。

| 参数名 | 类型 | 默认值 | 约束 | 说明 |
|-------|------|--------|------|------|
| `target_duration_id` | `integer` | `30` | 5-300秒 | 目标视频时长（秒） |
| `keywords_id` | `array[string]` | `[]` | - | 关键词列表，用于辅助理解内容 |
| `reference_media` | `object` | `null` | - | 参考媒体（产品图片、参考视频等） |
| `template` | `string` | `"vgp_new_pipeline"` | 见下方 | 工作流模板名称 |
| `session_id` | `string` | `null` | - | 会话ID，用于关联多次请求 |
| `user_id` | `string` | `null` | - | 用户ID，用于用户行为分析 |

---

## 参数详细说明

### 1. `theme_id` *（必需）*

**类型**: `string`

**说明**: 定义视频的主题类型，影响情感分析、镜头选择、音乐风格等。

**常用值**:
- `"产品展示"` - 产品介绍、功能演示
- `"产品广告"` - 营销宣传视频
- `"品牌宣传"` - 品牌形象片
- `"知识讲解"` - 教学、科普内容
- `"技能教学"` - 操作演示、教程
- `"微电影"` - 故事类短片
- `"VLOG"` - 个人记录
- `"社交媒体内容"` - 社交平台短视频

**示例**:
```json
"theme_id": "产品展示"
```

---

### 2. `user_description_id` *（必需）*

**类型**: `string`

**说明**: 详细描述视频内容、镜头、场景、动作、产品特性等。描述越详细，生成的视频越准确。

**写作建议**:
- ✅ 描述具体镜头：如"特写"、"环绕"、"推进"
- ✅ 描述产品外观：如"黑色磨砂机身"、"圆润边角"
- ✅ 描述动作：如"缓慢旋转"、"快速切换"
- ✅ 描述功能演示：如"投影启动"、"4K画面投射"
- ⚠️ 避免过于简短：如"产品视频"（太模糊）

**示例**:
```json
"user_description_id": "智能投影仪产品展示。开场展示黑色磨砂质感的机身特写，镜头从正面缓慢环绕到侧面。接着展示开机过程，投影镜头亮起蓝色指示灯。然后在白墙上投射4K清晰画面，展示色彩鲜艳的风景图。最后演示语音控制功能，画面切换流畅。"
```

---

### 3. `target_duration_id` *（可选，默认30）*

**类型**: `integer`

**范围**: `5` - `300` 秒

**说明**: 目标视频时长，系统会根据此时长生成相应数量的镜头。

**镜头数量参考**:
- 10秒 = 约2个镜头（每镜头5秒）
- 20秒 = 约4个镜头
- 30秒 = 约6个镜头
- 60秒 = 约12个镜头

**示例**:
```json
"target_duration_id": 10
```

---

### 4. `keywords_id` *（可选，默认[]）*

**类型**: `array[string]`

**说明**: 关键词列表，用于辅助理解视频主题和内容特征，影响视觉风格、情感基调。

**建议**:
- 包含3-5个关键词
- 包含产品名称
- 包含核心特性
- 包含使用场景

**示例**:
```json
"keywords_id": [
    "智能投影仪",
    "4K高清",
    "便携",
    "语音控制"
]
```

---

### 5. `reference_media` *（可选，默认null）*

**类型**: `object`

**说明**: 提供参考媒体，用于视觉一致性控制和产品识别。

**结构**:
```typescript
{
  "product_images": [    // 产品图片列表（可选）
    {
      "url": string,     // 图片URL或OSS路径
      "type": string,    // 类型："product"（产品图）
      "weight": number   // 权重 0.0-1.0，默认1.0
    }
  ],
  "videos": [           // 参考视频列表（可选）
    {
      "url": string,     // 视频URL
      "type": string,    // 类型："style_reference"（风格参考）
      "weight": number   // 权重 0.0-1.0
    }
  ]
}
```

**使用场景**:
- ✅ **产品展示视频**: 必须提供产品图片，确保生成的视频中产品外观一致
- ✅ **品牌宣传**: 可提供品牌元素图片
- ⚠️ **纯文字内容**: 可不提供

**重要**:
- 产品图片URL必须可访问（建议使用阿里云OSS）
- 第一个镜头会直接使用产品原图（避免变形）
- 后续镜头会基于产品图进行风格化处理

**示例**:
```json
"reference_media": {
    "product_images": [
        {
            "url": "https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/测试商品.jpg",
            "type": "product",
            "weight": 1.0
        }
    ]
}
```

---

### 6. `template` *（可选，默认"vgp_new_pipeline"）*

**类型**: `string`

**说明**: 工作流模板选择。

**可选值**:
- `"vgp_new_pipeline"` ✅ **推荐** - 新版16节点DAG工作流，优化的素材生成和并行处理
- `"vgp_full_pipeline"` - 旧版工作流（暂不可用）
- `"basic_video_generation"` - 基础生成（使用/generate接口）

**示例**:
```json
"template": "vgp_new_pipeline"
```

---

## 完整请求示例

### 最简请求（仅必需参数）

```json
{
    "theme_id": "产品展示",
    "user_description_id": "展示智能投影仪的外观和投影功能"
}
```

### 标准请求（推荐）

```json
{
    "theme_id": "产品展示",
    "user_description_id": "智能投影仪产品展示。黑色磨砂机身特写，镜头环绕产品，展示投影功能启动，在白墙上投射4K画面，展示语音控制",
    "target_duration_id": 10,
    "keywords_id": [
        "智能投影仪",
        "4K高清",
        "便携",
        "语音控制"
    ],
    "reference_media": {
        "product_images": [
            {
                "url": "https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/测试商品.jpg",
                "type": "product",
                "weight": 1.0
            }
        ]
    },
    "template": "vgp_new_pipeline"
}
```

### 完整请求（包含所有可选参数）

```json
{
    "theme_id": "产品展示",
    "user_description_id": "智能投影仪产品展示。黑色磨砂机身特写，镜头环绕产品，展示投影功能启动，在白墙上投射4K画面，展示语音控制",
    "target_duration_id": 30,
    "keywords_id": [
        "智能投影仪",
        "4K高清",
        "便携",
        "语音控制"
    ],
    "reference_media": {
        "product_images": [
            {
                "url": "https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/测试商品.jpg",
                "type": "product",
                "weight": 1.0
            }
        ]
    },
    "template": "vgp_new_pipeline",
    "session_id": "session_abc123",
    "user_id": "user_12345"
}
```

---

## 响应说明

### 成功响应 (200 OK)

```json
{
    "success": true,
    "instance_id": "0800c92a-d804-41a0-a137-0d8dea4e49cb",
    "task_id": "0800c92a-d804-41a0-a137-0d8dea4e49cb",
    "message": "VGP视频生成任务已启动（模板: vgp_new_pipeline）",
    "status": "started",
    "estimated_time": 60.0
}
```

**字段说明**:
- `instance_id`: 任务唯一ID，用于查询状态
- `status`: 任务状态（`started` = 已提交）
- `estimated_time`: 预计完成时间（秒）

---

## 状态查询

**接口**: `GET /vgp/status/{instance_id}`

**示例**:
```bash
curl http://localhost:8000/vgp/status/0800c92a-d804-41a0-a137-0d8dea4e49cb
```

**响应**:
```json
{
    "instance_id": "0800c92a-d804-41a0-a137-0d8dea4e49cb",
    "status": "processing",
    "progress": 62.5,
    "current_node": "DAG进度: 10/16 节点 - 节点10执行完成: sfx_integration",
    "execution_time": 45.2,
    "result": null,
    "error_message": null
}
```

**状态值说明**:
- `submitted`: 已提交
- `processing`: 处理中
- `completed`: 已完成
- `failed`: 失败

---

## 常见使用场景

### 场景1: 产品展示视频（有产品图）

```json
{
    "theme_id": "产品展示",
    "user_description_id": "智能手机产品展示，展示流线型金属机身、摄像头模组、屏幕显示效果",
    "target_duration_id": 15,
    "keywords_id": ["智能手机", "超清摄像", "高刷屏幕"],
    "reference_media": {
        "product_images": [{
            "url": "https://your-oss.com/phone.jpg",
            "type": "product",
            "weight": 1.0
        }]
    }
}
```

### 场景2: 品牌宣传视频（无产品图）

```json
{
    "theme_id": "品牌宣传",
    "user_description_id": "科技公司品牌宣传片，展示团队协作、创新研发、产品发布等场景",
    "target_duration_id": 30,
    "keywords_id": ["创新", "科技", "未来"]
}
```

### 场景3: 教学视频

```json
{
    "theme_id": "知识讲解",
    "user_description_id": "Python编程入门教学，讲解变量、函数、循环的基本概念",
    "target_duration_id": 60,
    "keywords_id": ["Python", "编程", "入门"]
}
```

---

## 错误处理

### 验证错误 (422)

```json
{
    "detail": [
        {
            "loc": ["body", "theme_id"],
            "msg": "field required",
            "type": "value_error.missing"
        }
    ]
}
```

### 服务器错误 (500)

```json
{
    "detail": "Failed to create VGP task: ..."
}
```

---

## 最佳实践

### ✅ 推荐做法

1. **产品展示类视频**：必须提供 `reference_media.product_images`
2. **描述要详细**：包含镜头、动作、场景、产品特性
3. **合理设置时长**：10-30秒适合社交媒体，60秒适合完整展示
4. **提供关键词**：帮助系统理解内容主题
5. **轮询查询状态**：每5-10秒查询一次任务状态

### ⚠️ 避免的做法

1. ❌ 描述过于简短：如"产品视频"
2. ❌ 时长设置过长：超过120秒会导致生成时间过长
3. ❌ 产品图URL无法访问：导致视觉一致性失效
4. ❌ 频繁查询状态：建议间隔至少5秒

---

## 技术支持

如有问题，请查看：
- API文档: http://localhost:8000/docs
- 快速开始: `API_QUICK_START.md`
- 项目GitHub: https://github.com/LumingMelody/aura_render
