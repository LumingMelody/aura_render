# 12步提示词优化器集成指南

## 概述

12步提示词优化器已成功集成到 `ShotBlockGenerationNode`（分镜块生成节点）中。

## 功能说明

当启用优化器时，视频生成流程将使用12步优化流程来生成更专业、更细致的分镜描述：

### 12步优化流程：
1. **产品描述提取** - 理解产品核心特性
2. **宣传偏好分析** - 分析营销导向
3. **时代偏好判断** - 确定视觉时代感
4. **粗排分镜** - 生成初步分镜框架
5. **视觉基因确定** - 确立全局视觉风格
6. **连续性分析** - 判断镜头衔接策略
7. **首帧和过程描述** - 生成60字结构化描述
8-9. **首帧细化** - 添加运镜、构图、光影、色彩
10. **全局美学检查** - 确保视觉统一性
11. **图生图策略** - 决定生成方式
12. **最终输出** - 生成完整的优化分镜

### 优化后的提升：
- ✅ **60字结构化首帧描述** - 专业精确
- ✅ **运镜术语** - 推拉摇移升降跟等
- ✅ **构图描述** - 特写/中景/全景、俯仰角、黄金分割
- ✅ **光影细节** - 硬光/柔光/体积光/伦勃朗光
- ✅ **色彩调色板** - 主色调、辅助色、点缀色
- ✅ **全局视觉统一** - 保持风格一致性
- ✅ **图生图策略** - 智能选择生成方式

## 配置方式

### 方式1：全局启用/禁用（推荐）

修改 `video_generate_protocol/nodes/shot_block_generation_node.py`:

```python
system_parameters = {
    "min_shot_duration": 2,
    "max_shot_duration": 10,
    "use_advanced_optimizer": True  # True=启用优化器，False=使用旧版
}
```

### 方式2：代码中动态控制

```python
# 创建节点时启用优化器
node = ShotBlockGenerationNode(node_id="my_node")
node.system_parameters["use_advanced_optimizer"] = True

# 创建节点时禁用优化器
node = ShotBlockGenerationNode(node_id="my_node")
node.system_parameters["use_advanced_optimizer"] = False
```

### 方式3：环境变量控制（未实现，可扩展）

```bash
export VGP_USE_OPTIMIZER=true  # 启用
export VGP_USE_OPTIMIZER=false # 禁用
```

## 降级策略

当优化器不可用或失败时，系统会自动降级到旧版分镜生成方式：

```
2025-10-29 14:28:50 - INFO - ✅ 12步提示词优化器已启用

# 如果优化器失败：
2025-10-29 14:28:50 - ERROR - ❌ 优化器生成失败: xxx，降级为旧版生成
2025-10-29 14:28:50 - INFO - 🎬 使用旧版分镜生成...
```

## 生成结果格式

使用优化器生成的分镜会包含额外的 `_optimized` 字段：

```json
{
  "shot_blocks_id": [
    {
      "shot_type": "特写",
      "duration": 3.5,
      "visual_description": "【60字精细化首帧描述】黑色金属质感的智能投影仪特写，镜头缓慢推进...",
      "pacing": "慢镜头",
      "caption": "开场吸引注意力",
      "start_time": 0.0,
      "end_time": 3.5,

      "_optimized": {
        "first_frame_refined": "黑色金属质感的智能投影仪特写...",
        "middle_process_refined": "镜头缓慢推进，投影仪logo清晰可见...",
        "generation_strategy": "文生图",
        "reference_source": "产品官方素材",
        "visual_style": {
          "target_style": "科技感产品广告",
          "core_theme": "未来科技",
          "core_emotion": "惊艳、向往",
          "color_palette": "主色调：深空黑、科技蓝；辅助色：冷灰；点缀色：荧光蓝",
          "lighting_rules": "硬光+体积光，营造科幻氛围"
        }
      }
    }
  ]
}
```

## 性能统计

节点会跟踪优化器的使用情况：

```python
node.stats = {
    "total_requests": 10,      # 总请求数
    "optimizer_calls": 8,      # 优化器调用次数
    "llm_calls": 2,            # 旧版LLM调用次数
    "fallback_calls": 0,       # 降级次数
    "cache_hits": 5,           # 缓存命中次数
    "avg_response_time": 2.5   # 平均响应时间
}
```

## 测试验证

运行测试脚本验证集成：

```bash
python3 test_optimizer_integration.py
```

预期输出：
```
============================================================
🧪 测试12步优化器集成
============================================================
✅ 成功导入 ShotBlockGenerationNode
✅ VideoPromptOptimizer 可用
✅ 节点创建成功
✅ 优化器已启用: VideoPromptOptimizer

============================================================
🎬 测试使用优化器生成分镜
============================================================
📦 测试产品: 智能投影仪
⏱️ 目标时长: 15秒
✅ 生成成功!
   分镜数量: 4

   镜头 1:
      时长: 3秒
      描述: 黑色金属质感的智能投影仪特写...
      ✨ 使用了优化器!
      生成策略: 文生图
      视觉风格: 科技感产品广告
```

## API调用示例

在 VGP API 中使用：

```bash
curl -X POST http://localhost:8001/vgp/generate \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["智能投影仪"],
    "user_description": "一款能投射100寸巨幕的便携智能投影仪",
    "video_type": "产品广告",
    "target_duration": 30
  }'
```

## 对比旧版与新版

| 特性 | 旧版 | 新版（优化器） |
|------|------|----------------|
| 描述长度 | 1-2句简单描述 | 60字结构化描述 |
| 运镜信息 | ❌ 无 | ✅ 专业术语 |
| 构图细节 | ❌ 无 | ✅ 精确描述 |
| 光影描述 | ❌ 无 | ✅ 专业光效 |
| 色彩调色板 | ❌ 无 | ✅ 完整调色板 |
| 全局统一 | ❌ 每镜独立 | ✅ 视觉基因统一 |
| 生成策略 | ❌ 无 | ✅ 文生图/图生图 |
| 处理时间 | ~2秒 | ~15秒（12步） |

## 注意事项

1. **API密钥要求**: 优化器依赖 Qwen LLM，需要配置 `DASHSCOPE_API_KEY`
2. **性能影响**: 优化器会增加生成时间（约10-15秒），但质量大幅提升
3. **成本考虑**: 12步流程会调用多次LLM，token消耗更大
4. **降级保护**: 如果优化器失败，会自动降级到旧版，保证可用性

## 后续优化方向

1. **并行化**: 将12步中独立的步骤并行执行，减少总时间
2. **缓存策略**: 缓存产品描述、视觉风格等可复用结果
3. **流式输出**: 支持分步返回结果，提升用户体验
4. **A/B测试**: 对比优化器效果，持续改进
5. **可配置步骤**: 允许用户选择性启用某些优化步骤

## 版本历史

- **v1.0** (2025-10-29) - 初始集成，支持12步优化流程
- 降级策略和性能统计
- 完整的错误处理和日志记录

---

**文档维护**: 如有问题请查看 `test_optimizer_integration.py` 或联系开发团队。
