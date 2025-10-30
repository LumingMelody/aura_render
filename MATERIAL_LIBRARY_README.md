# 🎉 素材库集成完成

## ✅ 本次更新内容

### 1. 花字大小优化
- 调整花字大小为原来的**50%**
- 避免文字遮挡视频内容
- 更适合720p视频显示

### 2. BGM问题修复
- 定位问题：假的占位符URL导致IMS API失败
- 解决方案：接入真实素材库API
- 过滤无效URL，确保视频生成成功

### 3. 素材库API集成
- 接口：`https://agent.cstlanbaai.com/gateway/admin-api/agent/resource/page`
- 支持视频素材 (type=1) 和音频素材 (type=2)
- 多级fallback搜索策略
- 自动提取tenant_id进行认证

## 📚 快速开始

### 步骤1: 配置环境

编辑 `.env` 文件：

```bash
# 添加素材库认证Token
MATERIAL_LIBRARY_AUTH=你的Authorization_Token
```

### 步骤2: 测试连接

```bash
python test_material_library.py
```

### 步骤3: 启动服务

```bash
python3 app.py
```

### 步骤4: 生成视频

```bash
curl -X POST "http://localhost:8001/vgp/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "1",
    "id": "test_001",
    "theme_id": "产品展示",
    "user_description_id": "智能产品展示",
    "target_duration_id": 10
  }'
```

## 📖 文档

- [完整配置说明](docs/MATERIAL_LIBRARY_SETUP.md)
- [快速开始指南](docs/MATERIAL_LIBRARY_QUICKSTART.md)
- [完成报告](docs/COMPLETION_REPORT.md)

## 🔍 日志监控

```bash
# 查看BGM匹配日志
tail -f logs/aura_render.log | grep "🎵"

# 查看素材库调用
tail -f logs/aura_render.log | grep "素材库"
```

## 📁 文件清单

### 新增文件
- `materials_supplies/material_library_client.py` - 素材库API客户端
- `test_material_library.py` - 集成测试脚本
- `docs/MATERIAL_LIBRARY_SETUP.md` - 配置文档
- `docs/MATERIAL_LIBRARY_QUICKSTART.md` - 快速指南
- `docs/COMPLETION_REPORT.md` - 完成报告

### 修改文件
- `materials_supplies/matcher/bgm_matcher.py` - BGM匹配逻辑
- `ims_converter/converter.py` - URL验证和日志
- `ims_converter/utils.py` - 花字大小
- `vgp_api.py` - 客户端初始化
- `.env` - 配置项

## 🎯 关键特性

### BGM匹配策略

```
策略1: tag="风格" (如"极简电子")
  ↓ 失败
策略2: tag="情绪" (如"冷静")
  ↓ 失败
策略3: tag="风格" + name="情绪"
  ↓ 失败
策略4: name="背景音乐" (兜底)
  ↓ 全部失败
返回空列表 (视频无BGM)
```

### 花字大小对比

| 类型 | 之前 | 现在 | 效果 |
|------|------|------|------|
| 小字 | 40 | 20 | 不遮挡 |
| 中等 | 55 | 28 | 刚好 |
| 大字 | 70 | 35 | 醒目 |
| 超大 | 85 | 43 | 突出 |

## ⚠️ 注意事项

1. **必需配置 `MATERIAL_LIBRARY_AUTH`**
   - 从素材库管理后台获取
   - 添加到 `.env` 文件

2. **tenant_id 必须传递**
   - 在 `/vgp/generate` 请求中
   - 用于素材库认证

3. **ffprobe 需要安装**
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu
   sudo apt install ffmpeg
   ```

## 🐛 故障排查

### BGM没有声音？

```bash
# 1. 检查日志
tail -100 logs/aura_render.log | grep "BGM"

# 2. 运行测试
python test_material_library.py

# 3. 检查配置
grep MATERIAL_LIBRARY_AUTH .env
```

### 花字还是太大？

修改 `ims_converter/utils.py:366` 的字号值

### API调用失败？

1. 检查网络: `curl https://agent.cstlanbaai.com`
2. 验证token: 运行测试脚本
3. 查看日志: `logs/aura_render.log`

## 📞 获取帮助

- 查看完整文档: `docs/`
- 运行测试: `python test_material_library.py`
- 查看日志: `logs/aura_render.log`

---

**更新日期:** 2025-10-28
**版本:** v1.0.0
**状态:** ✅ 已完成，可投入使用
