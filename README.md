# Aura Render

一个智能视频生成和渲染平台，基于 FastAPI 构建，集成多种 AI 服务，提供完整的视频制作工作流。

> **💡 前端项目**: 本项目的前端界面位于 [video_editing](https://github.com/LumingMelody/video_editing) 仓库，提供可视化的视频编辑和管理界面。

## 功能特性

### 核心功能
- **视频生成协议(VGP)**: 基于 DAG 的视频生成流程编排
- **智能素材管理**: 支持图片、视频、音频素材的管理和检索
- **多平台素材集成**: 集成 Pexels、Pixabay、Unsplash、Freesound 等素材平台
- **AI 图像生成**: 支持 OpenAI DALL-E、Stability AI、Midjourney 等图像生成服务
- **TTS 语音合成**: 集成 Azure TTS、Edge TTS、OpenAI TTS 等多种语音服务
- **视频智能剪辑**: 自动化视频编辑、转场、特效处理
- **IMS 转换器**: 阿里云智能媒体服务集成

### 技术特性
- **异步任务处理**: 基于 Celery 的分布式任务队列
- **实时通信**: WebSocket 支持实时进度推送
- **缓存系统**: Redis 分布式缓存和内存缓存
- **监控和分析**: Prometheus 指标收集和性能监控
- **模板系统**: 可复用的视频生成模板
- **批量处理**: 支持批量视频生成任务
- **用户认证**: 完整的用户认证和权限管理系统

## 技术栈

### 后端框架
- **FastAPI** - 现代异步 Web 框架
- **SQLAlchemy** - ORM 数据库管理
- **Celery** - 分布式任务队列
- **Redis** - 缓存和消息队列

### AI/ML
- **PyTorch** - 深度学习框架
- **Transformers** - Hugging Face 模型库
- **Diffusers** - 图像生成模型
- **Dashscope** - 阿里云通义千问 API
- **Cozepy** - Coze AI API
- **Ultralytics** - YOLO 目标检测

### 媒体处理
- **OpenCV** - 图像处理
- **Librosa** - 音频分析
- **Pillow** - 图像处理库
- **阿里云 IMS** - 智能媒体服务

### 云服务
- **阿里云 OSS** - 对象存储
- **AWS S3** - 对象存储
- **Azure Blob Storage** - 对象存储
- **Google Cloud Storage** - 对象存储

## 快速开始

### 环境要求
- Python 3.9+
- Redis 服务器
- SQLite (或 PostgreSQL)

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/LumingMelody/aura_render.git
cd aura_render
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，配置必要的 API 密钥和服务配置
```

4. 初始化数据库
```bash
python init_db.py
```

5. 启动服务

使用启动脚本:
```bash
./run.sh
```

或手动启动:
```bash
# 启动 FastAPI 应用
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 启动 Celery Worker (新终端)
celery -A task_queue.celery_app worker --loglevel=info

# 启动 Celery Beat (新终端) - 可选，用于定时任务
celery -A task_queue.celery_app beat --loglevel=info
```

6. 访问 API 文档
打开浏览器访问: `http://localhost:8000/docs`

### Docker 部署

使用 Docker Compose 一键部署:

```bash
docker-compose up -d
```

开发环境:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

## 配置说明

主要配置文件位于 `.env` 文件中，参考 `.env.example`:

### 必需配置
- `OSS_ACCESS_KEY_ID` - 阿里云 OSS 访问密钥 ID
- `OSS_ACCESS_KEY_SECRET` - 阿里云 OSS 访问密钥
- `OSS_BUCKET_NAME` - OSS 存储桶名称
- `OSS_ENDPOINT` - OSS 服务端点

### 可选配置
- `REDIS_URL` - Redis 连接 URL
- `DATABASE_URL` - 数据库连接 URL
- `OPENAI_API_KEY` - OpenAI API 密钥
- `AZURE_TTS_KEY` - Azure TTS API 密钥
- `DASHSCOPE_API_KEY` - 阿里云通义千问 API 密钥

## API 端点

### 主要 API 路由
- `/api/vgp/*` - 视频生成协议相关接口
- `/api/materials/*` - 素材管理接口
- `/api/render/*` - 渲染服务接口
- `/api/tasks/*` - 任务管理接口
- `/api/templates/*` - 模板管理接口
- `/api/analytics/*` - 分析统计接口
- `/api/export/*` - 导出服务接口
- `/api/auth/*` - 用户认证接口
- `/ws/*` - WebSocket 接口

详细 API 文档请访问运行中的服务的 `/docs` 端点。

## 项目结构

```
aura_render/
├── api/                      # API 路由模块
├── config/                   # 配置管理
├── database/                 # 数据库模型和服务
├── materials/                # 素材管理
├── materials_supplies/       # 素材供应商集成
├── video_generator/          # 视频生成器
├── video_processing/         # 视频处理引擎
├── image_generation/         # AI 图像生成
├── tts_services/            # TTS 语音服务
├── ai_optimization/         # AI 优化服务
├── task_queue/              # Celery 任务队列
├── workflow/                # 工作流引擎
├── video_generate_protocol/ # VGP 协议实现
├── ims_converter/           # 阿里云 IMS 转换器
├── utils/                   # 工具函数
├── app.py                   # FastAPI 主应用
├── vgp_api.py              # VGP API 实现
├── requirements.txt         # Python 依赖
└── docker-compose.yml       # Docker 配置
```

## 开发指南

### 运行测试
```bash
pytest
```

### 代码风格
项目遵循 PEP 8 代码规范。

### 贡献指南
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 监控和维护

### Prometheus 监控
系统集成了 Prometheus 指标收集，可以监控:
- API 请求性能
- 任务队列状态
- 系统资源使用
- 错误率和成功率

### Celery Flower
使用 Flower 监控 Celery 任务:
```bash
celery -A task_queue.celery_app flower
```
访问: `http://localhost:5555`

### 日志管理
日志文件位于 `logs/` 目录，包含:
- 应用日志
- 任务日志
- 错误日志
- 性能日志

## 常见问题

### 1. OSS 配置未加载
确保 `.env` 文件中配置了正确的 OSS 凭证。

### 2. Redis 连接失败
检查 Redis 服务是否运行，确认 `REDIS_URL` 配置正确。

### 3. 视频生成失败
- 检查 IMS 服务配置
- 确认素材文件路径正确
- 查看任务日志获取详细错误信息

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目主页: [https://github.com/LumingMelody/aura_render](https://github.com/LumingMelody/aura_render)
- 问题反馈: [GitHub Issues](https://github.com/LumingMelody/aura_render/issues)

## 致谢

感谢以下开源项目和服务:
- FastAPI
- Celery
- PyTorch
- Transformers
- 阿里云智能媒体服务
- 所有素材提供商 API

---

**注意**: 本项目仍在积极开发中，API 可能会有变动。建议在生产环境使用前进行充分测试。
