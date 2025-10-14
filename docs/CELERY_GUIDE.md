# Aura Render Celery 异步任务系统指南

## 📋 系统概述

Aura Render 现已集成 Celery 分布式任务队列系统，提供强大的异步视频生成能力：

- **多队列优先级处理** - 不同优先级任务分离处理
- **分布式 Worker 管理** - 支持多机器部署和横向扩展  
- **实时任务监控** - 完整的任务状态跟踪和进度显示
- **自动故障恢复** - 任务重试和错误处理机制
- **资源管理** - 智能资源分配和负载均衡

## 🏗️ 架构设计

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │     Redis        │    │   Celery        │
│   Application   │◄──►│     Broker       │◄──►│   Workers       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Task Manager   │    │  Message Queue   │    │  Video Pipeline │
│  API Endpoints  │    │  Task Storage    │    │  Processing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 1. 启动 Redis 服务
```bash
# 使用 Docker
docker run -d --name redis -p 6379:6379 redis:latest

# 或者本地安装
brew install redis
redis-server
```

### 2. 启动 Celery 系统
```bash
# 启动完整的 Celery 环境（推荐）
python scripts/start_celery.py default

# 或分别启动组件
python scripts/start_celery.py worker video_worker --queue video_generation --concurrency 2
python scripts/start_celery.py start --beat --flower
```

### 3. 启动 FastAPI 应用
```bash
python app.py
```

### 4. 访问监控面板
- **Flower 监控**: http://localhost:5555
- **FastAPI 文档**: http://localhost:8000/docs

## 📝 API 使用指南

### 提交异步视频生成任务

```bash
curl -X POST "http://localhost:8000/tasks/video/async" \
  -H "Content-Type: application/json" \
  -d '{
    "theme_id": "产品宣传",
    "keywords_id": ["科技", "创新", "未来"],
    "target_duration_id": 60,
    "user_description_id": "一个展示AI技术发展的宣传视频",
    "priority": "high",
    "config": {
      "quality": "high",
      "resolution": "1920x1080"
    }
  }'
```

**响应示例**:
```json
{
  "task_id": "video_1703123456_abc123",
  "status": "queued",
  "priority": "high",
  "estimated_duration": 45,
  "message": "视频生成任务已提交，优先级: HIGH",
  "timestamp": "2023-12-20T10:30:45Z"
}
```

### 查询任务状态

```bash
curl "http://localhost:8000/tasks/status/{task_id}"
```

**响应示例**:
```json
{
  "task_id": "video_1703123456_abc123",
  "status": "processing", 
  "priority": "high",
  "progress": 65.0,
  "message": "正在执行音频处理...",
  "created_at": "2023-12-20T10:30:45Z",
  "updated_at": "2023-12-20T10:32:15Z",
  "estimated_duration": 45,
  "actual_duration": null,
  "result": null,
  "error": null
}
```

### 任务状态说明

| 状态 | 描述 |
|------|------|
| `pending` | 任务已创建，等待处理 |
| `queued` | 任务已加入队列，等待分配 |
| `processing` | 任务正在处理中 |
| `completed` | 任务成功完成 |
| `failed` | 任务处理失败 |
| `cancelled` | 任务已取消 |
| `retry` | 任务正在重试 |

### 任务优先级

| 优先级 | 数值 | 描述 | 使用场景 |
|--------|------|------|----------|
| `urgent` | 10 | 紧急任务 | 紧急需求、演示用途 |
| `high` | 8 | 高优先级 | VIP用户、付费用户 |
| `normal` | 5 | 普通优先级 | 常规用户请求 |
| `low` | 1 | 低优先级 | 批量处理、测试任务 |

## 🔧 管理和监控

### 查看队列状态
```bash
curl "http://localhost:8000/tasks/queue/status"
```

### 查看任务历史
```bash
curl "http://localhost:8000/tasks/history?limit=20&status=completed"
```

### Worker 控制
```bash
# 启动新 Worker
curl -X POST "http://localhost:8000/tasks/workers/control" \
  -H "Content-Type: application/json" \
  -d '{
    "worker_id": "new_worker_1",
    "action": "start",
    "queue": "video_generation",
    "concurrency": 4
  }'

# 停止 Worker
curl -X POST "http://localhost:8000/tasks/workers/control" \
  -H "Content-Type: application/json" \
  -d '{
    "worker_id": "worker_1", 
    "action": "stop"
  }'
```

### 自动扩缩容
```bash
curl -X POST "http://localhost:8000/tasks/workers/autoscale?target_workers=5&queue=video_generation"
```

## ⚙️ 配置参数

### Celery 配置 (config.py)

```python
# Redis 连接
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Worker 配置
CELERY_WORKER_CONCURRENCY = 4
CELERY_WORKER_MAX_MEMORY = 512  # MB

# 任务限制
TASK_SOFT_TIME_LIMIT = 300  # 5 minutes
TASK_TIME_LIMIT = 600       # 10 minutes
TASK_MAX_RETRIES = 3
```

### 队列配置

| 队列名 | 优先级 | 用途 | 建议并发数 |
|--------|--------|------|------------|
| `video_generation` | 10 | 视频生成主流程 | 2-4 |
| `video_processing` | 8 | 视频处理任务 | 2-4 |
| `audio_processing` | 6 | 音频处理任务 | 1-2 |
| `image_processing` | 5 | 图像处理任务 | 2-4 |
| `maintenance` | 2 | 系统维护任务 | 1 |
| `monitoring` | 1 | 监控和健康检查 | 1 |
| `default` | 4 | 默认任务队列 | 2-4 |

## 🔍 故障排查

### 常见问题

#### 1. Redis 连接失败
```bash
# 检查 Redis 状态
redis-cli ping

# 检查连接配置
grep -r "redis" config.py
```

#### 2. Worker 启动失败
```bash
# 检查 Python 路径
echo $PYTHONPATH

# 手动启动 Worker 调试
celery -A task_queue.celery_app:app worker --loglevel=debug
```

#### 3. 任务处理超时
```bash
# 查看任务日志
celery -A task_queue.celery_app:app events

# 调整超时设置
# 在 config.py 中修改 TASK_TIME_LIMIT
```

#### 4. 内存不足
```bash
# 监控内存使用
celery -A task_queue.celery_app:app inspect stats

# 调整 Worker 参数
--max-tasks-per-child=50  # 减少每个 Worker 处理的任务数
--concurrency=2           # 减少并发数
```

### 日志和监控

#### Celery 日志位置
```bash
# Worker 日志
tail -f /var/log/celery/worker.log

# Beat 日志  
tail -f /var/log/celery/beat.log
```

#### 使用 Flower 监控
访问 http://localhost:5555 查看：
- 实时任务状态
- Worker 资源使用
- 任务执行历史
- 错误统计和分析

## 📈 性能优化

### 1. Worker 配置优化
```python
# 针对不同任务类型优化
video_workers = 2      # CPU 密集型，少而精
audio_workers = 1      # I/O 密集型  
general_workers = 4    # 混合型任务
```

### 2. 资源分配策略
- **CPU 密集型任务**: 降低并发数，避免过载
- **I/O 密集型任务**: 提高并发数，提升吞吐量  
- **内存密集型任务**: 限制每个 Worker 处理任务数

### 3. 缓存优化
- 启用 Redis 结果缓存
- 设置合理的 TTL (生存时间)
- 定期清理过期缓存

## 🔒 生产部署

### Docker 部署示例

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  celery_worker:
    build: .
    command: celery -A task_queue.celery_app:app worker --concurrency=4
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
    
  celery_beat:
    build: .
    command: celery -A task_queue.celery_app:app beat
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      
  flower:
    build: .
    command: flower -A task_queue.celery_app:app --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
```

### 环境变量配置

```bash
export REDIS_HOST=production-redis-host
export REDIS_PORT=6379
export REDIS_PASSWORD=your-redis-password
export CELERY_WORKER_CONCURRENCY=8
export ENVIRONMENT=production
```

## 🎯 最佳实践

1. **任务设计**
   - 保持任务幂等性
   - 避免长时间运行的任务
   - 合理设置超时时间

2. **监控告警**
   - 监控队列积压情况
   - 设置任务失败率告警
   - 监控 Worker 资源使用

3. **容错处理**  
   - 实现任务重试逻辑
   - 记录详细错误信息
   - 提供任务取消机制

4. **性能调优**
   - 根据任务特性调整 Worker 参数
   - 使用连接池减少 Redis 连接开销
   - 定期清理历史任务数据

通过这个完整的 Celery 系统，Aura Render 现在具备了企业级的异步任务处理能力，能够支持大规模并发视频生成需求！🚀