# Aura Render Production Monitoring

Comprehensive production monitoring and alerting system for the Aura Render platform.

## Overview

This monitoring stack provides complete observability for your Aura Render deployment including:

- **Metrics Collection**: Prometheus-based metrics for application, system, and business metrics
- **Alerting**: Rule-based alerting with email and Slack notifications
- **Visualization**: Pre-configured Grafana dashboards for monitoring key metrics
- **Health Checks**: Comprehensive system health monitoring
- **Performance Tracking**: Request latency, throughput, and error rate monitoring

## Components

### Core Monitoring Services

- **Prometheus** (port 9090): Metrics collection and alerting engine
- **Grafana** (port 3000): Visualization and dashboard platform  
- **Alertmanager** (port 9093): Alert routing and notification management

### Exporters and Collectors

- **Node Exporter** (port 9100): System metrics (CPU, memory, disk)
- **cAdvisor** (port 8080): Container metrics
- **Redis Exporter** (port 9121): Redis performance metrics
- **PostgreSQL Exporter** (port 9187): Database performance metrics
- **DCGM Exporter** (port 9400): GPU metrics (if available)

### Custom Metrics

- **Application Metrics**: HTTP requests, response times, error rates
- **Video Generation**: Task queues, success/failure rates, processing times
- **AI Optimization**: Optimization performance, quality scores, processing efficiency
- **Business Metrics**: User activity, revenue tracking, system utilization

## Quick Start

### 1. Start Monitoring Stack

```bash
# Start all monitoring services
./scripts/monitoring/start_monitoring.sh
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090  
- **Alertmanager**: http://localhost:9093

### 3. Configure Notifications

Edit `.env.monitoring` to configure email and Slack notifications:

```bash
# SMTP Configuration
ALERTMANAGER_SMTP_FROM=alerts@yourcompany.com
ALERTMANAGER_SMTP_PASSWORD=your_smtp_password

# Slack Integration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

Then restart Alertmanager:
```bash
docker-compose -f docker-compose.monitoring.yml restart alertmanager
```

## Monitoring Configuration

### Alert Rules

Critical alerts are configured in `monitoring/alert_rules.yml`:

#### System Health Alerts
- **Application Down**: Triggers when main application is unreachable
- **High CPU/Memory**: Resource usage thresholds
- **Disk Space Low**: Storage availability warnings
- **Database/Redis Down**: Critical service failures

#### Application Performance Alerts  
- **High Request Latency**: 95th percentile > 2 seconds
- **High Error Rate**: Error rate > 10%
- **Queue Backlog**: Task queue length > 100

#### Business Logic Alerts
- **Video Generation Failures**: Failure rate > 20%
- **AI Optimization Issues**: Processing failures or quality drops
- **Long Running Tasks**: Tasks exceeding expected duration

### Metrics Collection

The application exposes metrics on `/metrics` endpoint. Key metrics include:

#### HTTP Metrics
```python
# Request count by method, endpoint, status
http_requests_total{method="POST", endpoint="/generate", status="200"}

# Request duration histogram  
http_request_duration_seconds{method="POST", endpoint="/generate"}
```

#### Video Generation Metrics
```python
# Total video generations
video_generation_total{theme="tech", user_id="user123"}

# Generation failures
video_generation_failures_total{error_type="TimeoutError", stage="rendering"}

# Processing duration
video_generation_duration_seconds{theme="tech", duration_category="medium"}
```

#### AI Optimization Metrics
```python
# Optimization requests
ai_optimization_total{optimization_type="video", level="standard"}

# Size reduction achieved
optimization_size_reduction_percent{optimization_type="video"}

# Quality scores
video_quality_score{theme="tech", optimization_level="standard"}
```

## Grafana Dashboards

### Aura Render Overview Dashboard

Pre-configured dashboard showing:

- **Service Health**: Component status indicators
- **Request Metrics**: Traffic, latency, and error rates  
- **Video Generation**: Queue status and success/failure rates
- **System Resources**: CPU, memory, and disk usage
- **AI Optimization**: Performance and quality metrics
- **GPU Monitoring**: Utilization and temperature (if available)

### Custom Dashboards

You can create additional dashboards by:

1. Accessing Grafana at http://localhost:3000
2. Using the "+" icon to create new dashboard
3. Adding panels with Prometheus queries
4. Saving and sharing dashboard JSON configuration

## Health Checks

The health check system monitors:

### Critical Components
- Database connectivity and performance
- Redis availability and memory usage
- Application response times
- System resource availability

### Service Dependencies  
- FFmpeg availability for video processing
- AI model file availability
- External API connectivity
- Task queue functionality

Access health status via:
```bash
curl http://localhost:8000/health
```

## Production Deployment

### Environment Configuration

Set environment variables for production:

```bash
# Prometheus retention
PROMETHEUS_RETENTION_TIME=90d

# Grafana security
GRAFANA_ADMIN_PASSWORD=secure_password_here
GRAFANA_SECRET_KEY=your_secret_key

# Alert notifications
ALERTMANAGER_SMTP_PASSWORD=your_smtp_password
SLACK_WEBHOOK_URL=your_slack_webhook
```

### Security Considerations

1. **Change Default Passwords**: Update Grafana admin password
2. **Network Security**: Use proper network segmentation
3. **TLS/SSL**: Configure HTTPS for production access
4. **Authentication**: Set up proper user authentication
5. **Access Control**: Limit dashboard and metrics access

### Scaling Monitoring

For high-scale deployments:

1. **Prometheus Federation**: Set up multiple Prometheus instances
2. **Long-term Storage**: Configure Prometheus remote storage
3. **High Availability**: Deploy redundant monitoring components
4. **Load Balancing**: Distribute metrics collection load

## Troubleshooting

### Common Issues

#### Metrics Not Appearing
```bash
# Check if application metrics endpoint is accessible
curl http://localhost:8000/metrics

# Verify Prometheus can scrape targets
# Go to http://localhost:9090/targets
```

#### Alerts Not Firing
```bash
# Check alert rules syntax
docker exec aura-prometheus promtool check rules /etc/prometheus/alert_rules.yml

# Verify Alertmanager configuration
curl http://localhost:9093/api/v1/status
```

#### Grafana Dashboard Issues
```bash
# Check Grafana logs
docker logs aura-grafana

# Verify Prometheus datasource connectivity
# Go to Grafana > Configuration > Data Sources
```

### Log Analysis

View component logs:
```bash
# All monitoring services
docker-compose -f docker-compose.monitoring.yml logs

# Specific service
docker-compose -f docker-compose.monitoring.yml logs prometheus
docker-compose -f docker-compose.monitoring.yml logs grafana
docker-compose -f docker-compose.monitoring.yml logs alertmanager
```

### Performance Tuning

#### Prometheus Optimization
```yaml
# Adjust scrape intervals based on needs
scrape_interval: 15s     # Default
scrape_interval: 30s     # For less critical metrics
scrape_interval: 5s      # For critical real-time metrics
```

#### Storage Management
```bash
# Monitor Prometheus storage usage
docker exec aura-prometheus df -h /prometheus

# Clean up old metrics data if needed
docker exec aura-prometheus rm -rf /prometheus/data/*
```

## Integration with Application

### Adding Custom Metrics

```python
from monitoring.metrics import track_requests, track_video_generation

@track_requests(endpoint="custom_endpoint")
async def my_endpoint():
    # Your endpoint logic
    pass

@track_video_generation(theme="custom", user_id="user123")
async def generate_custom_video():
    # Video generation logic
    pass
```

### Health Check Integration

```python
from monitoring.health_check import initialize_health_checker, get_system_health

# Initialize health checker
config = {
    'database': {'host': 'localhost', 'port': 5432},
    'redis': {'host': 'localhost', 'port': 6379}
}
initialize_health_checker(config)

# Get health status
health = await get_system_health()
print(health.to_dict())
```

This monitoring system provides comprehensive observability for your Aura Render platform. Regular monitoring of dashboards and prompt response to alerts will help maintain optimal system performance and user experience.