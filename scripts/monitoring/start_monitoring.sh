#!/bin/bash

# Start Monitoring Stack
# Starts Prometheus, Alertmanager, Grafana, and related services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MONITORING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/monitoring"
DOCKER_COMPOSE_FILE="docker-compose.monitoring.yml"
ENV_FILE=".env.monitoring"

echo -e "${BLUE}Aura Render Monitoring Stack${NC}"
echo -e "${BLUE}=============================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check if monitoring directory exists
if [ ! -d "$MONITORING_DIR" ]; then
    echo -e "${RED}Error: Monitoring directory not found: $MONITORING_DIR${NC}"
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Creating monitoring environment file...${NC}"
    cat > "$ENV_FILE" << EOF
# Monitoring Configuration
PROMETHEUS_RETENTION_TIME=30d
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin123
ALERTMANAGER_SMTP_FROM=alerts@aurarender.com
ALERTMANAGER_SMTP_SMARTHOST=smtp.gmail.com:587
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Security
GRAFANA_SECRET_KEY=$(openssl rand -base64 32)
PROMETHEUS_WEB_CONFIG_FILE=/etc/prometheus/web.yml

# Network
MONITORING_NETWORK=monitoring
EOF
    echo -e "${GREEN}Environment file created: $ENV_FILE${NC}"
    echo -e "${YELLOW}Please edit $ENV_FILE to configure your SMTP and Slack settings${NC}"
fi

# Create Docker Compose file for monitoring stack
echo -e "${YELLOW}Creating monitoring Docker Compose configuration...${NC}"
cat > "$DOCKER_COMPOSE_FILE" << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.40.0
    container_name: aura-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--storage.tsdb.retention.time=\${PROMETHEUS_RETENTION_TIME:-30d}'
    networks:
      - monitoring
      - app-network

  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: aura-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    environment:
      - SMTP_USERNAME=\${ALERTMANAGER_SMTP_FROM}
      - SMTP_PASSWORD=\${ALERTMANAGER_SMTP_PASSWORD}
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:9.3.0
    container_name: aura-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      - GF_SECURITY_ADMIN_USER=\${GRAFANA_ADMIN_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD:-admin123}
      - GF_SECURITY_SECRET_KEY=\${GRAFANA_SECRET_KEY}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_USERS_ALLOW_ORG_CREATE=false
      - GF_USERS_AUTO_ASSIGN_ORG=true
      - GF_USERS_AUTO_ASSIGN_ORG_ROLE=Viewer
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:v1.5.0
    container_name: aura-node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.46.0
    container_name: aura-cadvisor
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - monitoring

  redis-exporter:
    image: oliver006/redis_exporter:v1.45.0
    container_name: aura-redis-exporter
    restart: unless-stopped
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    networks:
      - monitoring
      - app-network

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:v0.11.1
    container_name: aura-postgres-exporter
    restart: unless-stopped
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://postgres:password@postgres:5432/aura_render?sslmode=disable
    networks:
      - monitoring
      - app-network

  # Optional: DCGM Exporter for GPU monitoring (only if GPU available)
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.3-3.1.2-ubuntu20.04
    container_name: aura-dcgm-exporter
    restart: unless-stopped
    ports:
      - "9400:9400"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    cap_add:
      - SYS_ADMIN
    networks:
      - monitoring
    profiles:
      - gpu

volumes:
  prometheus_data:
  alertmanager_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
  app-network:
    external: true

EOF

echo -e "${GREEN}Monitoring Docker Compose configuration created${NC}"

# Function to check if service is healthy
check_service_health() {
    local service_name="$1"
    local port="$2"
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Checking $service_name health...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $service_name is healthy${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}Attempt $attempt/$max_attempts - waiting for $service_name...${NC}"
        sleep 2
        ((attempt++))
    done
    
    echo -e "${RED}✗ $service_name failed to start properly${NC}"
    return 1
}

# Start monitoring stack
echo -e "${YELLOW}Starting monitoring stack...${NC}"

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Create network if it doesn't exist
docker network create app-network 2>/dev/null || true
docker network create monitoring 2>/dev/null || true

# Start services
echo -e "${BLUE}Starting Prometheus, Alertmanager, and Grafana...${NC}"
docker-compose -f "$DOCKER_COMPOSE_FILE" up -d

# Wait for services to start
sleep 10

# Check service health
echo -e "${BLUE}Performing health checks...${NC}"

services_healthy=true

if ! check_service_health "Prometheus" "9090"; then
    services_healthy=false
fi

if ! check_service_health "Grafana" "3000"; then
    services_healthy=false
fi

# Check Alertmanager
if curl -s "http://localhost:9093" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Alertmanager is healthy${NC}"
else
    echo -e "${RED}✗ Alertmanager is not responding${NC}"
    services_healthy=false
fi

# Display status
echo ""
echo -e "${BLUE}Monitoring Stack Status${NC}"
echo -e "${BLUE}=======================${NC}"

if [ "$services_healthy" = true ]; then
    echo -e "${GREEN}✓ All monitoring services are running${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo -e "  Prometheus: ${GREEN}http://localhost:9090${NC}"
    echo -e "  Grafana:    ${GREEN}http://localhost:3000${NC} (admin/admin123)"
    echo -e "  Alertmanager: ${GREEN}http://localhost:9093${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Configure your SMTP and Slack settings in $ENV_FILE"
    echo "2. Restart Alertmanager: docker-compose -f $DOCKER_COMPOSE_FILE restart alertmanager"
    echo "3. Import additional Grafana dashboards if needed"
    echo "4. Set up alert notification channels in Grafana"
else
    echo -e "${RED}✗ Some monitoring services failed to start${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "1. Check logs: docker-compose -f $DOCKER_COMPOSE_FILE logs"
    echo "2. Verify port availability: netstat -tlnp | grep ':9090\\|:3000\\|:9093'"
    echo "3. Check configuration files in $MONITORING_DIR"
    echo "4. Restart services: docker-compose -f $DOCKER_COMPOSE_FILE restart"
fi

# Show running containers
echo ""
echo -e "${BLUE}Running Containers:${NC}"
docker-compose -f "$DOCKER_COMPOSE_FILE" ps

echo -e "${BLUE}Monitoring stack startup complete${NC}"