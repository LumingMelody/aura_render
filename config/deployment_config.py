"""
部署配置 - Docker和Kubernetes部署配置管理
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel
import yaml
import json
from pathlib import Path


@dataclass
class DockerConfig:
    """Docker配置"""
    # 基础镜像配置
    base_image: str = "python:3.9-slim"
    image_name: str = "aura-render"
    image_tag: str = "latest"
    registry: Optional[str] = None

    # 构建配置
    dockerfile_path: str = "Dockerfile"
    build_context: str = "."
    build_args: Dict[str, str] = field(default_factory=dict)

    # 运行时配置
    container_name: str = "aura-render-app"
    ports: Dict[str, str] = field(default_factory=lambda: {"8000": "8000"})
    volumes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    networks: List[str] = field(default_factory=list)

    # 资源限制
    memory_limit: str = "2g"
    cpu_limit: str = "2"
    memory_reservation: str = "1g"
    cpu_reservation: str = "1"

    def generate_dockerfile(self) -> str:
        """生成Dockerfile"""
        dockerfile_content = f"""# Aura Render Dockerfile
FROM {self.base_image}

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    ffmpeg \\
    libsm6 \\
    libxext6 \\
    libfontconfig1 \\
    libxrender1 \\
    libgl1-mesa-glx \\
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p logs output temp uploads

# 设置环境变量
ENV PYTHONPATH=/app
ENV AURA_ENV=production

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        return dockerfile_content

    def generate_docker_compose(self) -> str:
        """生成docker-compose.yml"""
        compose_config = {
            'version': '3.8',
            'services': {
                'aura-render': {
                    'build': {
                        'context': self.build_context,
                        'dockerfile': self.dockerfile_path
                    },
                    'image': f"{self.image_name}:{self.image_tag}",
                    'container_name': self.container_name,
                    'ports': [f"{host}:{container}" for container, host in self.ports.items()],
                    'environment': self.environment,
                    'volumes': [f"{host}:{container}" for host, container in self.volumes.items()],
                    'restart': 'unless-stopped',
                    'deploy': {
                        'resources': {
                            'limits': {
                                'memory': self.memory_limit,
                                'cpus': self.cpu_limit
                            },
                            'reservations': {
                                'memory': self.memory_reservation,
                                'cpus': self.cpu_reservation
                            }
                        }
                    },
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    }
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'container_name': 'aura-render-redis',
                    'ports': ['6379:6379'],
                    'volumes': ['redis_data:/data'],
                    'restart': 'unless-stopped'
                },
                'postgres': {
                    'image': 'postgres:15-alpine',
                    'container_name': 'aura-render-postgres',
                    'ports': ['5432:5432'],
                    'environment': {
                        'POSTGRES_DB': 'aura_render',
                        'POSTGRES_USER': 'aura_user',
                        'POSTGRES_PASSWORD': 'aura_password'
                    },
                    'volumes': ['postgres_data:/var/lib/postgresql/data'],
                    'restart': 'unless-stopped'
                }
            },
            'volumes': {
                'redis_data': {},
                'postgres_data': {}
            },
            'networks': {
                'aura-network': {
                    'driver': 'bridge'
                }
            }
        }

        # 添加网络配置
        if self.networks:
            for service in compose_config['services'].values():
                service['networks'] = self.networks
        else:
            for service in compose_config['services'].values():
                service['networks'] = ['aura-network']

        return yaml.dump(compose_config, default_flow_style=False)

    def generate_dockerignore(self) -> str:
        """生成.dockerignore文件"""
        dockerignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Temporary files
temp/
tmp/

# Output files
output/

# Test files
.pytest_cache/
.coverage
htmlcov/

# Git
.git/
.gitignore

# Docker
Dockerfile
docker-compose*.yml
.dockerignore

# Documentation
docs/
*.md

# Config files (security)
config/local.yaml
config/production.yaml
.env
"""
        return dockerignore_content


@dataclass
class KubernetesConfig:
    """Kubernetes配置"""
    # 基础配置
    namespace: str = "aura-render"
    app_name: str = "aura-render"
    version: str = "v1"

    # 镜像配置
    image: str = "aura-render:latest"
    image_pull_policy: str = "IfNotPresent"

    # 副本配置
    replicas: int = 3
    max_surge: int = 1
    max_unavailable: int = 1

    # 资源配置
    cpu_request: str = "500m"
    memory_request: str = "1Gi"
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"

    # 服务配置
    service_type: str = "ClusterIP"
    service_port: int = 8000
    target_port: int = 8000

    # 存储配置
    storage_class: str = "standard"
    storage_size: str = "10Gi"

    # 环境变量
    environment: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)

    def generate_namespace(self) -> str:
        """生成命名空间配置"""
        return f"""apiVersion: v1
kind: Namespace
metadata:
  name: {self.namespace}
  labels:
    app: {self.app_name}
"""

    def generate_deployment(self) -> str:
        """生成Deployment配置"""
        env_vars = []
        for key, value in self.environment.items():
            env_vars.append({
                'name': key,
                'value': value
            })

        for key, secret_key in self.secrets.items():
            env_vars.append({
                'name': key,
                'valueFrom': {
                    'secretKeyRef': {
                        'name': f"{self.app_name}-secrets",
                        'key': secret_key
                    }
                }
            })

        deployment_config = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.app_name,
                'namespace': self.namespace,
                'labels': {
                    'app': self.app_name,
                    'version': self.version
                }
            },
            'spec': {
                'replicas': self.replicas,
                'strategy': {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxSurge': self.max_surge,
                        'maxUnavailable': self.max_unavailable
                    }
                },
                'selector': {
                    'matchLabels': {
                        'app': self.app_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.app_name,
                            'version': self.version
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.app_name,
                            'image': self.image,
                            'imagePullPolicy': self.image_pull_policy,
                            'ports': [{
                                'containerPort': self.target_port,
                                'name': 'http'
                            }],
                            'env': env_vars,
                            'resources': {
                                'requests': {
                                    'cpu': self.cpu_request,
                                    'memory': self.memory_request
                                },
                                'limits': {
                                    'cpu': self.cpu_limit,
                                    'memory': self.memory_limit
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 'http'
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 'http'
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            },
                            'volumeMounts': [{
                                'name': 'app-storage',
                                'mountPath': '/app/data'
                            }]
                        }],
                        'volumes': [{
                            'name': 'app-storage',
                            'persistentVolumeClaim': {
                                'claimName': f"{self.app_name}-pvc"
                            }
                        }]
                    }
                }
            }
        }

        return yaml.dump(deployment_config, default_flow_style=False)

    def generate_service(self) -> str:
        """生成Service配置"""
        service_config = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.app_name}-service",
                'namespace': self.namespace,
                'labels': {
                    'app': self.app_name
                }
            },
            'spec': {
                'type': self.service_type,
                'ports': [{
                    'port': self.service_port,
                    'targetPort': self.target_port,
                    'protocol': 'TCP',
                    'name': 'http'
                }],
                'selector': {
                    'app': self.app_name
                }
            }
        }

        return yaml.dump(service_config, default_flow_style=False)

    def generate_persistent_volume_claim(self) -> str:
        """生成PVC配置"""
        pvc_config = {
            'apiVersion': 'v1',
            'kind': 'PersistentVolumeClaim',
            'metadata': {
                'name': f"{self.app_name}-pvc",
                'namespace': self.namespace
            },
            'spec': {
                'accessModes': ['ReadWriteOnce'],
                'storageClassName': self.storage_class,
                'resources': {
                    'requests': {
                        'storage': self.storage_size
                    }
                }
            }
        }

        return yaml.dump(pvc_config, default_flow_style=False)

    def generate_secret(self) -> str:
        """生成Secret配置"""
        import base64

        secret_data = {}
        for key, value in self.secrets.items():
            secret_data[key] = base64.b64encode(value.encode()).decode()

        secret_config = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': f"{self.app_name}-secrets",
                'namespace': self.namespace
            },
            'type': 'Opaque',
            'data': secret_data
        }

        return yaml.dump(secret_config, default_flow_style=False)

    def generate_configmap(self) -> str:
        """生成ConfigMap配置"""
        configmap_config = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.app_name}-config",
                'namespace': self.namespace
            },
            'data': self.environment
        }

        return yaml.dump(configmap_config, default_flow_style=False)

    def generate_hpa(self) -> str:
        """生成HPA配置"""
        hpa_config = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.app_name}-hpa",
                'namespace': self.namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.app_name
                },
                'minReplicas': 2,
                'maxReplicas': 10,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }

        return yaml.dump(hpa_config, default_flow_style=False)


class DeploymentConfig:
    """部署配置管理器"""

    def __init__(self, output_dir: str = "deploy"):
        self.output_dir = Path(output_dir)
        self.docker_config = DockerConfig()
        self.k8s_config = KubernetesConfig()

    def generate_docker_files(self) -> Dict[str, str]:
        """生成Docker相关文件"""
        files = {
            'Dockerfile': self.docker_config.generate_dockerfile(),
            'docker-compose.yml': self.docker_config.generate_docker_compose(),
            '.dockerignore': self.docker_config.generate_dockerignore()
        }

        return files

    def generate_kubernetes_files(self) -> Dict[str, str]:
        """生成Kubernetes相关文件"""
        files = {
            'namespace.yaml': self.k8s_config.generate_namespace(),
            'deployment.yaml': self.k8s_config.generate_deployment(),
            'service.yaml': self.k8s_config.generate_service(),
            'pvc.yaml': self.k8s_config.generate_persistent_volume_claim(),
            'secret.yaml': self.k8s_config.generate_secret(),
            'configmap.yaml': self.k8s_config.generate_configmap(),
            'hpa.yaml': self.k8s_config.generate_hpa()
        }

        return files

    def write_docker_files(self):
        """写入Docker文件"""
        files = self.generate_docker_files()

        for filename, content in files.items():
            file_path = self.output_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ Generated {filename}")

    def write_kubernetes_files(self):
        """写入Kubernetes文件"""
        k8s_dir = self.output_dir / "k8s"
        k8s_dir.mkdir(parents=True, exist_ok=True)

        files = self.generate_kubernetes_files()

        for filename, content in files.items():
            file_path = k8s_dir / filename

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ Generated k8s/{filename}")

    def generate_all_files(self):
        """生成所有部署文件"""
        print("🚀 Generating deployment files...")

        self.write_docker_files()
        self.write_kubernetes_files()

        # 生成部署脚本
        self._generate_deploy_scripts()

        print("✅ All deployment files generated successfully!")

    def _generate_deploy_scripts(self):
        """生成部署脚本"""
        # Docker部署脚本
        docker_script = """#!/bin/bash
# Docker deployment script

set -e

echo "Building Docker image..."
docker build -t aura-render:latest .

echo "Starting services with docker-compose..."
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 30

echo "Checking service health..."
docker-compose ps

echo "Docker deployment completed!"
"""

        docker_script_path = self.output_dir / "deploy-docker.sh"
        with open(docker_script_path, 'w') as f:
            f.write(docker_script)
        docker_script_path.chmod(0o755)

        # Kubernetes部署脚本
        k8s_script = """#!/bin/bash
# Kubernetes deployment script

set -e

NAMESPACE="aura-render"

echo "Creating namespace..."
kubectl apply -f k8s/namespace.yaml

echo "Creating secrets and configmaps..."
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml

echo "Creating persistent volume claim..."
kubectl apply -f k8s/pvc.yaml

echo "Deploying application..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/aura-render -n $NAMESPACE

echo "Getting service information..."
kubectl get services -n $NAMESPACE

echo "Kubernetes deployment completed!"
"""

        k8s_script_path = self.output_dir / "deploy-k8s.sh"
        with open(k8s_script_path, 'w') as f:
            f.write(k8s_script)
        k8s_script_path.chmod(0o755)

        print("✅ Generated deployment scripts")

    def update_docker_config(self, **kwargs):
        """更新Docker配置"""
        for key, value in kwargs.items():
            if hasattr(self.docker_config, key):
                setattr(self.docker_config, key, value)

    def update_k8s_config(self, **kwargs):
        """更新Kubernetes配置"""
        for key, value in kwargs.items():
            if hasattr(self.k8s_config, key):
                setattr(self.k8s_config, key, value)