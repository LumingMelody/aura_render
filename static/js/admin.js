/**
 * Aura Render Admin Dashboard
 * Advanced admin panel with real-time monitoring and WebSocket integration
 */

class AdminDashboard {
    constructor() {
        this.apiBaseUrl = window.location.origin + '/api';
        this.wsConnection = null;
        this.connectionId = this.generateConnectionId();
        
        // Chart instances
        this.charts = {};
        
        // Data storage
        this.dashboardData = {};
        this.systemMetrics = [];
        this.logs = [];
        
        // Initialize
        this.init();
    }
    
    generateConnectionId() {
        return 'admin_' + Math.random().toString(36).substring(2, 15);
    }
    
    async init() {
        console.log('🚀 Initializing Admin Dashboard...');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize WebSocket connection
        this.initWebSocket();
        
        // Load initial data
        await this.loadDashboardData();
        
        // Initialize charts
        this.initCharts();
        
        // Start periodic updates
        this.startPeriodicUpdates();
        
        console.log('✅ Admin Dashboard initialized');
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.closest('.nav-link').dataset.section;
                this.switchSection(section);
            });
        });
        
        // Analytics time range change
        const timeRangeSelect = document.getElementById('analyticsTimeRange');
        if (timeRangeSelect) {
            timeRangeSelect.addEventListener('change', () => {
                this.loadAnalyticsData();
            });
        }
    }
    
    switchSection(section) {
        // Hide all sections
        document.querySelectorAll('.content-section').forEach(sec => {
            sec.style.display = 'none';
        });
        
        // Show selected section
        const targetSection = document.getElementById(section);
        if (targetSection) {
            targetSection.style.display = 'block';
        }
        
        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-section="${section}"]`).classList.add('active');
        
        // Load section-specific data
        this.loadSectionData(section);
    }
    
    async loadSectionData(section) {
        try {
            switch (section) {
                case 'dashboard':
                    await this.loadDashboardData();
                    break;
                case 'analytics':
                    await this.loadAnalyticsData();
                    break;
                case 'system':
                    await this.loadSystemMonitoring();
                    break;
                case 'logs':
                    await this.loadSystemLogs();
                    break;
                default:
                    console.log(`Section ${section} data loading not implemented`);
            }
        } catch (error) {
            console.error(`Error loading ${section} data:`, error);
            this.showError(`Failed to load ${section} data`);
        }
    }
    
    initWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/${this.connectionId}`;
        
        console.log('🔌 Connecting to WebSocket:', wsUrl);
        
        this.wsConnection = new WebSocket(wsUrl);
        
        this.wsConnection.onopen = () => {
            console.log('✅ WebSocket connected');
            this.updateConnectionStatus(true);
            
            // Subscribe to admin topics
            this.subscribeToTopics([
                'system_alerts',
                'system_metrics',
                'video_progress',
                'batch_progress',
                'admin'
            ]);
        };
        
        this.wsConnection.onclose = () => {
            console.log('❌ WebSocket disconnected');
            this.updateConnectionStatus(false);
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => {
                console.log('🔄 Attempting to reconnect...');
                this.initWebSocket();
            }, 3000);
        };
        
        this.wsConnection.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        this.wsConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
    }
    
    subscribeToTopics(topics) {
        topics.forEach(topic => {
            if (this.wsConnection.readyState === WebSocket.OPEN) {
                this.wsConnection.send(JSON.stringify({
                    type: 'subscribe',
                    topic: topic
                }));
            }
        });
    }
    
    handleWebSocketMessage(message) {
        console.log('📨 WebSocket message:', message);
        
        switch (message.type) {
            case 'system_metrics':
                this.updateSystemMetrics(message.metrics);
                break;
            case 'system_alert':
                this.handleSystemAlert(message);
                break;
            case 'video_generation':
                this.handleVideoProgress(message);
                break;
            case 'batch_job':
                this.handleBatchProgress(message);
                break;
            case 'connection':
                console.log('Connection status:', message.status);
                break;
            default:
                console.log('Unhandled message type:', message.type);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            if (connected) {
                statusElement.className = 'connection-status';
                statusElement.innerHTML = '<i class="fas fa-wifi"></i> 已连接';
            } else {
                statusElement.className = 'connection-status disconnected';
                statusElement.innerHTML = '<i class="fas fa-wifi"></i> 连接中断';
            }
        }
    }
    
    async loadDashboardData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/analytics/dashboard`);
            const data = await response.json();
            
            this.dashboardData = data;
            this.updateDashboardMetrics(data);
            this.updateRecentActivities();
            
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showError('Failed to load dashboard data');
        }
    }
    
    updateDashboardMetrics(data) {
        // Update key metrics
        if (data.video) {
            document.getElementById('totalVideos').textContent = data.video.total_videos_generated || 0;
            document.getElementById('successRate').textContent = 
                Math.round((data.video.success_rate || 0) * 100) + '%';
        }
        
        if (data.performance) {
            document.getElementById('systemLoad').textContent = 
                Math.round(data.performance.cpu_usage || 0) + '%';
        }
        
        // Mock active users (would come from real analytics)\n        document.getElementById('activeUsers').textContent = '24';\n        \n        // Update trend indicators\n        this.updateMetricChange('videoChange', '+12% 较昨日', true);\n        this.updateMetricChange('userChange', '+8% 较昨日', true);\n        this.updateMetricChange('loadChange', 'CPU 使用率');\n        this.updateMetricChange('successChange', '视频生成成功率');\n    }\n    \n    updateMetricChange(elementId, text, isPositive = null) {\n        const element = document.getElementById(elementId);\n        if (element) {\n            element.textContent = text;\n            if (isPositive !== null) {\n                element.className = `metric-change ${isPositive ? 'positive' : 'negative'}`;\n            }\n        }\n    }\n    \n    updateRecentActivities() {\n        const activitiesContainer = document.getElementById('recentActivities');\n        if (!activitiesContainer) return;\n        \n        // Mock recent activities (would come from real data)\n        const activities = [\n            {\n                type: 'success',\n                message: '用户 user123 成功生成了一个营销视频',\n                time: '2 分钟前',\n                icon: 'fa-video'\n            },\n            {\n                type: 'info',\n                message: '批处理任务 batch_001 已开始执行',\n                time: '5 分钟前',\n                icon: 'fa-layer-group'\n            },\n            {\n                type: 'warning',\n                message: '系统 CPU 使用率达到 85%',\n                time: '8 分钟前',\n                icon: 'fa-exclamation-triangle'\n            },\n            {\n                type: 'success',\n                message: '新用户 user456 完成注册',\n                time: '12 分钟前',\n                icon: 'fa-user-plus'\n            }\n        ];\n        \n        activitiesContainer.innerHTML = activities.map(activity => `\n            <div class=\"d-flex align-items-center mb-3\">\n                <div class=\"me-3\">\n                    <i class=\"fas ${activity.icon} text-${this.getActivityColor(activity.type)}\"></i>\n                </div>\n                <div class=\"flex-grow-1\">\n                    <div class=\"fw-medium\">${activity.message}</div>\n                    <small class=\"text-muted\">${activity.time}</small>\n                </div>\n            </div>\n        `).join('');\n    }\n    \n    getActivityColor(type) {\n        const colors = {\n            success: 'success',\n            info: 'info',\n            warning: 'warning',\n            error: 'danger'\n        };\n        return colors[type] || 'secondary';\n    }\n    \n    async loadAnalyticsData() {\n        try {\n            const timeRange = document.getElementById('analyticsTimeRange')?.value || '24';\n            const response = await fetch(`${this.apiBaseUrl}/analytics/summary?hours=${timeRange}`);\n            const data = await response.json();\n            \n            this.updateAnalyticsCharts(data);\n            \n        } catch (error) {\n            console.error('Error loading analytics data:', error);\n            this.showError('Failed to load analytics data');\n        }\n    }\n    \n    updateAnalyticsCharts(data) {\n        // Update video analytics chart\n        if (this.charts.videoAnalytics) {\n            // Mock data - in reality, this would come from the analytics API\n            const videoData = {\n                labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],\n                datasets: [{\n                    label: '视频生成数量',\n                    data: [12, 19, 15, 25, 22, 18],\n                    borderColor: '#3b82f6',\n                    backgroundColor: 'rgba(59, 130, 246, 0.1)'\n                }]\n            };\n            \n            this.charts.videoAnalytics.data = videoData;\n            this.charts.videoAnalytics.update();\n        }\n        \n        // Update user activity chart\n        if (this.charts.userActivity) {\n            const userData = {\n                labels: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],\n                datasets: [{\n                    label: '活跃用户',\n                    data: [65, 78, 90, 81, 56, 55, 70],\n                    borderColor: '#10b981',\n                    backgroundColor: 'rgba(16, 185, 129, 0.1)'\n                }]\n            };\n            \n            this.charts.userActivity.data = userData;\n            this.charts.userActivity.update();\n        }\n    }\n    \n    async loadSystemMonitoring() {\n        try {\n            const response = await fetch(`${this.apiBaseUrl}/analytics/performance`);\n            const data = await response.json();\n            \n            this.updateSystemMetrics(data);\n            \n        } catch (error) {\n            console.error('Error loading system monitoring data:', error);\n            this.showError('Failed to load system monitoring data');\n        }\n    }\n    \n    updateSystemMetrics(metrics) {\n        // Update progress bars\n        this.updateProgressBar('cpuProgress', 'cpuText', metrics.cpu_usage || 0, '%');\n        this.updateProgressBar('memoryProgress', 'memoryText', metrics.memory_usage || 0, '%');\n        this.updateProgressBar('diskProgress', 'diskText', metrics.disk_usage || 0, '%');\n        \n        // Update real-time chart\n        if (this.charts.systemMonitor) {\n            const now = new Date();\n            this.systemMetrics.push({\n                time: now.toLocaleTimeString(),\n                cpu: metrics.cpu_usage || 0,\n                memory: metrics.memory_usage || 0,\n                timestamp: now.getTime()\n            });\n            \n            // Keep only last 20 data points\n            if (this.systemMetrics.length > 20) {\n                this.systemMetrics = this.systemMetrics.slice(-20);\n            }\n            \n            this.charts.systemMonitor.data.labels = this.systemMetrics.map(m => m.time);\n            this.charts.systemMonitor.data.datasets[0].data = this.systemMetrics.map(m => m.cpu);\n            this.charts.systemMonitor.data.datasets[1].data = this.systemMetrics.map(m => m.memory);\n            this.charts.systemMonitor.update('none');\n        }\n    }\n    \n    updateProgressBar(progressId, textId, value, unit) {\n        const progressBar = document.getElementById(progressId);\n        const textElement = document.getElementById(textId);\n        \n        if (progressBar) {\n            progressBar.style.width = `${value}%`;\n            \n            // Change color based on value\n            if (value > 80) {\n                progressBar.className = 'progress-bar bg-danger';\n            } else if (value > 60) {\n                progressBar.className = 'progress-bar bg-warning';\n            } else {\n                progressBar.className = 'progress-bar bg-success';\n            }\n        }\n        \n        if (textElement) {\n            textElement.textContent = `${Math.round(value)}${unit}`;\n        }\n    }\n    \n    async loadSystemLogs() {\n        // Mock system logs - in reality, these would come from a logging API\n        const mockLogs = [\n            {\n                level: 'info',\n                message: '视频生成任务 task_001 已开始处理',\n                timestamp: new Date(Date.now() - 60000)\n            },\n            {\n                level: 'success',\n                message: '批处理任务 batch_001 已完成，成功处理 25 个项目',\n                timestamp: new Date(Date.now() - 120000)\n            },\n            {\n                level: 'warning',\n                message: 'Redis 连接池使用率达到 80%',\n                timestamp: new Date(Date.now() - 180000)\n            },\n            {\n                level: 'error',\n                message: '视频渲染失败：模板文件不存在',\n                timestamp: new Date(Date.now() - 240000)\n            },\n            {\n                level: 'info',\n                message: '系统定时任务执行完成',\n                timestamp: new Date(Date.now() - 300000)\n            }\n        ];\n        \n        this.displaySystemLogs(mockLogs);\n    }\n    \n    displaySystemLogs(logs) {\n        const logsContainer = document.getElementById('systemLogs');\n        if (!logsContainer) return;\n        \n        logsContainer.innerHTML = logs.map(log => `\n            <div class=\"log-entry ${log.level}\">\n                <div class=\"d-flex justify-content-between align-items-start\">\n                    <div class=\"flex-grow-1\">\n                        <div class=\"fw-medium\">${log.message}</div>\n                    </div>\n                    <small class=\"text-muted ms-3\">${log.timestamp.toLocaleString()}</small>\n                </div>\n            </div>\n        `).join('');\n    }\n    \n    handleSystemAlert(alert) {\n        console.log('🚨 System Alert:', alert);\n        \n        // Add alert to logs\n        this.logs.unshift({\n            level: alert.level,\n            message: alert.message,\n            timestamp: new Date(alert.timestamp)\n        });\n        \n        // Show browser notification if supported\n        if ('Notification' in window && Notification.permission === 'granted') {\n            new Notification('Aura Render Alert', {\n                body: alert.message,\n                icon: '/static/favicon.ico'\n            });\n        }\n        \n        // Update logs display if we're on the logs section\n        const logsSection = document.getElementById('logs');\n        if (logsSection && logsSection.style.display !== 'none') {\n            this.displaySystemLogs(this.logs);\n        }\n    }\n    \n    handleVideoProgress(message) {\n        console.log('📹 Video Progress:', message);\n        // Handle video generation progress updates\n        // Could update a progress indicator or notifications\n    }\n    \n    handleBatchProgress(message) {\n        console.log('📦 Batch Progress:', message);\n        // Handle batch job progress updates\n        // Could update batch job status displays\n    }\n    \n    initCharts() {\n        // Video trend chart\n        const videoTrendCtx = document.getElementById('videoTrendChart');\n        if (videoTrendCtx) {\n            this.charts.videoTrend = new Chart(videoTrendCtx, {\n                type: 'line',\n                data: {\n                    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],\n                    datasets: [{\n                        label: '视频生成数量',\n                        data: [12, 19, 15, 25, 22, 18],\n                        borderColor: '#3b82f6',\n                        backgroundColor: 'rgba(59, 130, 246, 0.1)',\n                        tension: 0.4,\n                        fill: true\n                    }]\n                },\n                options: {\n                    responsive: true,\n                    maintainAspectRatio: false,\n                    plugins: {\n                        legend: {\n                            display: false\n                        }\n                    },\n                    scales: {\n                        y: {\n                            beginAtZero: true\n                        }\n                    }\n                }\n            });\n        }\n        \n        // Video analytics chart\n        const videoAnalyticsCtx = document.getElementById('videoAnalyticsChart');\n        if (videoAnalyticsCtx) {\n            this.charts.videoAnalytics = new Chart(videoAnalyticsCtx, {\n                type: 'bar',\n                data: {\n                    labels: [],\n                    datasets: [{\n                        label: '视频生成',\n                        data: [],\n                        backgroundColor: '#3b82f6'\n                    }]\n                },\n                options: {\n                    responsive: true,\n                    maintainAspectRatio: false\n                }\n            });\n        }\n        \n        // User activity chart\n        const userActivityCtx = document.getElementById('userActivityChart');\n        if (userActivityCtx) {\n            this.charts.userActivity = new Chart(userActivityCtx, {\n                type: 'line',\n                data: {\n                    labels: [],\n                    datasets: [{\n                        label: '活跃用户',\n                        data: [],\n                        borderColor: '#10b981',\n                        backgroundColor: 'rgba(16, 185, 129, 0.1)'\n                    }]\n                },\n                options: {\n                    responsive: true,\n                    maintainAspectRatio: false\n                }\n            });\n        }\n        \n        // Performance chart\n        const performanceCtx = document.getElementById('performanceChart');\n        if (performanceCtx) {\n            this.charts.performance = new Chart(performanceCtx, {\n                type: 'line',\n                data: {\n                    labels: [],\n                    datasets: [\n                        {\n                            label: 'CPU 使用率',\n                            data: [],\n                            borderColor: '#f59e0b',\n                            backgroundColor: 'rgba(245, 158, 11, 0.1)'\n                        },\n                        {\n                            label: '内存使用率',\n                            data: [],\n                            borderColor: '#ef4444',\n                            backgroundColor: 'rgba(239, 68, 68, 0.1)'\n                        }\n                    ]\n                },\n                options: {\n                    responsive: true,\n                    maintainAspectRatio: false,\n                    scales: {\n                        y: {\n                            beginAtZero: true,\n                            max: 100\n                        }\n                    }\n                }\n            });\n        }\n        \n        // System monitor chart\n        const systemMonitorCtx = document.getElementById('systemMonitorChart');\n        if (systemMonitorCtx) {\n            this.charts.systemMonitor = new Chart(systemMonitorCtx, {\n                type: 'line',\n                data: {\n                    labels: [],\n                    datasets: [\n                        {\n                            label: 'CPU',\n                            data: [],\n                            borderColor: '#3b82f6',\n                            backgroundColor: 'rgba(59, 130, 246, 0.1)',\n                            tension: 0.4\n                        },\n                        {\n                            label: '内存',\n                            data: [],\n                            borderColor: '#10b981',\n                            backgroundColor: 'rgba(16, 185, 129, 0.1)',\n                            tension: 0.4\n                        }\n                    ]\n                },\n                options: {\n                    responsive: true,\n                    maintainAspectRatio: false,\n                    animation: {\n                        duration: 0\n                    },\n                    scales: {\n                        y: {\n                            beginAtZero: true,\n                            max: 100\n                        }\n                    },\n                    plugins: {\n                        legend: {\n                            display: true,\n                            position: 'top'\n                        }\n                    }\n                }\n            });\n        }\n    }\n    \n    startPeriodicUpdates() {\n        // Update dashboard every 30 seconds\n        setInterval(async () => {\n            if (document.getElementById('dashboard').style.display !== 'none') {\n                await this.loadDashboardData();\n            }\n        }, 30000);\n        \n        // Update system monitoring every 10 seconds\n        setInterval(async () => {\n            if (document.getElementById('system').style.display !== 'none') {\n                await this.loadSystemMonitoring();\n            }\n        }, 10000);\n    }\n    \n    showError(message) {\n        console.error('❌ Error:', message);\n        \n        // Create a toast or alert to show the error\n        // This is a simple implementation - you might want to use a more sophisticated notification system\n        const alert = document.createElement('div');\n        alert.className = 'alert alert-danger alert-dismissible fade show position-fixed';\n        alert.style.cssText = 'top: 80px; right: 20px; z-index: 2000; min-width: 300px;';\n        alert.innerHTML = `\n            <i class=\"fas fa-exclamation-triangle\"></i> ${message}\n            <button type=\"button\" class=\"btn-close\" data-bs-dismiss=\"alert\"></button>\n        `;\n        \n        document.body.appendChild(alert);\n        \n        // Auto-remove after 5 seconds\n        setTimeout(() => {\n            if (alert.parentNode) {\n                alert.parentNode.removeChild(alert);\n            }\n        }, 5000);\n    }\n}\n\n// Global functions\nfunction refreshDashboard() {\n    if (window.adminDashboard) {\n        window.adminDashboard.loadDashboardData();\n    }\n}\n\nfunction clearLogs() {\n    const logsContainer = document.getElementById('systemLogs');\n    if (logsContainer) {\n        logsContainer.innerHTML = '<div class=\"text-center text-muted py-4\">日志已清空</div>';\n    }\n    \n    if (window.adminDashboard) {\n        window.adminDashboard.logs = [];\n    }\n}\n\n// Request notification permission on load\nif ('Notification' in window && Notification.permission === 'default') {\n    Notification.requestPermission();\n}\n\n// Initialize admin dashboard when DOM is loaded\ndocument.addEventListener('DOMContentLoaded', () => {\n    window.adminDashboard = new AdminDashboard();\n});