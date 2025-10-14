/**
 * Aura Render Web UI - JavaScript Application
 * 
 * Handles all frontend functionality including:
 * - Video generation form handling
 * - Real-time progress tracking
 * - Task management
 * - Gallery display
 * - API communication
 */

class AuraRenderApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.currentTaskId = null;
        this.progressInterval = null;
        this.keywords = [];
        
        this.init();
    }

    async init() {
        this.bindEvents();
        this.loadTemplates();
        await this.checkSystemStatus();
        this.showSection('generator');
    }

    bindEvents() {
        // Form submission
        document.getElementById('videoGeneratorForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitVideoGeneration();
        });

        // Keyword input
        document.getElementById('keywordInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.addKeyword();
            }
        });

        // Duration controls
        document.getElementById('duration').addEventListener('input', this.updateDurationDisplay);
        document.getElementById('durationInput').addEventListener('change', this.updateDurationSlider);

        // Character count for description
        document.getElementById('description').addEventListener('input', this.updateCharCount);

        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.getAttribute('href').substring(1);
                this.showSection(section);
                this.updateNavigation(link);
            });
        });
    }

    // Section Management
    showSection(sectionName) {
        // Hide all sections
        const sections = ['generator', 'progress', 'results', 'gallery', 'tasks'];
        sections.forEach(section => {
            const element = document.getElementById(section);
            if (element) {
                element.style.display = 'none';
            }
        });

        // Show target section
        const targetSection = document.getElementById(sectionName);
        if (targetSection) {
            targetSection.style.display = 'block';
        }

        // Load section-specific data
        switch (sectionName) {
            case 'gallery':
                this.loadGallery();
                break;
            case 'tasks':
                this.loadTasks();
                break;
        }
    }

    updateNavigation(activeLink) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        activeLink.classList.add('active');
    }

    // Video Generation
    async submitVideoGeneration() {
        const formData = this.getFormData();
        
        if (!this.validateForm(formData)) {
            return;
        }

        try {
            this.showLoading();
            
            // Submit to async endpoint
            const response = await axios.post(`${this.apiBaseUrl}/tasks/video/async`, formData);
            
            if (response.data.task_id) {
                this.currentTaskId = response.data.task_id;
                this.showProgress();
                this.startProgressTracking();
                
                this.showToast('视频生成已开始', 'success');
            } else {
                throw new Error('未收到任务ID');
            }
        } catch (error) {
            console.error('Video generation failed:', error);
            this.showToast('视频生成失败: ' + (error.response?.data?.detail || error.message), 'error');
        } finally {
            this.hideLoading();
        }
    }

    getFormData() {
        return {
            theme_id: document.getElementById('theme').value,
            keywords_id: this.keywords,
            target_duration_id: parseInt(document.getElementById('duration').value),
            user_description_id: document.getElementById('description').value,
            priority: document.getElementById('priority').value,
            config: {
                quality: document.getElementById('quality').value,
                format: document.getElementById('format').value,
                resolution: document.getElementById('resolution').value
            }
        };
    }

    validateForm(formData) {
        if (!formData.theme_id) {
            this.showToast('请选择视频主题', 'warning');
            return false;
        }

        if (formData.keywords_id.length === 0) {
            this.showToast('请至少添加一个关键词', 'warning');
            return false;
        }

        if (!formData.user_description_id || formData.user_description_id.length < 10) {
            this.showToast('请提供详细描述（至少10个字符）', 'warning');
            return false;
        }

        return true;
    }

    // Progress Tracking
    showProgress() {
        this.showSection('progress');
        document.getElementById('taskId').textContent = this.currentTaskId;
        document.getElementById('startTime').textContent = new Date().toLocaleString();
        this.resetProgressIndicators();
    }

    resetProgressIndicators() {
        const steps = ['step1', 'step2', 'step3', 'step4'];
        steps.forEach(stepId => {
            const step = document.getElementById(stepId);
            step.classList.remove('active', 'completed');
        });
        
        this.updateProgress(0, '初始化', '任务已提交，等待处理...');
    }

    async startProgressTracking() {
        this.progressInterval = setInterval(async () => {
            try {
                const response = await axios.get(`${this.apiBaseUrl}/tasks/status/${this.currentTaskId}`);
                const taskData = response.data;
                
                this.updateProgressFromTask(taskData);
                
                if (taskData.status === 'completed') {
                    this.onTaskCompleted(taskData);
                } else if (taskData.status === 'failed') {
                    this.onTaskFailed(taskData);
                }
            } catch (error) {
                console.error('Failed to fetch task status:', error);
                this.stopProgressTracking();
            }
        }, 2000); // Check every 2 seconds
    }

    updateProgressFromTask(taskData) {
        const progress = taskData.progress || 0;
        const status = taskData.status;
        const message = taskData.message || '处理中...';
        
        this.updateProgress(progress, this.getStageFromStatus(status, progress), message);
        this.updateStepIndicators(progress);
    }

    getStageFromStatus(status, progress) {
        if (progress < 25) return '内容分析';
        if (progress < 50) return '素材匹配';
        if (progress < 75) return '音频生成';
        if (progress < 100) return '视频合成';
        return '完成';
    }

    updateProgress(percent, stage, message) {
        document.getElementById('progressPercent').textContent = Math.round(percent);
        document.getElementById('progressBar').style.width = `${percent}%`;
        document.getElementById('currentStage').textContent = stage;
        document.getElementById('progressMessage').textContent = message;
    }

    updateStepIndicators(progress) {
        const steps = [
            { id: 'step1', threshold: 25 },
            { id: 'step2', threshold: 50 },
            { id: 'step3', threshold: 75 },
            { id: 'step4', threshold: 100 }
        ];

        steps.forEach((step, index) => {
            const element = document.getElementById(step.id);
            
            if (progress >= step.threshold) {
                element.classList.add('completed');
                element.classList.remove('active');
            } else if (progress >= (steps[index - 1]?.threshold || 0)) {
                element.classList.add('active');
                element.classList.remove('completed');
            } else {
                element.classList.remove('active', 'completed');
            }
        });
    }

    stopProgressTracking() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    async onTaskCompleted(taskData) {
        this.stopProgressTracking();
        this.showResults(taskData);
        this.showToast('视频生成完成！', 'success');
    }

    onTaskFailed(taskData) {
        this.stopProgressTracking();
        this.showToast(`视频生成失败: ${taskData.error || '未知错误'}`, 'error');
    }

    // Results Display
    showResults(taskData) {
        this.showSection('results');
        
        if (taskData.result && taskData.result.output_video_path) {
            const videoElement = document.getElementById('resultVideo');
            videoElement.src = `/static/outputs/${taskData.result.output_video_path}`;
            
            // Update video info
            this.updateVideoInfo(taskData);
        }
    }

    updateVideoInfo(taskData) {
        const result = taskData.result;
        
        if (result.output_metadata) {
            document.getElementById('fileSize').textContent = 
                `${(result.output_metadata.file_size_mb || 0).toFixed(1)} MB`;
            document.getElementById('videoResolution').textContent = 
                result.output_metadata.resolution || '1920x1080';
            document.getElementById('videoDuration').textContent = 
                `${result.output_metadata.duration_seconds || 0}秒`;
        }
        
        document.getElementById('generationTime').textContent = 
            `${(taskData.actual_duration || 0)}秒`;
    }

    // Keywords Management
    addKeyword() {
        const input = document.getElementById('keywordInput');
        const keyword = input.value.trim();
        
        if (keyword && !this.keywords.includes(keyword)) {
            this.keywords.push(keyword);
            this.updateKeywordTags();
            this.updateKeywordsInput();
            input.value = '';
        }
    }

    removeKeyword(keyword) {
        this.keywords = this.keywords.filter(k => k !== keyword);
        this.updateKeywordTags();
        this.updateKeywordsInput();
    }

    updateKeywordTags() {
        const container = document.getElementById('keywordTags');
        container.innerHTML = this.keywords.map(keyword => 
            `<span class="keyword-tag">
                ${keyword}
                <span class="remove-tag" onclick="app.removeKeyword('${keyword}')">&times;</span>
            </span>`
        ).join('');
    }

    updateKeywordsInput() {
        document.getElementById('keywords').value = JSON.stringify(this.keywords);
    }

    // UI Controls
    updateDurationDisplay() {
        const slider = document.getElementById('duration');
        const input = document.getElementById('durationInput');
        input.value = slider.value;
    }

    updateDurationSlider() {
        const slider = document.getElementById('duration');
        const input = document.getElementById('durationInput');
        slider.value = input.value;
    }

    updateCharCount() {
        const textarea = document.getElementById('description');
        const counter = document.getElementById('charCount');
        counter.textContent = textarea.value.length;
        
        if (textarea.value.length > 1000) {
            counter.style.color = '#ef4444';
        } else {
            counter.style.color = '#6b7280';
        }
    }

    // Templates
    loadTemplates() {
        // Templates are already in HTML, could be loaded dynamically here
    }

    useTemplate(templateType) {
        const templates = {
            product: {
                theme: '产品宣传',
                keywords: ['产品', '功能', '特色', '品质'],
                description: '展示我们产品的核心功能和独特优势，突出产品的创新性和实用性，吸引目标用户关注。'
            },
            tech: {
                theme: '科技介绍',
                keywords: ['科技', '创新', '未来', '智能'],
                description: '介绍最新的科技发展和创新应用，展示技术如何改变我们的生活和工作方式。'
            },
            company: {
                theme: '企业展示',
                keywords: ['企业', '团队', '文化', '价值'],
                description: '展示企业文化、团队实力和核心价值观，建立品牌形象和增强客户信任。'
            },
            education: {
                theme: '教育培训',
                keywords: ['学习', '成长', '知识', '技能'],
                description: '提供优质的教育内容和培训服务，帮助学习者获得新知识和技能提升。'
            }
        };

        const template = templates[templateType];
        if (template) {
            document.getElementById('theme').value = template.theme;
            this.keywords = [...template.keywords];
            this.updateKeywordTags();
            this.updateKeywordsInput();
            document.getElementById('description').value = template.description;
            this.updateCharCount();
            
            this.showToast(`已应用${template.theme}模板`, 'success');
        }
    }

    // Gallery Management
    async loadGallery() {
        try {
            const response = await axios.get(`${this.apiBaseUrl}/images/gallery`);
            const images = response.data.images || [];
            
            this.displayGallery(images);
        } catch (error) {
            console.error('Failed to load gallery:', error);
            this.showToast('加载作品库失败', 'error');
        }
    }

    displayGallery(images) {
        const grid = document.getElementById('galleryGrid');
        
        if (images.length === 0) {
            grid.innerHTML = `
                <div class="col-12 text-center py-5">
                    <i class="fas fa-images fa-3x text-muted mb-3"></i>
                    <p class="text-muted">还没有生成的作品</p>
                </div>
            `;
            return;
        }

        grid.innerHTML = images.map(image => `
            <div class="col-md-6 col-lg-4 gallery-item">
                <div class="gallery-card card">
                    <img src="${this.apiBaseUrl}/images/file/${image.thumbnail_path || image.path}" 
                         class="gallery-thumbnail" alt="Generated Image">
                    <div class="gallery-info">
                        <div class="gallery-title">${image.filename}</div>
                        <div class="gallery-meta">
                            <small>创建时间: ${new Date(image.created_at).toLocaleString()}</small><br>
                            <small>文件大小: ${(image.size_bytes / 1024).toFixed(1)} KB</small>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    async refreshGallery() {
        this.showLoading();
        await this.loadGallery();
        this.hideLoading();
        this.showToast('作品库已刷新', 'success');
    }

    // Task Management
    async loadTasks() {
        try {
            const response = await axios.get(`${this.apiBaseUrl}/tasks/history?limit=20`);
            const tasks = response.data || [];
            
            this.displayTasks(tasks);
        } catch (error) {
            console.error('Failed to load tasks:', error);
            this.showToast('加载任务列表失败', 'error');
        }
    }

    displayTasks(tasks) {
        const tbody = document.getElementById('tasksTableBody');
        
        if (tasks.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center py-4">
                        <i class="fas fa-tasks fa-2x text-muted mb-2"></i>
                        <p class="text-muted mb-0">暂无任务记录</p>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = tasks.map(task => `
            <tr>
                <td>
                    <code class="small">${task.task_id}</code>
                </td>
                <td>${this.truncateText(task.message || '视频生成', 20)}</td>
                <td>
                    <span class="status-badge ${task.status}">
                        ${this.getStatusText(task.status)}
                    </span>
                </td>
                <td>
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar" style="width: ${task.progress}%"></div>
                    </div>
                    <small class="text-muted">${Math.round(task.progress)}%</small>
                </td>
                <td>
                    <small>${new Date(task.created_at).toLocaleString()}</small>
                </td>
                <td>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary btn-sm" 
                                onclick="app.viewTask('${task.task_id}')">
                            <i class="fas fa-eye"></i>
                        </button>
                        ${task.status === 'processing' ? `
                            <button class="btn btn-outline-danger btn-sm" 
                                    onclick="app.cancelTask('${task.task_id}')">
                                <i class="fas fa-times"></i>
                            </button>
                        ` : ''}
                    </div>
                </td>
            </tr>
        `).join('');
    }

    getStatusText(status) {
        const statusMap = {
            'pending': '待处理',
            'queued': '队列中',
            'processing': '处理中',
            'completed': '已完成',
            'failed': '失败',
            'cancelled': '已取消'
        };
        return statusMap[status] || status;
    }

    async refreshTasks() {
        this.showLoading();
        await this.loadTasks();
        this.hideLoading();
        this.showToast('任务列表已刷新', 'success');
    }

    async viewTask(taskId) {
        try {
            const response = await axios.get(`${this.apiBaseUrl}/tasks/status/${taskId}`);
            const task = response.data;
            
            if (task.status === 'completed' && task.result) {
                this.currentTaskId = taskId;
                this.showResults(task);
            } else if (task.status === 'processing') {
                this.currentTaskId = taskId;
                this.showProgress();
                this.startProgressTracking();
            } else {
                this.showToast(`任务状态: ${this.getStatusText(task.status)}`, 'info');
            }
        } catch (error) {
            console.error('Failed to view task:', error);
            this.showToast('获取任务信息失败', 'error');
        }
    }

    async cancelTask(taskId = null) {
        const targetTaskId = taskId || this.currentTaskId;
        
        if (!targetTaskId) {
            this.showToast('没有可取消的任务', 'warning');
            return;
        }

        try {
            await axios.delete(`${this.apiBaseUrl}/tasks/cancel/${targetTaskId}`);
            this.showToast('任务已取消', 'success');
            
            if (taskId === this.currentTaskId) {
                this.stopProgressTracking();
                this.showSection('generator');
            }
            
            // Refresh tasks if on tasks page
            if (document.getElementById('tasks').style.display !== 'none') {
                await this.loadTasks();
            }
        } catch (error) {
            console.error('Failed to cancel task:', error);
            this.showToast('取消任务失败', 'error');
        }
    }

    // Utility Functions
    async checkSystemStatus() {
        try {
            const response = await axios.get(`${this.apiBaseUrl}/health`);
            if (response.data.status === 'healthy') {
                this.updateStatusIndicator('系统正常', 'success');
            }
        } catch (error) {
            this.updateStatusIndicator('连接异常', 'error');
        }
    }

    updateStatusIndicator(text, type) {
        const indicator = document.getElementById('status-indicator');
        const icon = indicator.previousElementSibling.querySelector('i');
        
        indicator.textContent = text;
        
        icon.classList.remove('text-success', 'text-warning', 'text-danger');
        switch (type) {
            case 'success':
                icon.classList.add('text-success');
                break;
            case 'warning':
                icon.classList.add('text-warning');
                break;
            case 'error':
                icon.classList.add('text-danger');
                break;
        }
    }

    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('liveToast');
        const toastMessage = document.getElementById('toastMessage');
        
        // Update icon and message
        const icon = toast.querySelector('.toast-header i');
        icon.classList.remove('fa-info-circle', 'fa-check-circle', 'fa-exclamation-triangle', 'fa-times-circle');
        icon.classList.remove('text-primary', 'text-success', 'text-warning', 'text-danger');
        
        switch (type) {
            case 'success':
                icon.classList.add('fa-check-circle', 'text-success');
                break;
            case 'warning':
                icon.classList.add('fa-exclamation-triangle', 'text-warning');
                break;
            case 'error':
                icon.classList.add('fa-times-circle', 'text-danger');
                break;
            default:
                icon.classList.add('fa-info-circle', 'text-primary');
        }
        
        toastMessage.textContent = message;
        
        // Show toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }

    truncateText(text, maxLength) {
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }

    // Action Handlers
    downloadVideo() {
        if (this.currentTaskId) {
            const link = document.createElement('a');
            link.href = `/api/tasks/${this.currentTaskId}/download`;
            link.download = `aura-render-${this.currentTaskId}.mp4`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    shareVideo() {
        if (navigator.share && this.currentTaskId) {
            navigator.share({
                title: 'Aura Render 生成的视频',
                text: '查看我用 AI 生成的视频！',
                url: window.location.origin + `/share/${this.currentTaskId}`
            });
        } else {
            // Fallback: copy link to clipboard
            const url = `${window.location.origin}/share/${this.currentTaskId}`;
            navigator.clipboard.writeText(url).then(() => {
                this.showToast('分享链接已复制到剪贴板', 'success');
            });
        }
    }

    generateNew() {
        this.currentTaskId = null;
        this.showSection('generator');
        // Clear form
        document.getElementById('videoGeneratorForm').reset();
        this.keywords = [];
        this.updateKeywordTags();
        this.updateCharCount();
    }

    showDemo() {
        // Placeholder for demo functionality
        this.showToast('演示功能开发中...', 'info');
    }
}

// Utility Functions
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

// Initialize App
let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new AuraRenderApp();
});

// Export for global access
window.AuraRenderApp = AuraRenderApp;