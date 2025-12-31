/**
 * LLM Training Dashboard - Enhanced Frontend
 * 
 * Features:
 * - Pipeline timeline visualization
 * - Quick stats summary
 * - Training sparklines
 * - Resource monitoring
 * - Expandable detail panels
 * - Event log filtering
 * - Health indicators
 * - ETA and elapsed time displays
 */

const API = {
    state: '/api/state',
    events: '/api/events',
    metrics: '/api/metrics',
    disk: '/api/disk',
    lossHistory: '/api/loss-history'
};

let currentFilter = 'all';
let expandedItems = new Set();
let lossHistory = [];

// ============================================================================
// Utility Functions
// ============================================================================

async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (e) {
        console.error(`Fetch failed: ${url}`, e);
        return null;
    }
}

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

function formatNumber(num) {
    if (!num) return '0';
    return num.toLocaleString();
}

function getStatusIcon(status) {
    switch (status) {
        case 'complete': return '✅';
        case 'running': return '🔄';
        case 'failed': return '❌';
        default: return '⏳';
    }
}

function getStatusClass(status) {
    return `status-${status || 'pending'}`;
}

// ============================================================================
// Pipeline Timeline
// ============================================================================

function renderPipeline(pipeline) {
    const container = document.getElementById('pipeline-timeline');
    if (!container || !pipeline) return;

    container.innerHTML = pipeline.map((stage, i) => `
        <div class="pipeline-step ${getStatusClass(stage.status)}">
            <div class="step-dot"></div>
            <div class="step-label">${stage.label}</div>
            ${i < pipeline.length - 1 ? '<div class="step-connector"></div>' : ''}
        </div>
    `).join('');
}

function updateOverallProgress(progress) {
    const bar = document.getElementById('overall-progress');
    const text = document.getElementById('overall-progress-text');

    if (bar) bar.style.width = `${progress}%`;
    if (text) text.textContent = `${progress.toFixed(1)}%`;
}

// ============================================================================
// Quick Stats
// ============================================================================

function updateQuickStats(state) {
    // Downloads count
    const downloads = Object.values(state.downloads || {});
    const dlComplete = downloads.filter(d => d.status === 'complete').length;
    const dlTotal = downloads.length;
    document.getElementById('downloads-count').textContent = `${dlComplete}/${dlTotal}`;

    // Current activity
    document.getElementById('current-activity').textContent = state.current_activity || 'Ready';

    // Current loss
    const lossEl = document.getElementById('current-loss');
    if (state.current_loss) {
        lossEl.textContent = state.current_loss.toFixed(4);
    } else {
        lossEl.textContent = '—';
    }

    // Total ETA
    const etaEl = document.getElementById('total-eta');
    etaEl.textContent = state.total_eta || '—';
}

// ============================================================================
// Health Indicators
// ============================================================================

function updateHealthIndicators(issues) {
    const container = document.getElementById('health-indicators');
    if (!container) return;

    if (!issues || issues.length === 0) {
        container.innerHTML = '<span class="health-badge health-ok">🟢 Healthy</span>';
        return;
    }

    container.innerHTML = issues.map(issue => `
        <span class="health-badge health-${issue.level}">
            ${issue.level === 'error' ? '🔴' : '🟡'} ${issue.message}
        </span>
    `).join('');
}

// ============================================================================
// Downloads List
// ============================================================================

function renderDownloads(downloads) {
    const container = document.getElementById('downloads-list');
    if (!container) return;

    const items = Object.entries(downloads || {});

    container.innerHTML = items.map(([id, data]) => {
        const isExpanded = expandedItems.has(`dl-${id}`);

        // Build meta text based on status
        let metaText = data.status;
        if (data.status === 'complete') {
            metaText = formatBytes(data.size_bytes);
        } else if (data.status === 'running') {
            if (data.progress_pct !== null && data.progress_pct !== undefined) {
                metaText = `${data.progress_pct.toFixed(1)}%`;
                if (data.documents_done) {
                    metaText += ` • ${formatNumber(data.documents_done)} docs`;
                }
            } else {
                metaText = 'running...';
            }
        }

        return `
            <div class="item-row ${getStatusClass(data.status)}" data-id="dl-${id}">
                <div class="item-main" onclick="toggleExpand('dl-${id}')">
                    <div class="item-info">
                        <span class="item-name">${id}</span>
                        <span class="item-meta">${metaText}</span>
                    </div>
                    ${data.status === 'running' && data.progress_pct ? `
                        <div class="mini-progress">
                            <div class="mini-progress-bar" style="width: ${data.progress_pct}%"></div>
                        </div>
                    ` : ''}
                    <div class="status-indicator ${getStatusClass(data.status)}"></div>
                </div>
                ${isExpanded ? `
                    <div class="item-details">
                        <div class="detail-row"><span>Status:</span><span>${data.status}</span></div>
                        ${data.current_progress ? `<div class="detail-row"><span>Progress:</span><span>${data.current_progress}</span></div>` : ''}
                        ${data.shards_done ? `<div class="detail-row"><span>Shards:</span><span>${data.shards_done}</span></div>` : ''}
                        ${data.documents_done ? `<div class="detail-row"><span>Documents:</span><span>${formatNumber(data.documents_done)}</span></div>` : ''}
                        ${data.file_path ? `<div class="detail-row"><span>Path:</span><span class="mono">${data.file_path}</span></div>` : ''}
                        ${data.size_bytes ? `<div class="detail-row"><span>Size:</span><span>${formatBytes(data.size_bytes)}</span></div>` : ''}
                        ${data.downloaded_at ? `<div class="detail-row"><span>Downloaded:</span><span>${data.downloaded_at}</span></div>` : ''}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// ============================================================================
// Processing List
// ============================================================================

function renderProcessing(stages) {
    const container = document.getElementById('processing-list');
    if (!container) return;

    const relevantStages = Object.entries(stages || {})
        .filter(([id]) => id.includes('clean') || id.includes('tokenize') || id.includes('train_tokenizer'));

    container.innerHTML = relevantStages.map(([id, data]) => {
        const isExpanded = expandedItems.has(`stage-${id}`);
        const details = data.details || {};
        let metaText = data.status;

        if (details.n_documents) metaText = `${formatNumber(details.n_documents)} docs`;
        if (details.tokens_so_far) metaText = `${formatNumber(details.tokens_so_far)} tokens`;

        return `
            <div class="item-row ${getStatusClass(data.status)}" data-id="stage-${id}">
                <div class="item-main" onclick="toggleExpand('stage-${id}')">
                    <div class="item-info">
                        <span class="item-name">${id.replace(/_/g, ' ')}</span>
                        <span class="item-meta">${metaText} ${data.elapsed ? `• ${data.elapsed}` : ''}</span>
                    </div>
                    <div class="status-indicator ${getStatusClass(data.status)}"></div>
                </div>
                ${isExpanded ? `
                    <div class="item-details">
                        <div class="detail-row"><span>Status:</span><span>${data.status}</span></div>
                        ${data.started_at ? `<div class="detail-row"><span>Started:</span><span>${data.started_at}</span></div>` : ''}
                        ${data.completed_at ? `<div class="detail-row"><span>Completed:</span><span>${data.completed_at}</span></div>` : ''}
                        ${data.elapsed ? `<div class="detail-row"><span>Elapsed:</span><span>${data.elapsed}</span></div>` : ''}
                        ${Object.entries(details).map(([k, v]) =>
            `<div class="detail-row"><span>${k}:</span><span>${typeof v === 'number' ? formatNumber(v) : v}</span></div>`
        ).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// ============================================================================
// Training Phases with Sparklines
// ============================================================================

function renderTrainingPhases(phases) {
    const container = document.getElementById('training-list');
    if (!container) return;

    const phaseNames = { '1': 'Grammar (P1)', '2': 'Vocabulary (P2)', '2b': 'Scientific (P2b)' };

    container.innerHTML = Object.entries(phases || {}).map(([id, data]) => {
        const isExpanded = expandedItems.has(`phase-${id}`);
        const current = data.current_step || 0;
        const total = data.total_steps || 1;
        const progress = (current / total * 100).toFixed(1);

        // Get loss history for this phase
        const phaseHistory = lossHistory.filter(h => h.phase === id).slice(-50);
        const sparkline = createSparkline(phaseHistory);

        return `
            <div class="item-row training-item ${getStatusClass(data.status)}" data-id="phase-${id}">
                <div class="item-main" onclick="toggleExpand('phase-${id}')">
                    <div class="item-info">
                        <span class="item-name">${phaseNames[id] || `Phase ${id}`}</span>
                        <span class="item-meta">
                            ${data.status === 'running' ?
                `Step ${formatNumber(current)} / ${formatNumber(total)} (${progress}%)` :
                data.status}
                        </span>
                    </div>
                    <div class="status-indicator ${getStatusClass(data.status)}"></div>
                </div>
                
                ${data.status === 'running' || data.status === 'complete' ? `
                    <div class="phase-progress">
                        <div class="progress-bar-bg small">
                            <div class="progress-bar" style="width: ${progress}%"></div>
                        </div>
                    </div>
                    <div class="phase-metrics">
                        ${data.loss ? `
                            <div class="metric">
                                <span class="metric-label">Loss</span>
                                <span class="metric-value">${data.loss.toFixed(4)}</span>
                            </div>
                        ` : ''}
                        ${sparkline ? `
                            <div class="metric sparkline-container">
                                ${sparkline}
                            </div>
                        ` : ''}
                        ${data.elapsed ? `
                            <div class="metric">
                                <span class="metric-label">Elapsed</span>
                                <span class="metric-value">${data.elapsed}</span>
                            </div>
                        ` : ''}
                        ${data.eta ? `
                            <div class="metric">
                                <span class="metric-label">ETA</span>
                                <span class="metric-value">${data.eta}</span>
                            </div>
                        ` : ''}
                    </div>
                ` : ''}
                
                ${isExpanded ? `
                    <div class="item-details">
                        <div class="detail-row"><span>Status:</span><span>${data.status}</span></div>
                        <div class="detail-row"><span>Steps:</span><span>${formatNumber(current)} / ${formatNumber(total)}</span></div>
                        ${data.loss ? `<div class="detail-row"><span>Loss:</span><span>${data.loss.toFixed(6)}</span></div>` : ''}
                        ${data.started_at ? `<div class="detail-row"><span>Started:</span><span>${data.started_at}</span></div>` : ''}
                        ${data.completed_at ? `<div class="detail-row"><span>Completed:</span><span>${data.completed_at}</span></div>` : ''}
                        ${data.checkpoint_path ? `<div class="detail-row"><span>Checkpoint:</span><span class="mono">${data.checkpoint_path}</span></div>` : ''}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

function createSparkline(history) {
    if (!history || history.length < 2) return '';

    const losses = history.map(h => h.loss).filter(l => l != null);
    if (losses.length < 2) return '';

    const width = 80;
    const height = 24;
    const padding = 2;

    const min = Math.min(...losses);
    const max = Math.max(...losses);
    const range = max - min || 1;

    const points = losses.map((loss, i) => {
        const x = padding + (i / (losses.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((loss - min) / range) * (height - 2 * padding);
        return `${x},${y}`;
    }).join(' ');

    // Determine trend color
    const firstLoss = losses[0];
    const lastLoss = losses[losses.length - 1];
    const trendColor = lastLoss < firstLoss ? 'var(--success)' : 'var(--error)';

    return `
        <svg class="sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <polyline
                fill="none"
                stroke="${trendColor}"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
                points="${points}"
            />
        </svg>
    `;
}

// ============================================================================
// Resources Panel
// ============================================================================

async function updateResources() {
    const metrics = await fetchData(API.metrics);
    if (!metrics) return;

    // CPU
    const cpuPercent = metrics.cpu_percent || 0;
    document.getElementById('cpu-percent').textContent = `${cpuPercent.toFixed(1)}%`;
    document.getElementById('cpu-bar').style.width = `${cpuPercent}%`;

    // Memory
    if (metrics.memory) {
        const memPercent = metrics.memory.percent || 0;
        const memUsed = formatBytes(metrics.memory.used);
        const memTotal = formatBytes(metrics.memory.total);
        document.getElementById('memory-percent').textContent = `${memUsed} / ${memTotal}`;
        document.getElementById('memory-bar').style.width = `${memPercent}%`;

        // Color based on usage
        const memBar = document.getElementById('memory-bar');
        if (memPercent > 90) memBar.classList.add('critical');
        else if (memPercent > 80) memBar.classList.add('warning');
        else memBar.classList.remove('critical', 'warning');
    }

    // GPU
    const gpuSection = document.getElementById('gpu-section');
    if (metrics.gpu) {
        gpuSection.style.display = 'block';
        const gpuUtil = metrics.gpu.utilization || 0;
        const gpuMemUsed = formatBytes(metrics.gpu.memory_used);
        const gpuMemTotal = formatBytes(metrics.gpu.memory_total);
        document.getElementById('gpu-percent').textContent = `${gpuUtil.toFixed(0)}% • ${gpuMemUsed}`;
        document.getElementById('gpu-bar').style.width = `${gpuUtil}%`;
    } else {
        gpuSection.style.display = 'none';
    }
}

// ============================================================================
// Disk Usage Panel
// ============================================================================

async function updateDiskUsage() {
    const disk = await fetchData(API.disk);
    if (!disk) return;

    const maxSize = Math.max(disk.raw || 0, disk.cleaned || 0, disk.tokenized || 0, disk.checkpoints || 0, 1);

    const updateBar = (id, size) => {
        const percent = (size / maxSize) * 100;
        document.getElementById(`${id}-bar`).style.width = `${percent}%`;
        document.getElementById(`${id}-size`).textContent = formatBytes(size);
    };

    updateBar('raw', disk.raw || 0);
    updateBar('cleaned', disk.cleaned || 0);
    updateBar('tokenized', disk.tokenized || 0);
    updateBar('checkpoints', disk.checkpoints || 0);
}

// ============================================================================
// Event Log with Filtering
// ============================================================================

async function updateEvents() {
    const filterParam = currentFilter !== 'all' ? `&category=${currentFilter}` : '';
    const url = `${API.events}?limit=100${filterParam}`;
    const events = await fetchData(url);

    if (!events) return;

    const logEl = document.getElementById('event-log');
    if (!logEl) return;

    logEl.innerHTML = events.slice(0, 50).map(e => {
        const time = e.timestamp ? e.timestamp.split('T')[1]?.split('.')[0] : '';
        const levelClass = e.level === 'ERROR' ? 'log-error' :
            e.level === 'WARNING' ? 'log-warning' :
                e.level === 'DEBUG' ? 'log-debug' : 'log-info';
        return `
            <div class="log-entry ${levelClass}">
                <span class="log-time">[${time}]</span>
                <span class="log-category">[${e.category}]</span>
                <span class="log-message">${e.message || e.action}</span>
            </div>
        `;
    }).join('');
}

function setupEventFilters() {
    const buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            updateEvents();
        });
    });
}

// ============================================================================
// Expandable Panels
// ============================================================================

window.toggleExpand = function (id) {
    if (expandedItems.has(id)) {
        expandedItems.delete(id);
    } else {
        expandedItems.add(id);
    }
    // Re-render will happen on next tick
};

// ============================================================================
// Loss History
// ============================================================================

async function updateLossHistory() {
    const history = await fetchData(API.lossHistory);
    if (history) {
        lossHistory = history;
    }
}

// ============================================================================
// Main Update Loop
// ============================================================================

async function updateUI(state) {
    if (!state) {
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.getElementById('connection-status').classList.add('status-failed');
        return;
    }

    // Update connection status
    document.getElementById('connection-status').textContent = 'Connected';
    document.getElementById('connection-status').classList.remove('status-failed');
    document.getElementById('connection-status').classList.add('status-complete');

    // Update all sections
    renderPipeline(state.pipeline);
    updateOverallProgress(state.overall_progress || 0);
    updateQuickStats(state);
    updateHealthIndicators(state.health_issues);
    renderDownloads(state.downloads);
    renderProcessing(state.stages);
    renderTrainingPhases(state.phases);
}

async function tick() {
    const [state] = await Promise.all([
        fetchData(API.state),
        updateLossHistory(),
        updateResources(),
        updateDiskUsage(),
        updateEvents()
    ]);

    updateUI(state);
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventFilters();
    tick();
    setInterval(tick, 2000);
});
