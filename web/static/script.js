// Global variables
let modelChart = null;
let jobPollInterval = null;
let currentJobId = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadLeaderboard();
    loadActiveJobs();
    
    // Set up auto-refresh
    setInterval(loadStats, 5000);
    setInterval(loadLeaderboard, 10000);
    setInterval(loadActiveJobs, 2000);
    
    // Show early stop options when checkbox is checked
    document.getElementById('early_stop').addEventListener('change', function() {
        document.getElementById('early-stop-options').style.display = 
            this.checked ? 'block' : 'none';
    });
});

// Tab switching
function showTab(tabName, buttonElement) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    if (buttonElement) {
        buttonElement.classList.add('active');
    }
    
    // Load data for specific tabs
    if (tabName === 'leaderboard') {
        loadLeaderboard();
    } else if (tabName === 'dashboard') {
        loadStats();
        loadActiveJobs();
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        document.getElementById('total-runs').textContent = data.total_runs || 0;
        document.getElementById('total-energy').textContent = (data.total_energy_kwh || 0).toFixed(4);
        document.getElementById('total-co2').textContent = (data.total_co2_g || 0).toFixed(2);
        document.getElementById('total-time').textContent = (data.total_time_hours || 0).toFixed(2);
        
        // Update model comparison chart
        if (data.models && Object.keys(data.models).length > 0) {
            updateModelChart(data.models);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Update model comparison chart
function updateModelChart(models) {
    const ctx = document.getElementById('modelChart');
    if (!ctx) return;
    
    const modelNames = Object.keys(models);
    const avgAccuracies = modelNames.map(m => models[m].avg_accuracy);
    const avgCo2 = modelNames.map(m => models[m].avg_co2_g);
    
    if (modelChart) {
        modelChart.destroy();
    }
    
    modelChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [
                {
                    label: 'Average Accuracy (%)',
                    data: avgAccuracies,
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Average CO₂ (g)',
                    data: avgCo2,
                    backgroundColor: 'rgba(244, 67, 54, 0.6)',
                    borderColor: 'rgba(244, 67, 54, 1)',
                    borderWidth: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'CO₂ (g)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Load leaderboard
async function loadLeaderboard() {
    try {
        const response = await fetch('/api/leaderboard');
        const data = await response.json();
        
        const tbody = document.getElementById('leaderboard-body');
        tbody.innerHTML = '';
        
        if (data.runs && data.runs.length > 0) {
            data.runs.forEach((run, index) => {
                const row = document.createElement('tr');
                const formatTime = (sec) => {
                    if (!sec && sec !== 0) return '-';
                    sec = Number(sec);
                    if (isNaN(sec)) return '-';
                    if (sec < 60) return `${Math.round(sec)} s`;
                    if (sec < 3600) return `${(sec/60).toFixed(1)} min`;
                    const h = Math.floor(sec / 3600);
                    const m = Math.round((sec % 3600) / 60);
                    return `${h} h ${m} m`;
                };

                const safeNum = (v, decimals=2) => {
                    if (v === null || v === undefined || isNaN(Number(v))) return '-';
                    return Number(v).toFixed(decimals);
                };

                const eff = (run.accuracy_per_kwh === null || run.accuracy_per_kwh === undefined) ? 'N/A' : Number(run.accuracy_per_kwh).toFixed(1);

                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td><strong>${run.model}</strong></td>
                    <td>${safeNum(run.accuracy,2)}%</td>
                    <td>${safeNum(run.energy_kwh,4)}</td>
                    <td>${safeNum(run.co2_g,2)}</td>
                    <td>${formatTime(run.training_time_sec)}</td>
                    <td><strong>${eff}</strong></td>
                `;
                tbody.appendChild(row);
            });
        } else {
            tbody.innerHTML = '<tr><td colspan="7">No runs yet. Start training to see results!</td></tr>';
        }
    } catch (error) {
        console.error('Error loading leaderboard:', error);
    }
}

// Load active jobs
async function loadActiveJobs() {
    try {
        const response = await fetch('/api/jobs');
        const data = await response.json();
        
        const container = document.getElementById('active-jobs');
        container.innerHTML = '';
        
        const jobs = Object.values(data).filter(job => 
            job.status === 'running' || job.status === 'queued'
        );
        
        if (jobs.length === 0) {
            container.innerHTML = '<p>No active training jobs</p>';
            return;
        }
        
        jobs.forEach(job => {
            const jobCard = document.createElement('div');
            jobCard.className = `job-card ${job.status}`;
            jobCard.dataset.jobid = job.id || job.id;
            
            const progress = job.status === 'running' && job.total_epochs > 0
                ? (job.current_epoch / job.total_epochs * 100).toFixed(1)
                : 0;
            
            jobCard.innerHTML = `
                <div class="job-header">
                    <div>
                        <strong>${job.config.model.toUpperCase()}</strong> - ${job.config.dataset}
                        <span class="job-status ${job.status}">${job.status}</span>
                    </div>
                    <div>${new Date(job.created_at).toLocaleString()}</div>
                </div>
                ${job.status === 'running' ? `
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: ${progress}%">${progress}%</div>
                    </div>
                    <p>Epoch ${job.current_epoch} / ${job.total_epochs}</p>
                    ${job.metrics && Object.keys(job.metrics).length > 0 ? `
                        <div id="training-metrics">
                            <div class="metric-item">
                                <label>Train Acc</label>
                                <value>${job.metrics.train_acc ? job.metrics.train_acc.toFixed(2) : '-'}%</value>
                            </div>
                            <div class="metric-item">
                                <label>Val Acc</label>
                                <value>${job.metrics.val_acc ? job.metrics.val_acc.toFixed(2) : '-'}%</value>
                            </div>
                            <div class="metric-item">
                                <label>Train Loss</label>
                                <value>${job.metrics.train_loss ? job.metrics.train_loss.toFixed(4) : '-'}</value>
                            </div>
                            <div class="metric-item">
                                <label>Val Loss</label>
                                <value>${job.metrics.val_loss ? job.metrics.val_loss.toFixed(4) : '-'}</value>
                            </div>
                        </div>
                    ` : ''}
                ` : ''}
            `;
            // Add action buttons
            const actions = document.createElement('div');
            actions.style.marginTop = '10px';
            actions.innerHTML = `<button class="btn-ghost" onclick="viewJobLog('${job.id}')"><i class='fa fa-file-alt'></i> View Log</button>`;
            jobCard.appendChild(actions);

            container.appendChild(jobCard);
        });
    } catch (error) {
        console.error('Error loading jobs:', error);
    }
}

// View job log (toggle)
async function viewJobLog(jobId) {
    try {
        const jobCard = document.querySelector(`.job-card[data-jobid="${jobId}"]`);
        if (!jobCard) return;

        const existing = jobCard.querySelector('.job-log');
        if (existing) {
            existing.remove();
            return;
        }

        const res = await fetch(`/api/job_log/${jobId}`);
        if (!res.ok) {
            alert('No log available for this job');
            return;
        }

        const data = await res.json();
        const pre = document.createElement('pre');
        pre.className = 'job-log';
        pre.style.background = '#f7fafc';
        pre.style.padding = '10px';
        pre.style.borderRadius = '8px';
        pre.style.marginTop = '10px';
        pre.style.maxHeight = '240px';
        pre.style.overflow = 'auto';
        pre.textContent = data.log || '';

        jobCard.appendChild(pre);
    } catch (error) {
        console.error('Error fetching job log:', error);
        alert('Error fetching job log');
    }
}

// Download leaderboard as CSV client-side (builds CSV from /api/leaderboard)
async function downloadLeaderboard() {
    try {
        const res = await fetch('/api/leaderboard');
        const data = await res.json();
        if (!data.runs || data.runs.length === 0) {
            alert('No leaderboard data available');
            return;
        }

        const runs = data.runs;
        const headers = ['model','accuracy','energy_kwh','co2_g','training_time_sec','epochs','batch_size','fp16','early_stop','accuracy_per_kwh'];
        const rows = runs.map(r => headers.map(h => (r[h] !== undefined ? r[h] : '')).join(','));
        const csv = [headers.join(','), ...rows].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'leaderboard.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error downloading leaderboard:', error);
        alert('Failed to download leaderboard');
    }
}

// Basic client-side table sorting for leaderboard
function enableTableSorting() {
    const table = document.getElementById('leaderboard-table');
    if (!table) return;
    const headers = table.querySelectorAll('thead th');
    headers.forEach((th, index) => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => {
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const isNumeric = (cell) => !isNaN(parseFloat(cell)) && isFinite(cell);

            const sorted = rows.sort((a, b) => {
                const aText = a.children[index].innerText.replace('%','').replace('min','').trim();
                const bText = b.children[index].innerText.replace('%','').replace('min','').trim();
                if (isNumeric(aText) && isNumeric(bText)) {
                    return parseFloat(bText) - parseFloat(aText);
                }
                return aText.localeCompare(bText);
            });

            // Re-append sorted rows
            tbody.innerHTML = '';
            sorted.forEach(r => tbody.appendChild(r));
        });
    });
}

// enable sorting on load
document.addEventListener('DOMContentLoaded', enableTableSorting);

// Start training
async function startTraining(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const data = {
        model: formData.get('model'),
        dataset: formData.get('dataset'),
        epochs: parseInt(formData.get('epochs')),
        batch_size: parseInt(formData.get('batch_size')),
        learning_rate: parseFloat(formData.get('learning_rate')),
        optimizer: formData.get('optimizer'),
        momentum: parseFloat(formData.get('momentum') || 0.9),
        weight_decay: parseFloat(formData.get('weight_decay') || 0.0001),
        fp16: formData.get('fp16') === 'on',
        early_stop: formData.get('early_stop') === 'on',
        early_stop_patience: parseInt(formData.get('early_stop_patience') || 10),
        device: formData.get('device'),
        num_workers: parseInt(formData.get('num_workers') || 4),
        region: formData.get('region') || null,
        seed: parseInt(formData.get('seed') || 42),
        demo: (formData.get('demo_mode') === 'on' || document.getElementById('demo_mode').checked)
    };
    
    try {
        const response = await fetch('/api/jobs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentJobId = result.job_id;
            document.getElementById('training-status').style.display = 'block';
            document.getElementById('training-form').reset();
            
            // Start polling for job updates
            startJobPolling(result.job_id);
            
            // Switch to dashboard to see progress
            const dashboardBtn = document.querySelectorAll('.tab-btn')[0];
            showTab('dashboard', dashboardBtn);
        } else {
            alert('Error starting training: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error starting training: ' + error.message);
    }
}

// Poll for job updates
function startJobPolling(jobId) {
    if (jobPollInterval) {
        clearInterval(jobPollInterval);
    }
    
    jobPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/jobs/${jobId}`);
            const job = await response.json();
            
            if (job.status === 'completed' || job.status === 'failed') {
                clearInterval(jobPollInterval);
                jobPollInterval = null;
                
                if (job.status === 'completed') {
                    alert('Training completed! Check the leaderboard for results.');
                    loadLeaderboard();
                    loadStats();
                } else {
                    alert('Training failed: ' + (job.error || 'Unknown error'));
                }
            }
            
            loadActiveJobs();
        } catch (error) {
            console.error('Error polling job:', error);
        }
    }, 2000);
}

// Recommendation tabs
function showRecommendTab(type, buttonElement) {
    document.querySelectorAll('.recommend-form').forEach(form => {
        form.classList.remove('active');
    });
    document.querySelectorAll('.recommend-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(`recommend-${type}`).classList.add('active');
    if (buttonElement) {
        buttonElement.classList.add('active');
    }
}

// Get recommendation
async function getRecommendation(type) {
    const resultDiv = document.getElementById('recommendation-result');
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '<p>Loading...</p>';
    resultDiv.style.display = 'block';
    
    let data = { type: type };
    
    if (type === 'accuracy') {
        data.target_accuracy = parseFloat(document.getElementById('target-accuracy').value);
        data.tolerance = parseFloat(document.getElementById('tolerance').value);
    } else if (type === 'co2') {
        data.max_co2_g = parseFloat(document.getElementById('max-co2').value);
    } else if (type === 'time') {
        data.max_time_hours = parseFloat(document.getElementById('max-time').value);
    }
    
    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const recommendation = await response.json();
        
        if (response.ok) {
            resultDiv.innerHTML = `
                <h3>Recommended Configuration</h3>
                <p><strong>Reason:</strong> ${recommendation.reason}</p>
                <div class="recommendation-item">
                    <strong>Model:</strong> ${recommendation.model}
                </div>
                <div class="recommendation-item">
                    <strong>Expected Accuracy:</strong> ${recommendation.expected_accuracy.toFixed(2)}%
                </div>
                <div class="recommendation-item">
                    <strong>Expected CO₂:</strong> ${recommendation.expected_co2_g.toFixed(2)} g
                </div>
                <div class="recommendation-item">
                    <strong>Expected Energy:</strong> ${recommendation.expected_energy_kwh.toFixed(4)} kWh
                </div>
                <div class="recommendation-item">
                    <strong>Expected Time:</strong> ${(recommendation.expected_time_sec / 60).toFixed(1)} minutes
                </div>
                <div class="recommendation-item">
                    <strong>Recommended Epochs:</strong> ${recommendation.recommended_epochs}
                </div>
                <div class="recommendation-item">
                    <strong>Recommended Batch Size:</strong> ${recommendation.recommended_batch_size}
                </div>
                <div class="recommendation-item">
                    <strong>Use FP16:</strong> ${recommendation.use_fp16 ? 'Yes' : 'No'}
                </div>
                <div class="recommendation-item">
                    <strong>Use Early Stop:</strong> ${recommendation.use_early_stop ? 'Yes' : 'No'}
                </div>
                <div class="recommendation-item">
                    <strong>Efficiency Score:</strong> ${recommendation.efficiency_score.toFixed(1)} (accuracy/kWh)
                </div>
            `;
        } else {
            resultDiv.innerHTML = `<p style="color: red;">Error: ${recommendation.error}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
}

