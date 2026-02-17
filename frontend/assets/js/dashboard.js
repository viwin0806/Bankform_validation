// Dashboard Logic for BankForm-AI

document.addEventListener('DOMContentLoaded', () => {
    loadDashboardData();
    // Refresh every 30 seconds
    setInterval(loadDashboardData, 30000);
});

async function loadDashboardData() {
    try {
        const response = await fetch('http://localhost:5050/api/dashboard');
        const data = await response.json();

        updateStats(data);
        renderCharts(data);
        updateRecentTable(data.recent_transactions);

    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function updateStats(data) {
    const statsGrid = document.getElementById('statsGrid');
    statsGrid.innerHTML = `
        <div class="stat-card">
            <h3>Total Processed</h3>
            <p>${data.total_forms}</p>
        </div>
        <div class="stat-card">
            <h3>Avg. Confidence</h3>
            <p>${(data.average_confidence * 100).toFixed(1)}%</p>
        </div>
        <div class="stat-card">
            <h3>Total Value</h3>
            <p>₹${formatCurrency(data.total_processed_amount)}</p>
        </div>
        <div class="stat-card success">
            <h3>Auto-Approved</h3>
            <p>${data.auto_approved}</p>
        </div>
        <div class="stat-card warning">
            <h3>Needs Review</h3>
            <p>${data.flagged_for_review}</p>
        </div>
    `;
}

function updateRecentTable(transactions) {
    const tableBody = document.getElementById('recentTableBody');
    tableBody.innerHTML = '';

    transactions.forEach(tx => {
        const row = document.createElement('tr');
        const date = new Date(tx.date).toLocaleDateString();
        const confClass = tx.confidence > 0.9 ? 'text-green' : (tx.confidence > 0.7 ? 'text-yellow' : 'text-red');
        const statusClass = tx.status === 'approved' ? 'approved' : 'flagged';

        row.innerHTML = `
            <td>${date}</td>
            <td>${tx.type.charAt(0).toUpperCase() + tx.type.slice(1)}</td>
            <td class="amount-highlight">₹${tx.amount}</td>
            <td class="${confClass}">${(tx.confidence * 100).toFixed(1)}%</td>
            <td><span class="status-badge ${statusClass}">${tx.status.toUpperCase()}</span></td>
        `;
        tableBody.appendChild(row);
    });
}

// Chart Instances
let volumeChart, typeChart, statusChart;

function renderCharts(data) {
    // 1. Volume Chart
    const ctxVol = document.getElementById('volumeChart').getContext('2d');
    if (volumeChart) volumeChart.destroy();

    volumeChart = new Chart(ctxVol, {
        type: 'line',
        data: {
            labels: data.daily_trend.dates,
            datasets: [{
                label: 'Forms Processed',
                data: data.daily_trend.counts,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, suggestedMax: 5 } }
        }
    });

    // 2. Type Chart (Doughnut)
    const ctxType = document.getElementById('typeChart').getContext('2d');
    if (typeChart) typeChart.destroy();

    const types = Object.keys(data.by_type);
    const typeCounts = Object.values(data.by_type);

    typeChart = new Chart(ctxType, {
        type: 'doughnut',
        data: {
            labels: types.map(t => t.charAt(0).toUpperCase() + t.slice(1)),
            datasets: [{
                data: typeCounts,
                backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'],
            }]
        },
        options: { responsive: true }
    });

    // 3. Status Chart (Pie)
    const ctxStatus = document.getElementById('statusChart').getContext('2d');
    if (statusChart) statusChart.destroy();

    statusChart = new Chart(ctxStatus, {
        type: 'pie',
        data: {
            labels: ['Auto-Approved', 'Flagged for Review'],
            datasets: [{
                data: [data.auto_approved, data.flagged_for_review],
                backgroundColor: ['#10b981', '#f59e0b'],
            }]
        },
        options: { responsive: true }
    });
}

function formatCurrency(num) {
    return new Intl.NumberFormat('en-IN', { maximumSignificantDigits: 3 }).format(num);
}
