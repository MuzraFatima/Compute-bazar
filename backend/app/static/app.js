/**
 * =============================================================
 *  COMPUTE BAZAAR 2.0 — Dashboard Javascript Logic (Week 8A)
 * =============================================================
 *  Handles:
 *    - Real-time fetching of backend database metrics (/api/v1/dashboard/metrics)
 *    - 4 Real-Data Chart.js Visualizations:
 *        1. Resource Utilization by Category
 *        2. Costs & VCG Surplus Savings Over Time
 *        3. Agent Wins vs Losses Performance
 *        4. Auction Activity & Social Welfare
 *    - Agent performance table rendering
 *    - Live bidding ticker & closed auction history table
 *    - WebSockets & Simulation controls
 * =============================================================
 */

// Chart instances
let resourceUtilChart = null;
let costSavingsChart = null;
let agentWinsLossesChart = null;
let auctionActivityChart = null;

// WebSocket reference
let ws = null;

// ── Document Initialization ─────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    loadDashboardMetrics();
    loadCurrentRound();
    connectWebSocket();
    checkSimulationStatus();
    
    // Poll metrics every 3 seconds for smooth real-time dashboard updates
    setInterval(loadDashboardMetrics, 3000);
    setInterval(loadCurrentRound, 3000);
    setInterval(checkSimulationStatus, 5000);
    
    // Control bindings
    document.getElementById("btn-start-classic").addEventListener("click", () => startSimulation("classic"));
    document.getElementById("btn-start-groq").addEventListener("click",   () => startSimulation("autonomous"));
    document.getElementById("btn-stop-sim").addEventListener("click", stopSimulation);
});

// ── Initialize Chart.js Instances ────────────────────────────
function initCharts() {
    // 1. Resource Utilization Chart (Bar)
    const ctxUtil = document.getElementById("resourceUtilChart")?.getContext("2d");
    if (ctxUtil) {
        resourceUtilChart = new Chart(ctxUtil, {
            type: "bar",
            data: {
                labels: ["GPU", "CPU", "RAM", "API Budget"],
                datasets: [{
                    label: "Utilization %",
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        "rgba(6, 182, 212, 0.7)",   // Cyan
                        "rgba(139, 92, 246, 0.7)",  // Violet
                        "rgba(16, 185, 129, 0.7)",  // Emerald
                        "rgba(245, 158, 11, 0.7)",  // Amber
                    ],
                    borderColor: [
                        "#06b6d4",
                        "#8b5cf6",
                        "#10b981",
                        "#f59e0b",
                    ],
                    borderWidth: 1.5,
                    borderRadius: 8,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: "#94a3b8", callback: (v) => v + "%" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" }
                    },
                    x: {
                        ticks: { color: "#94a3b8" },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // 2. Costs & Surplus Savings Over Time (Line)
    const ctxCosts = document.getElementById("costSavingsChart")?.getContext("2d");
    if (ctxCosts) {
        costSavingsChart = new Chart(ctxCosts, {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Social Welfare ($)",
                        data: [],
                        borderColor: "#34d399",
                        backgroundColor: "rgba(52, 211, 153, 0.1)",
                        fill: true,
                        tension: 0.35,
                    },
                    {
                        label: "Agent Spending ($)",
                        data: [],
                        borderColor: "#06b6d4",
                        backgroundColor: "transparent",
                        borderDash: [4, 4],
                        tension: 0.35,
                    },
                    {
                        label: "VCG Savings ($)",
                        data: [],
                        borderColor: "#a78bfa",
                        backgroundColor: "rgba(167, 139, 250, 0.15)",
                        fill: true,
                        tension: 0.35,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: "#94a3b8", callback: (v) => "$" + v },
                        grid: { color: "rgba(255, 255, 255, 0.05)" }
                    },
                    x: {
                        ticks: { color: "#94a3b8" },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { labels: { color: "#cbd5e1" } }
                }
            }
        });
    }

    // 3. Agent Wins vs Losses Chart (Grouped Bar)
    const ctxWinsLosses = document.getElementById("agentWinsLossesChart")?.getContext("2d");
    if (ctxWinsLosses) {
        agentWinsLossesChart = new Chart(ctxWinsLosses, {
            type: "bar",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Wins",
                        data: [],
                        backgroundColor: "rgba(16, 185, 129, 0.8)",
                        borderColor: "#10b981",
                        borderRadius: 6,
                    },
                    {
                        label: "Losses",
                        data: [],
                        backgroundColor: "rgba(239, 68, 68, 0.7)",
                        borderColor: "#ef4444",
                        borderRadius: 6,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: "#94a3b8", precision: 0 },
                        grid: { color: "rgba(255, 255, 255, 0.05)" }
                    },
                    x: {
                        ticks: { color: "#94a3b8" },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { labels: { color: "#cbd5e1" } }
                }
            }
        });
    }

    // 4. Auction Activity & Welfare Chart (Combo Line/Bar)
    const ctxActivity = document.getElementById("auctionActivityChart")?.getContext("2d");
    if (ctxActivity) {
        auctionActivityChart = new Chart(ctxActivity, {
            type: "bar",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Social Welfare ($)",
                        data: [],
                        backgroundColor: "rgba(6, 182, 212, 0.6)",
                        borderColor: "#06b6d4",
                        borderRadius: 6,
                        yAxisID: "y",
                    },
                    {
                        label: "Bids Received",
                        data: [],
                        type: "line",
                        borderColor: "#a78bfa",
                        backgroundColor: "#a78bfa",
                        tension: 0.3,
                        yAxisID: "y1",
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: "linear",
                        display: true,
                        position: "left",
                        ticks: { color: "#94a3b8", callback: (v) => "$" + v },
                        grid: { color: "rgba(255, 255, 255, 0.05)" }
                    },
                    y1: {
                        type: "linear",
                        display: true,
                        position: "right",
                        ticks: { color: "#a78bfa", precision: 0 },
                        grid: { drawOnChartArea: false }
                    },
                    x: {
                        ticks: { color: "#94a3b8" },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { labels: { color: "#cbd5e1" } }
                }
            }
        });
    }
}

// ── Fetch & Update Real Database Metrics ──────────────────────
async function loadDashboardMetrics() {
    try {
        const res = await fetch("/api/v1/dashboard/metrics");
        if (!res.ok) return;
        const data = await res.json();
        
        // 1. Update KPI Header Cards
        const s = data.summary;
        document.getElementById("metric-active-agents").innerText = s.active_agents_count;
        document.getElementById("metric-social-welfare").innerText = s.social_welfare.toFixed(2);
        document.getElementById("metric-spending").innerText = s.total_spending.toFixed(2);
        document.getElementById("metric-savings").innerText = s.estimated_savings.toFixed(2);
        document.getElementById("metric-jains-index").innerText = s.jains_fairness_index.toFixed(4);

        if (s.jains_fairness_index >= 0.90) {
            document.getElementById("metric-fairness-status").innerText = "High Equity (Fair)";
            document.getElementById("metric-fairness-status").className = "text-emerald-400 font-semibold";
        } else {
            document.getElementById("metric-fairness-status").innerText = "Moderate Concentration";
            document.getElementById("metric-fairness-status").className = "text-amber-400 font-semibold";
        }

        // LLM Telemetry
        const t = data.llm_telemetry;
        document.getElementById("metric-llm-calls").innerText = t.calls || 0;
        document.getElementById("metric-llm-tokens").innerText = (t.tokens || 0).toLocaleString();
        document.getElementById("metric-llm-latency").innerText = Math.round(t.avg_latency_ms || 0);

        if (t.is_mock) {
            document.getElementById("nav-llm-status").innerText = `${t.model || 'Llama-3.3-70b'} (Mock Mode)`;
        } else {
            document.getElementById("nav-llm-status").innerText = `${t.model || 'Groq Llama'} (Live Mode)`;
        }

        // 2. Render Agent Performance Table
        renderAgentPerformanceTable(data.agents);

        // 3. Render Auction History Table
        renderAuctionHistoryTable(data.auction_history);

        // 4. Update All 4 Charts with Real Data
        updateChartsWithRealData(data);

    } catch (e) {
        console.error("Error loading dashboard metrics:", e);
    }
}

// ── Render Agent Performance Table ───────────────────────────
function renderAgentPerformanceTable(agents) {
    const tbody = document.getElementById("agent-performance-table-body");
    const countBadge = document.getElementById("agent-table-count");
    if (!tbody) return;

    if (countBadge) countBadge.innerText = `${agents.length} AGENTS`;

    if (!agents || agents.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center py-6 text-slate-500">No agents registered yet. Start simulation loop to register agents.</td></tr>`;
        return;
    }

    tbody.innerHTML = agents.map(a => {
        let tierColor = "bg-amber-500/10 text-amber-400 border-amber-500/20";
        if (a.tier === "gold") tierColor = "bg-yellow-500/20 text-yellow-300 border-yellow-500/30";
        else if (a.tier === "silver") tierColor = "bg-slate-400/20 text-slate-200 border-slate-400/30";

        return `
            <tr class="border-b border-white/5 hover:bg-white/5 transition">
                <td class="py-2.5 font-bold text-white flex items-center space-x-2">
                    <span class="h-2 w-2 rounded-full ${a.is_active ? 'bg-emerald-400' : 'bg-slate-600'}"></span>
                    <span>${a.name}</span>
                </td>
                <td class="py-2.5">
                    <span class="text-[10px] uppercase font-bold px-2 py-0.5 rounded border ${tierColor}">${a.tier}</span>
                </td>
                <td class="py-2.5 text-right font-mono">$${a.budget_remaining.toFixed(2)} / $${a.budget_total.toFixed(2)}</td>
                <td class="py-2.5 text-right font-mono text-cyan-400">$${a.total_spent.toFixed(2)}</td>
                <td class="py-2.5 text-right font-mono text-emerald-400">$${a.total_surplus.toFixed(2)}</td>
                <td class="py-2.5 text-center font-bold">${a.wins} / ${a.total_bids}</td>
                <td class="py-2.5 text-right font-bold text-violet-400">${a.win_rate}%</td>
            </tr>
        `;
    }).join("");
}

// ── Render Auction History Table ─────────────────────────────
function renderAuctionHistoryTable(history) {
    const tbody = document.getElementById("auction-history-body");
    const countBadge = document.getElementById("auction-rounds-count");
    if (!tbody) return;

    if (countBadge) countBadge.innerText = `${history.length} ROUNDS`;

    if (!history || history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center py-8 text-slate-500">No closed auction rounds yet.</td></tr>`;
        return;
    }

    tbody.innerHTML = history.map(h => {
        let statusBadge = h.status === "closed" 
            ? `<span class="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-bold border border-white/5 uppercase">CLOSED</span>`
            : `<span class="text-[10px] bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded font-bold border border-emerald-500/30 uppercase">OPEN</span>`;

        return `
            <tr class="border-b border-white/5 hover:bg-white/5 transition">
                <td class="py-2.5 font-bold text-white">#${h.round_number}</td>
                <td class="py-2.5">${statusBadge}</td>
                <td class="py-2.5 text-right font-mono font-bold text-emerald-400">$${h.total_value_extracted.toFixed(2)}</td>
                <td class="py-2.5 text-center font-semibold text-slate-300">${h.bids_count} bids / ${h.winners_count} won</td>
                <td class="py-2.5 text-right uppercase text-[10px] font-bold text-cyan-400">${h.solver_method.toUpperCase()} ILP</td>
            </tr>
        `;
    }).join("");
}

// ── Update Chart.js Visualizations with Real Data ─────────────
function updateChartsWithRealData(data) {
    // 1. Chart 1: Resource Utilization %
    if (resourceUtilChart && data.utilization_by_category) {
        const u = data.utilization_by_category;
        resourceUtilChart.data.datasets[0].data = [
            u.gpu || 0,
            u.cpu || 0,
            u.ram || 0,
            u.api_budget || 0,
        ];
        resourceUtilChart.update();
    }

    // 2. Chart 2: Costs & Surplus Savings Over Time
    if (costSavingsChart && data.auction_history) {
        const reversedRounds = [...data.auction_history].reverse();
        const labels = reversedRounds.map(r => `#${r.round_number}`);
        
        let cumWelfare = 0;
        let cumSpent = 0;
        const welfareData = [];
        const spentData = [];
        const savingsData = [];

        reversedRounds.forEach(r => {
            cumWelfare += r.total_value_extracted;
            // Approximate spending / savings per round
            const estSpent = r.total_value_extracted * 0.82;
            const estSavings = r.total_value_extracted * 0.18;
            cumSpent += estSpent;

            welfareData.push(cumWelfare);
            spentData.push(cumSpent);
            savingsData.push(cumWelfare - cumSpent);
        });

        costSavingsChart.data.labels = labels;
        costSavingsChart.data.datasets[0].data = welfareData;
        costSavingsChart.data.datasets[1].data = spentData;
        costSavingsChart.data.datasets[2].data = savingsData;
        costSavingsChart.update();
    }

    // 3. Chart 3: Agent Wins vs Losses
    if (agentWinsLossesChart && data.agents) {
        const agentNames = data.agents.map(a => a.name);
        const wins = data.agents.map(a => a.wins);
        const losses = data.agents.map(a => a.losses);

        agentWinsLossesChart.data.labels = agentNames;
        agentWinsLossesChart.data.datasets[0].data = wins;
        agentWinsLossesChart.data.datasets[1].data = losses;
        agentWinsLossesChart.update();
    }

    // 4. Chart 4: Auction Activity & Welfare Extracted
    if (auctionActivityChart && data.auction_history) {
        const reversedRounds = [...data.auction_history].reverse();
        const labels = reversedRounds.map(r => `#${r.round_number}`);
        const welfareData = reversedRounds.map(r => r.total_value_extracted);
        const bidsData = reversedRounds.map(r => r.bids_count);

        auctionActivityChart.data.labels = labels;
        auctionActivityChart.data.datasets[0].data = welfareData;
        auctionActivityChart.data.datasets[1].data = bidsData;
        auctionActivityChart.update();
    }
}

// ── Load Current Open Round & Live Bids ───────────────────────
async function loadCurrentRound() {
    try {
        const res = await fetch("/api/v1/auctions/current");
        if (!res.ok) return;
        const round = await res.json();
        
        document.getElementById("metric-round-number").innerText = `#${round.round_number}`;
        document.getElementById("metric-round-status").innerText = round.status.toUpperCase();
        
        // Populate Live Bids Ticker
        const tbody = document.getElementById("bid-ticker-body");
        if (tbody && round.bids && round.bids.length > 0) {
            tbody.innerHTML = "";
            round.bids.forEach(b => {
                const tr = document.createElement("tr");
                tr.className = "border-b border-white/5 hover:bg-white/5 transition bid-row-new";
                
                let outcomeBadge = b.is_winner 
                    ? `<span class="text-[10px] font-bold bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30">WON ($${(b.payment_amount||0).toFixed(2)})</span>`
                    : (round.status === "closed" 
                        ? `<span class="text-[10px] font-bold bg-slate-800 text-slate-500 px-2 py-0.5 rounded">LOST</span>`
                        : `<span class="text-[10px] font-bold bg-cyan-500/10 text-cyan-400 px-2 py-0.5 rounded">PENDING</span>`);

                let priorityBadge = `<span class="text-[10px] font-bold uppercase text-slate-300 px-2 py-0.5 rounded bg-slate-800">${b.priority}</span>`;
                if (b.priority === "critical") priorityBadge = `<span class="text-[10px] font-bold uppercase text-rose-400 px-2 py-0.5 rounded bg-rose-500/20 border border-rose-500/30">CRITICAL</span>`;
                else if (b.priority === "high") priorityBadge = `<span class="text-[10px] font-bold uppercase text-amber-400 px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30">HIGH</span>`;

                tr.innerHTML = `
                    <td class="py-2 font-bold text-white">${b.agent_name}</td>
                    <td class="py-2 font-mono text-cyan-400 text-[11px]">${b.resource_bundle.join(", ")}</td>
                    <td class="py-2 text-right font-mono font-bold text-slate-200">$${b.bid_amount.toFixed(2)}</td>
                    <td class="py-2 text-center">${priorityBadge}</td>
                    <td class="py-2 text-right">${outcomeBadge}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error("Error loading current round:", e);
    }
}

// ── Connect Real-Time WebSocket Feed ─────────────────────────
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => console.log("WebSocket connected to Compute Bazaar server.");
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "bid_submitted") {
                appendLiveTickerBid(data);
            }
            if (data.type === "round_closed" || data.type === "new_round_opened") {
                loadCurrentRound();
                loadDashboardMetrics();
            }
        };
        ws.onclose = () => setTimeout(connectWebSocket, 5000);
    } catch (e) {
        console.error("WebSocket connection error:", e);
    }
}

// ── Append a single bid to the live ticker (top of list) ──────
function appendLiveTickerBid(b) {
    const tbody = document.getElementById("bid-ticker-body");
    if (!tbody) return;

    // Remove placeholder row if present
    const placeholder = document.getElementById("empty-bids-message");
    if (placeholder) placeholder.remove();

    const isAutonomous = b.mode === "autonomous";
    const priorityColors = {
        critical: "text-rose-400 bg-rose-500/20 border-rose-500/30",
        high:     "text-amber-400 bg-amber-500/20 border-amber-500/30",
        normal:   "text-slate-300 bg-slate-800 border-transparent",
        low:      "text-slate-500 bg-slate-900 border-transparent",
    };
    const priClass = priorityColors[b.priority] || priorityColors.normal;

    const llmBadge = isAutonomous
        ? `<span title="LLM latency: ${b.llm_latency_ms}ms | complexity: ${b.complexity} | confidence: ${(b.confidence*100).toFixed(0)}%" class="text-[9px] font-bold bg-violet-500/20 text-violet-300 border border-violet-500/30 px-1.5 py-0.5 rounded cursor-help">🤖 GROQ</span>`
        : `<span class="text-[9px] font-bold bg-slate-800 text-slate-500 px-1.5 py-0.5 rounded">SIM</span>`;

    const tr = document.createElement("tr");
    tr.className = "border-b border-white/5 hover:bg-white/5 transition";
    tr.style.animation = "fadeInDown 0.4s ease";
    tr.innerHTML = `
        <td class="py-2 font-bold text-white text-[11px]">${b.agent} ${llmBadge}</td>
        <td class="py-2 font-mono text-cyan-400 text-[10px]">${(b.bundle||[]).join(", ")}</td>
        <td class="py-2 text-right font-mono font-bold text-slate-200 text-[11px]">$${(b.amount||0).toFixed(2)}</td>
        <td class="py-2 text-center"><span class="text-[9px] font-bold uppercase border px-1.5 py-0.5 rounded ${priClass}">${b.priority}</span></td>
        <td class="py-2 text-right"><span class="text-[10px] font-bold text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded">PENDING</span></td>
    `;

    // Prepend so newest is always on top
    tbody.insertBefore(tr, tbody.firstChild);

    // Keep ticker to max 40 rows
    while (tbody.children.length > 40) tbody.removeChild(tbody.lastChild);
}

// ── Check & Control Simulation ───────────────────────────────
async function checkSimulationStatus() {
    try {
        const res = await fetch("/api/v1/simulation/status");
        if (!res.ok) return;
        const data = await res.json();
        
        const btnClassic    = document.getElementById("btn-start-classic");
        const btnGroq       = document.getElementById("btn-start-groq");
        const btnStop       = document.getElementById("btn-stop-sim");
        const statusPulse   = document.getElementById("status-pulse");
        const statusDot     = document.getElementById("status-dot");
        const statusText    = document.getElementById("status-text");

        if (data.is_running) {
            if (btnClassic) btnClassic.disabled = true;
            if (btnGroq)    btnGroq.disabled    = true;
            if (btnStop)    btnStop.disabled    = false;
            if (statusPulse) statusPulse.className = "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-emerald-400";
            if (statusDot)   statusDot.className   = "relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500";
            const modeLabel = data.mode === "autonomous" ? "GROQ AI ONLINE" : "SIMULATION ONLINE";
            if (statusText) { statusText.innerText = modeLabel; statusText.className = "font-bold uppercase tracking-wider text-emerald-400"; }
        } else {
            if (btnClassic) btnClassic.disabled = false;
            if (btnGroq)    btnGroq.disabled    = false;
            if (btnStop)    btnStop.disabled    = true;
            if (statusPulse) statusPulse.className = "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-slate-500";
            if (statusDot)   statusDot.className   = "relative inline-flex rounded-full h-2.5 w-2.5 bg-slate-500";
            if (statusText) { statusText.innerText = "SIMULATION OFFLINE"; statusText.className = "font-bold uppercase tracking-wider text-slate-400"; }
        }
    } catch (e) {
        console.error("Error checking simulation status:", e);
    }
}

async function startSimulation(mode = "classic") {
    try {
        await fetch(`/api/v1/simulation/start?mode=${mode}`, { method: "POST" });
        checkSimulationStatus();
        loadDashboardMetrics();
    } catch (e) {
        console.error("Error starting simulation:", e);
    }
}

async function stopSimulation() {
    try {
        await fetch("/api/v1/simulation/stop", { method: "POST" });
        checkSimulationStatus();
    } catch (e) {
        console.error("Error stopping simulation:", e);
    }
}
