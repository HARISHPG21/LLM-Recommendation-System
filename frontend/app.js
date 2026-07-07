// --- Item Database ---
const itemsDb = {
    Scientific: [],
    Office: [],
    Instruments: [],
    Pantry: [],
    Arts: []
};

const domainColors = {
    Scientific: "#38bdf8",
    Office: "#8b5cf6",
    Instruments: "#ec4899",
    Pantry: "#10b981",
    Arts: "#f59e0b"
};

// --- App State ---
let userTimeline = [];
let currentCandidates = [];
let modelGateValue = 0.5;
let isManualOverride = false;
let selectedHeatmapCell = null;

// --- Navigation ---
const navItems = document.querySelectorAll(".nav-item");
const sections = document.querySelectorAll(".content-section");

navItems.forEach(item => {
    item.addEventListener("click", (e) => {
        e.preventDefault();
        
        // Update active class in menu
        navItems.forEach(n => n.classList.remove("active"));
        item.classList.add("active");
        selectedHeatmapCell = null;
        
        // Show correct section
        const targetId = item.getAttribute("href").substring(1);
        sections.forEach(sec => {
            if (sec.id === targetId) {
                sec.style.display = "block";
                if (targetId === "tsne") {
                    initTsneCanvas();
                } else if (targetId === "backbones") {
                    renderLatencyChart();
                } else if (targetId === "heatmap") {
                    initHeatmap();
                }
            } else {
                sec.style.display = "none";
            }
        });
    });
});

// --- Timeline Simulator Logic ---
function updateItemOptions() {
    const domainSelect = document.getElementById("item-domain");
    const itemSelect = document.getElementById("item-title");
    const domain = domainSelect.value;
    
    itemSelect.innerHTML = "";
    itemsDb[domain].forEach(item => {
        const opt = document.createElement("option");
        opt.value = item.id;
        opt.textContent = item.title;
        itemSelect.appendChild(opt);
    });
}

function addItemToTimeline(itemObj = null) {
    let item;
    let domain;
    
    if (itemObj) {
        item = itemObj;
        domain = itemObj.domain;
    } else {
        const domainSelect = document.getElementById("item-domain");
        const itemSelect = document.getElementById("item-title");
        const itemId = parseInt(itemSelect.value);
        domain = domainSelect.value;
        item = itemsDb[domain].find(i => i.id === itemId);
    }
    
    // Check if item is already in timeline
    if (userTimeline.some(i => i.id === item.id)) {
        return;
    }

    userTimeline.push({ ...item, domain });
    renderTimeline();
    
    // Reset prediction output when history changes
    document.getElementById("inference-output-area").style.display = "none";
    resetSandbox();
}

function removeTimelineItem(itemId) {
    userTimeline = userTimeline.filter(i => i.id !== itemId);
    renderTimeline();
    document.getElementById("inference-output-area").style.display = "none";
    resetSandbox();
}

function renderTimeline() {
    const flowContainer = document.getElementById("timeline-flow");
    const emptyMsg = document.getElementById("timeline-empty-msg");
    
    flowContainer.innerHTML = "";
    
    if (userTimeline.length === 0) {
        emptyMsg.style.display = "block";
        return;
    }
    
    emptyMsg.style.display = "none";
    
    userTimeline.forEach((item, index) => {
        // Timeline Item Box
        const itemCard = document.createElement("div");
        itemCard.className = "timeline-item";
        itemCard.style.setProperty("--domain-color", domainColors[item.domain]);
        itemCard.innerHTML = `
            <div class="item-domain-label">${item.domain}</div>
            <div class="item-title-text" title="${item.title}">${item.title}</div>
            <div class="timeline-item-controls">
                <button class="btn-move" onclick="moveTimelineItem(${index}, -1)" ${index === 0 ? 'disabled' : ''}>◀</button>
                <button class="btn-move" onclick="moveTimelineItem(${index}, 1)" ${index === userTimeline.length - 1 ? 'disabled' : ''}>▶</button>
                <button class="btn-remove" onclick="removeTimelineItem(${item.id})">×</button>
            </div>
        `;
        flowContainer.appendChild(itemCard);
        
        // Add connector arrow if not the last item
        if (index < userTimeline.length - 1) {
            const connector = document.createElement("span");
            connector.className = "timeline-connector";
            connector.textContent = "➔";
            flowContainer.appendChild(connector);
        }
    });
}

// Preset Loader
function loadPreset(type) {
    userTimeline = [];
    resetSandbox();
    if (type === 'medical') {
        const sciItems = itemsDb["Scientific"] || [];
        const offItems = itemsDb["Office"] || [];
        if (sciItems.length > 0) addItemToTimeline({ ...sciItems[0], domain: 'Scientific' });
        if (offItems.length > 0) addItemToTimeline({ ...offItems[0], domain: 'Office' });
        if (offItems.length > 1) addItemToTimeline({ ...offItems[1], domain: 'Office' });
        if (sciItems.length > 1) addItemToTimeline({ ...sciItems[1], domain: 'Scientific' });
    } else if (type === 'office_arts') {
        const offItems = itemsDb["Office"] || [];
        const artItems = itemsDb["Arts"] || [];
        if (offItems.length > 0) addItemToTimeline({ ...offItems[0], domain: 'Office' });
        if (offItems.length > 1) addItemToTimeline({ ...offItems[1], domain: 'Office' });
        if (artItems.length > 0) addItemToTimeline({ ...artItems[0], domain: 'Arts' });
    } else if (type === 'music_office') {
        const instItems = itemsDb["Instruments"] || [];
        const offItems = itemsDb["Office"] || [];
        if (instItems.length > 0) addItemToTimeline({ ...instItems[0], domain: 'Instruments' });
        if (offItems.length > 0) addItemToTimeline({ ...offItems[0], domain: 'Office' });
        if (instItems.length > 1) addItemToTimeline({ ...instItems[1], domain: 'Instruments' });
    }
}

// Prediction Simulator Run
async function runPrediction() {
    if (userTimeline.length === 0) {
        alert("Please add at least one interaction to the timeline.");
        return;
    }
    
    const targetDomain = document.getElementById("target-domain").value;
    const btn = document.getElementById("btn-predict");
    const spinner = document.getElementById("predict-spinner");
    const outputArea = document.getElementById("inference-output-area");
    
    // Start spinner & hide output
    btn.disabled = true;
    spinner.style.display = "inline-block";
    outputArea.style.display = "none";
    
    // Clear old sandbox rerank list and selection
    resetSandbox();
    
    try {
        const sequence = userTimeline.map(i => i.id);
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                sequence: sequence,
                target_domain: targetDomain
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Populate App State
            currentCandidates = data.candidates || [];
            modelGateValue = data.gate_value || 0.5;
            
            // Activate architecture flow visualizer animation
            triggerArchFlowAnimation();
            
            // Enable manual override button
            const overrideBtn = document.getElementById("btn-manual-override");
            if (overrideBtn) overrideBtn.disabled = false;
            
            // Populate sandbox values
            const slider = document.getElementById("sandbox-gate-slider");
            if (slider) {
                slider.value = modelGateValue;
                document.getElementById("sandbox-gate-val-text").textContent = modelGateValue.toFixed(4);
            }
            
            // Update recommendation info
            document.getElementById("out-target-domain").textContent = targetDomain;
            document.getElementById("recommended-item-title").textContent = data.title;
            document.querySelector(".rec-desc").textContent = "Recommended product resolved by fusing collaborative history patterns and textual BERT semantic features.";
            
            // Update fusion bars
            const semBar = document.getElementById("semantic-bar");
            const colBar = document.getElementById("collaborative-bar");
            
            semBar.style.width = `${data.semantic_pct}%`;
            semBar.querySelector(".percentage").textContent = `${data.semantic_pct}%`;
            
            colBar.style.width = `${data.collab_pct}%`;
            colBar.querySelector(".percentage").textContent = `${data.collab_pct}%`;
            
            document.getElementById("gating-explanation").textContent = data.explanation;
            
            // Display main recommendation panel
            outputArea.style.display = "block";
            
            // Render Sandbox Rerank list
            renderRerankList(modelGateValue);
        } else {
            alert("Error: Server responded with status " + response.status);
        }
    } catch (e) {
        console.error("Prediction failed:", e);
        alert("Prediction failed. Make sure the Flask server is running.");
    } finally {
        btn.disabled = false;
        spinner.style.display = "none";
    }
}

// --- t-SNE Projection Canvas Visualizer ---
let tsneMode = 'llm'; // 'llm' or 'id'
let tsnePoints = [];
let tsneSearchQuery = '';
const tsneCanvas = document.getElementById("tsne-canvas");
const ctx = tsneCanvas.getContext("2d");
const tooltip = document.getElementById("canvas-tooltip");

function handleTsneSearch(val) {
    tsneSearchQuery = val.trim().toLowerCase();
    drawTsne();
}

function generateTsneData() {
    const domains = ['Scientific', 'Pantry', 'Instruments', 'Arts', 'Office'];
    const pointCount = 180;
    const items = [
        "Cardiology Stethoscope", "Bandage Shears", "pilot g2 pens", "Oatmeal Packet", 
        "Fender guitar cable", "Blue snowball mic", "Hi-polymer erasers", "Kleenex Tissues", 
        "Fiskars rotary cutter", "Microphone boom stand", "Whole Cashews", "Nylon Paint Brushes"
    ];
    
    tsnePoints = [];
    
    // Generate distinct clustered positions (for LLM-Rec)
    const clusters = {
        Scientific: { cx: 200, cy: 150, r: 80 },
        Pantry: { cx: 600, cy: 150, r: 80 },
        Instruments: { cx: 400, cy: 380, r: 80 },
        Arts: { cx: 220, cy: 350, r: 80 },
        Office: { cx: 580, cy: 350, r: 80 }
    };
    
    for (let i = 0; i < pointCount; i++) {
        const domain = domains[i % domains.length];
        const title = items[i % items.length] + ` #${i}`;
        const isHead = (i % 5 === 0);
        
        // 1. Clustered Coordinates (LLM-Rec)
        const cl = clusters[domain];
        const angle = Math.random() * Math.PI * 2;
        const dist = Math.random() * cl.r;
        const lx = cl.cx + Math.cos(angle) * dist;
        const ly = cl.cy + Math.sin(angle) * dist;
        
        // 2. Scattered/Intermixed Coordinates (ID-based SASRec)
        // Highly mixed, representing poor domain-agnostic separation
        const ix = 150 + Math.random() * 500;
        const iy = 100 + Math.random() * 300;
        
        tsnePoints.push({
            title,
            domain,
            freq: isHead ? "Head" : "Tail",
            color: domainColors[domain],
            lx, ly, // LLM coords
            ix, iy, // ID coords
            // Current animated positions
            cx: tsneMode === 'llm' ? lx : ix,
            cy: tsneMode === 'llm' ? ly : iy,
            size: isHead ? 6 : 4
        });
    }
}

function drawTsne() {
    ctx.clearRect(0, 0, tsneCanvas.width, tsneCanvas.height);
    
    // Draw background grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 1;
    for (let x = 0; x < tsneCanvas.width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, tsneCanvas.height);
        ctx.stroke();
    }
    for (let y = 0; y < tsneCanvas.height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(tsneCanvas.width, y);
        ctx.stroke();
    }
    
    // Draw points
    tsnePoints.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.cx, p.cy, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
        
        // Glow effect for Head items
        if (p.size > 5) {
            ctx.shadowBlur = 10;
            ctx.shadowColor = p.color;
            ctx.fill();
            ctx.shadowBlur = 0; // reset
        }
        
        // Search Match Highlight
        if (tsneSearchQuery && (p.title.toLowerCase().includes(tsneSearchQuery) || p.domain.toLowerCase().includes(tsneSearchQuery))) {
            ctx.beginPath();
            ctx.arc(p.cx, p.cy, p.size + 6, 0, Math.PI * 2);
            ctx.strokeStyle = "#00f2fe";
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw text title right above it
            ctx.font = "bold 9px sans-serif";
            ctx.fillStyle = "#fff";
            ctx.textAlign = "center";
            ctx.fillText(p.title.split("#")[0].trim(), p.cx, p.cy - 12);
        }
    });
}

// Smooth transition animation
function animateTsne() {
    let done = true;
    const speed = 0.12; // animation interpolation speed
    
    tsnePoints.forEach(p => {
        const targetX = tsneMode === 'llm' ? p.lx : p.ix;
        const targetY = tsneMode === 'llm' ? p.ly : p.iy;
        
        const dx = targetX - p.cx;
        const dy = targetY - p.cy;
        
        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
            p.cx += dx * speed;
            p.cy += dy * speed;
            done = false;
        } else {
            p.cx = targetX;
            p.cy = targetY;
        }
    });
    
    drawTsne();
    
    if (!done) {
        requestAnimationFrame(animateTsne);
    }
}

function toggleTsneMode(mode) {
    if (tsneMode === mode) return;
    
    tsneMode = mode;
    document.getElementById("btn-tsne-llm").classList.toggle("active", mode === 'llm');
    document.getElementById("btn-tsne-id").classList.toggle("active", mode === 'id');
    
    animateTsne();
}

function initTsneCanvas() {
    generateTsneData();
    drawTsne();
}

// Canvas Mouse Hover Tooltip
tsneCanvas.addEventListener("mousemove", (e) => {
    const rect = tsneCanvas.getBoundingClientRect();
    // Scale coordinates for mobile and responsive device compatibility
    const mx = (e.clientX - rect.left) * (tsneCanvas.width / rect.width);
    const my = (e.clientY - rect.top) * (tsneCanvas.height / rect.height);
    
    let hoverItem = null;
    
    // Search for closest hovered point
    for (let p of tsnePoints) {
        const dist = Math.sqrt((p.cx - mx) ** 2 + (p.cy - my) ** 2);
        if (dist < 8) {
            hoverItem = p;
            break;
        }
    }
    
    if (hoverItem) {
        // Redraw to highlight hovered item
        drawTsne();
        ctx.beginPath();
        ctx.arc(hoverItem.cx, hoverItem.cy, hoverItem.size + 4, 0, Math.PI * 2);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Position tooltip using local element relative coordinates
        tooltip.style.display = "block";
        tooltip.style.left = `${(e.clientX - rect.left) + 15}px`;
        tooltip.style.top = `${(e.clientY - rect.top) - 15}px`;
        tooltip.querySelector(".tooltip-title").textContent = hoverItem.title;
        tooltip.querySelector(".tooltip-domain").textContent = `Category: ${hoverItem.domain}`;
        tooltip.style.setProperty("--color-accent", hoverItem.color);
        tooltip.querySelector(".tooltip-freq").textContent = `Frequency class: ${hoverItem.freq}`;
    } else {
        tooltip.style.display = "none";
        drawTsne();
    }
});

tsneCanvas.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
    drawTsne();
});


// --- Backbone Specs database & Custom Latency Chart ---
const modelSpecDb = {
    'bert-base': {
        name: "BERT-Base (Encoder-Only)",
        recall: "25.61%",
        latency: "7.3 ms",
        zeroshot: "Moderate",
        latencies: [3.1, 4.8, 5.9, 7.3, 8.4, 9.5]
    },
    'opt-125m': {
        name: "OPT-125M (Decoder-Only)",
        recall: "24.26%",
        latency: "7.8 ms",
        zeroshot: "High (OPT Food: 18.2%)",
        latencies: [3.4, 5.1, 6.2, 7.8, 8.9, 10.1]
    },
    'flan-t5': {
        name: "FLAN-T5-Base (Encoder-Decoder)",
        recall: "25.56%",
        latency: "8.6 ms",
        zeroshot: "Moderate (T5 Food: 12.3%)",
        latencies: [4.0, 5.9, 7.1, 8.6, 9.8, 11.2]
    },
    'opt-1.3b': {
        name: "OPT-1.3B (Decoder-Only)",
        recall: "29.16%",
        latency: "17.3 ms",
        zeroshot: "Excellent (OPT Food: 21.1%)",
        latencies: [7.2, 10.4, 13.5, 17.3, 20.8, 24.1]
    },
    'opt-6.7b': {
        name: "OPT-6.7B (Decoder-Only)",
        recall: "30.57%",
        latency: "32.4 ms (est.)",
        zeroshot: "Outstanding (Exceeds IDRec)",
        latencies: [12.5, 19.8, 25.1, 32.4, 39.8, 46.2]
    }
};

let currentBackbone = 'bert-base';
let isCompareMode = false;
let compareBackbone = '';

function selectBackbone(modelKey, element = null) {
    currentBackbone = modelKey;
    
    // Update active state in grid cards
    const cards = document.querySelectorAll(".model-spec-card");
    cards.forEach(c => c.classList.remove("active"));
    
    if (element) {
        element.classList.add("active");
    } else {
        // Fallback: find the card matching the modelKey attribute
        const targetCard = Array.from(cards).find(c => {
            const attr = c.getAttribute("onclick") || "";
            return attr.includes(`'${modelKey}'`);
        });
        if (targetCard) targetCard.classList.add("active");
    }
    
    updateBackboneMetrics();
}

function updateBackboneMetrics() {
    const db1 = modelSpecDb[currentBackbone];
    document.getElementById("selected-model-name").textContent = isCompareMode ? `${db1.name} vs ${compareBackbone ? modelSpecDb[compareBackbone].name : '...'}` : db1.name;
    
    if (isCompareMode) {
        document.getElementById("compare-model-1-title").textContent = db1.name;
        document.getElementById("compare-recall-1").textContent = db1.recall;
        document.getElementById("compare-latency-1").textContent = db1.latency;
        
        if (compareBackbone) {
            const db2 = modelSpecDb[compareBackbone];
            document.getElementById("compare-model-2-title").textContent = db2.name;
            document.getElementById("compare-recall-2").textContent = db2.recall;
            document.getElementById("compare-latency-2").textContent = db2.latency;
        } else {
            document.getElementById("compare-model-2-title").textContent = "Select model";
            document.getElementById("compare-recall-2").textContent = "-";
            document.getElementById("compare-latency-2").textContent = "-";
        }
    } else {
        document.getElementById("metric-recall").textContent = db1.recall;
        document.getElementById("metric-latency").textContent = db1.latency;
        document.getElementById("metric-zeroshot").textContent = db1.zeroshot;
    }
    
    renderLatencyChart();
}

function toggleCompareMode(checked) {
    isCompareMode = checked;
    const selectEl = document.getElementById("compare-model-select");
    const normalGrid = document.getElementById("metrics-grid-normal");
    const compareGrid = document.getElementById("metrics-grid-compare");
    
    if (isCompareMode) {
        selectEl.disabled = false;
        normalGrid.style.display = "none";
        compareGrid.style.display = "grid";
    } else {
        selectEl.disabled = true;
        selectEl.value = "";
        compareBackbone = "";
        normalGrid.style.display = "grid";
        compareGrid.style.display = "none";
    }
    
    updateBackboneMetrics();
}

function selectCompareBackbone(modelKey) {
    compareBackbone = modelKey;
    updateBackboneMetrics();
}

function renderLatencyChart() {
    const barsContainer = document.getElementById("chart-bars-flow");
    const labelsContainer = document.getElementById("chart-axis-x");
    
    barsContainer.innerHTML = "";
    labelsContainer.innerHTML = "";
    
    const seqLengths = [5, 9, 13, 17, 21, 25];
    const data1 = modelSpecDb[currentBackbone].latencies;
    const data2 = isCompareMode && compareBackbone ? modelSpecDb[compareBackbone].latencies : null;
    
    const maxVal = 50.0;
    
    seqLengths.forEach((len, idx) => {
        const latVal1 = data1[idx];
        const heightPct1 = (latVal1 / maxVal) * 100;
        
        // Bar Column
        const col = document.createElement("div");
        col.className = "chart-bar-col";
        
        if (data2) {
            const latVal2 = data2[idx];
            const heightPct2 = (latVal2 / maxVal) * 100;
            
            col.style.width = "45px";
            col.style.flexDirection = "row";
            col.style.alignItems = "flex-end";
            col.style.gap = "4px";
            col.style.justifyContent = "center";
            
            col.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; flex:1; height:100%; justify-content:flex-end;">
                    <span class="val-lbl" style="font-size:8px;">${latVal1}</span>
                    <div class="chart-bar-fill" style="height: ${heightPct1}%; width:100%;"></div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; flex:1; height:100%; justify-content:flex-end;">
                    <span class="val-lbl" style="font-size:8px; color:#8b5cf6;">${latVal2}</span>
                    <div class="chart-bar-fill compare-bar" style="height: ${heightPct2}%; width:100%;"></div>
                </div>
            `;
        } else {
            col.innerHTML = `
                <span class="val-lbl">${latVal1}ms</span>
                <div class="chart-bar-fill" style="height: ${heightPct1}%;"></div>
            `;
        }
        barsContainer.appendChild(col);
        
        // Axis label
        const lbl = document.createElement("span");
        lbl.textContent = len;
        labelsContainer.appendChild(lbl);
    });
}


// --- Initialization ---
let isInitialized = false;

async function initCatalog() {
    try {
        const response = await fetch("/api/items");
        if (response.ok) {
            const data = await response.json();
            Object.assign(itemsDb, data);
            updateItemOptions();
            loadPreset('medical');
            isInitialized = true;
            console.log("Catalog initialized from backend successfully!");
            const banner = document.getElementById("connection-warning");
            if (banner) banner.style.display = "none";
            return true;
        }
    } catch (err) {
        console.warn("Backend server not ready yet. Retrying...", err);
    }
    return false;
}

window.onload = async () => {
    const success = await initCatalog();
    if (!success) {
        showConnectionWarning();
        const interval = setInterval(async () => {
            const ok = await initCatalog();
            if (ok) {
                clearInterval(interval);
            }
        }, 3000);
    }
};

function showConnectionWarning() {
    let banner = document.getElementById("connection-warning");
    if (!banner) {
        banner = document.createElement("div");
        banner.id = "connection-warning";
        banner.style.position = "fixed";
        banner.style.top = "20px";
        banner.style.left = "50%";
        banner.style.transform = "translateX(-50%)";
        banner.style.background = "rgba(239, 68, 68, 0.15)";
        banner.style.border = "1px solid rgba(239, 68, 68, 0.3)";
        banner.style.backdropFilter = "blur(10px)";
        banner.style.color = "#fca5a5";
        banner.style.padding = "10px 20px";
        banner.style.borderRadius = "8px";
        banner.style.fontSize = "14px";
        banner.style.zIndex = "9999";
        banner.style.boxShadow = "0 10px 25px rgba(0, 0, 0, 0.5)";
        banner.style.display = "flex";
        banner.style.alignItems = "center";
        banner.style.gap = "10px";
        banner.innerHTML = `
            <span style="width:14px; height:14px; border:2px solid #ef4444; border-top-color:transparent; border-radius:50%; display:inline-block; animation: spin 1s linear infinite;"></span>
            <span>Connecting to backend model service (initializing PyTorch tensors)...</span>
        `;
        document.body.appendChild(banner);
        
        if (!document.getElementById("spin-style")) {
            const style = document.createElement("style");
            style.id = "spin-style";
            style.innerHTML = "@keyframes spin { to { transform: rotate(360deg); } }";
            document.head.appendChild(style);
        }
    } else {
        banner.style.display = "flex";
    }
}

// --- Interactive Sandbox Helper Functions ---
function resetSandbox() {
    currentCandidates = [];
    isManualOverride = false;
    
    const slider = document.getElementById("sandbox-gate-slider");
    if (slider) {
        slider.value = 0.5;
        slider.disabled = true;
    }
    
    const overrideBtn = document.getElementById("btn-manual-override");
    if (overrideBtn) {
        overrideBtn.disabled = true;
        overrideBtn.textContent = "🔧 Enable Manual Tuning Override";
    }
    
    const modeBadge = document.getElementById("sandbox-mode-badge");
    if (modeBadge) {
        modeBadge.className = "mode-badge";
        modeBadge.textContent = "Auto (Model Resolved)";
    }
    
    const gateText = document.getElementById("sandbox-gate-val-text");
    if (gateText) gateText.textContent = "0.5000";
    
    const list = document.getElementById("sandbox-rerank-list");
    if (list) list.innerHTML = `<div class="rerank-list-placeholder">Run a prediction above to load candidates.</div>`;
    
    // Reset architecture nodes
    const nodes = ["node-history", "node-llm", "node-gru", "node-gate", "node-output"];
    const conns = ["conn-split", "conn-llm-gate", "conn-gru-gate", "conn-output"];
    
    nodes.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove("active-node");
    });
    
    conns.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove("active-conn");
            const particle = el.querySelector(".pulse-particle");
            if (particle) particle.classList.remove("animating");
        }
    });
    
    const branches = document.querySelector(".arch-branches");
    if (branches) branches.classList.remove("active-branches");
}

function triggerArchFlowAnimation() {
    const nodes = ["node-history", "node-llm", "node-gru", "node-gate", "node-output"];
    const conns = ["conn-split", "conn-llm-gate", "conn-gru-gate", "conn-output"];
    
    nodes.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add("active-node");
    });
    
    conns.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.add("active-conn");
            const particle = el.querySelector(".pulse-particle");
            if (particle) particle.classList.add("animating");
        }
    });
    
    const branches = document.querySelector(".arch-branches");
    if (branches) branches.classList.add("active-branches");
}

function toggleManualOverride() {
    isManualOverride = !isManualOverride;
    
    const slider = document.getElementById("sandbox-gate-slider");
    const overrideBtn = document.getElementById("btn-manual-override");
    const modeBadge = document.getElementById("sandbox-mode-badge");
    
    if (isManualOverride) {
        overrideBtn.textContent = "🔒 Restore Model Auto Tuning";
        modeBadge.className = "mode-badge manual";
        modeBadge.textContent = "Manual (User Override)";
        slider.disabled = false;
        renderRerankList(parseFloat(slider.value));
    } else {
        overrideBtn.textContent = "🔧 Enable Manual Tuning Override";
        modeBadge.className = "mode-badge";
        modeBadge.textContent = "Auto (Model Resolved)";
        slider.value = modelGateValue;
        document.getElementById("sandbox-gate-val-text").textContent = modelGateValue.toFixed(4);
        slider.disabled = true;
        renderRerankList(modelGateValue);
    }
}

function renderRerankList(g) {
    const list = document.getElementById("sandbox-rerank-list");
    if (!list) return;
    
    if (currentCandidates.length === 0) {
        list.innerHTML = `<div class="rerank-list-placeholder">No candidates found.</div>`;
        return;
    }
    
    // Sort candidates dynamically:
    // combinedScore = g * collab_score + (1 - g) * semantic_score
    const items = currentCandidates.map(c => {
        const combinedScore = g * c.collab_score + (1 - g) * c.semantic_score;
        return { ...c, combinedScore };
    });
    
    items.sort((a, b) => b.combinedScore - a.combinedScore);
    
    // Update main card recommendation title if manual override is active
    if (isManualOverride && items.length > 0) {
        const topTitle = document.getElementById("recommended-item-title");
        if (topTitle) topTitle.textContent = items[0].title;
    }
    
    list.innerHTML = "";
    items.forEach((item, index) => {
        const isRank1 = (index === 0);
        const card = document.createElement("div");
        card.className = `sandbox-rerank-item ${isRank1 ? 'rank-1' : ''}`;
        card.style.cursor = "pointer";
        card.onclick = () => toggleRerankAccordion(item.id);
        
        card.innerHTML = `
            <div style="width: 100%;">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <div class="rerank-item-info">
                        <span class="rank-badge">${index + 1}</span>
                        <span class="rerank-title-text" title="${item.title}">${item.title}</span>
                    </div>
                    <span class="rerank-score-badge">${item.combinedScore.toFixed(4)}</span>
                </div>
                <div class="rerank-accordion-content" id="rerank-acc-${item.id}" style="display: none; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05); font-size:11px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <span style="color:var(--text-secondary);">Collaborative Score:</span>
                        <strong style="color:#8b5cf6;">${item.collab_score.toFixed(4)}</strong>
                    </div>
                    <div class="progress-bar-wrap" style="height:4px; margin-bottom:10px;">
                        <div class="progress-fill" style="background:#8b5cf6; width:${Math.max(10, Math.min(100, (item.collab_score + 1) * 50))}%"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <span style="color:var(--text-secondary);">Semantic Textual Score:</span>
                        <strong style="color:var(--color-accent);">${item.semantic_score.toFixed(4)}</strong>
                    </div>
                    <div class="progress-bar-wrap" style="height:4px;">
                        <div class="progress-fill" style="background:var(--color-accent); width:${Math.max(10, Math.min(100, (item.semantic_score + 1) * 50))}%"></div>
                    </div>
                </div>
            </div>
        `;
        list.appendChild(card);
    });
    
    // Update main recommendation result title in real-time
    if (isManualOverride && items.length > 0) {
        document.getElementById("recommended-item-title").textContent = items[0].title;
    }
}

// --- Interactive Particles Background Canvas ---
function initBgParticles() {
    const canvas = document.getElementById("bg-particle-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);
    
    const particles = [];
    const particleCount = Math.min(60, Math.floor((width * height) / 25000));
    
    window.addEventListener("resize", () => {
        width = (canvas.width = window.innerWidth);
        height = (canvas.height = window.innerHeight);
    });
    
    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.3;
            this.vy = (Math.random() - 0.5) * 0.3;
            this.radius = Math.random() * 2 + 1;
        }
        
        update() {
            this.x += this.vx;
            this.y += this.vy;
            
            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;
        }
        
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(0, 242, 254, 0.4)";
            ctx.fill();
        }
    }
    
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }
    
    let mouse = { x: null, y: null };
    window.addEventListener("mousemove", (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });
    window.addEventListener("mouseleave", () => {
        mouse.x = null;
        mouse.y = null;
    });
    
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dist = Math.hypot(particles[i].x - particles[j].x, particles[i].y - particles[j].y);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(139, 92, 246, ${0.12 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
            
            if (mouse.x !== null) {
                const distToMouse = Math.hypot(particles[i].x - mouse.x, particles[i].y - mouse.y);
                if (distToMouse < 180) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.strokeStyle = `rgba(0, 242, 254, ${0.15 * (1 - distToMouse / 180)})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

// --- 3D Isometric Card Tilt Effect ---
function initTiltEffects() {
    const cards = document.querySelectorAll(".glass-card");
    cards.forEach(card => {
        card.addEventListener("mousemove", (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Radial spotlight
            card.style.setProperty("--mouse-x", `${x}px`);
            card.style.setProperty("--mouse-y", `${y}px`);
            
            // 3D Isometric Tilt
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -6; // max 6 degrees
            const rotateY = ((x - centerX) / centerX) * 6;  // max 6 degrees
            
            card.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.015, 1.015, 1.015)`;
            card.style.boxShadow = "0 20px 40px rgba(0, 242, 254, 0.15)";
        });
        
        card.addEventListener("mouseleave", () => {
            card.style.transform = "rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)";
            card.style.boxShadow = "";
        });
    });
}

// --- Timeline Item Moving Swaps ---
function moveTimelineItem(index, direction) {
    const targetIndex = index + direction;
    if (targetIndex < 0 || targetIndex >= userTimeline.length) return;
    
    const temp = userTimeline[index];
    userTimeline[index] = userTimeline[targetIndex];
    userTimeline[targetIndex] = temp;
    
    renderTimeline();
    resetSandbox();
    document.getElementById("inference-output-area").style.display = "none";
}

// --- Rerank List Accordion Content ---
let activeAccordionId = null;

function toggleRerankAccordion(itemId) {
    const acc = document.getElementById(`rerank-acc-${itemId}`);
    if (!acc) return;
    
    const isOpen = (acc.style.display === "block");
    
    if (activeAccordionId && activeAccordionId !== itemId) {
        const activeAcc = document.getElementById(`rerank-acc-${activeAccordionId}`);
        if (activeAcc) activeAcc.style.display = "none";
    }
    
    if (isOpen) {
        acc.style.display = "none";
        activeAccordionId = null;
    } else {
        acc.style.display = "block";
        activeAccordionId = itemId;
    }
}

// --- Synergy Heatmap Data & Layout ---
const heatmapData = {
    domains: ["Scientific", "Pantry", "Instruments", "Arts", "Office"],
    matrix: [
        // rows = source, cols = target
        // [Recall Improvement %, Similarity, Overlap Users]
        [ [0.0, 1.0, 12000], [25.4, 0.68, 2400], [18.2, 0.42, 1100], [32.1, 0.74, 3800], [14.6, 0.35, 950] ],
        [ [12.4, 0.54, 1800], [0.0, 1.0, 11000], [22.6, 0.61, 3100], [15.8, 0.49, 1400], [28.9, 0.79, 4500] ],
        [ [19.8, 0.59, 1500], [21.5, 0.63, 2900], [0.0, 1.0, 15000], [34.7, 0.81, 5200], [17.3, 0.46, 1200] ],
        [ [28.1, 0.71, 3400], [14.3, 0.44, 1200], [31.5, 0.76, 4900], [0.0, 1.0, 14000], [20.4, 0.53, 2100] ],
        [ [15.2, 0.48, 1100], [30.5, 0.82, 5500], [19.1, 0.51, 1600], [24.8, 0.65, 3300], [0.0, 1.0, 13000] ]
    ]
};

function initHeatmap() {
    const grid = document.getElementById("heatmap-grid");
    if (!grid) return;
    grid.innerHTML = "";
    
    // Reset selection state on rebuild
    selectedHeatmapCell = null;
    
    const domains = heatmapData.domains;
    
    // Top-left spacer
    const spacer = document.createElement("div");
    spacer.className = "heatmap-cell header-cell";
    grid.appendChild(spacer);
    
    // Target headers
    domains.forEach(d => {
        const hc = document.createElement("div");
        hc.className = "heatmap-cell header-cell";
        hc.textContent = d.substring(0, 5) + ".";
        hc.title = d;
        grid.appendChild(hc);
    });
    
    // Grid rows
    domains.forEach((sourceD, rIdx) => {
        const hr = document.createElement("div");
        hr.className = "heatmap-cell header-cell";
        hr.textContent = sourceD.substring(0, 5) + ".";
        hr.title = sourceD;
        grid.appendChild(hr);
        
        domains.forEach((targetD, cIdx) => {
            const cell = document.createElement("div");
            
            const handleCellClick = (cellData) => {
                const allCells = document.querySelectorAll(".heatmap-cell.data-cell");
                if (selectedHeatmapCell === cell) {
                    // Deselect
                    selectedHeatmapCell = null;
                    cell.classList.remove("selected-cell");
                    hideHeatmapDetails();
                } else {
                    // Remove selected class from others
                    allCells.forEach(c => c.classList.remove("selected-cell"));
                    // Select this cell
                    selectedHeatmapCell = cell;
                    cell.classList.add("selected-cell");
                    showHeatmapDetails(sourceD, targetD, cellData);
                }
            };
            
            if (rIdx === cIdx) {
                cell.className = "heatmap-cell data-cell";
                cell.style.background = "rgba(255,255,255,0.03)";
                cell.innerHTML = `
                    <span class="val-pct">-</span>
                    <span class="sub-val">Self</span>
                `;
                cell.onmouseenter = () => {
                    if (!selectedHeatmapCell) showHeatmapDetails(sourceD, targetD, null);
                };
                cell.onclick = () => handleCellClick(null);
            } else {
                const cellData = heatmapData.matrix[rIdx][cIdx];
                const val = cellData[0];
                const alpha = Math.max(0.1, Math.min(0.85, val / 40.0));
                
                cell.className = "heatmap-cell data-cell";
                cell.style.backgroundColor = `rgba(0, 242, 254, ${alpha})`;
                
                cell.innerHTML = `
                    <span class="val-pct" style="color: ${alpha > 0.45 ? '#000' : '#fff'}">+${val.toFixed(1)}%</span>
                    <span class="sub-val" style="color: ${alpha > 0.45 ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.5)'}">Synergy</span>
                `;
                cell.onmouseenter = () => {
                    if (!selectedHeatmapCell) showHeatmapDetails(sourceD, targetD, cellData);
                };
                cell.onclick = () => handleCellClick(cellData);
            }
            
            cell.onmouseleave = () => {
                if (!selectedHeatmapCell) hideHeatmapDetails();
            };
            grid.appendChild(cell);
        });
    });
}

function showHeatmapDetails(source, target, data) {
    const title = document.getElementById("heatmap-detail-title");
    const stats = document.getElementById("heatmap-detail-stats");
    
    if (!data) {
        title.textContent = `Intra-Domain Transition: ${source} ➔ ${target}`;
        stats.style.display = "none";
        return;
    }
    
    title.textContent = `Cross-Domain Transfer: ${source} ➔ ${target}`;
    stats.style.display = "grid";
    
    document.getElementById("heatmap-stat-recall").textContent = `+${data[0].toFixed(1)}%`;
    document.getElementById("heatmap-stat-sim").textContent = data[1].toFixed(2);
    document.getElementById("heatmap-stat-users").textContent = data[2].toLocaleString();
}

function hideHeatmapDetails() {
    document.getElementById("heatmap-detail-title").textContent = "Hover over a grid cell to inspect transfer characteristics";
    document.getElementById("heatmap-detail-stats").style.display = "none";
}

// Initializations
document.addEventListener("DOMContentLoaded", () => {
    initBgParticles();
    initTiltEffects();
    
    const slider = document.getElementById("sandbox-gate-slider");
    if (slider) {
        slider.addEventListener("input", (e) => {
            if (isManualOverride) {
                const val = parseFloat(e.target.value);
                document.getElementById("sandbox-gate-val-text").textContent = val.toFixed(4);
                renderRerankList(val);
            }
        });
    }
});
