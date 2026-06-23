// Global State
let dashboardData = null;
let currentChart = null;

// Simulator Constants (pre-calculated global sums from python run)
const GLOBAL_SIM = {
    baseline: { under: 111658, over: 192542 },
    model: { under: 108016, over: 108244 },
    testDays: 184
};

// DOM Elements
const storeSelect = document.getElementById('store-select');
const itemSelect = document.getElementById('item-select');
const skuMapeText = document.getElementById('sku-mape');

// Explorer Impact Elements
const expSavingsText = document.getElementById('explorer-savings');
const expStockoutText = document.getElementById('explorer-stockout');
const expStockoutUnitsText = document.getElementById('explorer-stockout-units');
const expOverstockText = document.getElementById('explorer-overstock');
const expOverstockUnitsText = document.getElementById('explorer-overstock-units');

// Simulator Elements
const simCostSlider = document.getElementById('sim-cost');
const simPriceSlider = document.getElementById('sim-price');
const simHoldingSlider = document.getElementById('sim-holding');
const simPenaltySlider = document.getElementById('sim-penalty');

const valCostText = document.getElementById('val-cost');
const valPriceText = document.getElementById('val-price');
const valHoldingText = document.getElementById('val-holding');
const valPenaltyText = document.getElementById('val-penalty');

const infoUnderstockText = document.getElementById('info-understock');
const infoOverstockText = document.getElementById('info-overstock');

const simSavingsText = document.getElementById('sim-savings');
const simBaseTotalText = document.getElementById('sim-base-total');
const simModelTotalText = document.getElementById('sim-model-total');

const simBaseStockoutText = document.getElementById('sim-base-stockout');
const simBaseHoldingText = document.getElementById('sim-base-holding');
const simBaseCostTotalText = document.getElementById('sim-base-cost-total');

const simModelStockoutText = document.getElementById('sim-model-stockout');
const simModelHoldingText = document.getElementById('sim-model-holding');
const simModelCostTotalText = document.getElementById('sim-model-cost-total');

// Initial Setup
document.addEventListener('DOMContentLoaded', async () => {
    setupTabs();
    await loadData();
    setupSelectors();
    setupSimulator();
    updateDashboard();
});

// Tab Switching Logic
function setupTabs() {
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.tab-content');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            item.classList.add('active');
            const targetTab = item.getAttribute('data-tab');
            document.getElementById(`tab-${targetTab}`).classList.add('active');
        });
    });
}

// Load JSON data
async function loadData() {
    try {
        const response = await fetch('dashboard_data.json');
        dashboardData = await response.json();
        
        // Populate backtest table
        populateBacktestTable(dashboardData.backtests);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        alert('Failed to load dashboard data. Verify dashboard_data.json exists.');
    }
}

// Setup Store & Item Selectors
function setupSelectors() {
    // 10 stores
    for (let s = 1; s <= 10; s++) {
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = `Store Location ${s}`;
        storeSelect.appendChild(opt);
    }
    
    // 50 items
    for (let i = 1; i <= 50; i++) {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `Product Item ${i}`;
        itemSelect.appendChild(opt);
    }
    
    storeSelect.addEventListener('change', updateDashboard);
    itemSelect.addEventListener('change', updateDashboard);
}

// Populate Backtest Results
function populateBacktestTable(backtests) {
    const tbody = document.querySelector('#backtest-table tbody');
    tbody.innerHTML = '';
    
    backtests.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="mono">${row.cutoff}</td>
            <td>${row.test_days} Days</td>
            <td class="mono">${row.mape_baseline.toFixed(2)}%</td>
            <td class="mono text-green">${row.mape_model.toFixed(2)}%</td>
            <td class="mono text-green font-bold">+${row.improvement_pct.toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

// Update Explorer metrics & charts
function updateDashboard() {
    if (!dashboardData) return;
    
    const store = storeSelect.value;
    const item = itemSelect.value;
    const skuKey = `S${store}_I${item}`;
    const skuData = dashboardData.skus[skuKey];
    
    if (!skuData) return;
    
    // Calculate individual SKU MAPE/RMSE
    const metrics = calculateSkuMetrics(skuData.y, skuData.p, skuData.b);
    skuMapeText.textContent = `${metrics.modelMape.toFixed(1)}%`;
    skuMapeText.className = metrics.modelMape < metrics.baselineMape ? 'stat-value text-green' : 'stat-value text-red';
    
    // Calculate and render financial inventory impact for this SKU
    updateSkuInventoryCosts(skuData.y, skuData.p, skuData.b);
    
    // Render Chart
    renderForecastChart(dashboardData.dates, skuData);
}

// Helper: Calculate MAPE for single SKU
function calculateSkuMetrics(actual, forecast, baseline) {
    let sumAbsErrModel = 0;
    let sumAbsErrBase = 0;
    let sumActual = 0;
    
    for (let i = 0; i < actual.length; i++) {
        sumAbsErrModel += Math.abs(actual[i] - forecast[i]);
        sumAbsErrBase += Math.abs(actual[i] - baseline[i]);
        sumActual += actual[i];
    }
    
    // We use WAPE (weighted MAPE) for safety against zero actual values
    return {
        modelMape: sumActual > 0 ? (sumAbsErrModel / sumActual) * 100 : 0,
        baselineMape: sumActual > 0 ? (sumAbsErrBase / sumActual) * 100 : 0
    };
}

// Calculate and render SKU specific inventory costs
function updateSkuInventoryCosts(actual, forecast, baseline) {
    const cost = parseFloat(simCostSlider.value);
    const price = parseFloat(simPriceSlider.value);
    const holdingRate = parseFloat(simHoldingSlider.value) / 100;
    const penalty = parseFloat(simPenaltySlider.value);
    
    const baseImpact = calculateImpact(actual, baseline, cost, price, holdingRate, penalty);
    const modelImpact = calculateImpact(actual, forecast, cost, price, holdingRate, penalty);
    
    const savings = baseImpact.totalCost - modelImpact.totalCost;
    
    expSavingsText.textContent = formatCurrency(savings);
    expSavingsText.className = savings >= 0 ? 'text-green' : 'text-red';
    
    expStockoutText.textContent = formatCurrency(modelImpact.stockoutCost);
    expStockoutUnitsText.textContent = `${modelImpact.understocked.toLocaleString()} Units Shortage`;
    
    expOverstockText.textContent = formatCurrency(modelImpact.overstockCost);
    expOverstockUnitsText.textContent = `${modelImpact.overstocked.toLocaleString()} Units Surplus`;
}

// Helper: Inventory calculation
function calculateImpact(actual, predicted, cost, price, holdingRate, penalty) {
    let understocked = 0;
    let overstocked = 0;
    
    for (let i = 0; i < actual.length; i++) {
        const diff = actual[i] - predicted[i];
        if (diff > 0) {
            understocked += diff;
        } else {
            overstocked += Math.abs(diff);
        }
    }
    
    const stockoutCost = understocked * price * penalty;
    const overstockCost = overstocked * cost * holdingRate / 365;
    
    return {
        understocked,
        overstocked,
        stockoutCost,
        overstockCost,
        totalCost: stockoutCost + overstockCost
    };
}

// Render ChartJS
function renderForecastChart(dates, skuData) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    if (currentChart) {
        currentChart.destroy();
    }
    
    // Format dates to look prettier (e.g. Jul 15)
    const formattedDates = dates.map(d => {
        const dateObj = new Date(d);
        return dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: formattedDates,
            datasets: [
                {
                    label: 'Lower Bound',
                    data: skuData.l,
                    borderColor: 'transparent',
                    pointRadius: 0,
                    tension: 0,
                    showLine: false
                },
                {
                    label: '90% Prediction Interval',
                    data: skuData.u,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(46, 134, 171, 0.12)',
                    pointRadius: 0,
                    fill: 0, // Fill down to lower bound
                    tension: 0
                },
                {
                    label: 'Actual Sales',
                    data: skuData.y,
                    borderColor: '#2e86ab',
                    borderWidth: 1.5,
                    pointRadius: 1,
                    pointHoverRadius: 4,
                    tension: 0.15
                },
                {
                    label: 'LightGBM Forecast',
                    data: skuData.p,
                    borderColor: '#e84855',
                    borderWidth: 1.8,
                    pointRadius: 1,
                    pointHoverRadius: 4,
                    borderDash: [5, 4],
                    tension: 0.15
                },
                {
                    label: 'Naive Baseline',
                    data: skuData.b,
                    borderColor: '#f4a261',
                    borderWidth: 1.1,
                    pointRadius: 0,
                    borderDash: [2, 2],
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false // We use our own legend header
                },
                tooltip: {
                    backgroundColor: '#151c2f',
                    titleColor: '#f3f4f6',
                    bodyColor: '#9ca3af',
                    borderColor: 'rgba(255, 255, 255, 0.08)',
                    borderWidth: 1,
                    padding: 10,
                    bodyFont: {
                        family: 'Outfit'
                    },
                    titleFont: {
                        family: 'Outfit',
                        weight: 'bold'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)'
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { family: 'Outfit', size: 10 },
                        maxTicksLimit: 12
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)'
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { family: 'Outfit', size: 10 }
                    }
                }
            }
        }
    });
}

// Setup Impact Simulator
function setupSimulator() {
    const handleSliderInput = () => {
        const cost = parseFloat(simCostSlider.value);
        const price = parseFloat(simPriceSlider.value);
        const holdingRate = parseFloat(simHoldingSlider.value) / 100;
        const penalty = parseFloat(simPenaltySlider.value);
        
        // Update slider texts
        valCostText.textContent = `₹${cost}`;
        valPriceText.textContent = `₹${price}`;
        valHoldingText.textContent = `${simHoldingSlider.value}%`;
        valPenaltyText.textContent = `${penalty.toFixed(1)}x`;
        
        // Update cost info box
        infoUnderstockText.textContent = `₹${(price * penalty).toFixed(2)}`;
        infoOverstockText.textContent = `₹${(cost * holdingRate / 365).toFixed(4)}`;
        
        // Run Global Simulation
        runGlobalSimulation(cost, price, holdingRate, penalty);
        
        // Run SKU-specific update in explorer
        updateDashboard();
    };
    
    simCostSlider.addEventListener('input', handleSliderInput);
    simPriceSlider.addEventListener('input', handleSliderInput);
    simHoldingSlider.addEventListener('input', handleSliderInput);
    simPenaltySlider.addEventListener('input', handleSliderInput);
    
    // Trigger initial values
    handleSliderInput();
}

// Run Global Simulator for all 500 SKUs
function runGlobalSimulation(cost, price, holdingRate, penalty) {
    const baseUnder = GLOBAL_SIM.baseline.under;
    const baseOver = GLOBAL_SIM.baseline.over;
    
    const modelUnder = GLOBAL_SIM.model.under;
    const modelOver = GLOBAL_SIM.model.over;
    
    // Calculations
    const baseStockout = baseUnder * price * penalty;
    const baseHolding = baseOver * cost * holdingRate / 365;
    const baseTotal = baseStockout + baseHolding;
    
    const modelStockout = modelUnder * price * penalty;
    const modelHolding = modelOver * cost * holdingRate / 365;
    const modelTotal = modelStockout + modelHolding;
    
    const savings = baseTotal - modelTotal;
    
    // Update texts
    simSavingsText.textContent = formatCurrency(savings);
    simBaseTotalText.textContent = formatCompactCurrency(baseTotal);
    simModelTotalText.textContent = formatCompactCurrency(modelTotal);
    
    simBaseStockoutText.textContent = formatCompactCurrency(baseStockout);
    simBaseHoldingText.textContent = formatCompactCurrency(baseHolding);
    simBaseCostTotalText.textContent = formatCompactCurrency(baseTotal);
    
    simModelStockoutText.textContent = formatCompactCurrency(modelStockout);
    simModelHoldingText.textContent = formatCompactCurrency(modelHolding);
    simModelCostTotalText.textContent = formatCompactCurrency(modelTotal);
}

// Helpers: Formatting Currency
function formatCurrency(value) {
    return 'INR ' + value.toLocaleString('en-IN', {
        maximumFractionDigits: 2,
        minimumFractionDigits: 2
    });
}

function formatCompactCurrency(value) {
    if (value >= 10000000) { // 1 Crore
        return '₹' + (value / 10000000).toFixed(2) + ' Cr';
    } else if (value >= 100000) { // 1 Lakh
        return '₹' + (value / 100000).toFixed(2) + ' L';
    } else {
        return '₹' + value.toLocaleString('en-IN', { maximumFractionDigits: 0 });
    }
}
