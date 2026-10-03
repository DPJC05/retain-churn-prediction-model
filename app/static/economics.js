const INK = "#001A32";
const ACCENT = "#2251FF";
const MARKER = "#D85A30";

Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.font.size = 13;
Chart.defaults.color = INK;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.boxHeight = 12;
Chart.defaults.plugins.legend.labels.padding = 12;

// Held at the top level of the script so updateChart() can reach evCurve created by init()
// Avoids destroying and rebuilding evCurve on every slider movement
let evCurve = null;

// Formats a number representing currency as pounds sterling to two decimal places
function formatCurrency(value) {
    return "£" + Number(value).toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// Replaces the text content of an element
function setText(id, text) {
    const element = document.getElementById(id);
    if (element) element.textContent = text;
}

// Converts separate X and Y data arrays into an array of paired (X, Y) coordinates, paired by index
function toPoints(xs, ys) {
    return xs.map((x, i) => ({ x: x, y: ys[i] }));
}

// Mirrors a slider's <input> value onto its <output>, as the <input> value is not displayed as text
function sliderOutput(inputId, outputId, format = (value) => value) {
    const rangeInput = document.getElementById(inputId);
    const rangeOutput = document.getElementById(outputId);
    if (!rangeInput || !rangeOutput) return;

    const update = () => { rangeOutput.textContent = format(rangeInput.value); };
    update();
    rangeInput.addEventListener("input", update);
}

// Reads the four campaign parameters using the same keys as parse_economics_parameters()
function readParameters() {
    return {
        cost_per_contact: Number(document.getElementById("cost-per-contact").value),
        average_customer_lifetime_value: Number(document.getElementById("customer-lifetime-value").value),
        save_rate: Number(document.getElementById("save-rate").value),
        campaign_capacity: Number(document.getElementById("campaign-capacity").value),
    };
}

// Expected Value Curve
function buildExpectedValueChart(canvas, payload) {
    const curve = payload.curve;

    return new Chart(canvas, {
        type: "line",
        data: {
            datasets: [{
                label: "Upper Bound",
                data: toPoints(curve.thresholds, curve.upper_bound),
                borderColor: "transparent",
                backgroundColor: ACCENT + "1C",
                borderWidth: 0,
                pointRadius: 0,
                fill: "+1"
            }, {
                label: "Lower Bound",
                data: toPoints(curve.thresholds, curve.lower_bound),
                borderColor: "transparent",
                borderWidth: 0,
                pointRadius: 0,
                fill: false
            }, {
                label: "Expected Value",
                data: toPoints(curve.thresholds, curve.expected_value),
                borderColor: ACCENT,
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }, {
                label: "Chosen Threshold",
                data: [{ x: payload.chosen_threshold, y: payload.expected_value }],
                borderColor: MARKER,
                backgroundColor: MARKER,
                pointRadius: 5,
                showLine: false
            }, {
                label: "Unconstrained Optimum",
                data: [{ x: payload.unconstrained_threshold, y: payload.unconstrained_expected_value }],
                borderColor: ACCENT,
                backgroundColor: "#FFFFFF",
                borderWidth: 2,
                pointRadius: 5,
                showLine: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "Contact Threshold" }
                },
                y: {
                    title: { display: true, text: "Expected Value (£)" },
                    ticks: {
                        callback: value => "£" + Number(value).toLocaleString()
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: "bottom",
                    labels: {
                        // Hides the two band datasets so the legend shows three entries rather than five
                        filter: item => !item.text.includes("Bound")
                    }
                },
                tooltip: {
                    callbacks: {
                        label: context =>
                            `${context.dataset.label}: £${context.parsed.y.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                    }
                }
            }
        }
    });
}

// Replaces evCurve's data in place and redraws evCurve
function updateChart(payload) {
    if (!evCurve) return;
    const curve = payload.curve;
    evCurve.data.datasets[0].data = toPoints(curve.thresholds, curve.upper_bound);
    evCurve.data.datasets[1].data = toPoints(curve.thresholds, curve.lower_bound);
    evCurve.data.datasets[2].data = toPoints(curve.thresholds, curve.expected_value);
    evCurve.data.datasets[3].data = [{ x: payload.chosen_threshold, y: payload.expected_value }];
    evCurve.data.datasets[4].data = [{ x: payload.unconstrained_threshold, y: payload.unconstrained_expected_value }];
    evCurve.update();
}

// Updates the outcome tiles and shows or hides the capacity banner
function updateOutcomes(payload) {
    setText("evValue", formatCurrency(payload.expected_value));
    setText("targeted", payload.n_flagged);
    setText("precision", payload.precision.toFixed(2));
    setText("recall", payload.recall.toFixed(2));
 
    const banner = document.getElementById("capacityBanner");
    if (banner) {
        banner.hidden = !payload.capacity_binding;
        setText("unconstrainedFlagged", payload.unconstrained_n_flagged);
        setText("unconstrainedValue", formatCurrency(payload.unconstrained_expected_value));
        setText("rawCapacity", payload.raw_capacity);
        setText("bannerValue", formatCurrency(payload.expected_value));
        setText("forgoneValue", formatCurrency(payload.forgone_value));
    }
}

// Rebuilds the sensitivity grid table rows
function updateSensitivityTable(payload) {
    const body = document.getElementById("sensitivityBody");
    if (!body) return;

    const entries = Object.entries(payload.expected_value_by_save_rate);
    const maxValue = Math.max(...entries.map(([, value]) => value));

    body.innerHTML = "";

    for (const [key, value] of entries) {
        const rate = parseFloat(key);

        const row = document.createElement("tr");

        const rateCell = document.createElement("td");
        rateCell.className = "row-left";
        rateCell.textContent = (rate * 100).toFixed(1) + "%";

        const valueCell = document.createElement("td");
        valueCell.className = "row-right";
        valueCell.textContent = formatCurrency(value);

        const barCell = document.createElement("td");
        barCell.className = "row-right";
        const track = document.createElement("div");
        track.className = "bar-track-economics";
        const fill = document.createElement("div");
        fill.className = "bar-fill-economics";
        fill.style.width = (maxValue > 0 ? Math.max(0, value / maxValue * 100) : 0) + "%";
        track.appendChild(fill);
        barCell.appendChild(track);

        row.append(rateCell, valueCell, barCell);
        body.appendChild(row);
    }
}

// Posts the current slider values to the API and updates the page with the result
let latestRequest = 0;
async function recalculate() {
    const requestId = ++latestRequest;
    let response;
    try {
        response = await fetch("/api/economics", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(readParameters()),
        });
    } catch (error) {
        console.error("economics request failed", error);
        return;
    }

    if (!response.ok) {
        const problem = await response.json().catch(() => ({}));
        console.error("economics endpoint returned", response.status, problem.error);
        return;
    }

    const payload = await response.json();
    if (requestId !== latestRequest) return;
    updateChart(payload);
    updateOutcomes(payload);
    updateSensitivityTable(payload);
}

// Limits the input function to running at most once every "wait" milliseconds
function throttle(func, wait) {
    let last = 0;
    let trailing = null;
    return () => {
        const remaining = wait - (Date.now() - last);
        clearTimeout(trailing);
        if (remaining <= 0) {
            last = Date.now();
            func();
        } else {
            trailing = setTimeout(() => {
                last = Date.now();
                func();
            }, remaining);
        }
    };
}

function init() {
    sliderOutput("cost-per-contact", "costRangeValue", value => formatCurrency(value));
    sliderOutput("customer-lifetime-value", "clvRangeValue", value => formatCurrency(value));
    sliderOutput("save-rate", "saveRangeValue", value => `${(Number(value) * 100).toFixed(0)}%`);
    sliderOutput("campaign-capacity", "capacityRangeValue", value => Number(value).toLocaleString());

    const evCanvas = document.getElementById("evCurve");
    if (evCanvas) {
        evCurve = buildExpectedValueChart(evCanvas, JSON.parse(evCanvas.dataset.payload));
    }

    const liveRecalculate = throttle(recalculate, 100);
    document.querySelectorAll("#economics-form input[type=range]").forEach(slider => {
        slider.addEventListener("input", liveRecalculate);
    });
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}
