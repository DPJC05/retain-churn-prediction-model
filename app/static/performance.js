const INK = "#001A32";
const ACCENT = "#2251FF";
const MUTED = "#A3B8E8";

Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.font.size = 13;
Chart.defaults.color = INK;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.boxHeight = 12;
Chart.defaults.plugins.legend.labels.padding = 12;

// Shared Chart.js options for all the model evaluation charts
function curveOptions(xLabel, yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { type: "linear", min: 0, max: 1, title: { display: true, text: xLabel } },
            y: { min: 0, max: 1, title: { display: true, text: yLabel } }
        },
        plugins: { legend: { display: false } }
    };
}

// Converts separate X and Y data arrays into an array of paired (X, Y) coordinates, paired by index
function toPoints(xs, ys) {
    return xs.map((x, i) => ({ x: x, y: ys[i] }));
}

// ROC Curve
const roc = document.getElementById("rocCurve");
const rocData = JSON.parse(roc.dataset.curve);
const rocCurve = new Chart(roc, {
    type: "line",
    data: {
        datasets: [{
            data: toPoints(rocData.fpr, rocData.tpr),
            borderColor: ACCENT,
            borderWidth: 2
        }, {
            data: toPoints(rocData.fpr, rocData.baseline),
            borderColor: MUTED,
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0
        }]
    },
    options: curveOptions("False Positive Rate", "True Positive Rate")
});

// PR Curve
const pr = document.getElementById("prCurve");
const prData = JSON.parse(pr.dataset.curve);
const prCurve = new Chart(pr, {
    type: "line",
    data: {
        datasets: [{
            data: toPoints(prData.recall, prData.precision),
            borderColor: ACCENT,
            borderWidth: 2
        }, {
            data: toPoints(prData.recall, prData.baseline),
            borderColor: MUTED,
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0
        }]
    },
    options: curveOptions("Recall", "Precision")
});

// Cumulative Gains Curve
const gains = document.getElementById("gainsCurve");
const gainsData = JSON.parse(gains.dataset.curve);
const gainsCurve = new Chart(gains, {
    type: "line",
    data: {
        datasets: [{
            data: toPoints(gainsData.contacted, gainsData.captured),
            borderColor: ACCENT,
            borderWidth: 2
        }, {
            data: toPoints(gainsData.contacted, gainsData.baseline),
            borderColor: MUTED,
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0
        }]
    },
    options: curveOptions("Proportion Contacted", "Proportion of Churners Caught")
});

// Calibration Curve
const calibration = document.getElementById("calibrationCurve");
const calibrationData = JSON.parse(calibration.dataset.curve);

// Normalises each bin count against the largest
const maxCount = Math.max(...calibrationData.bin_counts, 1);
const radii = calibrationData.bin_counts.map(n => 3 + 6 * Math.sqrt(n / maxCount));

const calibrationCurve = new Chart(calibration, {
    type: "line",
    data: {
        datasets: [{
                data: toPoints(calibrationData.mean_predicted, calibrationData.observed_frequency),
                borderColor: ACCENT,
                backgroundColor: ACCENT,
                borderWidth: 2,
                pointRadius: radii
            }, {
                data: toPoints(calibrationData.mean_predicted, calibrationData.baseline),
                borderColor: MUTED,
                borderWidth: 1,
                borderDash: [4, 4],
                pointRadius: 0
        }]
    },
    options: curveOptions("Mean Predicted Probability", "Observed Frequency")
});
