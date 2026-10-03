const canvas = document.getElementById("doughnutChart");
const doughnutChart = new Chart(canvas, {
    type: "doughnut",
    data: {
        labels: JSON.parse(canvas.dataset.labels),
        datasets: [{
            label: "Churn Risk Distribution",
            data: JSON.parse(canvas.dataset.values),
            backgroundColor: [
                "#2251FF",
                "#7B95E0",
                "#D0DAF5",
            ],
            borderColor: "#FFFFFF",
            borderWidth: 2,
            hoverOffset: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: "right",
                labels: {
                    font: {
                        family: "Inter, system-ui, sans-serif",
                        size: 13,
                        weight: 400
                    },
                    color: "#001A32",
                    boxWidth: 12,
                    boxHeight: 12,
                    padding: 12,
                    usePointStyle: false
                }  
            }
        }
    } 
});
