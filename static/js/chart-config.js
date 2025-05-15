// Chart.js configuration and utilities for Health Detection System

// Set default Chart.js theme to match our dark theme
Chart.defaults.color = '#fff';
Chart.defaults.borderColor = '#495057';
Chart.defaults.backgroundColor = 'rgba(54, 162, 235, 0.7)';

// Utility function to create a risk gauge chart
function createGaugeChart(elementId, value, maxValue = 100) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return null;
    
    // Calculate percentage and determine color
    const percentage = (value / maxValue) * 100;
    let color;
    
    if (percentage < 40) {
        color = '#198754'; // success/green
    } else if (percentage < 70) {
        color = '#ffc107'; // warning/yellow
    } else {
        color = '#dc3545'; // danger/red
    }
    
    // Create gauge chart
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: [color, 'rgba(200, 200, 200, 0.2)'],
                borderWidth: 0
            }]
        },
        options: {
            circumference: 180,
            rotation: -90,
            cutout: '80%',
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                }
            },
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

// Function to create feature importance chart
function createFeatureImportanceChart(elementId, features) {
    const ctx = document.getElementById(elementId);
    if (!ctx || !features) return null;
    
    // Process feature names to make them more readable
    const labels = Object.keys(features).map(key => {
        return key.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    });
    
    const values = Object.values(features);
    
    // Sort by importance (descending)
    const combined = labels.map((label, i) => ({ label, value: values[i] }));
    combined.sort((a, b) => b.value - a.value);
    
    const sortedLabels = combined.map(item => item.label);
    const sortedValues = combined.map(item => item.value);
    
    // Create horizontal bar chart
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedLabels,
            datasets: [{
                label: 'Importance',
                data: sortedValues,
                backgroundColor: 'rgba(54, 162, 235, 0.7)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Relative Importance'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Weight: ${(context.raw * 100).toFixed(2)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Function to create comparison chart for health metrics
function createMetricsComparisonChart(elementId, userData, normalRanges) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return null;
    
    const labels = Object.keys(userData);
    const userValues = Object.values(userData);
    const lowerBounds = labels.map(label => normalRanges[label]?.min || 0);
    const upperBounds = labels.map(label => normalRanges[label]?.max || 0);
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Your Values',
                    data: userValues,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                    pointRadius: 5,
                    tension: 0.1
                },
                {
                    label: 'Lower Normal Bound',
                    data: lowerBounds,
                    backgroundColor: 'rgba(75, 192, 192, 0)',
                    borderColor: 'rgba(75, 192, 192, 0.5)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointStyle: false,
                    tension: 0.1
                },
                {
                    label: 'Upper Normal Bound',
                    data: upperBounds,
                    backgroundColor: 'rgba(255, 99, 132, 0)',
                    borderColor: 'rgba(255, 99, 132, 0.5)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointStyle: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            },
            elements: {
                line: {
                    fill: false
                }
            }
        }
    });
}
