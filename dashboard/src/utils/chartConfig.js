/* ==============================================================================
   CHART.JS GLOBAL CONFIGURATION — SOC THEME
   ============================================================================== */

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Filler,
  Tooltip,
  Legend
);

// Global defaults
ChartJS.defaults.color = '#94a3b8';
ChartJS.defaults.borderColor = 'rgba(0, 240, 255, 0.08)';
ChartJS.defaults.font.family = "'JetBrains Mono', monospace";
ChartJS.defaults.font.size = 10;
ChartJS.defaults.plugins.legend.display = false;
ChartJS.defaults.animation.duration = 400;

export const lineDefaults = {
  tension: 0.4,
  borderWidth: 2,
  pointRadius: 0,
  pointHoverRadius: 4,
  fill: true,
};

export const gridStyle = {
  color: 'rgba(0, 240, 255, 0.06)',
  drawBorder: false,
};

export const createTimeSeriesOptions = (yLabel = '', suggestedMax) => ({
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  scales: {
    x: {
      grid: { display: false },
      ticks: { maxTicksLimit: 6 },
    },
    y: {
      grid: gridStyle,
      suggestedMin: 0,
      ...(suggestedMax !== undefined && { suggestedMax }),
      title: {
        display: !!yLabel,
        text: yLabel,
        font: { size: 10 },
      },
    },
  },
  plugins: {
    tooltip: {
      backgroundColor: 'rgba(10, 14, 39, 0.95)',
      borderColor: 'rgba(0, 240, 255, 0.3)',
      borderWidth: 1,
      titleFont: { family: "'Inter', sans-serif", weight: 600 },
      bodyFont: { family: "'JetBrains Mono', monospace" },
      padding: 10,
      cornerRadius: 8,
    },
  },
});
