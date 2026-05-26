import { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

/**
 * MetricsChart — shows success rate & avg steps over episodes.
 */
export default function MetricsChart({ metrics }) {
  const chartData = useMemo(() => {
    if (!metrics || metrics.length === 0) return null;

    const labels = metrics.map(m => `Ep ${m.episode + 1}`);

    // Compute running success rate
    const runningSuccess = [];
    let successes = 0;
    metrics.forEach((m, i) => {
      if (m.success) successes++;
      runningSuccess.push(((successes / (i + 1)) * 100).toFixed(1));
    });

    return {
      labels,
      datasets: [
        {
          label: 'Success Rate (%)',
          data: runningSuccess,
          borderColor: '#22d3ee',
          backgroundColor: 'rgba(34, 211, 238, 0.08)',
          fill: true,
          tension: 0.35,
          pointRadius: 2,
          pointHoverRadius: 5,
          borderWidth: 2,
          yAxisID: 'y',
        },
        {
          label: 'Steps',
          data: metrics.map(m => m.steps),
          borderColor: '#a78bfa',
          backgroundColor: 'rgba(167, 139, 250, 0.08)',
          fill: true,
          tension: 0.35,
          pointRadius: 2,
          pointHoverRadius: 5,
          borderWidth: 2,
          yAxisID: 'y1',
        },
        {
          label: 'Reward',
          data: metrics.map(m => m.total_reward),
          borderColor: '#34d399',
          backgroundColor: 'rgba(52, 211, 153, 0.08)',
          fill: false,
          tension: 0.35,
          pointRadius: 2,
          pointHoverRadius: 5,
          borderWidth: 2,
          yAxisID: 'y1',
        },
      ],
    };
  }, [metrics]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        labels: {
          color: '#94a3b8',
          font: { family: 'Inter', size: 11 },
          boxWidth: 12,
          boxHeight: 12,
          useBorderRadius: true,
          borderRadius: 3,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1,
        cornerRadius: 8,
        titleFont: { family: 'Inter', weight: 600 },
        bodyFont: { family: 'Inter' },
      },
    },
    scales: {
      x: {
        ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        type: 'linear',
        position: 'left',
        min: 0,
        max: 100,
        ticks: {
          color: '#22d3ee',
          font: { family: 'Inter', size: 10 },
          callback: v => v + '%',
        },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
      y1: {
        type: 'linear',
        position: 'right',
        ticks: { color: '#a78bfa', font: { family: 'Inter', size: 10 } },
        grid: { drawOnChartArea: false },
      },
    },
  };

  if (!chartData) {
    return (
      <div className="card">
        <div className="card-title">Training Metrics</div>
        <div className="chart-empty">No data yet — start a simulation</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Training Metrics</div>
      <div className="chart-container">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
