import { useEffect, useRef, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

// Brand colors, per policy (must match the backend _COMPARE_POLICIES).
const COLORS = {
  'Non-AI (APF)': '#94a3b8',
  'Expert 1': '#3b82f6',
  'Expert 2': '#10b981',
  'Expert 3': '#f59e0b',
  MoE: '#8b5cf6',
};

// Four normalized 0-100 axes so the policies are directly comparable in one
// grouped bar chart (raw steps live in the winner cards instead).
const AXES = ['Success %', 'Rescued %', 'Connectivity %', 'Efficiency %'];

function normalize(p, maxSteps) {
  const rescuedPct = p.targets ? (p.avg_rescued / p.targets) * 100 : 0;
  const efficiency = maxSteps ? Math.max(0, (1 - p.avg_steps / maxSteps) * 100) : 0;
  return [p.success_rate * 100, rescuedPct, p.peer_connectivity * 100, efficiency];
}

function WinnerCard({ label, name, detail }) {
  return (
    <div className="cmp-card" style={{ '--accent': COLORS[name] || '#8b5cf6' }}>
      <div className="cmp-card-label">{label}</div>
      <div className="cmp-card-name">{name}</div>
      <div className="cmp-card-detail">{detail}</div>
    </div>
  );
}

function Skeleton({ progress }) {
  return (
    <div className="cmp-skeleton">
      <div className="cmp-skel-bars">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="cmp-skel-bar" style={{ animationDelay: `${i * 0.12}s` }} />
        ))}
      </div>
      <div className="cmp-progress">
        <div className="cmp-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="cmp-skel-text">Running greedy rollouts on identical grids… (CPU-bound, can take a minute)</div>
    </div>
  );
}

export default function PolicyComparison({ episodes = 30, autoRunToken = 0 }) {
  const [state, setState] = useState('idle'); // idle | loading | done | error
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const seenToken = useRef(autoRunToken);

  const run = async () => {
    setState('loading');
    setError(null);
    setProgress(8);
    // Indeterminate-but-lively progress while the backend batches run.
    const timer = setInterval(() => setProgress((p) => Math.min(92, p + Math.random() * 9)), 400);
    try {
      const res = await fetch(`/api/compare_policies?episodes=${episodes}`);
      clearInterval(timer);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.message || `Comparison failed (${res.status})`);
      }
      const payload = await res.json();
      setProgress(100);
      setData(payload);
      setState('done');
    } catch (e) {
      clearInterval(timer);
      setError(e.message);
      setState('error');
    }
  };

  // Auto-run when the parent bumps the token (skip-playback runs jump
  // straight to the comparison). Runs after every render; the ref guard
  // makes it fire once per bump.
  useEffect(() => {
    if (autoRunToken > seenToken.current) {
      seenToken.current = autoRunToken;
      if (state !== 'loading') run();
    }
  });

  const maxSteps = data
    ? Math.max(...data.policies.map((p) => p.avg_steps), 1)
    : 1;

  const chartData = data && {
    labels: AXES,
    datasets: data.policies.map((p) => ({
      label: p.name,
      data: normalize(p, Math.max(maxSteps, 60)),
      backgroundColor: `${COLORS[p.name]}cc`,
      hoverBackgroundColor: COLORS[p.name],
      borderColor: COLORS[p.name],
      borderWidth: 1,
      borderRadius: 5,
      borderSkipped: false,
    })),
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 900, easing: 'easeOutQuart' },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: { color: '#cbd5e1', usePointStyle: true, pointStyleWidth: 10, font: { size: 11 } },
      },
      tooltip: {
        callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}` },
      },
    },
    scales: {
      x: { grid: { color: 'rgba(148,163,184,0.08)' }, ticks: { color: '#94a3b8', font: { size: 11 } } },
      y: {
        beginAtZero: true, max: 100,
        grid: { color: 'rgba(148,163,184,0.08)' },
        ticks: { color: '#94a3b8', font: { size: 10 }, callback: (v) => `${v}` },
      },
    },
  };

  return (
    <div className="card cmp-card-wrap">
      <div className="card-title cmp-title">
        <span>Experts vs. MoE — head-to-head</span>
        <button className="btn btn-primary cmp-run" onClick={run} disabled={state === 'loading'}>
          {state === 'loading' ? 'Running…' : state === 'done' ? '↻ Re-run' : '▶ Compare'}
        </button>
      </div>

      {state === 'idle' && (
        <div className="cmp-empty">
          Runs greedy rollouts of the raw non-AI APF baseline, each standalone expert head
          (E2 = the MAPPO + QMIX + TransfQMix distilled coordinator) and the blended MoE on the
          <strong> same {episodes} grids</strong> — so the gap between "Non-AI (APF)" and the
          rest is exactly what ML buys. Train the MoE first.
        </div>
      )}

      {state === 'loading' && <Skeleton progress={progress} />}

      {state === 'error' && <div className="cmp-error">⚠ {error}</div>}

      {state === 'done' && data && (
        <div className="cmp-body">
          <div className="cmp-cards">
            <WinnerCard label="Best success" name={data.winners.success.name}
              detail={`${(data.winners.success.value * 100).toFixed(0)}% solved`} />
            <WinnerCard label="Most efficient" name={data.winners.efficiency.name}
              detail={`${data.winners.efficiency.value.toFixed(0)} steps`} />
            <WinnerCard label="Most rescued" name={data.winners.rescued.name}
              detail={`${data.winners.rescued.value.toFixed(2)} / 4`} />
            <WinnerCard label="Best connectivity" name={data.winners.connectivity.name}
              detail={`${(data.winners.connectivity.value * 100).toFixed(0)}% linked`} />
          </div>

          <div className="cmp-chart">
            <Bar data={chartData} options={chartOptions} />
          </div>

          <div className="cmp-foot">
            {data.grid} · {data.num_agents} agents · {data.episodes} episodes on identical grids
          </div>
        </div>
      )}
    </div>
  );
}
