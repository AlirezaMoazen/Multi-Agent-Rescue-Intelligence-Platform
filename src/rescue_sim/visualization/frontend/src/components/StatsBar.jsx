/**
 * StatsBar — Live simulation statistics displayed at the top.
 */
export default function StatsBar({ episode, step, rescued, activeTargets, successRate, avgSteps, totalReward, explorationRate }) {
  const stats = [
    { icon: '📡', label: 'Episode',         value: episode,                              color: 'cyan'   },
    { icon: '👣', label: 'Steps',            value: step,                                 color: 'purple' },
    { icon: '🎯', label: 'Rescued',          value: `${rescued.length}`,                  color: 'green'  },
    { icon: '⏳', label: 'Remaining',        value: activeTargets,                        color: 'amber'  },
    { icon: '📈', label: 'Success Rate',     value: `${(successRate * 100).toFixed(1)}%`, color: 'cyan'   },
    { icon: '🏃', label: 'Avg Steps',        value: avgSteps.toFixed(0),                  color: 'purple' },
    { icon: '💰', label: 'Total Reward',     value: totalReward.toFixed(1),               color: 'green'  },
    { icon: '🔍', label: 'Exploration',      value: `${(explorationRate * 100).toFixed(1)}%`, color: 'amber' },
  ];

  return (
    <div className="stats-bar">
      {stats.map((s, i) => (
        <div key={i} className="card stat-card">
          <div className={`stat-icon ${s.color}`}>{s.icon}</div>
          <div className="stat-content">
            <span className="stat-value">{s.value}</span>
            <span className="stat-label">{s.label}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
