function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 1) {
  return Number(value ?? 0).toFixed(digits);
}

export default function EvaluationPanel({ report }) {
  return (
    <div className="card">
      <div className="card-title">Baseline Comparison</div>

      {!report && (
        <div className="evaluation-empty">
          Run Learned Policy to compare the learned agent against the baseline.
        </div>
      )}

      {report && (
        <>
          <div className="evaluation-grid">
            {report.aggregates.map((agent) => (
              <div className="evaluation-agent" key={agent.agent_name}>
                <div className="evaluation-agent-name">{agent.agent_name}</div>
                <div className="evaluation-main">
                  {formatPercent(agent.success_rate)}
                  <span>success</span>
                </div>
                <div className="evaluation-stats">
                  <span>steps</span>
                  <strong>{formatNumber(agent.average_steps)}</strong>
                  <span>reward</span>
                  <strong>{formatNumber(agent.average_accumulated_reward)}</strong>
                  <span>rescued</span>
                  <strong>{formatNumber(agent.average_rescued_targets)}</strong>
                  <span>explored</span>
                  <strong>{formatNumber(agent.average_explored_area_percentage)}%</strong>
                </div>
              </div>
            ))}
          </div>

          <div className="evaluation-summary">
            {report.sprint_demo_summary}
          </div>
        </>
      )}

      <div style={{ marginTop: 24, borderTop: '1px solid var(--border-glass)', paddingTop: 20 }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          MAPF Algorithmic Comparison
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="comparison-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.72rem', textAlign: 'left', color: 'var(--text-muted)' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-glass)', color: 'var(--text-primary)' }}>
                <th style={{ padding: '8px 6px', fontWeight: 600 }}>Algorithm</th>
                <th style={{ padding: '8px 6px', fontWeight: 600 }}>Type</th>
                <th style={{ padding: '8px 6px', fontWeight: 600 }}>Optimality</th>
                <th style={{ padding: '8px 6px', fontWeight: 600 }}>Description vs Q-Learning</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '8px 6px', fontWeight: 600, color: 'var(--text-primary)' }}>CBS</td>
                <td style={{ padding: '8px 6px' }}>Centralized</td>
                <td style={{ padding: '8px 6px', color: 'var(--accent-green)' }}>✅ Optimal</td>
                <td style={{ padding: '8px 6px' }}>Resolves agent collisions using a constraint tree. Guarantees shortest path but scales poorly as agent counts grow.</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '8px 6px', fontWeight: 600, color: 'var(--text-primary)' }}>ICBS</td>
                <td style={{ padding: '8px 6px' }}>Centralized</td>
                <td style={{ padding: '8px 6px', color: 'var(--accent-green)' }}>✅ Optimal</td>
                <td style={{ padding: '8px 6px' }}>Improved CBS that prioritizes conflicts first; executes faster than standard CBS.</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '8px 6px', fontWeight: 600, color: 'var(--text-primary)' }}>ECBS</td>
                <td style={{ padding: '8px 6px' }}>Centralized</td>
                <td style={{ padding: '8px 6px', color: 'var(--accent-amber)' }}>Bounded Suboptimal</td>
                <td style={{ padding: '8px 6px' }}>Enhanced CBS offering suboptimal paths within a specified bound; significantly faster for real-time applications.</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '8px 6px', fontWeight: 600, color: 'var(--text-primary)' }}>Prioritized Planning</td>
                <td style={{ padding: '8px 6px' }}>Decentralized</td>
                <td style={{ padding: '8px 6px', color: 'var(--accent-red)' }}>❌ Suboptimal</td>
                <td style={{ padding: '8px 6px' }}>Agents plan sequentially (highest priority first), treating other paths as dynamic obstacles. Fast and simple.</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <td style={{ padding: '8px 6px', fontWeight: 600, color: 'var(--text-primary)' }}>M*</td>
                <td style={{ padding: '8px 6px' }}>Centralized/Hybrid</td>
                <td style={{ padding: '8px 6px', color: 'var(--accent-green)' }}>✅ Optimal</td>
                <td style={{ padding: '8px 6px' }}>Dynamically scales configuration space dimensionality based on conflict density to find paths.</td>
              </tr>
              <tr style={{ backgroundColor: 'var(--accent-cyan-dim)' }}>
                <td style={{ padding: '8px 6px', fontWeight: 700, color: 'var(--accent-cyan)' }}>Q-Learning Model</td>
                <td style={{ padding: '8px 6px', fontWeight: 600 }}>Decentralized (execution)</td>
                <td style={{ padding: '8px 6px', color: 'var(--accent-cyan)' }}>Learned Suboptimal</td>
                <td style={{ padding: '8px 6px', color: 'var(--text-primary)' }}>Decentralized execution after training. Learns robust paths through trial-and-error; scales well to large agents but requires a training phase.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
