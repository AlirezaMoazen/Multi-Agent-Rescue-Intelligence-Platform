function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 1) {
  if (value == null) return '—';
  return Number(value ?? 0).toFixed(digits);
}

function groupLabel(agent) {
  const group = agent.algorithm_group;
  if (group === 'main') return 'Main algorithm';
  if (group === 'q_learning') return 'Q-learning';
  if (group === 'baseline') return 'Baselines';
  if (group === 'deep_marl') return 'Deep RL';

  const name = String(agent.agent_name || '').toLowerCase();
  if (name.includes('q-learning')) return 'Q-learning';
  return 'Baselines';
}

function groupedAgents(aggregates = []) {
  return aggregates.reduce((groups, agent) => {
    const label = groupLabel(agent);
    if (!groups[label]) groups[label] = [];
    groups[label].push(agent);
    return groups;
  }, {});
}

export default function EvaluationPanel({ report }) {
  const groups = groupedAgents(report?.aggregates);
  const hasDeepBenchmark = (report?.deep_benchmark || []).length > 0;

  return (
    <div className="card">
      <div className="card-title">Evaluation Comparison</div>

      {!report && (
        <div className="evaluation-empty">
          Run evaluation to compare algorithms on the current simulation grid.
        </div>
      )}

      {report && (
        <>
          {Object.entries(groups).map(([label, agents]) => (
            <section className="evaluation-section" key={label}>
              <div className="evaluation-section-title">{label}</div>
              <div className="evaluation-grid">
                {agents.map((agent) => (
                  <div className="evaluation-agent" key={agent.agent_name}>
                    <div className="evaluation-agent-header">
                      <div className="evaluation-agent-name">{agent.agent_name}</div>
                      {agent.status && (
                        <span className={`evaluation-status status-${agent.status}`}>
                          {agent.status}
                        </span>
                      )}
                    </div>
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
                      {agent.num_agents != null && (
                        <>
                          <span>agents</span>
                          <strong>{agent.num_agents}</strong>
                        </>
                      )}
                      {agent.communication_events != null && (
                        <>
                          <span>comms</span>
                          <strong>{formatNumber(agent.communication_events)}</strong>
                        </>
                      )}
                    </div>
                    {agent.error && <div className="evaluation-error">{agent.error}</div>}
                  </div>
                ))}
              </div>
            </section>
          ))}

          <div className="evaluation-summary">
            {report.sprint_demo_summary}
          </div>

          {hasDeepBenchmark && (
            <section className="evaluation-section evaluation-benchmark">
              <div className="evaluation-section-title">Deep RL benchmark</div>
              <div className="evaluation-note">
                {report.deep_benchmark_note}
              </div>
              <div className="evaluation-table-wrap">
                <table className="evaluation-table">
                  <thead>
                    <tr>
                      <th>Algorithm</th>
                      <th>Mode</th>
                      <th>Status</th>
                      <th>Success</th>
                      <th>Steps</th>
                      <th>Rescued</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.deep_benchmark.map((item) => (
                      <tr key={item.agent_name}>
                        <td>{item.agent_name}</td>
                        <td>{item.evaluation_mode}</td>
                        <td>{item.status}</td>
                        <td>{formatNumber(item.success_rate)}</td>
                        <td>{formatNumber(item.average_steps)}</td>
                        <td>{formatNumber(item.average_rescued_targets)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
