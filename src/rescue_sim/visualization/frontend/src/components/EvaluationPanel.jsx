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
    </div>
  );
}
