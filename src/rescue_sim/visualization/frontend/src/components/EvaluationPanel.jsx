import { useEffect, useState } from 'react';

function getApiBaseUrl() {
  const proto = window.location.protocol;
  const host = window.location.hostname;
  const port = import.meta.env.DEV ? '8000' : window.location.port || '8000';
  return `${proto}//${host}:${port}`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 1) {
  return Number(value ?? 0).toFixed(digits);
}

export default function EvaluationPanel({ refreshKey }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    fetch(`${getApiBaseUrl()}/api/evaluation?refresh=${refreshKey}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Evaluation request failed (${response.status})`);
        }
        return response.json();
      })
      .then((data) => {
        if (!cancelled) {
          setReport(data);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  return (
    <div className="card">
      <div className="card-title">Baseline Comparison</div>

      {loading && <div className="evaluation-empty">Loading evaluation...</div>}
      {error && <div className="evaluation-empty">Evaluation unavailable</div>}

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
