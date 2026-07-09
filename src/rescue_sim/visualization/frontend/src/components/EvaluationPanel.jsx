const SAME_GRID = [
  ['epidemic_hysteretic_q', 'Q-learning gossip', 'Multi-agent Q-learning with nearby-agent gossip.'],
  ['frontier', 'Baseline', 'Greedy frontier exploration.'],
  ['apf', 'Baseline', 'Artificial Potential Fields swarm navigation.'],
];

const DEEP = [
  ['QMIX', 'Deep RL', 'Value-factorization team Q method.'],
  ['TransfQMix', 'Deep RL', 'Transformer-based QMIX variant.'],
  ['MAPPO', 'Deep RL', 'Policy-gradient multi-agent method.'],
];

function byName(rows = []) {
  return rows.reduce((lookup, row) => {
    lookup[row.agent_name] = row;
    return lookup;
  }, {});
}

function pct(value) {
  if (value == null) return 'Pending';
  return `${(value * 100).toFixed(0)}%`;
}

function num(value, digits = 1) {
  if (value == null) return 'Pending';
  return Number(value).toFixed(digits);
}

function metricSet(row) {
  const pills = [
    ['Success', pct(row?.success_rate)],
    ['Steps', num(row?.average_steps)],
    ['Rescued', num(row?.average_rescued_targets)],
    ['Score', num(row?.average_accumulated_reward)],
  ];
  // Once a row has data, hide metrics that method does not report.
  return row ? pills.filter(([, value]) => value !== 'Pending') : pills;
}

function AlgorithmTable({ title, algorithms, rows }) {
  const metrics = byName(rows);

  return (
    <section className="evaluation-section">
      <div className="evaluation-section-title">{title}</div>
      <div className="evaluation-table-wrap">
        <table className="evaluation-table">
          <thead>
            <tr>
              <th>Algorithm</th>
              <th>Type</th>
              <th>Description</th>
              <th>Metrics</th>
            </tr>
          </thead>
          <tbody>
            {algorithms.map(([name, group, description]) => {
              const row = metrics[name];
              return (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{group}</td>
                  <td>{description}</td>
                  <td>
                    <div className="metric-pills">
                      {metricSet(row).map(([label, value]) => (
                        <span key={label}>
                          {label} <strong>{value}</strong>
                        </span>
                      ))}
                    </div>
                    {row?.communication_events != null && (
                      <div className="evaluation-muted">
                        comms {num(row.communication_events)}
                      </div>
                    )}
                    {row?.error && <div className="evaluation-error">{row.error}</div>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function EvaluationPanel({ report }) {
  return (
    <div className="card">
      <div className="card-title">Evaluation Comparison</div>
      <AlgorithmTable title="Same-grid comparison" algorithms={SAME_GRID} rows={report?.aggregates || []} />
      <AlgorithmTable title="Deep RL checkpoint benchmark" algorithms={DEEP} rows={report?.deep_benchmark || []} />
      {report?.deep_benchmark_note && (
        <div className="evaluation-note">{report.deep_benchmark_note}</div>
      )}
    </div>
  );
}
