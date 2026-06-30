const SAME_GRID = [
  ['epidemic_hysteretic_q', 'Q-learning gossip', 'Multi-agent Q-learning with nearby-agent gossip.'],
  ['frontier', 'Baseline', 'Greedy frontier exploration.'],
  ['dfs', 'Baseline', 'Depth-first exploration.'],
  ['prioritized_planning', 'Baseline', 'Sequential multi-agent path planner.'],
  ['cbs', 'Baseline', 'Conflict-Based Search planner.'],
  ['icbs', 'Baseline', 'Improved CBS planner.'],
  ['ecbs', 'Baseline', 'Bounded-suboptimal CBS planner.'],
  ['mstar', 'Baseline', 'Coupled multi-agent path planner.'],
  ['trained', 'Legacy Q-learning', 'Legacy tabular comparator.'],
];

const DEEP = [
  ['MAPPO', 'Deep RL', 'Policy-gradient multi-agent method.'],
  ['QMIX', 'Deep RL', 'Value-factorization team Q method.'],
  ['TransfQMix', 'Deep RL', 'Transformer-based QMIX variant.'],
  ['Ensemble', 'Deep RL', 'Combines QMIX and TransfQMix values.'],
  ['Distilled', 'Deep RL', 'Student model distilled from deep teachers.'],
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
  return [
    ['Success', pct(row?.success_rate)],
    ['Steps', num(row?.average_steps)],
    ['Rescued', num(row?.average_rescued_targets)],
    ['Score', num(row?.average_accumulated_reward)],
  ];
}

function Leaderboard({ report }) {
  const hybrid = report?.hybrid_report;
  if (!hybrid) {
    return (
      <div className="winner-empty">
        Winner appears here after the checkpoint-backed hybrid run finishes.
      </div>
    );
  }

  const entries = Object.entries(hybrid.leaderboard || {})
    .map(([name, metrics]) => ({ name, ...metrics }))
    .sort((a, b) => Number(b.score ?? -999) - Number(a.score ?? -999));
  const winner = entries.find((entry) => entry.name === hybrid.final_leader) || entries[0];
  const losers = entries.filter((entry) => entry.name !== winner?.name);

  return (
    <>
      <section className="winner-panel">
        <div>
          <div className="winner-label">Selected live expert</div>
          <div className="winner-name">{winner?.name || hybrid.final_leader}</div>
        </div>
        <div className="winner-metrics">
          <strong>{winner?.success ? 'Solved' : 'Not solved'}</strong>
          <span>{winner?.rescued ?? 0}/{winner?.targets ?? 0} rescued</span>
          <span>{winner?.steps ?? 'Pending'} steps</span>
          <span>score {num(winner?.score)}</span>
        </div>
      </section>

      <section className="contender-grid">
        {losers.map((entry) => (
          <div key={entry.name} className="contender-card">
            <div className="contender-name">{entry.name}</div>
            <div className="contender-stats">
              <strong>{entry.success ? 'Solved' : 'Not solved'}</strong>
              <span>{entry.rescued}/{entry.targets} rescued</span>
              <span>{entry.steps} steps</span>
              <span>score {num(entry.score)}</span>
            </div>
          </div>
        ))}
      </section>
    </>
  );
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
      <Leaderboard report={report} />

      <AlgorithmTable title="Same-grid comparison" algorithms={SAME_GRID} rows={report?.aggregates || []} />
      <AlgorithmTable title="Deep RL checkpoint benchmark" algorithms={DEEP} rows={report?.deep_benchmark || []} />

      {report?.deep_benchmark_note && (
        <div className="evaluation-note">{report.deep_benchmark_note}</div>
      )}
    </div>
  );
}
