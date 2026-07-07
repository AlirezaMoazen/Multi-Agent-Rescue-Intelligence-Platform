const ALGORITHMS = [
  {
    id: 'epidemic_fleet',
    label: 'Epidemic Fleet',
    hint: 'Tabular hysteretic Q-learning with peer-to-peer gossip.',
  },
  {
    id: 'neural_moe',
    label: 'Neural MoE',
    hint: '3 experts (heuristic / deep coordination / local Q) blended by an attention router; repeated tries on one fixed grid.',
  },
];

export default function ParameterPanel({ config, onChange, disabled }) {
  const algorithm = config.algorithm || 'epidemic_fleet';
  const isMoe = algorithm === 'neural_moe';

  const fields = [
    ['Grid', `${config.grid_width} x ${config.grid_height}`],
    ['Agents', config.num_agents],
    ['Sensor range', config.sensor_range],
    ['Targets', config.target_count],
    ['Max steps', config.max_steps],
    [isMoe ? 'Tries (same grid)' : 'Episodes', config.num_episodes],
    ['Obstacle probability', config.obstacle_probability],
  ];

  return (
    <div className="card">
      <div className="card-title">Scenario</div>

      <div className="algo-select" role="radiogroup" aria-label="Algorithm">
        {ALGORITHMS.map((algo) => (
          <button
            key={algo.id}
            type="button"
            role="radio"
            aria-checked={algorithm === algo.id}
            className={`algo-option ${algorithm === algo.id ? 'active' : ''}`}
            onClick={() => onChange({ ...config, algorithm: algo.id })}
            disabled={disabled}
          >
            {algo.label}
          </button>
        ))}
      </div>
      <div className="algo-hint">{ALGORITHMS.find((a) => a.id === algorithm)?.hint}</div>

      <div className="scenario-grid">
        {fields.map(([label, value]) => (
          <div key={label} className="scenario-field">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="checkpoint-note">
        {isMoe
          ? 'Train streams live BC / router progress, then the MoE solves the same grid on every try.'
          : 'The Deep RL benchmark table uses saved QMIX, TransfQMix and MAPPO checkpoints.'}
      </div>
    </div>
  );
}
