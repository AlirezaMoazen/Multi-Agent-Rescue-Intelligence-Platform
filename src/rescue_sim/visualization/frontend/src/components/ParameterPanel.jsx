const MOE_HINT =
  '3 experts (heuristic explorer / deep coordination / local Q) blended live by an attention router; repeated tries on one fixed grid.';

export default function ParameterPanel({ config }) {
  const fields = [
    ['Grid', `${config.grid_width} x ${config.grid_height}`],
    ['Agents', config.num_agents],
    ['Sensor range', config.sensor_range],
    ['Targets', config.target_count],
    ['Max steps', config.max_steps],
    ['Tries (same grid)', config.num_episodes],
    ['Obstacle probability', config.obstacle_probability],
  ];

  return (
    <div className="card">
      <div className="card-title">Scenario — Neural MoE</div>

      <div className="algo-hint">{MOE_HINT}</div>

      <div className="scenario-grid">
        {fields.map(([label, value]) => (
          <div key={label} className="scenario-field">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="checkpoint-note">
        Train streams live BC / router progress, then the MoE solves the same grid on every try.
      </div>
    </div>
  );
}
