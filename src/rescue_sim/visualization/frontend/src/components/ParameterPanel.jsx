export default function ParameterPanel({ config }) {
  const fields = [
    ['Grid', `${config.grid_width} x ${config.grid_height}`],
    ['Agents', config.num_agents],
    ['Sensor range', config.sensor_range],
    ['Targets', config.target_count],
    ['Max steps', config.max_steps],
    ['MoE trials', config.num_episodes],
    ['Obstacle probability', config.obstacle_probability],
  ];

  return (
    <div className="card">
      <div className="card-title">Scenario</div>
      <div className="scenario-grid">
        {fields.map(([label, value]) => (
          <div key={label} className="scenario-field">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="checkpoint-note">
        Hybrid live mode uses saved QMIX, TransfQMix and MAPPO checkpoints.
      </div>
    </div>
  );
}
