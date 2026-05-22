/**
 * ParameterPanel — Editable simulation parameters (grid, agent, RL).
 */
export default function ParameterPanel({ config, onChange, disabled }) {
  const handle = (key, type) => (e) => {
    const raw = e.target.value;
    const val = type === 'int' ? parseInt(raw, 10) : parseFloat(raw);
    if (!isNaN(val)) {
      onChange({ ...config, [key]: val });
    }
  };

  const fields = [
    { key: 'grid_width',            label: 'Grid Width',     type: 'int' },
    { key: 'grid_height',           label: 'Grid Height',    type: 'int' },
    { key: 'obstacle_probability',  label: 'Obstacle %',     type: 'float' },
    { key: 'target_count',          label: 'Targets',        type: 'int' },
    { key: 'num_agents',            label: 'Agents',         type: 'int' },
    { key: 'sensor_range',          label: 'Sensor Range',   type: 'int' },
    { key: 'max_steps',             label: 'Max Steps',      type: 'int' },
    { key: 'num_episodes',          label: 'Episodes',       type: 'int' },
    { key: 'learning_rate',         label: 'Learning Rate',  type: 'float' },
    { key: 'discount_factor',       label: 'Discount (γ)',   type: 'float' },
    { key: 'exploration_rate',      label: 'Exploration (ε)', type: 'float' },
  ];

  return (
    <div className="card">
      <div className="card-title">Simulation Parameters</div>
      <div className="param-grid">
        {fields.map(f => (
          <div key={f.key} className="param-field">
            <label htmlFor={`param-${f.key}`}>{f.label}</label>
            <input
              id={`param-${f.key}`}
              type="number"
              step={f.type === 'float' ? '0.01' : '1'}
              min={f.type === 'float' ? '0' : '1'}
              value={config[f.key]}
              onChange={handle(f.key, f.type)}
              disabled={disabled}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
