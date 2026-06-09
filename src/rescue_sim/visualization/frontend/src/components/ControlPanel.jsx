/**
 * ControlPanel — Start / Stop / Restart + speed slider.
 */
export default function ControlPanel({
  status,
  onStart,
  onRunLearned,
  onStop,
  onRestart,
  speed,
  onSpeedChange,
  disabled,
}) {
  const isRunning = status === 'running';
  const isIdle = status === 'idle' || status === 'stopped' || status === 'complete';

  return (
    <div className="controls">
      {isIdle && (
        <button id="btn-start" className="btn btn-primary" onClick={onStart} disabled={disabled}>
          ▶ Start Simulation
        </button>
      )}
      {isIdle && (
        <button id="btn-run-learned" className="btn" onClick={onRunLearned} disabled={disabled}>
          Run Learned Policy
        </button>
      )}
      {isRunning && (
        <button id="btn-stop" className="btn btn-danger" onClick={onStop}>
          ⏹ Stop
        </button>
      )}
      <button id="btn-restart" className="btn" onClick={onRestart} disabled={disabled || isRunning}>
        🔄 Restart
      </button>

      <div className="speed-control">
        <label htmlFor="speed-slider">Speed</label>
        <input
          id="speed-slider"
          className="speed-slider"
          type="range"
          min="10"
          max="500"
          step="10"
          value={speed}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
        />
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', minWidth: 44 }}>
          {speed}ms
        </span>
      </div>
    </div>
  );
}
