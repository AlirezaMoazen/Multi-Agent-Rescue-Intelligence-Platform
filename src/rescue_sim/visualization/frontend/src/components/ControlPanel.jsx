/**
 * ControlPanel — Start / Stop / Restart + speed slider + skip-playback toggle.
 */
export default function ControlPanel({
  status,
  onStart,
  onInstantTrain,
  onRunLearned,
  onStop,
  onRestart,
  speed,
  onSpeedChange,
  skipPlayback,
  onSkipPlaybackChange,
  disabled,
  labels,
}) {
  const isRunning = status === 'running';
  const isIdle = status === 'idle' || status === 'stopped' || status === 'complete';

  return (
    <div className="controls">
      {isIdle && (
        <button id="btn-start" className="btn btn-primary" onClick={onStart} disabled={disabled}>
          {labels?.start ?? '▶ Start Simulation'}
        </button>
      )}
      {isIdle && (
        <button id="btn-instant-train" className="btn btn-success" onClick={onInstantTrain} disabled={disabled}>
          {labels?.instant ?? '⚡ Train Instantly'}
        </button>
      )}
      {isIdle && labels?.evaluate !== null && (
        <button id="btn-run-learned" className="btn" onClick={onRunLearned} disabled={disabled}>
          {labels?.evaluate ?? 'Run Learned Policy'}
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

      <label
        className="skip-toggle"
        title="Don't animate each try — run at full speed and jump straight to the per-try numbers and comparison"
      >
        <input
          id="skip-playback"
          type="checkbox"
          checked={skipPlayback}
          onChange={(e) => onSkipPlaybackChange(e.target.checked)}
        />
        ⏭ Skip to results
      </label>

      <div className="speed-control">
        <label htmlFor="speed-slider">Speed</label>
        <input
          id="speed-slider"
          className="speed-slider"
          type="range"
          min="6"
          max="500"
          step="2"
          value={speed}
          disabled={skipPlayback}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
        />
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', minWidth: 44 }}>
          {skipPlayback ? '—' : `${speed}ms`}
        </span>
      </div>
    </div>
  );
}
