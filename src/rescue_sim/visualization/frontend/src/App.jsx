import { useState } from 'react';
import useSimulation from './hooks/useSimulation';
import StatsBar from './components/StatsBar';
import GridCanvas from './components/GridCanvas';
import ControlPanel from './components/ControlPanel';
import ParameterPanel from './components/ParameterPanel';
import MetricsChart from './components/MetricsChart';
import EvaluationPanel from './components/EvaluationPanel';

const DEFAULT_CONFIG = {
  grid_width: 10,
  grid_height: 10,
  obstacle_probability: 0.15,
  target_count: 4,
  num_agents: 1,
  sensor_range: 3,
  max_steps: 500,
  num_episodes: 10,
  learning_rate: 0.1,
  discount_factor: 0.9,
  exploration_rate: 1.0,
  speed_ms: 100,
  run_mode: 'train',
};

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const sim = useSimulation();

  const isRunning = sim.status === 'running';
  const isComplete = sim.status === 'complete';

  const handleStart = () => {
    const nextConfig = { ...config, run_mode: 'train' };
    setConfig(nextConfig);
    sim.start(nextConfig);
  };

  const handleInstantTrain = () => {
    const nextConfig = { ...config, run_mode: 'instant_train' };
    setConfig(nextConfig);
    sim.start(nextConfig);
  };

  const handleRunLearned = () => {
    const nextConfig = { ...config, run_mode: 'evaluate' };
    setConfig(nextConfig);
    sim.start(nextConfig);
  };

  const handleStop = () => {
    sim.stop();
  };

  const handleRestart = () => {
    sim.start({ ...config });
  };

  // Send live speed updates to the backend while running
  const handleSpeedChange = (ms) => {
    const newConfig = { ...config, speed_ms: ms };
    setConfig(newConfig);
    if (isRunning) {
      sim.send({ type: 'config', data: newConfig });
    }
  };

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="app-header">
        <h1>🚁 Rescue Sim — Multi-Agent Visualization</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span className={`connection-dot ${sim.connected ? 'connected' : 'disconnected'}`} />
            {sim.connected ? 'Connected' : 'Reconnecting…'}
          </span>
          <span className="team-badge">TUHH Group 05</span>
        </div>
      </header>

      {/* ── Status Banner ──────────────────────────────────────────── */}
      {sim.error && (
        <div className="status-banner error">
          ⚠ {sim.error}
        </div>
      )}
      {isComplete && (
        <div className="status-banner success">
          ✅ Training complete — {sim.episodeMetrics.length} episodes finished •
          Final success rate: {(sim.successRate * 100).toFixed(1)}%
        </div>
      )}
      {sim.status === 'stopped' && !sim.error && (
        <div className="status-banner warning">
          ⏸ Simulation stopped — adjust parameters and restart
        </div>
      )}
      {!sim.connected && (
        <div className="status-banner error">
          ⚠ Disconnected from backend — make sure the API server is running on port 8000
        </div>
      )}

      {/* ── Body ───────────────────────────────────────────────────── */}
      <main className="app-body">
        {/* Stats */}
        <StatsBar
          episode={sim.episode}
          step={sim.step}
          rescued={sim.rescued}
          activeTargets={sim.activeTargets}
          successRate={sim.successRate}
          avgSteps={sim.avgSteps}
          totalReward={sim.totalReward}
          explorationRate={sim.explorationRate}
        />

        {/* Grid + Controls */}
        <div className="grid-area">
          <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Rescue Grid</span>
              {sim.grid && (
                <span style={{ fontWeight: 400, fontSize: '0.65rem', textTransform: 'none', letterSpacing: 0 }}>
                  {sim.grid.width}×{sim.grid.height} • {sim.grid.obstacles.length} obstacles • {sim.grid.targets.length} targets
                </span>
              )}
            </div>
            <div className="grid-wrapper">
              <GridCanvas grid={sim.grid} agents={sim.agents} rescued={sim.rescued} trails={sim.trails} />
            </div>
            <div className="legend">
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: '#1e293b' }} /> Wall
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: '#f87171' }} /> Target A
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: '#fb923c' }} /> Target B
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: 'rgba(52,211,153,0.4)' }} /> Rescued
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ background: '#22d3ee', borderRadius: '50%' }} /> Agent
              </div>
            </div>
          </div>

          <ControlPanel
            status={sim.status}
            onStart={handleStart}
            onInstantTrain={handleInstantTrain}
            onRunLearned={handleRunLearned}
            onStop={handleStop}
            onRestart={handleRestart}
            speed={config.speed_ms}
            onSpeedChange={handleSpeedChange}
            disabled={!sim.connected}
          />
        </div>

        {/* Side Panel */}
        <div className="side-panel">
          <ParameterPanel config={config} onChange={setConfig} disabled={isRunning} />
          <MetricsChart metrics={sim.episodeMetrics} />
        </div>

        <section className="evaluation-wide">
          <EvaluationPanel report={sim.baselineComparison} />
        </section>
      </main>
    </div>
  );
}
