export const EXPERT_META = {
  exploration: { label: 'E1 · Heuristic Explorer', hint: 'non-AI frontier', color: 'var(--accent-cyan)' },
  coordination: { label: 'E2 · Deep Coordination', hint: 'distilled QMIX / TransfQMix', color: 'var(--accent-green)' },
  fallback: { label: 'E3 · Local Hysteretic Q', hint: 'GRU fallback (isolated)', color: 'var(--accent-amber)' },
};

const EXPERT_ORDER = ['exploration', 'coordination', 'fallback'];

function WeightBar({ value, color }) {
  return (
    <div className="moe-bar">
      <div className="moe-bar-fill" style={{ width: `${Math.round(value * 100)}%`, background: color }} />
    </div>
  );
}

function TrainingProgress({ training }) {
  const stageLabel = training.stage === 'distillation'
    ? 'Stage 1 · Behavioral cloning (3 expert heads)'
    : 'Stage 2 · Attention router (blackout penalties)';
  const pct = Math.round((training.epoch / training.total) * 100);
  return (
    <div className="moe-training">
      <div className="moe-training-stage">{stageLabel}</div>
      <div className="moe-bar moe-bar-lg">
        <div className="moe-bar-fill" style={{ width: `${pct}%`, background: 'var(--accent-purple)' }} />
      </div>
      <div className="moe-training-stats">
        <span>{training.epoch}/{training.total}</span>
        <span>loss {training.loss?.toFixed(4)}</span>
        {training.stage === 'distillation' && <span>BC acc {training.accuracy?.toFixed(1)}%</span>}
      </div>
    </div>
  );
}

/**
 * MoePanel — live view of the neural Mixture-of-Experts router.
 *
 * Top: the three experts with the fleet-average gating weight (the "router
 * diagram"). Middle: per-agent softmax routing vectors with peer count and an
 * ISOLATED badge under communication blackout. Bottom: baseline parameters.
 * While training, shows the streamed stage/epoch/loss progress instead.
 */
export default function MoePanel({ moe, training, status }) {
  const isTraining = status === 'running' && !moe && training;

  return (
    <div className="card">
      <div className="card-title">MoE Router — 3 Experts</div>

      {isTraining && <TrainingProgress training={training} />}

      {!moe && !isTraining && (
        <div className="moe-empty">
          Select <strong>Neural MoE</strong> and press Start: the router trains
          live, then solves the same grid over repeated tries.
        </div>
      )}

      {moe && (
        <>
          <div className="moe-experts">
            {EXPERT_ORDER.map((name, j) => {
              const meta = EXPERT_META[name];
              const avg = moe.weights.reduce((sum, w) => sum + w[j], 0) / moe.weights.length;
              return (
                <div key={name} className="moe-expert">
                  <div className="moe-expert-head">
                    <span className="moe-expert-dot" style={{ background: meta.color }} />
                    <span className="moe-expert-label">{meta.label}</span>
                    <span className="moe-expert-avg">{(avg * 100).toFixed(0)}%</span>
                  </div>
                  <WeightBar value={avg} color={meta.color} />
                  <div className="moe-expert-hint">{meta.hint}</div>
                </div>
              );
            })}
          </div>

          <div className="moe-agents">
            {moe.weights.map((w, i) => {
              const isolated = moe.peer_count[i] <= 1;
              const active = EXPERT_META[moe.active_expert[i]];
              return (
                <div key={i} className="moe-agent-row">
                  <span className="moe-agent-id" style={{ borderColor: active.color, color: active.color }}>
                    {i + 1}
                  </span>
                  <div className="moe-agent-bars">
                    {EXPERT_ORDER.map((name, j) => (
                      <WeightBar key={name} value={w[j]} color={EXPERT_META[name].color} />
                    ))}
                  </div>
                  <span className="moe-agent-peers" title="agents in the 3-block radius (incl. self)">
                    ⛓ {moe.peer_count[i]}
                  </span>
                  {isolated
                    ? <span className="moe-badge moe-badge-isolated">BLACKOUT</span>
                    : <span className="moe-badge moe-badge-linked">LINKED</span>}
                </div>
              );
            })}
          </div>

          <div className="moe-baselines">
            Hyst Q α={moe.baselines.hysteretic_alpha} · β={moe.baselines.hysteretic_beta} ·
            frontier γ={moe.baselines.frontier_decay}
          </div>
        </>
      )}
    </div>
  );
}
