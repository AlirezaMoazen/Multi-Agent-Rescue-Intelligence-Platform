import { useRef, useEffect, useMemo } from 'react';

const AGENT_COLORS = [
  '#22d3ee', // cyan
  '#a78bfa', // purple
  '#fbbf24', // amber
  '#34d399', // green
  '#f472b6', // pink
  '#fb923c', // orange
];

/**
 * GridCanvas — renders the rescue grid on a <canvas> element.
 * Walls = dark, empty = slightly lighter, targets = red glow, agents = colored dots.
 */
export default function GridCanvas({ grid, agents, rescued, trails }) {
  const canvasRef = useRef(null);

  // Build a lookup set for rescued positions
  const rescuedSet = useMemo(() => {
    const s = new Set();
    (rescued || []).forEach(r => s.add(`${r.x},${r.y}`));
    return s;
  }, [rescued]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !grid) return;

    const ctx = canvas.getContext('2d');
    const { width: gw, height: gh, obstacles, targets } = grid;

    // Calculate cell size to fit in the container
    const container = canvas.parentElement;
    const maxW = container.clientWidth - 20;
    const maxH = container.clientHeight - 20;
    const cellSize = Math.max(4, Math.min(Math.floor(maxW / gw), Math.floor(maxH / gh), 32));

    canvas.width  = gw * cellSize;
    canvas.height = gh * cellSize;

    // Build obstacle set
    const obstacleSet = new Set();
    obstacles.forEach(o => obstacleSet.add(`${o.x},${o.y}`));

    // Build target set
    const targetSet = new Set();
    targets.forEach(t => targetSet.add(`${t.x},${t.y}`));

    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw cells
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const key = `${x},${y}`;
        const px = x * cellSize;
        const py = y * cellSize;

        if (obstacleSet.has(key)) {
          ctx.fillStyle = '#1e293b';
          ctx.fillRect(px, py, cellSize, cellSize);
        } else if (targetSet.has(key) && !rescuedSet.has(key)) {
          // Active target — glowing red
          ctx.fillStyle = '#450a0a';
          ctx.fillRect(px, py, cellSize, cellSize);
          ctx.fillStyle = '#f87171';
          const pad = Math.max(2, cellSize * 0.2);
          ctx.beginPath();
          ctx.arc(px + cellSize / 2, py + cellSize / 2, cellSize / 2 - pad, 0, Math.PI * 2);
          ctx.fill();
        } else if (rescuedSet.has(key)) {
          // Rescued target — dim green
          ctx.fillStyle = '#052e16';
          ctx.fillRect(px, py, cellSize, cellSize);
          ctx.fillStyle = 'rgba(52,211,153,0.4)';
          ctx.fillRect(px + 2, py + 2, cellSize - 4, cellSize - 4);
        } else {
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(px, py, cellSize, cellSize);
        }

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.strokeRect(px, py, cellSize, cellSize);
      }
    }

    // Draw agent history trails
    const agentTrails = trails || {};
    Object.keys(agentTrails).forEach(agentIdKey => {
      const idx = parseInt(agentIdKey, 10);
      const trail = agentTrails[idx] || [];
      if (trail.length < 2) return;

      const color = AGENT_COLORS[idx % AGENT_COLORS.length];

      ctx.beginPath();
      const startX = trail[0].x * cellSize + cellSize / 2;
      const startY = trail[0].y * cellSize + cellSize / 2;
      ctx.moveTo(startX, startY);

      for (let i = 1; i < trail.length; i++) {
        const tx = trail[i].x * cellSize + cellSize / 2;
        const ty = trail[i].y * cellSize + cellSize / 2;
        ctx.lineTo(tx, ty);
      }

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, cellSize * 0.15);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.globalAlpha = 0.5; // semi-transparent trail
      ctx.stroke();
      ctx.restore();
    });

    // Draw agents
    (agents || []).forEach((agent, idx) => {
      const color = AGENT_COLORS[idx % AGENT_COLORS.length];
      const cx = agent.x * cellSize + cellSize / 2;
      const cy = agent.y * cellSize + cellSize / 2;
      const r  = Math.max(3, cellSize * 0.35);

      // Glow
      ctx.save();
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Label
      if (cellSize >= 16) {
        ctx.fillStyle = '#0a0e1a';
        ctx.font = `bold ${Math.max(8, cellSize * 0.4)}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(idx + 1, cx, cy + 1);
      }
    });

  }, [grid, agents, rescuedSet, trails]);

  if (!grid) {
    return (
      <div className="grid-placeholder">
        <span>🗺️</span>
        Click <strong>Start Simulation</strong> to generate a rescue grid
      </div>
    );
  }

  return <canvas ref={canvasRef} />;
}
