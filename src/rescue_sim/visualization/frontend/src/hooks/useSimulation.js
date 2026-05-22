import { useRef, useState, useCallback, useEffect } from 'react';

/**
 * Resolve the WebSocket URL dynamically so it works both in dev
 * (localhost:5173 proxied to localhost:8000) and production / Docker
 * (same host, port 8000).
 */
function getWsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  // In dev mode, Vite runs on 5173 and the backend on 8000
  const host = window.location.hostname;
  const port = import.meta.env.DEV ? '8000' : window.location.port || '8000';
  return `${proto}://${host}:${port}/ws/simulation`;
}

const RECONNECT_DELAY = 2000;
const MAX_RECONNECT = 10;

/**
 * Custom hook for managing the WebSocket connection to the simulation backend.
 * Handles connect / disconnect / auto-reconnect, sending commands, and dispatching incoming messages.
 */
export default function useSimulation() {
  const wsRef = useRef(null);
  const reconnectCount = useRef(0);
  const reconnectTimer = useRef(null);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('idle');        // idle | running | stopped | complete
  const [grid, setGrid] = useState(null);              // current grid layout
  const [agents, setAgents] = useState([]);             // current agent positions
  const [episode, setEpisode] = useState(0);
  const [step, setStep] = useState(0);
  const [rescued, setRescued] = useState([]);
  const [activeTargets, setActiveTargets] = useState(0);
  const [trails, setTrails] = useState({});             // history of agent coordinates
  const [error, setError] = useState(null);            // any backend validation errors
  const [episodeMetrics, setEpisodeMetrics] = useState([]);
  const [successRate, setSuccessRate] = useState(0);
  const [avgSteps, setAvgSteps] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [explorationRate, setExplorationRate] = useState(1.0);

  const connectFn = useRef(null);

  // Connect
  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    // Clean up existing connection
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
    }

    const url = getWsUrl();
    let ws;
    try {
      ws = new WebSocket(url);
    } catch {
      return; // Will retry via reconnect logic
    }

    ws.onopen = () => {
      setConnected(true);
      reconnectCount.current = 0;
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;

      // Auto-reconnect if not intentionally disconnected
      if (reconnectCount.current < MAX_RECONNECT) {
        reconnectCount.current += 1;
        reconnectTimer.current = setTimeout(() => {
          if (connectFn.current) connectFn.current();
        }, RECONNECT_DELAY);
      }
    };

    ws.onerror = () => {
      // onclose will fire after onerror
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case 'episode_start':
          setGrid(msg.grid);
          setAgents(msg.agents);
          setEpisode(msg.episode);
          setStep(0);
          setRescued([]);
          setActiveTargets(msg.grid.targets.length);
          setStatus('running');
          setError(null);
          
          // Initialize trails
          const initialTrails = {};
          (msg.agents || []).forEach(a => {
            initialTrails[a.id] = [{ x: a.x, y: a.y }];
          });
          setTrails(initialTrails);
          break;

        case 'step':
          setAgents(msg.agents);
          setStep(msg.step);
          setRescued(msg.rescued);
          setActiveTargets(msg.active_targets);

          // Append to trails
          setTrails(prev => {
            const next = { ...prev };
            (msg.agents || []).forEach(a => {
              if (!next[a.id]) {
                next[a.id] = [];
              }
              const currentTrail = next[a.id];
              const last = currentTrail[currentTrail.length - 1];
              if (!last || last.x !== a.x || last.y !== a.y) {
                next[a.id] = [...currentTrail, { x: a.x, y: a.y }];
              }
            });
            return next;
          });
          break;

        case 'episode_end':
          setEpisodeMetrics(prev => [...prev, {
            episode: msg.episode,
            steps: msg.steps,
            rescued_count: msg.rescued_count,
            target_count: msg.target_count,
            success: msg.success,
            total_reward: msg.total_reward,
            success_rate: msg.success_rate,
            exploration_rate: msg.exploration_rate,
          }]);
          setSuccessRate(msg.success_rate);
          setAvgSteps(msg.avg_steps);
          setTotalReward(msg.total_reward);
          setExplorationRate(msg.exploration_rate);
          break;

        case 'training_complete':
          setStatus('complete');
          break;

        case 'stopped':
          setStatus('stopped');
          break;

        case 'error':
          setError(msg.message);
          setStatus('stopped');
          break;

        case 'config_ack':
          break;

        default:
          break;
      }
    };

    wsRef.current = ws;
  }, []);

  // Keep a stable ref to connect for the reconnect timer
  connectFn.current = connect;

  // Disconnect
  const disconnect = useCallback(() => {
    reconnectCount.current = MAX_RECONNECT; // prevent auto-reconnect
    clearTimeout(reconnectTimer.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // Send a message
  const send = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  // Start simulation
  const start = useCallback((config) => {
    setEpisodeMetrics([]);
    setSuccessRate(0);
    setAvgSteps(0);
    setTotalReward(0);
    setExplorationRate(1.0);
    setGrid(null);
    setAgents([]);
    setRescued([]);
    setStep(0);
    setEpisode(0);
    setActiveTargets(0);
    setError(null);
    setTrails({});
    setStatus('running');
    if (config) {
      send({ type: 'config', data: config });
    }
    // Small delay to let config apply
    setTimeout(() => send({ type: 'start' }), 100);
  }, [send]);

  // Stop simulation
  const stop = useCallback(() => {
    send({ type: 'stop' });
  }, [send]);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    connected,
    status,
    grid,
    agents,
    episode,
    step,
    rescued,
    activeTargets,
    episodeMetrics,
    successRate,
    avgSteps,
    totalReward,
    explorationRate,
    trails,
    error,
    start,
    stop,
    send,
    connect,
    disconnect,
  };
}
