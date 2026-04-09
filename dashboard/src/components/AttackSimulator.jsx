/* ==============================================================================
   ATTACK SIMULATOR — INTERACTIVE ADVERSARIAL ATTACK CONTROLS
   ============================================================================== */

import React, { useState, useEffect } from 'react';
import { colors } from '../utils/theme';
import { api } from '../services/api';

const ATTACK_TYPES = [
  { value: 'pgd', label: 'PGD (White-box)' },
  { value: 'carlini_wagner', label: 'C&W L2' },
  { value: 'auto_attack', label: 'AutoAttack' },
  { value: 'boundary', label: 'Boundary (Black-box)' },
  { value: 'feature_constrained', label: 'Feature Constrained' },
  { value: 'slow_drip', label: 'Slow Drip' },
  { value: 'label_flip', label: 'Label Flip Poison' },
  { value: 'calibration', label: 'Calibration Poison' },
];

export default function AttackSimulator({ wsEvents }) {
  const [attackType, setAttackType] = useState('pgd');
  const [epsilon, setEpsilon] = useState(0.1);
  const [nSamples, setNSamples] = useState(1000);
  const [busy, setBusy] = useState(false);          // API call in flight
  const [activeAttack, setActiveAttack] = useState(null);  // { sim_id, attack_type, epsilon }
  const [lastResult, setLastResult] = useState(null);

  // Sync with WebSocket events from parent (simulation_started / simulation_stopped)
  useEffect(() => {
    if (!wsEvents) return;
    if (wsEvents.type === 'simulation_started') {
      setActiveAttack(wsEvents.data);
    } else if (wsEvents.type === 'simulation_stopped') {
      setActiveAttack(null);
      setLastResult({ status: 'stopped', sim_id: wsEvents.data.sim_id });
    }
  }, [wsEvents]);

  // Restore state on mount (e.g. page reload while attack is running)
  useEffect(() => {
    api.simulationStatus?.().then((s) => {
      if (s?.active) setActiveAttack({ sim_id: s.sim_id, epsilon: s.epsilon });
    }).catch(() => {});
  }, []);

  const handleLaunch = async () => {
    setBusy(true);
    setLastResult(null);
    try {
      const res = await api.simulate({ attack_type: attackType, epsilon, n_samples: nSamples });
      setActiveAttack(res);
      setLastResult({ status: 'started', ...res });
    } catch (err) {
      setLastResult({ status: 'error', message: err.message });
    }
    setBusy(false);
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      const res = await api.stopSimulation();
      setActiveAttack(null);
      setLastResult({ status: 'stopped', sim_id: res.sim_id });
    } catch (err) {
      setLastResult({ status: 'error', message: err.message });
    }
    setBusy(false);
  };

  const isRunning = !!activeAttack;

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="glass-panel-header">
        <span>Attack Simulator</span>
        {isRunning && <span className="badge badge-warning pulse">ACTIVE</span>}
      </div>
      <div className="glass-panel-body" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>

        {/* Attack type */}
        <div>
          <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
            ATTACK TYPE
          </label>
          <select value={attackType} onChange={(e) => setAttackType(e.target.value)}
            disabled={isRunning}
            style={{ width: '100%', opacity: isRunning ? 0.5 : 1 }}>
            {ATTACK_TYPES.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
        </div>

        {/* Epsilon */}
        <div>
          <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
            EPSILON:{' '}
            <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>
              {epsilon.toFixed(2)}
            </span>
          </label>
          <input
            type="range"
            className="input-range"
            min="0" max="0.5" step="0.01"
            value={epsilon}
            disabled={isRunning}
            onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            style={{ opacity: isRunning ? 0.5 : 1 }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
            <span>0.00</span><span>0.25</span><span>0.50</span>
          </div>
        </div>

        {/* Samples */}
        <div>
          <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
            SAMPLES
          </label>
          <input
            type="number"
            value={nSamples}
            onChange={(e) => setNSamples(parseInt(e.target.value) || 100)}
            min={100} max={50000} step={100}
            disabled={isRunning}
            style={{ width: '100%', opacity: isRunning ? 0.5 : 1 }}
          />
        </div>

        {/* Launch / Stop button */}
        {isRunning ? (
          <button
            className="btn"
            onClick={handleStop}
            disabled={busy}
            style={{
              width: '100%', justifyContent: 'center', padding: '10px 0',
              fontSize: 12, fontWeight: 700, letterSpacing: '0.06em',
              background: 'var(--red-dim)',
              border: '1px solid var(--red)',
              color: 'var(--red)',
              opacity: busy ? 0.5 : 1,
            }}>
            {busy ? 'STOPPING...' : '⬛ STOP ATTACK'}
          </button>
        ) : (
          <button
            className="btn btn-danger"
            onClick={handleLaunch}
            disabled={busy}
            style={{
              width: '100%', justifyContent: 'center', padding: '10px 0',
              fontSize: 12, fontWeight: 700, letterSpacing: '0.06em',
              opacity: busy ? 0.5 : 1,
            }}>
            {busy ? 'LAUNCHING...' : 'LAUNCH ATTACK'}
          </button>
        )}

        {/* Status card */}
        {isRunning && (
          <div style={{
            padding: 10, borderRadius: 8, fontSize: 10,
            fontFamily: 'var(--font-mono)',
            background: 'rgba(255,80,80,0.08)',
            border: '1px solid rgba(255,80,80,0.3)',
          }}>
            <div style={{ color: colors.red }}>▶ Attack in progress</div>
            <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>
              id={activeAttack.sim_id} | {activeAttack.attack_type || attackType} | eps={activeAttack.epsilon ?? epsilon}
            </div>
          </div>
        )}

        {/* Result message (stopped / error) */}
        {!isRunning && lastResult && (
          <div style={{
            padding: 10, borderRadius: 8, fontSize: 10,
            fontFamily: 'var(--font-mono)',
            background: lastResult.status === 'error'
              ? 'var(--red-dim)'
              : lastResult.status === 'stopped'
              ? 'rgba(100,100,120,0.2)'
              : 'var(--cyan-dim)',
            border: `1px solid ${
              lastResult.status === 'error' ? colors.red
              : lastResult.status === 'stopped' ? 'rgba(150,150,170,0.4)'
              : colors.cyan}33`,
          }}>
            {lastResult.status === 'error' && (
              <span style={{ color: colors.red }}>Error: {lastResult.message}</span>
            )}
            {lastResult.status === 'stopped' && (
              <>
                <div style={{ color: 'var(--text-muted)' }}>⬛ Attack stopped</div>
                <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>
                  id={lastResult.sim_id} — severity decaying
                </div>
              </>
            )}
            {lastResult.status === 'started' && (
              <>
                <div style={{ color: colors.cyan }}>Simulation #{lastResult.sim_id} started</div>
                <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>
                  {lastResult.attack_type} | eps={lastResult.epsilon} | n={lastResult.n_samples}
                </div>
              </>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
