/* ==============================================================================
   ATTACK SIMULATOR — INTERACTIVE ADVERSARIAL ATTACK CONTROLS
   ============================================================================== */

import React, { useState, useEffect, useRef } from 'react';
import { colors } from '../utils/theme';
import { api } from '../services/api';

const ATTACK_TYPES = [
  { value: 'pgd',                 label: 'PGD (White-box)' },
  { value: 'carlini_wagner',      label: 'C&W L2' },
  { value: 'auto_attack',         label: 'AutoAttack' },
  { value: 'boundary',            label: 'Boundary (Black-box)' },
  { value: 'feature_constrained', label: 'Feature Constrained' },
  { value: 'slow_drip',           label: 'Slow Drip' },
  { value: 'label_flip',          label: 'Label Flip Poison' },
  { value: 'calibration',         label: 'Calibration Poison' },
];

function fmt(sec) {
  if (sec < 60) return `${sec}s`;
  return `${Math.floor(sec / 60)}m ${sec % 60}s`;
}

export default function AttackSimulator({ wsEvents }) {
  const [attackType, setAttackType]   = useState('pgd');
  const [epsilon, setEpsilon]         = useState(0.1);
  const [nSamples, setNSamples]       = useState(1000);
  const [busy, setBusy]               = useState(false);
  const [activeAttack, setActiveAttack] = useState(null);
  const [countdown, setCountdown]     = useState(null);
  const [history, setHistory]         = useState([]);   // completed attack log
  const countdownRef  = useRef(null);
  const startedAtRef  = useRef(null);  // wall-clock when attack started

  // ── countdown helpers ──────────────────────────────────────
  const startCountdown = (seconds) => {
    if (countdownRef.current) clearInterval(countdownRef.current);
    startedAtRef.current = Date.now() - ((activeAttack?.duration_seconds ?? seconds) - seconds) * 1000;
    setCountdown(seconds);
    countdownRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) { clearInterval(countdownRef.current); return 0; }
        return prev - 1;
      });
    }, 1000);
  };

  const clearCountdown = () => {
    if (countdownRef.current) clearInterval(countdownRef.current);
    setCountdown(null);
  };

  // ── finish an attack and push to history ───────────────────
  const finishAttack = (attack, how) => {
    const elapsed = startedAtRef.current
      ? Math.round((Date.now() - startedAtRef.current) / 1000)
      : attack?.duration_seconds ?? 0;

    setHistory((prev) => [
      {
        sim_id:       attack?.sim_id,
        attack_type:  attack?.attack_type ?? attackType,
        epsilon:      attack?.epsilon ?? epsilon,
        duration:     elapsed,
        ended:        how,          // 'completed' | 'stopped'
        at:           new Date().toLocaleTimeString('en-GB', { hour12: false }),
      },
      ...prev.slice(0, 4),          // keep last 5
    ]);
    setActiveAttack(null);
    clearCountdown();
    startedAtRef.current = null;
  };

  // ── WebSocket events ───────────────────────────────────────
  useEffect(() => {
    if (!wsEvents) return;
    if (wsEvents.type === 'simulation_started') {
      startedAtRef.current = Date.now();
      setActiveAttack(wsEvents.data);
      if (wsEvents.data.duration_seconds) startCountdown(wsEvents.data.duration_seconds);
    } else if (wsEvents.type === 'simulation_stopped') {
      finishAttack(activeAttack, wsEvents.data.status === 'completed' ? 'completed' : 'stopped');
    }
  }, [wsEvents]);

  // ── restore state on page load ─────────────────────────────
  useEffect(() => {
    api.simulationStatus?.().then((s) => {
      if (s?.active) {
        startedAtRef.current = Date.now() - (s.duration_seconds - s.time_remaining) * 1000;
        setActiveAttack({ sim_id: s.sim_id, epsilon: s.epsilon, duration_seconds: s.duration_seconds });
        if (s.time_remaining > 0) startCountdown(s.time_remaining);
      }
    }).catch(() => {});
  }, []);

  // ── poll while active (self-corrects if WS event missed) ──
  useEffect(() => {
    if (!activeAttack) return;
    const id = setInterval(() => {
      api.simulationStatus?.().then((s) => {
        if (!s?.active) finishAttack(activeAttack, 'completed');
      }).catch(() => {});
    }, 2000);
    return () => clearInterval(id);
  }, [activeAttack]);

  // ── launch / stop handlers ─────────────────────────────────
  const handleLaunch = async () => {
    setBusy(true);
    clearCountdown();
    try {
      const res = await api.simulate({ attack_type: attackType, epsilon, n_samples: nSamples });
      startedAtRef.current = Date.now();
      setActiveAttack(res);
      if (res.duration_seconds) startCountdown(res.duration_seconds);
    } catch (err) {
      setHistory((prev) => [{ error: err.message, at: new Date().toLocaleTimeString() }, ...prev.slice(0, 4)]);
    }
    setBusy(false);
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      await api.stopSimulation();
      finishAttack(activeAttack, 'stopped');
    } catch (err) { /* ignore */ }
    setBusy(false);
  };

  const isRunning = !!activeAttack;

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="glass-panel-header">
        <span>Attack Simulator</span>
        {isRunning && <span className="badge badge-warning pulse">ACTIVE</span>}
      </div>
      <div className="glass-panel-body" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto' }}>

        {/* Controls */}
        <div>
          <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>ATTACK TYPE</label>
          <select value={attackType} onChange={(e) => setAttackType(e.target.value)}
            disabled={isRunning} style={{ width: '100%', opacity: isRunning ? 0.5 : 1 }}>
            {ATTACK_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
        </div>

        <div>
          <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
            EPSILON: <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{epsilon.toFixed(2)}</span>
          </label>
          <input type="range" className="input-range" min="0" max="0.5" step="0.01"
            value={epsilon} disabled={isRunning}
            onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            style={{ opacity: isRunning ? 0.5 : 1 }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
            <span>0.00</span><span>0.25</span><span>0.50</span>
          </div>
        </div>

        <div>
          <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>SAMPLES</label>
          <input type="number" value={nSamples}
            onChange={(e) => setNSamples(parseInt(e.target.value) || 100)}
            min={100} max={50000} step={100} disabled={isRunning}
            style={{ width: '100%', opacity: isRunning ? 0.5 : 1 }} />
        </div>

        {/* Button */}
        {isRunning ? (
          <button className="btn" onClick={handleStop} disabled={busy} style={{
            width: '100%', justifyContent: 'center', padding: '10px 0',
            fontSize: 12, fontWeight: 700, letterSpacing: '0.06em',
            background: 'var(--red-dim)', border: '1px solid var(--red)',
            color: 'var(--red)', opacity: busy ? 0.5 : 1,
          }}>
            {busy ? 'STOPPING...' : '⬛ STOP ATTACK'}
          </button>
        ) : (
          <button className="btn btn-danger" onClick={handleLaunch} disabled={busy} style={{
            width: '100%', justifyContent: 'center', padding: '10px 0',
            fontSize: 12, fontWeight: 700, letterSpacing: '0.06em',
            opacity: busy ? 0.5 : 1,
          }}>
            {busy ? 'LAUNCHING...' : 'LAUNCH ATTACK'}
          </button>
        )}

        {/* Active attack card with countdown */}
        {isRunning && (
          <div style={{
            padding: 10, borderRadius: 8, fontSize: 10,
            fontFamily: 'var(--font-mono)',
            background: 'rgba(255,80,80,0.08)',
            border: '1px solid rgba(255,80,80,0.3)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: colors.red }}>▶ Attack in progress</span>
              {countdown !== null && (
                <span style={{ color: countdown <= 5 ? colors.red : 'var(--amber)', fontWeight: 700, fontSize: 12 }}>
                  {fmt(countdown)} left
                </span>
              )}
            </div>
            <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>
              {activeAttack.attack_type || attackType} | eps={activeAttack.epsilon ?? epsilon} | id={activeAttack.sim_id}
            </div>
            {countdown !== null && activeAttack.duration_seconds && (
              <div style={{ marginTop: 6, height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.1)' }}>
                <div style={{
                  height: '100%', borderRadius: 2,
                  background: countdown <= 5 ? colors.red : colors.amber,
                  width: `${(countdown / activeAttack.duration_seconds) * 100}%`,
                  transition: 'width 1s linear',
                }} />
              </div>
            )}
          </div>
        )}

        {/* Attack history log */}
        {history.length > 0 && (
          <div>
            <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, letterSpacing: '0.06em' }}>
              ATTACK LOG
            </div>
            {history.map((h, i) => h.error ? (
              <div key={i} style={{ fontSize: 10, color: colors.red, fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
                ✕ {h.error}
              </div>
            ) : (
              <div key={i} style={{
                padding: '6px 8px', borderRadius: 6, marginBottom: 4,
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.07)',
                fontFamily: 'var(--font-mono)', fontSize: 10,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: h.ended === 'completed' ? colors.green : colors.amber }}>
                    {h.ended === 'completed' ? '✓ Completed' : '⬛ Stopped'}
                  </span>
                  <span style={{ color: 'var(--text-muted)' }}>{h.at}</span>
                </div>
                <div style={{ color: 'var(--text-secondary)', marginTop: 3 }}>
                  {h.attack_type} | ε={h.epsilon} | <span style={{ color: 'var(--cyan)' }}>{fmt(h.duration)}</span>
                </div>
                <div style={{ color: 'var(--text-muted)', marginTop: 1, fontSize: 9 }}>
                  id={h.sim_id}
                </div>
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
