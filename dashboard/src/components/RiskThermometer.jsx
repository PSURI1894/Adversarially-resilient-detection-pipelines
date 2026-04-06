/* ==============================================================================
   RISK THERMOMETER — ANIMATED FSM STATE DISPLAY
   ============================================================================== */

import React from 'react';
import { stateColors, severityColor } from '../utils/theme';

const STATE_LABELS = {
  STABLE: 'STABLE',
  SUSPICIOUS: 'SUSPICIOUS',
  EVASION_LOCKED: 'EVASION LOCKED',
  FAILURE: 'FAILURE',
};

const STATE_ICONS = {
  STABLE: '\u2713',
  SUSPICIOUS: '\u26A0',
  EVASION_LOCKED: '\u26D4',
  FAILURE: '\u2716',
};

export default function RiskThermometer({ state, severity }) {
  const color = stateColors[state] || '#94a3b8';
  const sevColor = severityColor(severity);
  const pct = Math.min(severity, 100);

  return (
    <div className="glass-panel" style={{ height: '100%' }}>
      <div className="glass-panel-header">
        <span>Risk Thermostat</span>
        <span className={`badge ${severity > 60 ? 'badge-critical' : severity > 30 ? 'badge-warning' : 'badge-stable'}`}>
          {STATE_LABELS[state] || state}
        </span>
      </div>
      <div className="glass-panel-body" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
        {/* Severity arc */}
        <div style={{ position: 'relative', width: 130, height: 80 }}>
          <svg viewBox="0 0 130 80" style={{ width: '100%', height: '100%' }}>
            {/* Background arc */}
            <path
              d="M 10 75 A 55 55 0 0 1 120 75"
              fill="none"
              stroke="rgba(255,255,255,0.06)"
              strokeWidth="10"
              strokeLinecap="round"
            />
            {/* Severity arc */}
            <path
              d="M 10 75 A 55 55 0 0 1 120 75"
              fill="none"
              stroke={sevColor}
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={`${pct * 1.72} 172`}
              style={{ filter: `drop-shadow(0 0 6px ${sevColor})`, transition: 'stroke-dasharray 0.6s ease' }}
            />
          </svg>
          <div style={{
            position: 'absolute', bottom: 0, left: '50%', transform: 'translateX(-50%)',
            textAlign: 'center',
          }}>
            <div className="metric-value" style={{ color: sevColor, fontSize: 32 }}>
              {severity.toFixed(0)}
            </div>
          </div>
        </div>

        <div className="metric-label">SEVERITY SCORE (0-100)</div>

        {/* State indicator */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 14px', borderRadius: 8,
          background: `${color}15`, border: `1px solid ${color}33`,
        }}>
          <span style={{ fontSize: 16 }}>{STATE_ICONS[state]}</span>
          <span style={{
            fontFamily: 'var(--font-mono)', fontWeight: 700,
            fontSize: 13, color, letterSpacing: '0.06em',
          }}>
            {STATE_LABELS[state] || state}
          </span>
        </div>
      </div>
    </div>
  );
}
