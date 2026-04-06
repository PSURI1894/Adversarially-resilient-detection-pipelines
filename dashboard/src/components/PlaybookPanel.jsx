/* ==============================================================================
   PLAYBOOK PANEL — CURRENT SOC PLAYBOOK WITH ACTION DISPLAY
   ============================================================================== */

import React from 'react';
import { stateColors, colors } from '../utils/theme';

const PLAYBOOKS = {
  STABLE: {
    action: 'Auto-block',
    description: 'Low uncertainty. Automated blocking of confirmed threats. No analyst intervention required.',
    steps: ['Automated threat blocking active', 'Conformal sets tight (size ~1)', 'Normal alert processing'],
    icon: '\u2713',
  },
  SUSPICIOUS: {
    action: 'Analyst Review',
    description: 'Elevated uncertainty detected. Flagging ambiguous alerts for manual analyst review.',
    steps: ['Route uncertain alerts to analyst queue', 'Increase logging verbosity', 'Monitor drift detectors'],
    icon: '\u26A0',
  },
  EVASION_LOCKED: {
    action: 'Throttle + Honeypot',
    description: 'Alert debt exceeds analyst capacity. Deploying deception and throttling ingress.',
    steps: ['Activate honeypot network segments', 'Throttle non-critical traffic', 'Trigger adaptive retraining', 'Notify SOC lead'],
    icon: '\u26D4',
  },
  FAILURE: {
    action: 'Fail-safe Shutdown',
    description: 'Critical uncertainty levels. Model reliability compromised. Engaging fail-safe protocols.',
    steps: ['Switch to allowlist-only mode', 'Block all unverified traffic', 'Emergency model rollback', 'Escalate to incident commander'],
    icon: '\u2716',
  },
};

export default function PlaybookPanel({ state, alertDebt }) {
  const playbook = PLAYBOOKS[state] || PLAYBOOKS.STABLE;
  const color = stateColors[state] || colors.green;

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="glass-panel-header">
        <span>SOC Playbook</span>
        <span className={`badge ${state === 'FAILURE' || state === 'EVASION_LOCKED' ? 'badge-critical' : state === 'SUSPICIOUS' ? 'badge-warning' : 'badge-stable'}`}>
          {playbook.action}
        </span>
      </div>
      <div className="glass-panel-body" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Action header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '8px 12px', borderRadius: 8,
          background: `${color}12`, border: `1px solid ${color}28`,
        }}>
          <span style={{ fontSize: 20 }}>{playbook.icon}</span>
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13, color }}>
              {playbook.action.toUpperCase()}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
              {playbook.description}
            </div>
          </div>
        </div>

        {/* Alert debt */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '6px 10px', borderRadius: 6,
          background: 'rgba(255,255,255,0.02)',
        }}>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Alert Debt</span>
          <span className="metric-sm" style={{
            color: alertDebt > 50 ? colors.red : alertDebt > 20 ? colors.amber : colors.green,
          }}>
            {alertDebt.toFixed(0)}
          </span>
        </div>

        {/* Steps */}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Active Procedures
          </div>
          {playbook.steps.map((step, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'flex-start', gap: 8,
              padding: '4px 0', fontSize: 11, color: 'var(--text-secondary)',
            }}>
              <span style={{
                color, fontFamily: 'var(--font-mono)', fontSize: 10,
                minWidth: 16, textAlign: 'center',
              }}>
                {String(i + 1).padStart(2, '0')}
              </span>
              <span>{step}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
