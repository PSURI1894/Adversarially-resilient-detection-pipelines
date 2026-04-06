/* ==============================================================================
   HEADER — SOC BRANDING, STATUS INDICATORS, LIVE CLOCK
   ============================================================================== */

import React, { useState, useEffect } from 'react';
import { stateColors } from '../utils/theme';

export default function Header({ socState, severity, isConnected, totalAlerts }) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const stateColor = stateColors[socState] || '#94a3b8';

  return (
    <header className="glass-panel" style={{ padding: '10px 20px' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        {/* Left: Branding */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: `linear-gradient(135deg, #00f0ff, #a855f7)`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 800, fontSize: 16, color: '#0a0e27',
          }}>
            SC
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: '0.02em' }}>
              SOC COMMAND CENTER
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
              ADVERSARIALLY RESILIENT DETECTION PIPELINE
            </div>
          </div>
        </div>

        {/* Center: State banner */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              padding: '4px 16px', borderRadius: 9999,
              background: `${stateColor}18`, border: `1px solid ${stateColor}44`,
            }}>
              <span className={`status-dot ${severity > 60 ? 'critical pulse' : severity > 30 ? 'warning' : 'stable'}`}
                    style={{ background: stateColor, boxShadow: `0 0 6px ${stateColor}` }} />
              <span style={{
                fontFamily: 'var(--font-mono)', fontWeight: 700,
                fontSize: 12, color: stateColor, letterSpacing: '0.08em',
              }}>
                {socState.replace('_', ' ')}
              </span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
              THREAT LEVEL
            </div>
          </div>

          <div style={{ textAlign: 'center' }}>
            <div className="metric-sm" style={{ color: severity > 60 ? 'var(--red)' : severity > 30 ? 'var(--amber)' : 'var(--green)' }}>
              {severity.toFixed(1)}
            </div>
            <div className="metric-label">SEVERITY</div>
          </div>

          <div style={{ textAlign: 'center' }}>
            <div className="metric-sm" style={{ color: 'var(--cyan)' }}>
              {totalAlerts}
            </div>
            <div className="metric-label">ALERTS</div>
          </div>
        </div>

        {/* Right: Clock & Connection */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 700,
              color: 'var(--cyan)', letterSpacing: '0.05em',
            }}>
              {time.toLocaleTimeString('en-GB', { hour12: false })}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              {time.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase()}
            </div>
          </div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '4px 10px', borderRadius: 6,
            background: isConnected ? 'var(--green-dim)' : 'var(--red-dim)',
            border: `1px solid ${isConnected ? 'var(--green)' : 'var(--red)'}33`,
          }}>
            <span className="status-dot" style={{
              background: isConnected ? 'var(--green)' : 'var(--red)',
              boxShadow: `0 0 6px ${isConnected ? 'var(--green)' : 'var(--red)'}`,
              width: 6, height: 6,
            }} />
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600,
              color: isConnected ? 'var(--green)' : 'var(--red)',
            }}>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
