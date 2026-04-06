/* ==============================================================================
   UNCERTAINTY GAUGE — RADIAL GAUGE + SPARKLINE
   ============================================================================== */

import React from 'react';
import { Line } from 'react-chartjs-2';
import { uncertaintyColor, colors } from '../utils/theme';
import { lineDefaults } from '../utils/chartConfig';

export default function UncertaintyGauge({ avgSetSize, history }) {
  const color = uncertaintyColor(avgSetSize);
  const pct = Math.min((avgSetSize - 1.0) / 1.0, 1.0) * 100; // 1.0→0%, 2.0→100%

  const sparkData = {
    labels: history.slice(-60).map((_, i) => i),
    datasets: [{
      data: history.slice(-60),
      borderColor: color,
      backgroundColor: `${color}15`,
      ...lineDefaults,
      borderWidth: 1.5,
    }],
  };

  const sparkOpts = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { display: false },
      y: { display: false, suggestedMin: 1.0, suggestedMax: 2.0 },
    },
    plugins: { tooltip: { enabled: false } },
    animation: false,
  };

  return (
    <div className="glass-panel" style={{ height: '100%' }}>
      <div className="glass-panel-header">
        <span>Conformal Uncertainty</span>
        <span className={`badge ${avgSetSize > 1.5 ? 'badge-critical' : avgSetSize > 1.1 ? 'badge-warning' : 'badge-stable'}`}>
          {avgSetSize > 1.5 ? 'HIGH' : avgSetSize > 1.1 ? 'ELEVATED' : 'NORMAL'}
        </span>
      </div>
      <div className="glass-panel-body" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
        {/* Radial gauge */}
        <div style={{ position: 'relative', width: 110, height: 110 }}>
          <svg viewBox="0 0 110 110" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
            <circle cx="55" cy="55" r="46" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
            <circle cx="55" cy="55" r="46" fill="none"
              stroke={color}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${pct * 2.89} 289`}
              style={{ filter: `drop-shadow(0 0 6px ${color})`, transition: 'stroke-dasharray 0.6s ease' }}
            />
          </svg>
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          }}>
            <div className="metric-value" style={{ color, fontSize: 24 }}>
              {avgSetSize.toFixed(2)}
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>AVG SET SIZE</div>
          </div>
        </div>

        {/* Sparkline */}
        <div style={{ width: '100%', height: 40 }}>
          {history.length > 1 && <Line data={sparkData} options={sparkOpts} />}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: 10, color: 'var(--text-muted)' }}>
          <span>Warn: 1.10</span>
          <span>Crit: 1.50</span>
        </div>
      </div>
    </div>
  );
}
