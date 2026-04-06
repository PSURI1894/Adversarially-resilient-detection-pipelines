/* ==============================================================================
   DRIFT INDICATOR — CONCEPT DRIFT STATUS WITH SPARKLINES
   ============================================================================== */

import React from 'react';
import { Line } from 'react-chartjs-2';
import { colors } from '../utils/theme';
import { lineDefaults } from '../utils/chartConfig';

export default function DriftIndicator({ driftHistory, calibrationDrift }) {
  const isDrifting = calibrationDrift > 0.5;
  const driftColor = calibrationDrift > 0.5 ? colors.red : calibrationDrift > 0.2 ? colors.amber : colors.green;

  const driftValues = driftHistory.map((d) => d.value ?? d);

  const sparkData = {
    labels: driftValues.slice(-40).map((_, i) => i),
    datasets: [{
      data: driftValues.slice(-40),
      borderColor: driftColor,
      backgroundColor: `${driftColor}10`,
      ...lineDefaults,
      borderWidth: 1.5,
    }],
  };

  const sparkOpts = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { display: false },
      y: { display: false, suggestedMin: 0, suggestedMax: 1 },
    },
    plugins: { tooltip: { enabled: false } },
    animation: false,
  };

  return (
    <div className="glass-panel" style={{ height: '100%' }}>
      <div className="glass-panel-header">
        <span>Concept Drift</span>
        <span className={`badge ${isDrifting ? 'badge-critical pulse' : calibrationDrift > 0.2 ? 'badge-warning' : 'badge-stable'}`}>
          {isDrifting ? 'DETECTED' : 'STABLE'}
        </span>
      </div>
      <div className="glass-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {/* Drift score */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div>
            <div className="metric-value" style={{ color: driftColor, fontSize: 28 }}>
              {calibrationDrift.toFixed(3)}
            </div>
            <div className="metric-label">DRIFT SCORE</div>
          </div>

          {/* Progress bar */}
          <div style={{ flex: 1 }}>
            <div style={{
              height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)',
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%', borderRadius: 3,
                width: `${Math.min(calibrationDrift * 100, 100)}%`,
                background: driftColor,
                boxShadow: `0 0 8px ${driftColor}`,
                transition: 'width 0.4s ease',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 9, color: 'var(--text-muted)' }}>
              <span>0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
          </div>
        </div>

        {/* Sparkline */}
        <div style={{ width: '100%', height: 44 }}>
          {driftValues.length > 1 && <Line data={sparkData} options={sparkOpts} />}
        </div>

        {/* Detectors */}
        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
          Consensus: ADWIN + Page-Hinkley + KS + MMD
        </div>
      </div>
    </div>
  );
}
