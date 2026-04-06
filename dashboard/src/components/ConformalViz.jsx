/* ==============================================================================
   CONFORMAL VISUALIZATION — SET SIZE DISTRIBUTION + COVERAGE
   ============================================================================== */

import React from 'react';
import { Line, Bar } from 'react-chartjs-2';
import { colors } from '../utils/theme';
import { lineDefaults, createTimeSeriesOptions, gridStyle } from '../utils/chartConfig';

export default function ConformalViz({ setSizeHistory, uncertaintyHistory }) {
  // Set size time series
  const timeData = {
    labels: uncertaintyHistory.slice(-80).map((_, i) => i),
    datasets: [
      {
        label: 'Avg Set Size',
        data: uncertaintyHistory.slice(-80),
        borderColor: colors.cyan,
        backgroundColor: `${colors.cyan}10`,
        ...lineDefaults,
      },
    ],
  };

  const timeOpts = {
    ...createTimeSeriesOptions('Set Size', 2.0),
    plugins: {
      ...createTimeSeriesOptions().plugins,
      annotation: {
        annotations: {
          warning: { type: 'line', yMin: 1.1, yMax: 1.1, borderColor: `${colors.amber}66`, borderWidth: 1, borderDash: [4, 4] },
          critical: { type: 'line', yMin: 1.5, yMax: 1.5, borderColor: `${colors.red}66`, borderWidth: 1, borderDash: [4, 4] },
        },
      },
    },
  };

  // Distribution histogram (mock from history)
  const buckets = [0, 0, 0];
  for (const s of uncertaintyHistory) {
    if (s <= 1.1) buckets[0]++;
    else if (s <= 1.5) buckets[1]++;
    else buckets[2]++;
  }

  const distData = {
    labels: ['1.0-1.1', '1.1-1.5', '1.5+'],
    datasets: [{
      data: buckets,
      backgroundColor: [colors.green + '55', colors.amber + '55', colors.red + '55'],
      borderColor: [colors.green, colors.amber, colors.red],
      borderWidth: 1,
      borderRadius: 4,
    }],
  };

  const distOpts = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { grid: { display: false } },
      y: { grid: gridStyle, beginAtZero: true },
    },
  };

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="glass-panel-header">
        <span>Conformal Prediction</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
          {uncertaintyHistory.length} evals
        </span>
      </div>
      <div className="glass-panel-body" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {/* Time series */}
        <div style={{ flex: 1, minHeight: 0 }}>
          {uncertaintyHistory.length > 1 ? (
            <Line data={timeData} options={timeOpts} />
          ) : (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 11 }}>
              Awaiting conformal data...
            </div>
          )}
        </div>

        {/* Distribution */}
        {uncertaintyHistory.length > 0 && (
          <div style={{ height: 80 }}>
            <Bar data={distData} options={distOpts} />
          </div>
        )}
      </div>
    </div>
  );
}
