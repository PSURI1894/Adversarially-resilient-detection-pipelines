/* ==============================================================================
   MODEL PERFORMANCE — F1, FDR, AUC TIME SERIES CHARTS
   ============================================================================== */

import React from 'react';
import { Line } from 'react-chartjs-2';
import { colors } from '../utils/theme';
import { lineDefaults, createTimeSeriesOptions } from '../utils/chartConfig';

export default function ModelPerformance({ metrics }) {
  if (!metrics) {
    return (
      <div className="glass-panel" style={{ height: '100%' }}>
        <div className="glass-panel-header"><span>Model Performance</span></div>
        <div className="glass-panel-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 12, height: 150 }}>
          Waiting for metrics...
        </div>
      </div>
    );
  }

  const latency = metrics.latency_history || [];
  const severity = metrics.severity_history || [];

  const latencyData = {
    labels: latency.slice(-100).map((_, i) => i),
    datasets: [{
      label: 'Latency (ms)',
      data: latency.slice(-100),
      borderColor: colors.cyan,
      backgroundColor: `${colors.cyan}10`,
      ...lineDefaults,
    }],
  };

  const severityData = {
    labels: severity.slice(-100).map((_, i) => i),
    datasets: [{
      label: 'Severity',
      data: severity.slice(-100),
      borderColor: colors.red,
      backgroundColor: `${colors.red}10`,
      ...lineDefaults,
    }],
  };

  // Metric cards from latest f1/fdr/auc
  const latest = (arr) => arr?.length ? arr[arr.length - 1] : null;
  const f1  = latest(metrics.f1_history);
  const fdr = latest(metrics.fdr_history);
  const auc = latest(metrics.auc_history);

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="glass-panel-header">
        <span>Model Performance</span>
      </div>
      <div className="glass-panel-body" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Metric cards */}
        <div style={{ display: 'flex', gap: 8 }}>
          {[
            { label: 'F1', value: f1?.value ?? metrics.severity?.toFixed?.(2) ?? '--', color: colors.green },
            { label: 'FDR', value: fdr?.value?.toFixed?.(3) ?? '--', color: colors.amber },
            { label: 'AUC', value: auc?.value?.toFixed?.(3) ?? '--', color: colors.cyan },
          ].map(({ label, value, color }) => (
            <div key={label} style={{
              flex: 1, textAlign: 'center', padding: '8px 0',
              borderRadius: 8, background: `${color}08`, border: `1px solid ${color}22`,
            }}>
              <div className="metric-sm" style={{ color }}>{typeof value === 'number' ? value.toFixed(3) : value}</div>
              <div className="metric-label">{label}</div>
            </div>
          ))}
        </div>

        {/* Latency chart */}
        <div style={{ flex: 1, minHeight: 0 }}>
          {latency.length > 1 ? (
            <Line data={latencyData} options={createTimeSeriesOptions('Latency (ms)')} />
          ) : (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 11 }}>
              Latency data loading...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
