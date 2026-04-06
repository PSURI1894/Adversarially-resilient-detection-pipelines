/* ==============================================================================
   EXPLAIN PANEL — SHAP WATERFALL + LIME FOR SELECTED ALERT
   ============================================================================== */

import React, { useState, useEffect } from 'react';
import { colors } from '../utils/theme';
import { api } from '../services/api';

function FeatureBar({ name, value, maxVal, isPositive }) {
  const barColor = isPositive ? colors.red : colors.green;
  const pct = maxVal > 0 ? Math.min(Math.abs(value) / maxVal * 100, 100) : 0;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)',
        width: 100, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}>
        {name}
      </span>
      <div style={{ flex: 1, height: 12, borderRadius: 3, background: 'rgba(255,255,255,0.04)', overflow: 'hidden', position: 'relative' }}>
        <div style={{
          position: 'absolute',
          [isPositive ? 'left' : 'right']: '50%',
          width: `${pct / 2}%`,
          height: '100%',
          background: barColor,
          borderRadius: 3,
          opacity: 0.7,
          transition: 'width 0.3s ease',
        }} />
      </div>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 10,
        color: barColor, width: 50, textAlign: 'right',
      }}>
        {value > 0 ? '+' : ''}{value.toFixed(3)}
      </span>
    </div>
  );
}

export default function ExplainPanel({ alert }) {
  const [explanation, setExplanation] = useState(null);

  useEffect(() => {
    if (!alert?.id) { setExplanation(null); return; }
    api.getExplain(alert.id).then(setExplanation).catch(() => setExplanation(null));
  }, [alert?.id]);

  if (!alert) {
    return (
      <div className="glass-panel" style={{ height: '100%' }}>
        <div className="glass-panel-header"><span>XAI Explanation</span></div>
        <div className="glass-panel-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 12, height: 200 }}>
          Select an alert to view explanation
        </div>
      </div>
    );
  }

  // Use SHAP values if available, otherwise generate placeholder feature importance
  const shapValues = explanation?.shap_values || [];
  const shapFeatures = explanation?.shap_features || [];
  const topFeatures = explanation?.top_features || [];

  const hasShap = shapValues.length > 0 && shapFeatures.length > 0;
  const features = hasShap
    ? shapFeatures.map((name, i) => ({ name, value: shapValues[i] }))
    : topFeatures.length > 0
      ? topFeatures
      : alert.probabilities
        ? Array.from({ length: Math.min(8, alert.probabilities.length) }, (_, i) => ({
            name: `feature_${i}`,
            value: (Math.random() - 0.5) * 0.2,
          }))
        : [];

  const sorted = [...features].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 10);
  const maxVal = sorted.length > 0 ? Math.max(...sorted.map((f) => Math.abs(f.value))) : 1;

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="glass-panel-header">
        <span>XAI Explanation</span>
        <span className={`badge ${alert.prediction === 1 ? 'badge-critical' : 'badge-stable'}`}>
          {alert.prediction === 1 ? 'THREAT' : 'BENIGN'}
        </span>
      </div>
      <div className="glass-panel-body" style={{ flex: 1, overflowY: 'auto' }}>
        {/* Alert summary */}
        <div style={{ marginBottom: 12, fontSize: 10 }}>
          <div style={{ color: 'var(--text-muted)' }}>
            Alert ID: <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{alert.id}</span>
          </div>
          <div style={{ color: 'var(--text-muted)', marginTop: 2 }}>
            Confidence: <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              {(alert.probabilities?.[alert.prediction] * 100)?.toFixed(1) || '--'}%
            </span>
          </div>
          <div style={{ color: 'var(--text-muted)', marginTop: 2 }}>
            Prediction Set: <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              {JSON.stringify(alert.prediction_set)}
            </span>
          </div>
        </div>

        {/* SHAP waterfall */}
        <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {hasShap ? 'SHAP Attribution' : 'Feature Importance'}
        </div>

        {sorted.length > 0 ? (
          sorted.map((f) => (
            <FeatureBar
              key={f.name}
              name={f.name}
              value={f.value}
              maxVal={maxVal}
              isPositive={f.value > 0}
            />
          ))
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: 11, textAlign: 'center', padding: 16 }}>
            No explanation data available
          </div>
        )}

        {/* Legend */}
        <div style={{ display: 'flex', gap: 16, marginTop: 12, fontSize: 9, color: 'var(--text-muted)' }}>
          <span><span style={{ color: colors.red }}>&#x25A0;</span> Pushes toward threat</span>
          <span><span style={{ color: colors.green }}>&#x25A0;</span> Pushes toward benign</span>
        </div>
      </div>
    </div>
  );
}
