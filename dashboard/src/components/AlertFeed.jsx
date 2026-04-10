/* ==============================================================================
   ALERT FEED — LIVE SCROLLING ALERT LIST WITH SEVERITY COLORING
   ============================================================================== */

import React, { useState } from 'react';
import { colors } from '../utils/theme';

function AlertRow({ alert, isSelected, onSelect }) {
  const isHigh = alert.uncertainty === 'HIGH';
  const borderColor = alert.prediction === 1
    ? (isHigh ? colors.red : colors.amber)
    : colors.green;

  const ts = alert.timestamp
    ? new Date(alert.timestamp * 1000).toLocaleTimeString('en-GB', { hour12: false })
    : '--:--:--';

  return (
    <div
      onClick={() => onSelect(alert)}
      style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '8px 12px', borderRadius: 8, cursor: 'pointer',
        borderLeft: `3px solid ${borderColor}`,
        background: isSelected ? 'rgba(0, 240, 255, 0.06)' : 'transparent',
        transition: 'background 0.15s',
      }}
      onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(0, 240, 255, 0.04)'}
      onMouseLeave={(e) => e.currentTarget.style.background = isSelected ? 'rgba(0, 240, 255, 0.06)' : 'transparent'}
    >
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 10,
        color: 'var(--text-muted)', minWidth: 60,
      }}>
        {ts}
      </span>

      <span className={`badge ${alert.prediction === 1 ? (isHigh ? 'badge-critical' : 'badge-warning') : 'badge-stable'}`}>
        {alert.prediction === 1 ? 'THREAT' : 'BENIGN'}
      </span>

      <span className={`badge ${isHigh ? 'badge-critical' : 'badge-info'}`}>
        {isHigh ? 'HIGH' : 'LOW'}
      </span>

      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 10,
        color: 'var(--text-secondary)', marginLeft: 'auto',
      }}>
        set:{alert.prediction_set?.length || 1}
      </span>

      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 10,
        color: 'var(--text-muted)',
      }}>
        {alert.latency_ms?.toFixed(1) || '0.0'}ms
      </span>
    </div>
  );
}

const PAGE_SIZE = 10;

export default function AlertFeed({ alerts, stats, onSelectAlert }) {
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState(null);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  const filtered = filter === 'all'
    ? alerts
    : filter === 'high'
      ? alerts.filter((a) => a.uncertainty === 'HIGH')
      : alerts.filter((a) => a.prediction === 1);

  const handleSelect = (alert) => {
    setSelected(alert.id);
    onSelectAlert(alert);
  };

  const handleFilterChange = (f) => {
    setFilter(f);
    setVisibleCount(PAGE_SIZE);
  };

  const visible = filtered.slice(0, visibleCount);
  const hasMore = visibleCount < filtered.length;

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="glass-panel-header">
        <span>Alert Feed</span>
        <div style={{ display: 'flex', gap: 6 }}>
          {['all', 'threats', 'high'].map((f) => (
            <button key={f} className="btn" onClick={() => handleFilterChange(f)}
              style={{
                padding: '2px 8px', fontSize: 10,
                borderColor: filter === f ? 'var(--cyan)' : undefined,
                color: filter === f ? 'var(--cyan)' : undefined,
              }}>
              {f.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Stats bar */}
      <div style={{
        display: 'flex', gap: 16, padding: '6px 16px',
        borderBottom: '1px solid var(--border)', fontSize: 10,
      }}>
        <span style={{ color: 'var(--text-muted)' }}>Total: <strong style={{ color: 'var(--cyan)' }}>{stats.total}</strong></span>
        <span style={{ color: 'var(--text-muted)' }}>High: <strong style={{ color: 'var(--red)' }}>{stats.high}</strong></span>
        <span style={{ color: 'var(--text-muted)' }}>Low: <strong style={{ color: 'var(--green)' }}>{stats.low}</strong></span>
      </div>

      {/* Scrollable list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px' }}>
        {filtered.length === 0 ? (
          <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
            No alerts yet. Waiting for pipeline data...
          </div>
        ) : (
          <>
            {visible.map((alert) => (
              <AlertRow
                key={alert.id}
                alert={alert}
                isSelected={selected === alert.id}
                onSelect={handleSelect}
              />
            ))}
            {(hasMore || visibleCount > PAGE_SIZE) && (
              <div style={{ padding: '8px 12px', textAlign: 'center', display: 'flex', gap: 8, justifyContent: 'center' }}>
                {hasMore && (
                  <button
                    className="btn"
                    onClick={() => setVisibleCount((c) => c + PAGE_SIZE)}
                    style={{ fontSize: 10, padding: '4px 16px', color: 'var(--cyan)', borderColor: 'var(--cyan)' }}
                  >
                    View More ({filtered.length - visibleCount} remaining)
                  </button>
                )}
                {visibleCount > PAGE_SIZE && (
                  <button
                    className="btn"
                    onClick={() => setVisibleCount(PAGE_SIZE)}
                    style={{ fontSize: 10, padding: '4px 16px', color: 'var(--text-muted)', borderColor: 'var(--border)' }}
                  >
                    Collapse
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
