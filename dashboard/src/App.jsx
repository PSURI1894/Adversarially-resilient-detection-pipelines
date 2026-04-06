/* ==============================================================================
   SOC COMMAND CENTER — MAIN APPLICATION
   ============================================================================== */

import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import RiskThermometer from './components/RiskThermometer';
import UncertaintyGauge from './components/UncertaintyGauge';
import AlertFeed from './components/AlertFeed';
import ThreatMap from './components/ThreatMap';
import ModelPerformance from './components/ModelPerformance';
import ConformalViz from './components/ConformalViz';
import DriftIndicator from './components/DriftIndicator';
import ExplainPanel from './components/ExplainPanel';
import AttackSimulator from './components/AttackSimulator';
import PlaybookPanel from './components/PlaybookPanel';
import useWebSocket from './hooks/useWebSocket';
import useAlerts from './hooks/useAlerts';
import { api } from './services/api';

// Poll interval for REST endpoints (ms)
const POLL_INTERVAL = 3000;

export default function App() {
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const { lastMessage, isConnected } = useWebSocket();
  const { alerts, stats, addAlert, addBatch } = useAlerts();

  // ── Poll REST endpoints ──────────────────────────────────
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const [s, m] = await Promise.all([api.getStatus(), api.getMetrics()]);
        if (active) { setStatus(s); setMetrics(m); }
      } catch { /* API not available yet */ }
    };
    poll();
    const id = setInterval(poll, POLL_INTERVAL);
    return () => { active = false; clearInterval(id); };
  }, []);

  // ── Handle WebSocket messages ────────────────────────────
  useEffect(() => {
    if (!lastMessage) return;
    if (lastMessage.type === 'alert') addAlert(lastMessage.data);
    if (lastMessage.type === 'state_update') {
      setStatus((prev) => prev ? { ...prev, ...lastMessage.data } : lastMessage.data);
    }
  }, [lastMessage, addAlert]);

  // ── Fetch initial alerts ─────────────────────────────────
  useEffect(() => {
    api.getAlerts(100).then((res) => addBatch(res.alerts)).catch(() => {});
  }, [addBatch]);

  // ── Derived state ────────────────────────────────────────
  const socState = status?.soc_state || 'STABLE';
  const severity = status?.severity ?? 0;
  const avgSetSize = metrics?.set_size_history?.length
    ? metrics.set_size_history[metrics.set_size_history.length - 1]?.value ?? 1.0
    : 1.0;

  return (
    <div className="dashboard-layout">
      {/* Row 0: Header */}
      <div className="dashboard-header">
        <Header
          socState={socState}
          severity={severity}
          isConnected={isConnected}
          totalAlerts={stats.total}
        />
      </div>

      {/* Row 1: Key indicators */}
      <div className="span-3">
        <RiskThermometer state={socState} severity={severity} />
      </div>
      <div className="span-3">
        <UncertaintyGauge
          avgSetSize={avgSetSize}
          history={metrics?.uncertainty_history || []}
        />
      </div>
      <div className="span-3">
        <DriftIndicator
          driftHistory={metrics?.drift_history || []}
          calibrationDrift={status?.calibration_drift ?? 0}
        />
      </div>
      <div className="span-3">
        <PlaybookPanel state={socState} alertDebt={status?.alert_debt ?? 0} />
      </div>

      {/* Row 2: Main content */}
      <div className="span-5">
        <AlertFeed
          alerts={alerts}
          stats={stats}
          onSelectAlert={setSelectedAlert}
        />
      </div>
      <div className="span-4">
        <ThreatMap alerts={alerts} socState={socState} />
      </div>
      <div className="span-3">
        <ExplainPanel alert={selectedAlert} />
      </div>

      {/* Row 3: Performance & simulation */}
      <div className="span-4">
        <ModelPerformance metrics={metrics} />
      </div>
      <div className="span-4">
        <ConformalViz
          setSizeHistory={metrics?.set_size_history || []}
          uncertaintyHistory={metrics?.uncertainty_history || []}
        />
      </div>
      <div className="span-4">
        <AttackSimulator />
      </div>
    </div>
  );
}
