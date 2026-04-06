/* ==============================================================================
   useAlerts — ALERT STATE MANAGEMENT HOOK
   ============================================================================== */

import { useState, useCallback, useRef } from 'react';

const MAX_ALERTS = 500;

export default function useAlerts() {
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({ total: 0, high: 0, low: 0 });
  const idSet = useRef(new Set());

  const addAlert = useCallback((alert) => {
    const id = alert.id || `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    if (idSet.current.has(id)) return;
    idSet.current.add(id);

    setAlerts((prev) => {
      const next = [{ ...alert, id }, ...prev];
      if (next.length > MAX_ALERTS) {
        const removed = next.pop();
        idSet.current.delete(removed.id);
      }
      return next;
    });

    setStats((prev) => ({
      total: prev.total + 1,
      high: prev.high + (alert.uncertainty === 'HIGH' ? 1 : 0),
      low: prev.low + (alert.uncertainty === 'LOW' ? 1 : 0),
    }));
  }, []);

  const addBatch = useCallback((batch) => {
    batch.forEach(addAlert);
  }, [addAlert]);

  const clear = useCallback(() => {
    setAlerts([]);
    setStats({ total: 0, high: 0, low: 0 });
    idSet.current.clear();
  }, []);

  return { alerts, stats, addAlert, addBatch, clear };
}
