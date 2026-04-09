/* ==============================================================================
   API CLIENT — REST + WEBSOCKET
   ============================================================================== */

const BASE_URL = '/api';

async function fetchJSON(url, options = {}) {
  const res = await fetch(`${BASE_URL}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  getStatus:   ()          => fetchJSON('/status'),
  getAlerts:   (limit=50, offset=0, severity=null) => {
    let url = `/alerts?limit=${limit}&offset=${offset}`;
    if (severity) url += `&severity=${severity}`;
    return fetchJSON(url);
  },
  getMetrics:  ()          => fetchJSON('/metrics'),
  getHistory:  (metric, limit=200) => fetchJSON(`/metrics/history?metric=${metric}&limit=${limit}`),
  getExplain:  (alertId)   => fetchJSON(`/explain/${alertId}`),
  simulate:         (params) => fetchJSON('/simulate', {
    method: 'POST',
    body: JSON.stringify(params),
  }),
  stopSimulation:   ()      => fetchJSON('/simulate/stop', { method: 'POST' }),
  simulationStatus: ()      => fetchJSON('/simulate/status'),
  demoStart:        ()      => fetchJSON('/demo/start', { method: 'POST' }),
  demoStop:         ()      => fetchJSON('/demo/stop', { method: 'POST' }),
  demoStatus:       ()      => fetchJSON('/demo/status'),
  getConnections:   ()      => fetchJSON('/connections'),
};

export function createWebSocket(onMessage) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch { /* ignore malformed */ }
  };

  ws.onclose = () => {
    // Auto-reconnect after 3s
    setTimeout(() => createWebSocket(onMessage), 3000);
  };

  return ws;
}
