/* ==============================================================================
   DESIGN TOKENS — SOC COMMAND CENTER THEME
   ============================================================================== */

export const colors = {
  bgPrimary: '#0a0e27',
  bgSecondary: '#0f1435',
  bgPanel: 'rgba(15, 20, 53, 0.65)',

  cyan: '#00f0ff',
  cyanDim: 'rgba(0, 240, 255, 0.15)',
  cyanGlow: 'rgba(0, 240, 255, 0.4)',
  green: '#00ff88',
  greenDim: 'rgba(0, 255, 136, 0.15)',
  red: '#ff3d3d',
  redDim: 'rgba(255, 61, 61, 0.15)',
  amber: '#ffaa00',
  amberDim: 'rgba(255, 170, 0, 0.15)',
  purple: '#a855f7',

  textPrimary: '#e2e8f0',
  textSecondary: '#94a3b8',
  textMuted: '#64748b',

  border: 'rgba(0, 240, 255, 0.12)',
  borderActive: 'rgba(0, 240, 255, 0.35)',
};

export const stateColors = {
  STABLE: colors.green,
  SUSPICIOUS: colors.amber,
  EVASION_LOCKED: colors.red,
  FAILURE: colors.red,
};

export const severityColor = (severity) => {
  if (severity < 30) return colors.green;
  if (severity < 60) return colors.amber;
  return colors.red;
};

export const uncertaintyColor = (avgSetSize) => {
  if (avgSetSize <= 1.1) return colors.green;
  if (avgSetSize <= 1.5) return colors.amber;
  return colors.red;
};
