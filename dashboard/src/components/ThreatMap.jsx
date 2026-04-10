/* ==============================================================================
   THREAT MAP — REAL-TIME ATTACK VISUALIZATION (CANVAS-BASED)
   ==============================================================================
   Renders a network-style visualization of recent alerts as particles.
   Threats appear as red pulses, benign as green, uncertain as amber.
   ============================================================================== */

import React, { useRef, useEffect } from 'react';
import { colors, stateColors } from '../utils/theme';

export default function ThreatMap({ alerts, socState }) {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);

  // Update particles from alerts
  useEffect(() => {
    const recent = alerts.slice(0, 20);
    particlesRef.current = recent.map((a, i) => ({
      x: 30 + Math.random() * 240,
      y: 20 + Math.random() * 160,
      r: a.uncertainty === 'HIGH' ? 5 : 3,
      color: a.prediction === 1
        ? (a.uncertainty === 'HIGH' ? colors.red : colors.amber)
        : colors.green,
      alpha: Math.max(0.3, 1 - i * 0.012),
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
    }));
  }, [alerts]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Background grid
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.04)';
      ctx.lineWidth = 0.5;
      for (let x = 0; x < w; x += 30) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y < h; y += 30) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // Draw particles
      for (const p of particlesRef.current) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 10 || p.x > w - 10) p.vx *= -1;
        if (p.y < 10 || p.y > h - 10) p.vy *= -1;

        ctx.globalAlpha = p.alpha;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();

        // Glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r + 4, 0, Math.PI * 2);
        ctx.fillStyle = p.color.replace(')', ', 0.15)').replace('rgb', 'rgba');
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // State label
      const stColor = stateColors[socState] || colors.cyan;
      ctx.font = '600 10px "JetBrains Mono"';
      ctx.fillStyle = stColor;
      ctx.textAlign = 'right';
      ctx.fillText(socState.replace('_', ' '), w - 12, 16);

      raf = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(raf);
  }, [socState]);

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="glass-panel-header">
        <span>Threat Visualization</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
          last {Math.min(20, alerts.length)} flows
        </span>
      </div>
      <div className="glass-panel-body" style={{ flex: 1, padding: 8 }}>
        <canvas
          ref={canvasRef}
          width={300}
          height={200}
          style={{ width: '100%', height: '100%', borderRadius: 8 }}
        />
      </div>
      {/* Legend */}
      <div style={{
        display: 'flex', gap: 12, padding: '6px 16px',
        borderTop: '1px solid var(--border)', fontSize: 10, color: 'var(--text-muted)',
      }}>
        <span><span style={{ color: colors.red }}>&#x25CF;</span> Threat (High)</span>
        <span><span style={{ color: colors.amber }}>&#x25CF;</span> Threat (Low)</span>
        <span><span style={{ color: colors.green }}>&#x25CF;</span> Benign</span>
      </div>
    </div>
  );
}
