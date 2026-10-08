interface Props {
  x: number[];
  y: number[];
  xLabel: string;
  yLabel: string;
  color?: string;
  height?: number;
  dots?: boolean;
}

const W = 640;
const M = { l: 52, r: 12, t: 10, b: 38 };

function ticks(min: number, max: number, n = 4): number[] {
  return Array.from({ length: n + 1 }, (_, i) => min + ((max - min) * i) / n);
}

function decimals(min: number, max: number): number {
  const range = Math.abs(max - min) || 1;
  return Math.max(0, Math.min(4, 2 - Math.floor(Math.log10(range))));
}

export function LineChart({ x, y, xLabel, yLabel, color = "var(--accent)", height = 200, dots }: Props) {
  const xmin = Math.min(...x);
  const xmax = Math.max(...x);
  let ymin = Math.min(...y);
  let ymax = Math.max(...y);
  const pad = (ymax - ymin) * 0.08 || 1e-3;
  ymin -= pad;
  ymax += pad;
  const yd = decimals(ymin, ymax);
  const xd = decimals(xmin, xmax);
  const sx = (v: number) => M.l + ((v - xmin) / (xmax - xmin || 1)) * (W - M.l - M.r);
  const sy = (v: number) => height - M.b - ((v - ymin) / (ymax - ymin)) * (height - M.t - M.b);
  const path = x.map((v, i) => `${i ? "L" : "M"}${sx(v).toFixed(1)},${sy(y[i]).toFixed(1)}`).join("");

  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`${yLabel} in funzione di ${xLabel}`} className="chart">
      {ticks(ymin, ymax).map((t) => (
        <g key={`y${t}`}>
          <line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} className="grid" />
          <text x={M.l - 6} y={sy(t) + 4} textAnchor="end" className="tick">
            {t.toFixed(yd)}
          </text>
        </g>
      ))}
      {ticks(xmin, xmax).map((t) => (
        <text key={`x${t}`} x={sx(t)} y={height - M.b + 16} textAnchor="middle" className="tick">
          {t.toFixed(xd)}
        </text>
      ))}
      {dots ? (
        x.map((v, i) => <circle key={i} cx={sx(v)} cy={sy(y[i])} r={1.6} fill={color} opacity={0.8} />)
      ) : (
        <path d={path} fill="none" stroke={color} strokeWidth={1.6} strokeLinejoin="round" />
      )}
      <text x={(M.l + W - M.r) / 2} y={height - 4} textAnchor="middle" className="axis">
        {xLabel}
      </text>
      <text transform={`translate(12 ${(M.t + height - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis">
        {yLabel}
      </text>
    </svg>
  );
}
