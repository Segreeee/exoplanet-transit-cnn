import type { MeanStd } from "./types";

export const f3 = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "n/d" : v.toFixed(3);

export const f2 = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "n/d" : v.toFixed(2);

export const pm = (m: MeanStd | undefined) => (m ? `${f3(m.mean)} ± ${f3(m.std)}` : "n/d");

export const signed = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "n/d" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

export const pct = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "n/d" : `${Math.round(v * 100)}%`;

/** Probabilità calibrata (isotonica: assume anche 0 e 1 esatti): non la si mostra come certezza. */
export function softProb(p: number): string {
  if (p < 0.01) return "meno dell'1%";
  if (p > 0.99) return "oltre il 99%";
  return `circa ${Math.round(p * 100)}%`;
}

const num = (s: string) => {
  const v = Number(s);
  return Math.abs(v) >= 100 ? v.toFixed(0) : String(Number(v.toFixed(2)));
};

/** "(10.0, 30.0]" -> "10–30"; primo intervallo "fino a", ultimo "oltre". */
export function binLabel(bin: string, index: number, last: number): string {
  const m = bin.match(/[\[(]\s*(-?[\d.eE+]+)\s*,\s*(-?[\d.eE+]+)\s*[\])]/);
  if (!m) return bin;
  if (index === 0) return `fino a ${num(m[2])}`;
  if (index === last) return `oltre ${num(m[1])}`;
  return `${num(m[1])}–${num(m[2])}`;
}
