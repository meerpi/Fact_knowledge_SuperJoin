/**
 * Formatting utilities for numbers, currencies, and confidence scores
 */

export function formatConfidence(score: number): string {
  return `${Math.round(score * 100)}%`;
}

export function formatMetricName(predicate: string): string {
  return predicate
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function formatValue(value: string | number | null | undefined, unit?: string | null): string {
  if (value === null || value === undefined) return '—';
  const valStr = typeof value === 'number' ? value.toLocaleString('en-US', { maximumFractionDigits: 2 }) : String(value);
  if (unit && !valStr.toLowerCase().includes(unit.toLowerCase())) {
    return `${valStr} ${unit}`;
  }
  return valStr;
}

export function formatCanonical(canonicalValue?: number | null, canonicalUnit?: string | null): string | null {
  if (canonicalValue === null || canonicalValue === undefined) return null;
  const numStr = canonicalValue.toLocaleString('en-US', { maximumFractionDigits: 2 });
  return canonicalUnit ? `${numStr} ${canonicalUnit}` : numStr;
}
