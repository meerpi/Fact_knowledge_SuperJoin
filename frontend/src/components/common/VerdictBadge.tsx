import React from 'react';

export type VerdictType =
  | 'corroborated'
  | 'contradicted'
  | 'reconciled'
  | 'reconciled_temporal'
  | 'reconciled_scope'
  | 'reconciled_unit'
  | 'reconciled_conditions'
  | 'extraction_failure';

interface VerdictBadgeProps {
  verdict: VerdictType | string;
  className?: string;
  size?: 'sm' | 'md';
}

export const VerdictBadge: React.FC<VerdictBadgeProps> = ({
  verdict,
  className = '',
  size = 'md',
}) => {
  const norm = (verdict || '').toLowerCase();
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs tracking-wider';

  if (norm.includes('contradict')) {
    return (
      <span
        className={`inline-flex items-center font-mono font-semibold uppercase border border-red-500 bg-red-50 text-red-900 dark:border-red-600 dark:bg-red-950/40 dark:text-red-200 ${sizeClasses} ${className}`}
      >
        Contradicted
      </span>
    );
  }

  if (norm.includes('corroborat')) {
    return (
      <span
        className={`inline-flex items-center font-mono font-medium uppercase border border-teal-600/40 bg-teal-50 text-teal-900 dark:border-teal-700/50 dark:bg-teal-950/40 dark:text-teal-300 ${sizeClasses} ${className}`}
      >
        Corroborated
      </span>
    );
  }

  if (norm.includes('reconcil')) {
    return (
      <span
        className={`inline-flex items-center font-mono font-medium uppercase border border-blue-600/40 bg-blue-50 text-blue-900 dark:border-blue-700/50 dark:bg-blue-950/40 dark:text-blue-300 ${sizeClasses} ${className}`}
      >
        Reconciled
      </span>
    );
  }

  if (norm.includes('failure')) {
    return (
      <span
        className={`inline-flex items-center font-mono font-medium uppercase border border-neutral-400 bg-neutral-100 text-neutral-800 dark:border-neutral-600 dark:bg-neutral-800 dark:text-neutral-200 ${sizeClasses} ${className}`}
      >
        Extraction Failure
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center font-mono font-medium uppercase border border-neutral-300 bg-neutral-50 text-neutral-700 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-300 ${sizeClasses} ${className}`}
    >
      {verdict}
    </span>
  );
};
