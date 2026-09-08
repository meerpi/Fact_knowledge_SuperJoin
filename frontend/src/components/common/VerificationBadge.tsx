import React from 'react';
import { CheckCircle2, AlertTriangle } from 'lucide-react';
import type { MatchType } from '../../api/types';

interface VerificationBadgeProps {
  verified: boolean;
  matchType: MatchType | string;
  confidence?: number;
  className?: string;
}

export const VerificationBadge: React.FC<VerificationBadgeProps> = ({
  verified,
  matchType,
  confidence,
  className = '',
}) => {
  const normType = (matchType || 'unverified').toLowerCase();

  if (!verified || normType === 'unverified') {
    return (
      <span
        className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-medium rounded border border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700/50 dark:bg-amber-950/30 dark:text-amber-300 ${className}`}
        title="Quote could not be grounded as a verbatim substring in the source PDF text"
      >
        <AlertTriangle className="w-3 h-3 text-amber-600 dark:text-amber-400 shrink-0" />
        <span>Unverified quote</span>
        {confidence !== undefined && (
          <span className="opacity-75">({Math.round(confidence * 100)}%)</span>
        )}
      </span>
    );
  }

  if (normType === 'exact') {
    return (
      <span
        className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-medium rounded border border-teal-300 bg-teal-50 text-teal-900 dark:border-teal-700/50 dark:bg-teal-950/30 dark:text-teal-300 ${className}`}
        title="Exact character-for-character match in source PDF text"
      >
        <CheckCircle2 className="w-3 h-3 text-teal-600 dark:text-teal-400 shrink-0" />
        <span>Verified (exact match)</span>
      </span>
    );
  }

  if (normType === 'normalized') {
    return (
      <span
        className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-medium rounded border border-teal-200 bg-teal-50/70 text-teal-800 dark:border-teal-800/40 dark:bg-teal-950/20 dark:text-teal-400 ${className}`}
        title="Verified after whitespace and character normalization"
      >
        <CheckCircle2 className="w-3 h-3 text-teal-500 shrink-0" />
        <span>Verified (normalized)</span>
      </span>
    );
  }

  // Fuzzy match
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono font-medium rounded border border-blue-200 bg-blue-50 text-blue-900 dark:border-blue-700/50 dark:bg-blue-950/30 dark:text-blue-300 ${className}`}
      title="Verified with high-confidence fuzzy string match"
    >
      <CheckCircle2 className="w-3 h-3 text-blue-600 dark:text-blue-400 shrink-0" />
      <span>Verified (fuzzy match)</span>
      {confidence !== undefined && (
        <span className="opacity-75">({Math.round(confidence * 100)}%)</span>
      )}
    </span>
  );
};
