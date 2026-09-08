import React from 'react';
import { getDisputeMeta } from '../../utils/disputeLabels';

interface DisputeBadgeProps {
  disputeCode?: string | null;
  className?: string;
}

export const DisputeBadge: React.FC<DisputeBadgeProps> = ({
  disputeCode,
  className = '',
}) => {
  const meta = getDisputeMeta(disputeCode);

  let colorClasses = 'border-neutral-300 bg-neutral-50 text-neutral-800 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-200';
  if (meta.severity === 'urgent') {
    colorClasses = 'border-red-300 bg-red-50 text-red-900 font-medium dark:border-red-700/60 dark:bg-red-950/30 dark:text-red-300';
  } else if (meta.severity === 'moderate') {
    colorClasses = 'border-blue-300 bg-blue-50 text-blue-900 dark:border-blue-700/60 dark:bg-blue-950/30 dark:text-blue-300';
  }

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-xs font-sans rounded border ${colorClasses} ${className}`}
      title={meta.description}
    >
      {meta.label}
    </span>
  );
};
