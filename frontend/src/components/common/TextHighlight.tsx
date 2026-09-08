import React from 'react';

interface TextHighlightProps {
  text: string;
  highlight?: string | null;
  type?: 'contradict' | 'corroborate' | 'reconcile' | 'default';
  className?: string;
}

export const TextHighlight: React.FC<TextHighlightProps> = ({
  text,
  highlight,
  type = 'default',
  className = '',
}) => {
  if (!text) return null;
  if (!highlight || !highlight.trim()) {
    return <span className={`source-quote ${className}`}>{text}</span>;
  }

  const highlightClass =
    type === 'contradict'
      ? 'evidence-highlight-contradict'
      : type === 'corroborate'
      ? 'evidence-highlight-corroborate'
      : type === 'reconcile'
      ? 'evidence-highlight-reconcile'
      : 'evidence-highlight';

  const lowerText = text.toLowerCase();
  const lowerHighlight = highlight.trim().toLowerCase();
  const index = lowerText.indexOf(lowerHighlight);

  if (index === -1) {
    return <span className={`source-quote ${className}`}>{text}</span>;
  }

  const before = text.slice(0, index);
  const matched = text.slice(index, index + highlight.trim().length);
  const after = text.slice(index + highlight.trim().length);

  return (
    <span className={`source-quote ${className}`}>
      {before}
      <mark className={highlightClass}>{matched}</mark>
      {after}
    </span>
  );
};
