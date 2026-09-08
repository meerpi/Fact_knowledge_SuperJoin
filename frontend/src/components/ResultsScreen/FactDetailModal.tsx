import React from 'react';
import { X, ExternalLink } from 'lucide-react';
import type { Fact } from '../../api/types';
import { VerificationBadge } from '../common/VerificationBadge';
import { TextHighlight } from '../common/TextHighlight';
import { formatMetricName, formatValue, formatCanonical } from '../../utils/formatters';

interface FactDetailModalProps {
  fact: Fact | null;
  onClose: () => void;
  onViewPdf?: (docId: string, page: number) => void;
}

export const FactDetailModal: React.FC<FactDetailModalProps> = ({
  fact,
  onClose,
  onViewPdf,
}) => {
  if (!fact) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-[1px] p-4">
      <div className="w-full max-w-2xl bg-white dark:bg-neutral-900 border border-neutral-300 dark:border-neutral-700 shadow-xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-800 flex items-center justify-between bg-neutral-50 dark:bg-neutral-850">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold uppercase text-neutral-500">
              Fact Evidence Detail
            </span>
            <span className="text-neutral-300 dark:text-neutral-700">•</span>
            <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100 font-mono">
              {fact.subject} / {formatMetricName(fact.predicate)}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 overflow-y-auto space-y-5 text-xs text-neutral-900 dark:text-neutral-100">
          {/* Values Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3 bg-neutral-50 dark:bg-neutral-850 border border-neutral-200 dark:border-neutral-800">
            <div>
              <div className="text-[10px] font-mono text-neutral-400 uppercase">Reported Value</div>
              <div className="text-sm font-bold font-mono mt-0.5">{formatValue(fact.value)}</div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-neutral-400 uppercase">Canonical Value</div>
              <div className="text-sm font-mono mt-0.5">
                {formatCanonical(fact.canonical_value, fact.canonical_unit) || '—'}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-neutral-400 uppercase">Scale Multiplier</div>
              <div className="text-sm font-mono mt-0.5">
                {fact.scale ? `10^${fact.scale}` : '0 (units)'}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-neutral-400 uppercase">Dimension</div>
              <div className="text-sm font-mono mt-0.5 capitalize">{fact.dimension || 'other'}</div>
            </div>
          </div>

          {/* Context Qualifiers */}
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-500 mb-2">
              Contextual Qualifiers
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <div className="p-2.5 border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 font-mono">
                <span className="text-[10px] text-neutral-400 block">Temporal Period:</span>
                <span className="font-medium">{fact.context.temporal || 'Unspecified'}</span>
              </div>
              <div className="p-2.5 border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 font-mono">
                <span className="text-[10px] text-neutral-400 block">Entity Scope:</span>
                <span className="font-medium">{fact.context.scope || 'General'}</span>
              </div>
              <div className="p-2.5 border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 font-mono">
                <span className="text-[10px] text-neutral-400 block">Conditions:</span>
                <span className="font-medium">{fact.context.conditions || 'None stated'}</span>
              </div>
            </div>
          </div>

          {/* Source Grounding */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-500">
                Verbatim Source Excerpt
              </div>
              {onViewPdf && fact.provenance.page !== null && fact.provenance.page !== undefined && (
                <button
                  onClick={() => {
                    onViewPdf(fact.provenance.doc_id, fact.provenance.page!);
                    onClose();
                  }}
                  className="inline-flex items-center gap-1 text-[11px] font-mono text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100 underline decoration-dotted"
                >
                  View in Source PDF (p. {fact.provenance.page})
                  <ExternalLink className="w-3 h-3" />
                </button>
              )}
            </div>

            <div className="p-3 bg-neutral-50 dark:bg-neutral-850 border border-neutral-200 dark:border-neutral-800 text-xs">
              <TextHighlight
                text={fact.provenance.evidence_quote}
                highlight={fact.value}
                type="corroborate"
              />
            </div>
          </div>

          {/* Verification Audit */}
          <div className="pt-2 border-t border-neutral-200 dark:border-neutral-800 flex flex-wrap items-center justify-between gap-2">
            <VerificationBadge
              verified={fact.provenance.verified}
              matchType={fact.provenance.match_type}
              confidence={fact.confidence}
            />
            <div className="text-[11px] font-mono text-neutral-500">
              Confidence Score: {(fact.confidence * 100).toFixed(0)}% • Grounded on page {fact.provenance.page ?? 'N/A'}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 flex justify-end">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs font-mono border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:bg-neutral-100 dark:hover:bg-neutral-800"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
