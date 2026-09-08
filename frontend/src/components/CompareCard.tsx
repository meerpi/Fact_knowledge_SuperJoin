import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, Scale, ExternalLink, AlertOctagon } from 'lucide-react';
import type { EvidenceEntry, ExtractionFailure } from '../api/types';
import { VerdictBadge, type VerdictType } from './common/VerdictBadge';
import { DisputeBadge } from './common/DisputeBadge';
import { VerificationBadge } from './common/VerificationBadge';
import { TextHighlight } from './common/TextHighlight';
import { formatMetricName, formatValue, formatCanonical } from '../utils/formatters';

interface CompareCardProps {
  caseTitle: string;
  subject?: string;
  predicate?: string;
  verdict: VerdictType | string;
  disputeCode?: string | null;
  disputeDetail?: string | null;
  evidence?: EvidenceEntry[];
  explanation: string;
  consensusValue?: number | null;
  credibilityScores?: Record<string | number, number>;
  reconciliationDimension?: string;
  failure?: ExtractionFailure | null;
  onViewPdf?: (docId: string, page: number) => void;
}

export const CompareCard: React.FC<CompareCardProps> = ({
  caseTitle,
  subject,
  predicate,
  verdict,
  disputeCode,
  disputeDetail,
  evidence = [],
  explanation,
  consensusValue,
  credibilityScores,
  reconciliationDimension,
  failure,
  onViewPdf,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // If this is Case 4 (Extraction Failure)
  if (failure) {
    return (
      <div className="border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 rounded-none p-5 text-neutral-900 dark:text-neutral-100">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-neutral-200 dark:border-neutral-800">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold tracking-wider text-neutral-500 uppercase">
              {caseTitle}
            </span>
            <span className="text-neutral-300 dark:text-neutral-700">•</span>
            <span className="text-sm font-semibold">Extraction & Verification Failure</span>
          </div>
          <div className="flex items-center gap-2">
            <VerdictBadge verdict="extraction_failure" />
            <span className="px-2 py-0.5 text-xs font-mono rounded border border-neutral-300 bg-neutral-100 text-neutral-700 dark:border-neutral-750 dark:bg-neutral-800 dark:text-neutral-300">
              {failure.failure_type.replace(/_/g, ' ')}
            </span>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 border border-red-200 bg-red-50/50 dark:border-red-900/40 dark:bg-red-950/20 text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-red-900 dark:text-red-300 mb-2">
              <AlertOctagon className="w-4 h-4 text-red-600 dark:text-red-400" />
              Observed Failure
            </div>
            <p className="text-neutral-700 dark:text-neutral-300 leading-relaxed font-mono">
              {failure.description}
            </p>
            <div className="mt-3 pt-2 border-t border-red-200 dark:border-red-900/40 flex items-center justify-between text-neutral-500">
              <span>Doc ID: {failure.doc_id.slice(0, 12)}</span>
              <span>Page: {failure.page !== null && failure.page !== undefined ? failure.page : 'N/A'}</span>
            </div>
          </div>

          <div className="p-4 border border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-850 text-xs">
            <div className="font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
              Deterministic Handling & Mitigation
            </div>
            <p className="text-neutral-700 dark:text-neutral-300 leading-relaxed">
              {failure.mitigation}
            </p>
            <div className="mt-3 pt-2 border-t border-neutral-200 dark:border-neutral-800">
              <VerificationBadge verified={false} matchType="unverified" confidence={0.1} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Two primary evidence entries (left vs right)
  const left = evidence[0];
  const right = evidence[1] || evidence[0];
  const isContradiction = verdict.toLowerCase().includes('contradict');
  const isCorroboration = verdict.toLowerCase().includes('corroborat');
  const isReconciled = verdict.toLowerCase().includes('reconcil');

  const highlightType = isContradiction
    ? 'contradict'
    : isCorroboration
    ? 'corroborate'
    : isReconciled
    ? 'reconcile'
    : 'default';

  return (
    <div
      className={`border bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 ${
        isContradiction
          ? 'border-red-400 dark:border-red-800/80 shadow-xs'
          : 'border-neutral-300 dark:border-neutral-800'
      }`}
    >
      {/* Top Bar: Case identifier, Claim Subject & Predicate, Verdict + Dispute Badge */}
      <div className="p-4 sm:p-5 pb-3 border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-850/40">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-mono font-bold tracking-wider text-neutral-500 uppercase">
              {caseTitle}
            </span>
            <span className="text-neutral-300 dark:text-neutral-700">•</span>
            <span className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              {subject ? `${subject} — ` : ''}
              <span className="font-mono text-neutral-700 dark:text-neutral-300">
                {predicate ? formatMetricName(predicate) : 'Metric'}
              </span>
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <VerdictBadge verdict={verdict} />
            <DisputeBadge disputeCode={disputeCode} />
          </div>
        </div>
      </div>

      {/* Side-by-side Evidence Display: Left Excerpt vs Right Excerpt */}
      <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-neutral-200 dark:divide-neutral-800">
        {/* Left Side */}
        {left && (
          <div className="p-4 sm:p-5 flex flex-col justify-between">
            <div>
              {/* Document and Provenance Header */}
              <div className="flex items-start justify-between gap-2 pb-2 mb-3 border-b border-neutral-100 dark:border-neutral-800 text-xs">
                <div className="flex items-center gap-1.5 font-medium text-neutral-800 dark:text-neutral-200 truncate">
                  <FileText className="w-3.5 h-3.5 text-neutral-500 shrink-0" />
                  <span className="truncate" title={left.doc_filename}>
                    {left.doc_filename}
                  </span>
                  <span className="text-neutral-400 font-mono">
                    (p. {left.page !== null && left.page !== undefined ? left.page : '—'})
                  </span>
                </div>
                {onViewPdf && left.page !== null && left.page !== undefined && (
                  <button
                    onClick={() => onViewPdf(left.doc_id, left.page!)}
                    className="inline-flex items-center gap-1 text-[11px] font-mono text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100 underline decoration-dotted"
                  >
                    View in PDF
                    <ExternalLink className="w-2.5 h-2.5" />
                  </button>
                )}
              </div>

              {/* Reported Value & Qualifiers */}
              <div className="mb-3 flex flex-wrap items-baseline gap-2">
                <span className="text-base font-bold font-mono text-neutral-900 dark:text-neutral-100">
                  {formatValue(left.value)}
                </span>
                {left.canonical_value !== null && left.canonical_value !== undefined && (
                  <span className="text-xs font-mono text-neutral-500" title="Canonical scale-resolved value">
                    (canonical: {formatCanonical(left.canonical_value, left.canonical_unit)})
                  </span>
                )}
              </div>

              {/* Context badges (temporal, scope, conditions) */}
              <div className="flex flex-wrap gap-1.5 mb-3 text-[11px]">
                {left.temporal && (
                  <span className="px-1.5 py-0.5 font-mono border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300">
                    Period: {left.temporal}
                  </span>
                )}
                {left.scope && (
                  <span className="px-1.5 py-0.5 font-mono border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300">
                    Scope: {left.scope}
                  </span>
                )}
                {left.conditions && (
                  <span className="px-1.5 py-0.5 font-mono border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300">
                    Conditions: {left.conditions}
                  </span>
                )}
              </div>

              {/* Verbatim Source Excerpt */}
              <div className="p-3 bg-neutral-50 dark:bg-neutral-850/50 border border-neutral-200/80 dark:border-neutral-800 text-xs text-neutral-800 dark:text-neutral-200">
                <span className="block text-[10px] font-mono text-neutral-400 uppercase tracking-wider mb-1">
                  Primary Source Grounding
                </span>
                <TextHighlight
                  text={left.evidence_quote}
                  highlight={left.value}
                  type={highlightType}
                />
              </div>
            </div>

            {/* Verification State Chip */}
            <div className="mt-4 pt-2 border-t border-neutral-100 dark:border-neutral-800/60 flex items-center justify-between">
              <VerificationBadge
                verified={left.match_type !== 'unverified'}
                matchType={left.match_type}
                confidence={left.confidence}
              />
              <span className="text-[11px] font-mono text-neutral-400">
                Match: {left.match_type}
              </span>
            </div>
          </div>
        )}

        {/* Right Side */}
        {right && (
          <div className="p-4 sm:p-5 flex flex-col justify-between">
            <div>
              {/* Document and Provenance Header */}
              <div className="flex items-start justify-between gap-2 pb-2 mb-3 border-b border-neutral-100 dark:border-neutral-800 text-xs">
                <div className="flex items-center gap-1.5 font-medium text-neutral-800 dark:text-neutral-200 truncate">
                  <FileText className="w-3.5 h-3.5 text-neutral-500 shrink-0" />
                  <span className="truncate" title={right.doc_filename}>
                    {right.doc_filename}
                  </span>
                  <span className="text-neutral-400 font-mono">
                    (p. {right.page !== null && right.page !== undefined ? right.page : '—'})
                  </span>
                </div>
                {onViewPdf && right.page !== null && right.page !== undefined && (
                  <button
                    onClick={() => onViewPdf(right.doc_id, right.page!)}
                    className="inline-flex items-center gap-1 text-[11px] font-mono text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100 underline decoration-dotted"
                  >
                    View in PDF
                    <ExternalLink className="w-2.5 h-2.5" />
                  </button>
                )}
              </div>

              {/* Reported Value & Qualifiers */}
              <div className="mb-3 flex flex-wrap items-baseline gap-2">
                <span className="text-base font-bold font-mono text-neutral-900 dark:text-neutral-100">
                  {formatValue(right.value)}
                </span>
                {right.canonical_value !== null && right.canonical_value !== undefined && (
                  <span className="text-xs font-mono text-neutral-500" title="Canonical scale-resolved value">
                    (canonical: {formatCanonical(right.canonical_value, right.canonical_unit)})
                  </span>
                )}
              </div>

              {/* Context badges (temporal, scope, conditions) */}
              <div className="flex flex-wrap gap-1.5 mb-3 text-[11px]">
                {right.temporal && (
                  <span className="px-1.5 py-0.5 font-mono border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300">
                    Period: {right.temporal}
                  </span>
                )}
                {right.scope && (
                  <span className="px-1.5 py-0.5 font-mono border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300">
                    Scope: {right.scope}
                  </span>
                )}
                {right.conditions && (
                  <span className="px-1.5 py-0.5 font-mono border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300">
                    Conditions: {right.conditions}
                  </span>
                )}
              </div>

              {/* Verbatim Source Excerpt */}
              <div className="p-3 bg-neutral-50 dark:bg-neutral-850/50 border border-neutral-200/80 dark:border-neutral-800 text-xs text-neutral-800 dark:text-neutral-200">
                <span className="block text-[10px] font-mono text-neutral-400 uppercase tracking-wider mb-1">
                  Primary Source Grounding
                </span>
                <TextHighlight
                  text={right.evidence_quote}
                  highlight={right.value}
                  type={highlightType}
                />
              </div>
            </div>

            {/* Verification State Chip */}
            <div className="mt-4 pt-2 border-t border-neutral-100 dark:border-neutral-800/60 flex items-center justify-between">
              <VerificationBadge
                verified={right.match_type !== 'unverified'}
                matchType={right.match_type}
                confidence={right.confidence}
              />
              <span className="text-[11px] font-mono text-neutral-400">
                Match: {right.match_type}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Expandable Reasoning Row: 1-Click Away From Verdict, Never Buried */}
      <div className="border-t border-neutral-200 dark:border-neutral-800 bg-neutral-50/70 dark:bg-neutral-850/30">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full px-4 py-2.5 flex items-center justify-between text-left text-xs font-medium text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Scale className="w-3.5 h-3.5 text-neutral-500" />
            <span>ArbGraph Arbitration Reasoning & Credibility Analysis</span>
          </div>
          <div className="flex items-center gap-1 text-neutral-500 font-mono text-[11px]">
            <span>{isExpanded ? 'Hide reasoning' : 'View full reasoning'}</span>
            {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </div>
        </button>

        {isExpanded && (
          <div className="px-4 pb-4 pt-1 border-t border-neutral-200/50 dark:border-neutral-800/50 text-xs text-neutral-800 dark:text-neutral-200 space-y-3">
            <div className="p-3 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 font-mono leading-relaxed text-[11.5px]">
              {explanation}
            </div>

            {/* Consensus & Credibility Breakdown (if contradiction) */}
            {consensusValue !== undefined && consensusValue !== null && (
              <div className="p-3 border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-neutral-500 font-medium">Consensus Value:</span>
                  <span className="font-mono font-bold text-neutral-900 dark:text-neutral-100">
                    {consensusValue}
                  </span>
                </div>
                {credibilityScores && Object.keys(credibilityScores).length > 0 && (
                  <div className="text-[11px] font-mono text-neutral-500">
                    Evaluated against {Object.keys(credibilityScores).length} fact sources with authority weighting
                  </div>
                )}
              </div>
            )}

            {reconciliationDimension && (
              <div className="text-[11px] text-neutral-600 dark:text-neutral-400 font-mono">
                <span className="font-medium text-neutral-700 dark:text-neutral-300">Reconciliation Dimension:</span>{' '}
                {reconciliationDimension}
              </div>
            )}

            {disputeDetail && (
              <div className="text-[11px] text-neutral-600 dark:text-neutral-400">
                <span className="font-medium text-neutral-700 dark:text-neutral-300">Category Note:</span>{' '}
                {disputeDetail}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
