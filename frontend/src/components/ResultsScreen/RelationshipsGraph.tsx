import React, { useMemo, useState, useRef } from 'react';
import type { FactCluster, EvidenceEntry } from '../../api/types';
import { VerdictBadge } from '../common/VerdictBadge';
import { DisputeBadge } from '../common/DisputeBadge';
import { getDisputeMeta } from '../../utils/disputeLabels';
import { formatMetricName, formatConfidence } from '../../utils/formatters';
import { FileText, ChevronDown, ChevronRight, Eye, Columns, ExternalLink } from 'lucide-react';
import { InlinePdfViewer } from './InlinePdfViewer';

interface RelationshipsGraphProps {
  clusters: FactCluster[];
  onViewPdf?: (docId: string, page: number, quote?: string) => void;
  docFilenames?: Record<string, string>;
}

type VerdictFilter = 'all' | 'contradicted' | 'reconciled' | 'corroborated';

export const RelationshipsGraph: React.FC<RelationshipsGraphProps> = ({
  clusters,
  onViewPdf,
  docFilenames = {},
}) => {
  const [expandedClusterId, setExpandedClusterId] = useState<string | null>(
    clusters.length > 0 ? clusters[0].cluster_id : null
  );
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>('all');
  const [selectedDoc, setSelectedDoc] = useState<string>('all');

  // Extract unique documents present in the clusters
  const clusterDocs = useMemo(() => {
    const docMap = new Map<string, string>();
    clusters.forEach((cl) => {
      cl.evidence?.forEach((ev) => {
        if (ev.doc_id && !docMap.has(ev.doc_id)) {
          docMap.set(ev.doc_id, ev.doc_filename || docFilenames[ev.doc_id] || ev.doc_id);
        }
      });
    });
    return Array.from(docMap.entries()).map(([id, name]) => ({ id, name }));
  }, [clusters, docFilenames]);

  // Filter clusters based on verdict and selected document
  const filteredClusters = useMemo(() => {
    return clusters.filter((c) => {
      if (verdictFilter === 'contradicted' && c.case_type !== 'contradicted') return false;
      if (verdictFilter === 'reconciled' && !c.case_type.startsWith('reconciled')) return false;
      if (verdictFilter === 'corroborated' && c.case_type !== 'corroborated') return false;

      if (selectedDoc !== 'all') {
        const hasDoc = c.evidence?.some((ev) => ev.doc_id === selectedDoc);
        if (!hasDoc) return false;
      }
      return true;
    });
  }, [clusters, verdictFilter, selectedDoc]);

  // Count by verdict type
  const counts = useMemo(() => {
    const c = { contradicted: 0, reconciled: 0, corroborated: 0 };
    clusters.forEach((cl) => {
      if (cl.case_type === 'contradicted') c.contradicted++;
      else if (cl.case_type.startsWith('reconciled')) c.reconciled++;
      else if (cl.case_type === 'corroborated') c.corroborated++;
    });
    return c;
  }, [clusters]);

  if (clusters.length === 0) {
    return (
      <div className="p-8 text-center border border-dashed border-neutral-300 dark:border-neutral-700 text-neutral-500 font-mono text-xs">
        No claim clusters found. Run cross-document reconciliation to view disputed facts.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-2 font-mono">
          <FilterButton
            label={`All (${clusters.length})`}
            active={verdictFilter === 'all'}
            onClick={() => setVerdictFilter('all')}
          />
          {counts.contradicted > 0 && (
            <FilterButton
              label={`Contradicted (${counts.contradicted})`}
              active={verdictFilter === 'contradicted'}
              onClick={() => setVerdictFilter('contradicted')}
              color="red"
            />
          )}
          {counts.reconciled > 0 && (
            <FilterButton
              label={`Reconciled (${counts.reconciled})`}
              active={verdictFilter === 'reconciled'}
              onClick={() => setVerdictFilter('reconciled')}
              color="blue"
            />
          )}
          {counts.corroborated > 0 && (
            <FilterButton
              label={`Corroborated (${counts.corroborated})`}
              active={verdictFilter === 'corroborated'}
              onClick={() => setVerdictFilter('corroborated')}
              color="teal"
            />
          )}

          {clusterDocs.length > 1 && (
            <div className="flex items-center gap-1.5 ml-2 font-mono text-xs">
              <span className="text-neutral-500 text-[11px]">Doc:</span>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(e.target.value)}
                className="px-2 py-1 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-850 text-neutral-800 dark:text-neutral-200 text-[11px] font-mono focus:outline-none cursor-pointer"
              >
                <option value="all">All Documents</option>
                {clusterDocs.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name.length > 25 ? d.name.slice(0, 22) + '...' : d.name}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
        <div className="text-[11px] font-mono text-neutral-500">
          {filteredClusters.length} cluster{filteredClusters.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Cluster list */}
      <div className="space-y-3">
        {filteredClusters.map((cluster) => {
          const isExpanded = expandedClusterId === cluster.cluster_id;
          return (
            <ClusterCard
              key={cluster.cluster_id}
              cluster={cluster}
              isExpanded={isExpanded}
              onToggle={() =>
                setExpandedClusterId(isExpanded ? null : cluster.cluster_id)
              }
              onViewPdf={onViewPdf}
            />
          );
        })}
      </div>
    </div>
  );
};

/* ─────────── Filter Button ─────────── */
const FilterButton: React.FC<{
  label: string;
  active: boolean;
  onClick: () => void;
  color?: 'red' | 'blue' | 'teal';
}> = ({ label, active, onClick, color }) => {
  let cls = 'px-2.5 py-1 border font-mono text-[11px] transition-colors cursor-pointer ';
  if (active) {
    if (color === 'red') cls += 'border-red-500 bg-red-50 text-red-900 dark:bg-red-950/40 dark:text-red-200 dark:border-red-600';
    else if (color === 'blue') cls += 'border-blue-500 bg-blue-50 text-blue-900 dark:bg-blue-950/40 dark:text-blue-200 dark:border-blue-600';
    else if (color === 'teal') cls += 'border-teal-500 bg-teal-50 text-teal-900 dark:bg-teal-950/40 dark:text-teal-200 dark:border-teal-600';
    else cls += 'border-neutral-900 bg-neutral-100 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100 dark:border-neutral-100';
  } else {
    cls += 'border-neutral-300 dark:border-neutral-700 text-neutral-600 dark:text-neutral-400 hover:bg-neutral-50 dark:hover:bg-neutral-850';
  }
  return <button onClick={onClick} className={cls}>{label}</button>;
};

/* ─────────── Cluster Card ─────────── */
const ClusterCard: React.FC<{
  cluster: FactCluster;
  isExpanded: boolean;
  onToggle: () => void;
  onViewPdf?: (docId: string, page: number, quote?: string) => void;
}> = ({ cluster, isExpanded, onToggle, onViewPdf }) => {
  const evidence = cluster.evidence || [];
  const edges = cluster.edges || [];
  const isReconciled = cluster.case_type.startsWith('reconciled');
  const isContradicted = cluster.case_type === 'contradicted';
  const disputeMeta = getDisputeMeta(cluster.dispute_code);

  // Border accent based on verdict
  let borderAccent = 'border-l-neutral-300 dark:border-l-neutral-700';
  if (isContradicted) borderAccent = 'border-l-red-500 dark:border-l-red-600';
  else if (isReconciled) borderAccent = 'border-l-blue-500 dark:border-l-blue-600';
  else if (cluster.case_type === 'corroborated') borderAccent = 'border-l-teal-500 dark:border-l-teal-600';

  return (
    <div
      className={`border border-neutral-300 dark:border-neutral-700 border-l-4 ${borderAccent} bg-white dark:bg-neutral-900`}
    >
      {/* Header row — always visible */}
      <button
        onClick={onToggle}
        className="w-full p-3 flex items-center justify-between gap-3 text-left hover:bg-neutral-50 dark:hover:bg-neutral-850 transition-colors cursor-pointer"
      >
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-neutral-900 dark:text-neutral-100">
              {cluster.subject}
            </span>
            <span className="text-[11px] font-mono text-neutral-500">
              {formatMetricName(cluster.predicate)}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <VerdictBadge verdict={cluster.case_type} size="sm" />
            <DisputeBadge disputeCode={cluster.dispute_code} />
            {/* Reconciled: show plain-language reason inline */}
            {isReconciled && (
              <span className="text-[11px] font-mono text-blue-800 dark:text-blue-300">
                — {disputeMeta.description}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <span className="text-[10px] font-mono text-neutral-400">
            {cluster.doc_count ?? evidence.length} doc{(cluster.doc_count ?? evidence.length) !== 1 ? 's' : ''} • {evidence.length} fact{evidence.length !== 1 ? 's' : ''} • {edges.length} edge{edges.length !== 1 ? 's' : ''}
          </span>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-neutral-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-neutral-400" />
          )}
        </div>
      </button>

      {/* Expanded detail */}
      {isExpanded && (
        <div className="border-t border-neutral-200 dark:border-neutral-800 p-4 space-y-4">
          {/* Evidence display — adaptive by document count */}
          {evidence.length <= 2 ? (
            <TwoDocSideBySide
              evidence={evidence}
              edges={edges}
              onViewPdf={onViewPdf}
            />
          ) : (
            <MultiDocTable
              evidence={evidence}
              onViewPdf={onViewPdf}
            />
          )}

          {/* Edge reasoning */}
          {edges.length > 0 && (
            <div className="pt-3 border-t border-neutral-200 dark:border-neutral-800">
              <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-500 mb-2">
                Evidence Graph Edges ({edges.length})
              </div>
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {edges.map((edge, i) => (
                  <div
                    key={i}
                    className="p-2 border border-neutral-200 dark:border-neutral-800 text-[11px] font-mono"
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className={`font-semibold uppercase tracking-wider text-[10px] ${
                          edge.edge_type === 'contradicts'
                            ? 'text-red-600'
                            : edge.edge_type === 'corroborates'
                            ? 'text-teal-600'
                            : 'text-blue-600'
                        }`}
                      >
                        {edge.edge_type} ({edge.detection_method})
                      </span>
                      <span className="text-neutral-400 text-[10px]">
                        {formatConfidence(edge.confidence)}
                      </span>
                    </div>
                    <p className="text-neutral-700 dark:text-neutral-300 mt-0.5">
                      {edge.explanation}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Cluster explanation */}
          {cluster.explanation && (
            <div className="p-3 bg-neutral-50 dark:bg-neutral-850 border border-neutral-200 dark:border-neutral-800 text-[11px] font-mono text-neutral-800 dark:text-neutral-200 leading-relaxed">
              {cluster.explanation}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/* ─────────── Two-Document Side-by-Side Card ─────────── */
const TwoDocSideBySide: React.FC<{
  evidence: EvidenceEntry[];
  edges: { edge_type: string; explanation: string }[];
  onViewPdf?: (docId: string, page: number, quote?: string) => void;
}> = ({ evidence, edges, onViewPdf }) => {
  const [showPdfExhibits, setShowPdfExhibits] = useState(true);

  if (evidence.length === 0) return null;

  // Determine the connector label from the dominant edge type
  const edgeTypes = edges.map((e) => e.edge_type);
  const hasContradicts = edgeTypes.includes('contradicts');
  const hasSupersedes = edgeTypes.includes('supersedes');
  const hasCorroborates = edgeTypes.includes('corroborates');

  let connectorLabel = 'vs';
  let connectorColor = 'text-neutral-500 border-neutral-300 dark:border-neutral-600';
  if (hasContradicts) {
    connectorLabel = 'contradicts';
    connectorColor = 'text-red-600 border-red-400 dark:border-red-600';
  } else if (hasSupersedes) {
    connectorLabel = 'supersedes';
    connectorColor = 'text-blue-600 border-blue-400 dark:border-blue-600';
  } else if (hasCorroborates) {
    connectorLabel = 'corroborates';
    connectorColor = 'text-teal-600 border-teal-400 dark:border-teal-600';
  }

  const left = evidence[0];
  const right = evidence.length > 1 ? evidence[1] : null;

  return (
    <div className="space-y-3">
      {/* Top summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr] gap-0 items-stretch">
        {/* Left side */}
        <EvidenceCard entry={left} onViewPdf={onViewPdf} />

        {/* Connector */}
        <div className="hidden sm:flex flex-col items-center justify-center px-3">
          <div className={`border-l-2 flex-1 ${connectorColor}`} />
          <div
            className={`my-1 px-2 py-0.5 text-[10px] font-mono font-semibold uppercase tracking-wider border ${connectorColor}`}
          >
            {connectorLabel}
          </div>
          <div className={`border-l-2 flex-1 ${connectorColor}`} />
        </div>
        {/* Mobile connector */}
        <div className="sm:hidden flex items-center justify-center py-1">
          <span
            className={`px-2 py-0.5 text-[10px] font-mono font-semibold uppercase border ${connectorColor}`}
          >
            {connectorLabel}
          </span>
        </div>

        {/* Right side */}
        {right ? (
          <EvidenceCard entry={right} onViewPdf={onViewPdf} />
        ) : (
          <div className="border border-dashed border-neutral-200 dark:border-neutral-800 p-3 flex items-center justify-center text-[11px] font-mono text-neutral-400">
            Single source
          </div>
        )}
      </div>

      {/* In-place PDF Exhibit Toggle */}
      <div className="flex items-center justify-between pt-1 border-t border-dashed border-neutral-200 dark:border-neutral-800 text-xs font-mono">
        <span className="text-neutral-500 text-[11px]">
          Inbuilt Source PDF Verification:
        </span>
        <button
          onClick={() => setShowPdfExhibits((v) => !v)}
          className="px-2.5 py-1 text-[11px] font-mono border border-neutral-300 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-850 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors flex items-center gap-1.5 cursor-pointer text-neutral-800 dark:text-neutral-200"
        >
          <Eye className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
          <span>{showPdfExhibits ? 'Hide PDF Exhibits' : 'View In-Place PDF Exhibits Side-by-Side'}</span>
        </button>
      </div>

      {/* In-place Side-by-Side PDF Exhibits */}
      {showPdfExhibits && left && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
          <InlinePdfViewer
            docId={left.doc_id}
            pageNum={left.page ?? 0}
            docFilename={left.doc_filename}
            highlightQuote={left.evidence_quote}
            highlightValue={left.value}
            maxHeight="380px"
          />
          {right && (
            <InlinePdfViewer
              docId={right.doc_id}
              pageNum={right.page ?? 0}
              docFilename={right.doc_filename}
              highlightQuote={right.evidence_quote}
              highlightValue={right.value}
              maxHeight="380px"
            />
          )}
        </div>
      )}
    </div>
  );
};

/* ─────────── Evidence Card (for side-by-side) ─────────── */
const EvidenceCard: React.FC<{
  entry: EvidenceEntry;
  onViewPdf?: (docId: string, page: number, quote?: string) => void;
}> = ({ entry, onViewPdf }) => {
  return (
    <div className="border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 p-3 space-y-2">
      {/* Document + page */}
      <div className="flex items-center justify-between gap-2">
        <button
          onClick={() =>
            onViewPdf?.(entry.doc_id, entry.page ?? 1, entry.evidence_quote)
          }
          className="flex items-center gap-1 text-[11px] font-mono text-neutral-700 dark:text-neutral-300 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors cursor-pointer truncate"
          title={`View ${entry.doc_filename} page ${entry.page ?? '—'} in source PDF`}
        >
          <FileText className="w-3 h-3 text-neutral-400 shrink-0" />
          <span className="truncate">{entry.doc_filename}</span>
          <span className="text-neutral-400 shrink-0">(p.{entry.page ?? '—'})</span>
        </button>
      </div>

      {/* Value */}
      <div className="text-sm font-mono font-bold text-neutral-900 dark:text-neutral-100">
        {entry.value}
        {entry.canonical_unit && (
          <span className="text-neutral-500 font-normal text-xs ml-1.5">
            {entry.canonical_unit}
          </span>
        )}
      </div>

      {/* Context qualifiers */}
      <div className="flex flex-wrap gap-1.5 text-[10px] font-mono">
        {entry.temporal && (
          <span className="px-1.5 py-0.5 border border-neutral-200 dark:border-neutral-700 text-neutral-600 dark:text-neutral-400">
            {entry.temporal}
          </span>
        )}
        {entry.scope && (
          <span className="px-1.5 py-0.5 border border-neutral-200 dark:border-neutral-700 text-neutral-600 dark:text-neutral-400">
            {entry.scope}
          </span>
        )}
      </div>

      {/* Verbatim quote */}
      <p className="text-[11px] font-mono text-neutral-500 italic leading-relaxed line-clamp-3">
        &ldquo;{entry.evidence_quote}&rdquo;
      </p>

      {/* Confidence */}
      <div className="flex items-center justify-between text-[10px] font-mono text-neutral-400">
        <span>Confidence: {formatConfidence(entry.confidence)}</span>
        <span className="uppercase">{entry.match_type}</span>
      </div>
    </div>
  );
};

/* ─────────── Multi-Document Table (3+ docs) ─────────── */
const MultiDocTable: React.FC<{
  evidence: EvidenceEntry[];
  onViewPdf?: (docId: string, page: number, quote?: string) => void;
}> = ({ evidence, onViewPdf }) => {
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [showViewer, setShowViewer] = useState<boolean>(true);
  const [compareMode, setCompareMode] = useState<boolean>(false);
  const [compareIdxA, setCompareIdxA] = useState<number>(0);
  const [compareIdxB, setCompareIdxB] = useState<number>(1);
  const viewerRef = useRef<HTMLDivElement>(null);

  // Determine majority value for outlier detection
  const values = evidence
    .map((e) => e.canonical_value)
    .filter((v): v is number => v !== null && v !== undefined);

  let majorityValue: number | null = null;
  if (values.length >= 2) {
    // Simple mode detection: find most common value (within 5% tolerance)
    const groups: { val: number; count: number }[] = [];
    for (const v of values) {
      const existing = groups.find(
        (g) => Math.abs(g.val - v) / Math.max(Math.abs(g.val), 1) < 0.05
      );
      if (existing) {
        existing.count++;
      } else {
        groups.push({ val: v, count: 1 });
      }
    }
    groups.sort((a, b) => b.count - a.count);
    if (groups.length > 1 && groups[0].count > groups[1].count) {
      majorityValue = groups[0].val;
    }
  }

  const isOutlier = (entry: EvidenceEntry): boolean => {
    if (majorityValue === null || entry.canonical_value === null || entry.canonical_value === undefined)
      return false;
    const diff =
      Math.abs(entry.canonical_value - majorityValue) /
      Math.max(Math.abs(majorityValue), 1);
    return diff > 0.05 && entry.confidence < 0.7;
  };

  const activeEntry = evidence[selectedIdx] || evidence[0];

  return (
    <div className="space-y-3">
      {/* Evidence Table */}
      <div className="border border-neutral-300 dark:border-neutral-700 overflow-x-auto max-h-64 overflow-y-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead className="sticky top-0 z-10">
            <tr className="border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 font-mono text-[11px] text-neutral-500 uppercase tracking-wider">
              <th className="py-2 px-3">Document</th>
              <th className="py-2 px-3">Page</th>
              <th className="py-2 px-3">Value</th>
              <th className="py-2 px-3">Period</th>
              <th className="py-2 px-3">Confidence</th>
              <th className="py-2 px-3 text-right">PDF Exhibit</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800/80">
            {evidence.map((entry, idx) => {
              const outlier = isOutlier(entry);
              const isSelected = selectedIdx === idx;
              return (
                <tr
                  key={idx}
                  onClick={() => {
                    setSelectedIdx(idx);
                    setShowViewer(true);
                    viewerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                  }}
                  className={`cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-blue-50/80 dark:bg-blue-950/40 border-l-4 border-l-blue-600 dark:border-l-blue-400 font-medium'
                      : outlier
                      ? 'bg-amber-50/50 dark:bg-amber-950/20 hover:bg-neutral-50 dark:hover:bg-neutral-850'
                      : 'hover:bg-neutral-50 dark:hover:bg-neutral-850'
                  }`}
                >
                  <td className="py-2 px-3">
                    <div className="flex items-center gap-1 text-[11px] font-mono text-neutral-800 dark:text-neutral-200 truncate max-w-[220px]">
                      <FileText className="w-3 h-3 text-neutral-400 shrink-0" />
                      <span className="truncate">{entry.doc_filename}</span>
                    </div>
                  </td>
                  <td className="py-2 px-3 font-mono text-neutral-500">
                    p.{entry.page ?? '—'}
                  </td>
                  <td className="py-2 px-3 font-mono font-semibold text-neutral-900 dark:text-neutral-100">
                    {entry.value}
                    {outlier && (
                      <span className="ml-1.5 text-[10px] font-normal text-amber-700 dark:text-amber-400">
                        ⚠ outlier
                      </span>
                    )}
                  </td>
                  <td className="py-2 px-3 font-mono text-neutral-500 text-[11px]">
                    {entry.temporal || '—'}
                  </td>
                  <td className="py-2 px-3 font-mono text-neutral-500 text-[11px]">
                    {formatConfidence(entry.confidence)}
                  </td>
                  <td className="py-2 px-3 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedIdx(idx);
                          setShowViewer(true);
                          if (onViewPdf) {
                            onViewPdf(entry.doc_id, entry.page ?? 0, entry.evidence_quote);
                          } else {
                            viewerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                          }
                        }}
                        className={`px-2 py-0.5 text-[10px] font-mono border transition-colors inline-flex items-center gap-1 cursor-pointer ${
                          isSelected && showViewer
                            ? 'border-blue-600 bg-blue-600 text-white dark:border-blue-400 dark:bg-blue-500'
                            : 'border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-600 dark:text-neutral-400'
                        }`}
                        title="Open full page in modal dialog"
                      >
                        <Eye className="w-2.5 h-2.5" />
                        <span>Inspect</span>
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Inbuilt PDF Segment Control Strip */}
      <div
        ref={viewerRef}
        className="p-2.5 bg-neutral-50 dark:bg-neutral-850 border border-neutral-200 dark:border-neutral-800 flex flex-wrap items-center justify-between gap-2 text-xs font-mono"
      >
        <div className="flex items-center gap-1.5 overflow-x-auto max-w-full pb-1">
          <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-wider mr-1 shrink-0">
            Inbuilt PDF Segment:
          </span>
          {evidence.map((ev, i) => (
            <button
              key={i}
              onClick={() => {
                setSelectedIdx(i);
                setShowViewer(true);
                if (compareMode) {
                  setCompareIdxA(i);
                }
              }}
              className={`px-2 py-1 text-[11px] border transition-colors flex items-center gap-1 cursor-pointer ${
                selectedIdx === i && showViewer && !compareMode
                  ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 font-bold border-neutral-900 dark:border-neutral-100'
                  : 'border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-700 dark:text-neutral-300'
              }`}
            >
              <span className="font-semibold text-neutral-400">#{i + 1}</span>
              <span className="truncate max-w-[120px]">{ev.doc_filename}</span>
              <span className="text-neutral-400">p.{ev.page ?? '—'}</span>
              <span className="font-bold text-neutral-900 dark:text-neutral-100 ml-0.5">({ev.value})</span>
            </button>
          ))}
        </div>

        {/* View mode toggle controls */}
        <div className="flex items-center gap-2">
          {evidence.length >= 2 && (
            <button
              onClick={() => {
                setCompareMode((c) => !c);
                setShowViewer(true);
              }}
              className={`px-2.5 py-1 text-[11px] font-mono border transition-colors flex items-center gap-1.5 cursor-pointer ${
                compareMode
                  ? 'bg-blue-600 text-white border-blue-600 dark:bg-blue-500 dark:border-blue-500 font-semibold'
                  : 'border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-700 dark:text-neutral-300'
              }`}
            >
              <Columns className="w-3 h-3" />
              <span>{compareMode ? 'Comparing 2 Sources' : 'Compare 2 Sources'}</span>
            </button>
          )}

          <button
            onClick={() => setShowViewer((v) => !v)}
            className="px-2 py-1 text-[11px] font-mono border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors flex items-center gap-1 cursor-pointer text-neutral-700 dark:text-neutral-300"
          >
            <Eye className="w-3 h-3" />
            <span>{showViewer ? 'Hide Exhibit' : 'Show Exhibit'}</span>
          </button>

          {onViewPdf && (
            <button
              onClick={() =>
                onViewPdf(
                  activeEntry.doc_id,
                  activeEntry.page ?? 0,
                  activeEntry.evidence_quote
                )
              }
              className="px-2 py-1 text-[11px] font-mono border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors flex items-center gap-1 cursor-pointer text-neutral-600 dark:text-neutral-400"
              title="Open full page in modal dialog"
            >
              <ExternalLink className="w-3 h-3" />
              <span>Fullscreen</span>
            </button>
          )}
        </div>
      </div>

      {/* Inbuilt PDF Segment Exhibit Viewer Body */}
      {showViewer && (
        <div className="pt-1">
          {!compareMode ? (
            /* Single Source Exhibit View */
            <InlinePdfViewer
              key={`${activeEntry.doc_id}-${activeEntry.page}`}
              docId={activeEntry.doc_id}
              pageNum={activeEntry.page ?? 0}
              docFilename={activeEntry.doc_filename}
              highlightQuote={activeEntry.evidence_quote}
              highlightValue={activeEntry.value}
              maxHeight="400px"
              onClose={() => setShowViewer(false)}
              showCloseButton={true}
            />
          ) : (
            /* Side-by-Side Comparison of 2 Sources */
            <div className="space-y-2">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* Left Source */}
                <div className="space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] font-mono px-1">
                    <span className="font-bold text-blue-700 dark:text-blue-400 shrink-0">Source Exhibit A:</span>
                    <select
                      value={compareIdxA}
                      onChange={(e) => setCompareIdxA(Number(e.target.value))}
                      className="px-2 py-0.5 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-850 text-neutral-800 dark:text-neutral-200 text-[11px] font-mono cursor-pointer max-w-[260px] truncate"
                    >
                      {evidence.map((ev, i) => (
                        <option key={i} value={i}>
                          #{i + 1}: {ev.doc_filename} (p.{ev.page ?? '—'}) — {ev.value}
                        </option>
                      ))}
                    </select>
                  </div>
                  {evidence[compareIdxA] && (
                    <InlinePdfViewer
                      key={`cmpA-${evidence[compareIdxA].doc_id}-${evidence[compareIdxA].page}`}
                      docId={evidence[compareIdxA].doc_id}
                      pageNum={evidence[compareIdxA].page ?? 0}
                      docFilename={evidence[compareIdxA].doc_filename}
                      highlightQuote={evidence[compareIdxA].evidence_quote}
                      highlightValue={evidence[compareIdxA].value}
                      maxHeight="380px"
                    />
                  )}
                </div>

                {/* Right Source */}
                <div className="space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] font-mono px-1">
                    <span className="font-bold text-red-700 dark:text-red-400 shrink-0">Source Exhibit B:</span>
                    <select
                      value={compareIdxB}
                      onChange={(e) => setCompareIdxB(Number(e.target.value))}
                      className="px-2 py-0.5 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-850 text-neutral-800 dark:text-neutral-200 text-[11px] font-mono cursor-pointer max-w-[260px] truncate"
                    >
                      {evidence.map((ev, i) => (
                        <option key={i} value={i}>
                          #{i + 1}: {ev.doc_filename} (p.{ev.page ?? '—'}) — {ev.value}
                        </option>
                      ))}
                    </select>
                  </div>
                  {evidence[compareIdxB] && (
                    <InlinePdfViewer
                      key={`cmpB-${evidence[compareIdxB].doc_id}-${evidence[compareIdxB].page}`}
                      docId={evidence[compareIdxB].doc_id}
                      pageNum={evidence[compareIdxB].page ?? 0}
                      docFilename={evidence[compareIdxB].doc_filename}
                      highlightQuote={evidence[compareIdxB].evidence_quote}
                      highlightValue={evidence[compareIdxB].value}
                      maxHeight="380px"
                    />
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
