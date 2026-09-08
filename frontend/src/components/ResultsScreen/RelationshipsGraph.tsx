import React, { useState } from 'react';
import type { FactCluster } from '../../api/types';
import { VerdictBadge } from '../common/VerdictBadge';
import { DisputeBadge } from '../common/DisputeBadge';
import { formatMetricName } from '../../utils/formatters';

interface RelationshipsGraphProps {
  clusters: FactCluster[];
}

export const RelationshipsGraph: React.FC<RelationshipsGraphProps> = ({ clusters }) => {
  const [selectedCluster, setSelectedCluster] = useState<FactCluster | null>(
    clusters[0] || null
  );

  const getClusterEdges = (c: FactCluster | null | undefined) =>
    c ? c.edges || (c as any).corroborating_edges || [] : [];
  const getClusterEvidence = (c: FactCluster | null | undefined) =>
    c ? c.evidence || [] : [];

  if (clusters.length === 0) {
    return (
      <div className="p-8 text-center border border-dashed border-neutral-300 dark:border-neutral-700 text-neutral-500 font-mono text-xs">
        No relational claim clusters found. Run cross-document reconciliation to view evidence relationships.
      </div>
    );
  }

  const currentEdges = getClusterEdges(selectedCluster);
  const currentEvidence = getClusterEvidence(selectedCluster);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div>
          <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">
            Cross-Document Claim Network
          </h3>
          <p className="text-neutral-500 text-[11px] mt-0.5">
            Supplementary topological view of assertion clusters and typed evidence edges.
          </p>
        </div>
        <div className="flex items-center gap-3 font-mono text-[11px]">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-teal-600 inline-block" /> Corroborates
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-red-600 inline-block" /> Contradicts
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-600 inline-block" /> Reconciled
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column: List of Claim Clusters */}
        <div className="lg:col-span-1 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 divide-y divide-neutral-200 dark:divide-neutral-800 max-h-[550px] overflow-y-auto">
          <div className="p-3 bg-neutral-50 dark:bg-neutral-850 font-mono text-[11px] text-neutral-500 uppercase tracking-wider font-semibold">
            Assertion Clusters ({clusters.length})
          </div>

          {clusters.map((cluster) => {
            const isSelected = selectedCluster?.cluster_id === cluster.cluster_id;
            const edges = getClusterEdges(cluster);
            const evidence = getClusterEvidence(cluster);
            return (
              <div
                key={cluster.cluster_id}
                onClick={() => setSelectedCluster(cluster)}
                className={`p-3 cursor-pointer transition-colors text-xs ${
                  isSelected
                    ? 'bg-neutral-100 dark:bg-neutral-800 border-l-4 border-l-neutral-900 dark:border-l-neutral-100'
                    : 'hover:bg-neutral-50 dark:hover:bg-neutral-850'
                }`}
              >
                <div className="flex items-start justify-between gap-1">
                  <span className="font-semibold text-neutral-900 dark:text-neutral-100">
                    {cluster.subject}
                  </span>
                  <VerdictBadge verdict={cluster.case_type} size="sm" />
                </div>
                <div className="font-mono text-[11px] text-neutral-600 dark:text-neutral-400 mt-1">
                  {formatMetricName(cluster.predicate)}
                </div>
                <div className="mt-2 flex items-center justify-between text-[10px] font-mono text-neutral-400">
                  <span>{cluster.doc_count ?? evidence.length} docs • {evidence.length} facts</span>
                  <span>{edges.length} edges</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Right Column: Selected Cluster Evidence & Edges Details */}
        <div className="lg:col-span-2 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 p-5 space-y-4">
          {selectedCluster ? (
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-neutral-200 dark:border-neutral-800">
                <div>
                  <span className="text-xs font-mono text-neutral-400 uppercase">
                    Cluster {selectedCluster.cluster_id}
                  </span>
                  <h4 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 mt-0.5">
                    {selectedCluster.subject} — {formatMetricName(selectedCluster.predicate)}
                  </h4>
                </div>
                <div className="flex items-center gap-2">
                  <VerdictBadge verdict={selectedCluster.case_type} />
                  <DisputeBadge disputeCode={selectedCluster.dispute_code} />
                </div>
              </div>

              {/* Cluster Explanation */}
              <div className="mt-3 p-3 bg-neutral-50 dark:bg-neutral-850 border border-neutral-200 dark:border-neutral-800 text-xs font-mono text-neutral-800 dark:text-neutral-200 leading-relaxed">
                {selectedCluster.explanation}
              </div>

              {/* Edges List */}
              <div className="mt-4">
                <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-500 mb-2">
                  Evaluated Evidence Graph Edges ({currentEdges.length})
                </div>

                {currentEdges.length === 0 ? (
                  <div className="p-3 text-neutral-400 font-mono text-xs border border-dashed border-neutral-200">
                    No intra-cluster conflict edges evaluated.
                  </div>
                ) : (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {currentEdges.map((edge, i) => (
                      <div
                        key={i}
                        className="p-2.5 border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 text-xs font-mono space-y-1"
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
                            Confidence: {(edge.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-neutral-700 dark:text-neutral-300 text-[11px]">
                          {edge.explanation}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Side-by-side Evidence sources */}
              <div className="mt-4 pt-3 border-t border-neutral-200 dark:border-neutral-800">
                <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-500 mb-2">
                  Supporting Document Evidence ({currentEvidence.length})
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                  {currentEvidence.map((ev, idx) => (
                    <div
                      key={idx}
                      className="p-2.5 border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 font-mono text-[11px] space-y-1"
                    >
                      <div className="font-semibold text-neutral-900 dark:text-neutral-100 truncate" title={ev.doc_filename}>
                        {ev.doc_filename} (p.{ev.page ?? '—'})
                      </div>
                      <div className="text-neutral-800 dark:text-neutral-200 font-bold">
                        {ev.value}
                      </div>
                      <p className="text-neutral-500 text-[10.5px] italic line-clamp-2">
                        &ldquo;{ev.evidence_quote}&rdquo;
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="p-12 text-center text-neutral-400 font-mono text-xs">
              Select an assertion cluster on the left to inspect its evidence edges.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
