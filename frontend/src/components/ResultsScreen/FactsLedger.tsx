import React, { useMemo, useState } from 'react';
import { Search, Download, FileText, ChevronRight } from 'lucide-react';
import type { Fact } from '../../api/types';
import { VerificationBadge } from '../common/VerificationBadge';
import { FactDetailModal } from './FactDetailModal';
import { formatMetricName, formatValue, formatCanonical } from '../../utils/formatters';

interface FactsLedgerProps {
  facts: Fact[];
  docFilenames?: Record<string, string>;
  onViewPdf?: (docId: string, page: number, quote?: string) => void;
}

export const FactsLedger: React.FC<FactsLedgerProps> = ({
  facts,
  docFilenames = {},
  onViewPdf,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDoc, setSelectedDoc] = useState<string>('all');
  const [verificationFilter, setVerificationFilter] = useState<string>('all');
  const [selectedFact, setSelectedFact] = useState<Fact | null>(null);

  // Available unique documents in facts list
  const uniqueDocIds = useMemo(() => {
    const ids = new Set<string>();
    facts.forEach((f) => {
      if (f.provenance.doc_id) ids.add(f.provenance.doc_id);
    });
    return Array.from(ids);
  }, [facts]);

  // Filtered facts
  const filteredFacts = useMemo(() => {
    return facts.filter((f) => {
      // Search match
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        const subjectMatch = f.subject.toLowerCase().includes(query);
        const predicateMatch = f.predicate.toLowerCase().includes(query);
        const valueMatch = f.value.toLowerCase().includes(query);
        const quoteMatch = f.provenance.evidence_quote.toLowerCase().includes(query);
        if (!subjectMatch && !predicateMatch && !valueMatch && !quoteMatch) {
          return false;
        }
      }

      // Doc filter
      if (selectedDoc !== 'all' && f.provenance.doc_id !== selectedDoc) {
        return false;
      }

      // Verification status filter
      if (verificationFilter !== 'all') {
        const matchType = (f.provenance.match_type || 'unverified').toLowerCase();
        if (verificationFilter === 'unverified') {
          if (f.provenance.verified && matchType !== 'unverified') return false;
        } else if (matchType !== verificationFilter) {
          return false;
        }
      }

      return true;
    });
  }, [facts, searchQuery, selectedDoc, verificationFilter]);

  // Export to CSV helper
  const handleExportCsv = () => {
    const headers = [
      'Subject',
      'Predicate',
      'Reported Value',
      'Canonical Value',
      'Canonical Unit',
      'Scale',
      'Dimension',
      'Temporal',
      'Scope',
      'Conditions',
      'Confidence',
      'Verified',
      'Match Type',
      'Doc ID',
      'Page',
      'Evidence Quote',
    ];

    const rows = filteredFacts.map((f) => [
      `"${(f.subject || '').replace(/"/g, '""')}"`,
      `"${(f.predicate || '').replace(/"/g, '""')}"`,
      `"${(f.value || '').replace(/"/g, '""')}"`,
      f.canonical_value !== null && f.canonical_value !== undefined ? f.canonical_value : '',
      `"${(f.canonical_unit || '').replace(/"/g, '""')}"`,
      f.scale ?? 0,
      `"${(f.dimension || '').replace(/"/g, '""')}"`,
      `"${(f.context.temporal || '').replace(/"/g, '""')}"`,
      `"${(f.context.scope || '').replace(/"/g, '""')}"`,
      `"${(f.context.conditions || '').replace(/"/g, '""')}"`,
      f.confidence,
      f.provenance.verified ? 'TRUE' : 'FALSE',
      `"${f.provenance.match_type || ''}"`,
      `"${f.provenance.doc_id || ''}"`,
      f.provenance.page !== null && f.provenance.page !== undefined ? f.provenance.page : '',
      `"${(f.provenance.evidence_quote || '').replace(/"/g, '""')}"`,
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `extracted_facts_ledger_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-4">
      {/* Control Bar: Search & Filter Toolbar */}
      <div className="p-3 bg-white dark:bg-neutral-900 border border-neutral-300 dark:border-neutral-700 flex flex-wrap items-center justify-between gap-3 text-xs">
        {/* Search */}
        <div className="relative flex-1 min-w-[220px]">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter by subject, metric, value, or quote text..."
            className="w-full pl-9 pr-3 py-1.5 border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-850 text-neutral-900 dark:text-neutral-100 placeholder-neutral-400 focus:outline-hidden focus:border-neutral-500 font-mono text-xs"
          />
        </div>

        {/* Dropdown Filters */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Document filter */}
          {uniqueDocIds.length > 1 && (
            <select
              value={selectedDoc}
              onChange={(e) => setSelectedDoc(e.target.value)}
              className="px-2.5 py-1.5 border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-850 text-neutral-800 dark:text-neutral-200 font-mono text-xs"
            >
              <option value="all">All Documents ({facts.length})</option>
              {uniqueDocIds.map((id) => (
                <option key={id} value={id}>
                  {docFilenames[id] || id.slice(0, 12)}
                </option>
              ))}
            </select>
          )}

          {/* Verification Status filter */}
          <select
            value={verificationFilter}
            onChange={(e) => setVerificationFilter(e.target.value)}
            className="px-2.5 py-1.5 border border-neutral-200 dark:border-neutral-750 bg-neutral-50 dark:bg-neutral-850 text-neutral-800 dark:text-neutral-200 font-mono text-xs"
          >
            <option value="all">All Grounding States</option>
            <option value="exact">Exact Match Only</option>
            <option value="normalized">Normalized Match</option>
            <option value="fuzzy">Fuzzy Match</option>
            <option value="unverified">Unverified Only</option>
          </select>

          {/* Export CSV Button */}
          <button
            onClick={handleExportCsv}
            disabled={filteredFacts.length === 0}
            className="px-3 py-1.5 border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-800 dark:text-neutral-200 font-mono text-xs inline-flex items-center gap-1.5 disabled:opacity-40"
          >
            <Download className="w-3 h-3" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Stats summary bar */}
      <div className="flex items-center justify-between text-[11px] font-mono text-neutral-500 px-1">
        <div>
          Showing {filteredFacts.length} of {facts.length} facts
        </div>
        {facts.some((f) => !f.provenance.verified || f.provenance.match_type === 'unverified') && (
          <div className="text-amber-700 dark:text-amber-400">
            Notice: Low-confidence/unverified quotes are visibly surfaced below
          </div>
        )}
      </div>

      {/* Ledger Table */}
      <div className="border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-850 font-mono text-[11px] text-neutral-500 uppercase tracking-wider">
              <th className="py-2.5 px-3">Subject</th>
              <th className="py-2.5 px-3">Metric (Predicate)</th>
              <th className="py-2.5 px-3">Reported Value</th>
              <th className="py-2.5 px-3">Canonical Value</th>
              <th className="py-2.5 px-3">Source & Page</th>
              <th className="py-2.5 px-3">Grounding State</th>
              <th className="py-2.5 px-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800/80">
            {filteredFacts.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-neutral-400 font-mono text-xs">
                  No facts found matching filter criteria.
                </td>
              </tr>
            ) : (
              filteredFacts.map((fact, idx) => {
                const docName = docFilenames[fact.provenance.doc_id] || fact.provenance.doc_id.slice(0, 10);
                return (
                  <tr
                    key={idx}
                    onClick={() => setSelectedFact(fact)}
                    className="hover:bg-neutral-50 dark:hover:bg-neutral-850 cursor-pointer transition-colors"
                  >
                    <td className="py-2 px-3 font-medium text-neutral-900 dark:text-neutral-100">
                      {fact.subject}
                    </td>
                    <td className="py-2 px-3 font-mono text-neutral-700 dark:text-neutral-300">
                      {formatMetricName(fact.predicate)}
                    </td>
                    <td className="py-2 px-3 font-mono font-semibold text-neutral-900 dark:text-neutral-100">
                      {formatValue(fact.value)}
                    </td>
                    <td className="py-2 px-3 font-mono text-neutral-500">
                      {formatCanonical(fact.canonical_value, fact.canonical_unit) || '—'}
                    </td>
                    <td className="py-2 px-3 text-neutral-600 dark:text-neutral-400">
                      <div className="flex items-center gap-1 font-mono text-[11px] truncate max-w-[180px]" title={docName}>
                        <FileText className="w-3 h-3 text-neutral-400 shrink-0" />
                        <span className="truncate">{docName}</span>
                        <span>(p.{fact.provenance.page ?? '—'})</span>
                      </div>
                    </td>
                    <td className="py-2 px-3">
                      <VerificationBadge
                        verified={fact.provenance.verified}
                        matchType={fact.provenance.match_type}
                        confidence={fact.confidence}
                      />
                    </td>
                    <td className="py-2 px-3 text-right text-neutral-400">
                      <ChevronRight className="w-4 h-4 inline-block" />
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Detail Slide-over / Modal on Row Click */}
      <FactDetailModal
        fact={selectedFact}
        onClose={() => setSelectedFact(null)}
        onViewPdf={onViewPdf}
      />
    </div>
  );
};
